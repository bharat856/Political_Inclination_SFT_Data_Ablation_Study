from pathlib import Path
import argparse
import json
from itertools import combinations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


LABELS = ["left", "neutral", "right"]

LABEL_MAP = {
    "1": "left",
    "1.0": "left",
    "left": "left",
    "l": "left",

    "2": "neutral",
    "2.0": "neutral",
    "neutral": "neutral",
    "center": "neutral",
    "centre": "neutral",
    "centrist": "neutral",
    "n": "neutral",

    "3": "right",
    "3.0": "right",
    "right": "right",
    "r": "right",
}

ID_CANDIDATES = [
    "validation_id",
    "prompt_id",
    "source_row_index",
    "row_id",
    "benchmark_id",
    "id",
]

PREDICTION_CANDIDATES = [
    "parsed_prediction_label",
    "prediction_label",
    "predicted_label",
    "pred_label",
    "prediction",
    "label_pred",
    "slant_prediction",
    "final_prediction",
    "pred_3way_dmitry",
    "nli_pred_3way",
    "sft_political_label",
    "sft_political_label_id",
    "parsed_prediction_num",
]

GOLD_CANDIDATES = [
    "gold_label",
    "slantlabel_final",
    "label",
    "gold",
    "target",
    "rile_label_3way",
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Model specification NAME=PATH. "
            "Supply this exactly three times."
        ),
    )

    parser.add_argument(
        "--fallback-model",
        default=None,
        help=(
            "Model used when all three models predict different labels. "
            "Defaults to the first model supplied."
        ),
    )

    parser.add_argument(
        "--gold-file",
        default="data/validation_tanise/slant_testset_for_4way.csv",
    )

    parser.add_argument(
        "--gold-label-column",
        default=None,
    )

    parser.add_argument(
        "--gold-id-column",
        default=None,
    )

    parser.add_argument(
        "--align-by-row-order",
        action="store_true",
        help="Ignore file IDs and align all files using their row positions.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/validation_tanise/ensemble_v4/ensemble_evaluation",
    )

    return parser.parse_args()


def normalize_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in LABEL_MAP:
        return LABEL_MAP[text]

    # Handle strings such as "1. LEFT" or "Prediction: 3".
    for token, label in [
        ("1", "left"),
        ("2", "neutral"),
        ("3", "right"),
    ]:
        if text == token or text.startswith(token + "."):
            return label

    return None


def parse_model_spec(spec):
    if "=" not in spec:
        raise ValueError(
            f"Invalid --model specification: {spec!r}. "
            "Expected NAME=PATH."
        )

    name, path = spec.split("=", 1)
    name = name.strip()
    path = Path(path.strip())

    if not name:
        raise ValueError(f"Empty model name in {spec!r}")

    if not path.exists():
        raise FileNotFoundError(path)

    return name, path


def choose_column(df, candidates, explicit=None, purpose="column"):
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(
                f"Requested {purpose} {explicit!r} is missing. "
                f"Columns: {df.columns.tolist()}"
            )
        return explicit

    for column in candidates:
        if column in df.columns:
            return column

    return None


def prepare_gold(
    path,
    label_column=None,
    id_column=None,
    align_by_row_order=False,
):
    df = pd.read_csv(path).copy()

    gold_col = choose_column(
        df,
        GOLD_CANDIDATES,
        explicit=label_column,
        purpose="gold-label column",
    )

    if gold_col is None:
        raise KeyError(
            f"Could not identify the gold-label column in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    id_col = None if align_by_row_order else choose_column(
        df,
        ID_CANDIDATES,
        explicit=id_column,
        purpose="gold ID column",
    )

    if id_col is None:
        print(
            "Warning: no stable ID found in the gold file; "
            "using source row order."
        )
        df["ensemble_id"] = [
            f"row_{i:06d}" for i in range(len(df))
        ]
    else:
        df["ensemble_id"] = df[id_col].astype(str)

    df["gold_label"] = df[gold_col].map(normalize_label)

    invalid = df["gold_label"].isna().sum()
    if invalid:
        raise ValueError(
            f"{invalid} gold labels could not be normalized "
            f"from column {gold_col!r}."
        )

    if df["ensemble_id"].duplicated().any():
        duplicates = df.loc[
            df["ensemble_id"].duplicated(),
            "ensemble_id",
        ].head(10).tolist()
        raise ValueError(
            f"Duplicate gold IDs found: {duplicates}"
        )

    keep = ["ensemble_id", "gold_label"]

    # Preserve useful source fields where available.
    for column in [
        "prompt_id",
        "content",
        "full_text",
        "source",
    ]:
        if column in df.columns and column not in keep:
            keep.append(column)

    return df[keep].copy()


def prepare_predictions(
    name,
    path,
    gold,
    align_by_row_order=False,
):
    df = pd.read_csv(path).copy()

    pred_col = choose_column(
        df,
        PREDICTION_CANDIDATES,
        purpose=f"{name} prediction column",
    )

    if pred_col is None:
        raise KeyError(
            f"Could not identify prediction column for {name} in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    id_col = None if align_by_row_order else choose_column(
        df,
        ID_CANDIDATES,
        purpose=f"{name} ID column",
    )

    if id_col is None:
        if len(df) != len(gold):
            raise ValueError(
                f"{name} has no stable ID and has {len(df)} rows, "
                f"but gold has {len(gold)} rows."
            )

        print(
            f"Warning: {name} has no stable ID; "
            "aligning predictions by row order."
        )

        df["ensemble_id"] = gold["ensemble_id"].tolist()
    else:
        df["ensemble_id"] = df[id_col].astype(str)

    output_column = f"pred_{name}"

    df[output_column] = df[pred_col].map(normalize_label)

    if df["ensemble_id"].duplicated().any():
        duplicates = df.loc[
            df["ensemble_id"].duplicated(),
            "ensemble_id",
        ].head(10).tolist()
        raise ValueError(
            f"{name} contains duplicate IDs: {duplicates}"
        )

    result = df[["ensemble_id", output_column]].copy()

    parse_failures = result[output_column].isna().sum()

    print(
        f"{name}: rows={len(result):,}, "
        f"prediction_column={pred_col!r}, "
        f"parse_failures={parse_failures:,}"
    )

    return result


def evaluate_predictions(gold, pred):
    valid = gold.notna() & pred.notna()
    y_true = gold[valid]
    y_pred = pred[valid]

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    summary = {
        "evaluated_rows": int(valid.sum()),
        "missing_predictions": int((~valid).sum()),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "weighted_f1": float(
            report["weighted avg"]["f1-score"]
        ),
        "per_class": {
            label: {
                "precision": float(
                    report[label]["precision"]
                ),
                "recall": float(
                    report[label]["recall"]
                ),
                "f1": float(
                    report[label]["f1-score"]
                ),
                "support": int(
                    report[label]["support"]
                ),
            }
            for label in LABELS
        },
    }

    matrix_df = pd.DataFrame(
        matrix,
        index=[f"gold_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )

    return summary, matrix_df


def majority_vote(row, model_names, fallback_model):
    predictions = [
        row[f"pred_{name}"]
        for name in model_names
        if pd.notna(row[f"pred_{name}"])
    ]

    if not predictions:
        return None, "no_predictions"

    counts = pd.Series(predictions).value_counts()

    if counts.iloc[0] >= 2:
        if counts.iloc[0] == 3:
            return counts.index[0], "unanimous"
        return counts.index[0], "two_to_one"

    # This occurs when all three models predict different classes.
    fallback = row.get(f"pred_{fallback_model}")

    if pd.notna(fallback):
        return fallback, f"all_different_fallback_{fallback_model}"

    return None, "all_different_no_fallback"


def main():
    args = parse_args()

    model_specs = [
        parse_model_spec(spec)
        for spec in args.model
    ]

    if len(model_specs) != 3:
        raise ValueError(
            "Supply exactly three --model NAME=PATH arguments."
        )

    model_names = [name for name, _ in model_specs]

    if len(set(model_names)) != len(model_names):
        raise ValueError(
            f"Model names must be unique: {model_names}"
        )

    fallback_model = args.fallback_model or model_names[0]

    if fallback_model not in model_names:
        raise ValueError(
            f"Fallback model {fallback_model!r} is not one of "
            f"{model_names}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = prepare_gold(
        Path(args.gold_file),
        label_column=args.gold_label_column,
        id_column=args.gold_id_column,
        align_by_row_order=args.align_by_row_order,
    )

    merged = gold.copy()

    for name, path in model_specs:
        prediction_df = prepare_predictions(
            name,
            path,
            gold,
            align_by_row_order=args.align_by_row_order,
        )

        merged = merged.merge(
            prediction_df,
            on="ensemble_id",
            how="left",
            validate="one_to_one",
        )

    results = {
        "gold_file": args.gold_file,
        "rows": int(len(merged)),
        "models": model_names,
        "fallback_model": fallback_model,
        "individual_models": {},
        "pairwise_agreement": {},
        "ensemble": {},
    }

    print()
    print("=" * 100)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 100)

    for name in model_names:
        pred_col = f"pred_{name}"

        summary, matrix = evaluate_predictions(
            merged["gold_label"],
            merged[pred_col],
        )

        results["individual_models"][name] = summary

        matrix.to_csv(
            output_dir / f"confusion_matrix_{name}.csv"
        )

        print(
            f"{name:20s} "
            f"accuracy={summary['accuracy']:.4f}  "
            f"macro-F1={summary['macro_f1']:.4f}  "
            f"missing={summary['missing_predictions']:,}"
        )

    print()
    print("=" * 100)
    print("PAIRWISE AGREEMENT")
    print("=" * 100)

    for first, second in combinations(model_names, 2):
        col_a = f"pred_{first}"
        col_b = f"pred_{second}"

        valid = merged[col_a].notna() & merged[col_b].notna()

        agreement = float(
            (
                merged.loc[valid, col_a]
                == merged.loc[valid, col_b]
            ).mean()
        )

        key = f"{first}__{second}"

        results["pairwise_agreement"][key] = {
            "rows": int(valid.sum()),
            "agreement": agreement,
        }

        print(
            f"{first:15s} vs {second:15s}: "
            f"{agreement:.4f}"
        )

    ensemble_outputs = merged.apply(
        lambda row: majority_vote(
            row,
            model_names,
            fallback_model,
        ),
        axis=1,
    )

    merged["ensemble_prediction"] = [
        item[0] for item in ensemble_outputs
    ]

    merged["ensemble_decision_type"] = [
        item[1] for item in ensemble_outputs
    ]

    ensemble_summary, ensemble_matrix = evaluate_predictions(
        merged["gold_label"],
        merged["ensemble_prediction"],
    )

    decision_counts = (
        merged["ensemble_decision_type"]
        .value_counts(dropna=False)
        .to_dict()
    )

    ensemble_summary["decision_counts"] = {
        str(key): int(value)
        for key, value in decision_counts.items()
    }

    results["ensemble"] = ensemble_summary

    ensemble_matrix.to_csv(
        output_dir / "confusion_matrix_ensemble.csv"
    )

    merged.to_csv(
        output_dir / "ensemble_predictions.csv",
        index=False,
    )

    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    comparison_rows = []

    for name in model_names:
        model_result = results["individual_models"][name]

        comparison_rows.append({
            "method": name,
            "accuracy": model_result["accuracy"],
            "macro_f1": model_result["macro_f1"],
            "weighted_f1": model_result["weighted_f1"],
            "missing_predictions": (
                model_result["missing_predictions"]
            ),
        })

    comparison_rows.append({
        "method": "ensemble",
        "accuracy": ensemble_summary["accuracy"],
        "macro_f1": ensemble_summary["macro_f1"],
        "weighted_f1": ensemble_summary["weighted_f1"],
        "missing_predictions": (
            ensemble_summary["missing_predictions"]
        ),
    })

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    )

    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("ENSEMBLE RESULT")
    print("=" * 100)

    print(
        f"Fallback model: {fallback_model}"
    )

    print(
        f"Accuracy:       {ensemble_summary['accuracy']:.4f}"
    )

    print(
        f"Macro-F1:       {ensemble_summary['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1:    {ensemble_summary['weighted_f1']:.4f}"
    )

    print("\nDecision types:")
    for key, value in decision_counts.items():
        print(f"  {key}: {value:,}")

    print()
    print("=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)
    print(comparison.to_string(index=False))

    print()
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()
