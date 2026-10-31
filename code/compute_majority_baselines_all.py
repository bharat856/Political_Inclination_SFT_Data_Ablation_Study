from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


DATASETS = [
    {
        "name": "20newsgroups",
        "path": Path(
            "outputs/validation_external/binary_union_same_rows/"
            "20newsgroups_binary_union_same_rows_predictions.csv"
        ),
        "label_column": "gold_binary",
    },
    {
        "name": "bbc_news",
        "path": Path(
            "outputs/validation_external/binary_union_same_rows/"
            "bbc_news_binary_union_same_rows_predictions.csv"
        ),
        "label_column": "gold_binary",
    },
    {
        "name": "manual_target_domain",
        "path": Path(
            "outputs/validation_target_domain/manual_binary_gate_comparison/"
            "manual_target_domain_binary_gate_predictions.csv"
        ),
        "label_column": "gold_binary",
    },
    {
        "name": "tanise_slant_900",
        "path": Path(
            "data/validation_tanise/slant_testset_for_4way.csv"
        ),
        "label_column": "label",
    },
]

OUT_DIR = Path("outputs/validation_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_baseline(name: str, path: Path, label_column: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if label_column not in df.columns:
        raise KeyError(
            f"{label_column!r} not found in {path}. "
            f"Columns: {df.columns.tolist()}"
        )

    y_true = df[label_column].dropna()

    if y_true.empty:
        raise ValueError(f"No labels found in {path}")

    counts = y_true.value_counts()
    majority_class = counts.index[0]
    majority_count = int(counts.iloc[0])

    y_pred = pd.Series(
        [majority_class] * len(y_true),
        index=y_true.index,
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "dataset": name,
        "input": str(path),
        "label_column": label_column,
        "rows": int(len(df)),
        "evaluated_rows": int(len(y_true)),
        "missing_labels": int(df[label_column].isna().sum()),
        "majority_class": str(majority_class),
        "majority_count": majority_count,
        "majority_accuracy": float(accuracy),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "label_distribution": {
            str(label): int(count)
            for label, count in counts.items()
        },
    }


def main():
    results = []

    for cfg in DATASETS:
        result = compute_baseline(
            name=cfg["name"],
            path=cfg["path"],
            label_column=cfg["label_column"],
        )
        results.append(result)

        out_path = OUT_DIR / f"{cfg['name']}_majority_baseline.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("=" * 100)
        print(cfg["name"])
        print("=" * 100)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Saved:", out_path)

    summary_path = OUT_DIR / "majority_baselines_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"],
            "rows": result["evaluated_rows"],
            "majority_class": result["majority_class"],
            "majority_count": result["majority_count"],
            "accuracy": result["majority_accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
        })

    summary_csv = OUT_DIR / "majority_baselines_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print("\nSaved combined JSON:", summary_path)
    print("Saved combined CSV:", summary_csv)


if __name__ == "__main__":
    main()
