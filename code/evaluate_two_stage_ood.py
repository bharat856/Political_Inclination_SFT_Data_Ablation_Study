import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABEL_NAMES = {
    0: "non_political",
    1: "left",
    2: "center_neutral",
    3: "right",
}


def normalize_fourway(value):
    s = str(value).strip().lower()
    if s in {"", "nan", "none", "u", "unclear"}:
        return None
    if s.endswith(".0"):
        s = s[:-2]

    mapping = {
        "0": 0,
        "non_political": 0,
        "non-political": 0,
        "non political": 0,
        "1": 1,
        "left": 1,
        "2": 2,
        "center": 2,
        "centre": 2,
        "neutral": 2,
        "center_neutral": 2,
        "center/neutral": 2,
        "3": 3,
        "right": 3,
    }
    return mapping.get(s)


def normalize_binary_from_fourway(value):
    y = normalize_fourway(value)
    if y is None:
        return None
    return 0 if y == 0 else 1


def find_first(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def merge_predictions(manual_df, pred_path, suffix):
    pred_df = pd.read_csv(pred_path)
    key = "row_id" if "row_id" in manual_df.columns and "row_id" in pred_df.columns else None

    if key:
        keep = [key] + [c for c in pred_df.columns if c != key]
        return manual_df.merge(pred_df[keep], on=key, how="left", suffixes=("", suffix))

    if len(pred_df) != len(manual_df):
        raise ValueError(
            f"Cannot align {pred_path}: no row_id key and row counts differ "
            f"({len(pred_df)} vs {len(manual_df)})."
        )
    joined = manual_df.reset_index(drop=True).copy()
    for col in pred_df.columns:
        joined[f"{col}{suffix}"] = pred_df[col].reset_index(drop=True)
    return joined


def metric_block(y_true, y_pred, labels, target_names):
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def print_metrics(title, y_true, y_pred, labels, target_names):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print("Rows:", len(y_true))
    if not len(y_true):
        return
    print("Accuracy:", accuracy_score(y_true, y_pred))
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"gold_{x}" for x in target_names],
        columns=[f"pred_{x}" for x in target_names],
    )
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stage-1 ModernBERT and optional full two-stage OOD behavior."
    )
    parser.add_argument(
        "--manual",
        default="outputs/modernbert_ood_test/manual_review_ood_sample_300_labeled.csv",
    )
    parser.add_argument(
        "--stage1",
        default="outputs/modernbert_ood_test/modernbert_ood_predictions.csv",
    )
    parser.add_argument(
        "--stage2",
        default=None,
        help="Optional 4-way SFT classifier predictions for the same rows.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/modernbert_ood_test/two_stage_eval",
    )
    args = parser.parse_args()

    manual_path = Path(args.manual)
    stage1_path = Path(args.stage1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manual_path)
    df["gold_4way_id"] = df["manual_label"].apply(normalize_fourway)
    df["gold_binary_id"] = df["manual_label"].apply(normalize_binary_from_fourway)
    df = df[df["gold_4way_id"].notna()].copy()
    df["gold_4way_id"] = df["gold_4way_id"].astype(int)
    df["gold_binary_id"] = df["gold_binary_id"].astype(int)
    df["gold_4way_label"] = df["gold_4way_id"].map(LABEL_NAMES)
    df["gold_binary_label"] = df["gold_binary_id"].map({0: "non_political", 1: "political"})

    if stage1_path.exists():
        df = merge_predictions(df, stage1_path, suffix="_stage1")
    else:
        raise FileNotFoundError(f"Stage-1 prediction file not found: {stage1_path}")

    stage1_col = find_first(df, ["modernbert_pred_binary", "binary_pred_id", "pred_label"])
    if stage1_col is None:
        raise ValueError(
            "Could not find a stage-1 prediction column. Expected one of "
            "modernbert_pred_binary, binary_pred_id, pred_label."
        )
    df["stage1_binary_pred"] = df[stage1_col].astype(float).astype(int)
    df["stage1_binary_label"] = df["stage1_binary_pred"].map({0: "non_political", 1: "political"})

    summary = {
        "manual": str(manual_path),
        "stage1_predictions": str(stage1_path),
        "stage2_predictions": args.stage2,
        "rows_with_gold": int(len(df)),
        "gold_4way_distribution": df["gold_4way_label"].value_counts().to_dict(),
        "stage1_pred_distribution": df["stage1_binary_label"].value_counts().to_dict(),
    }

    print("Rows with usable manual labels:", len(df))
    print("\nGold 4-way distribution:")
    print(df["gold_4way_label"].value_counts())
    print("\nStage-1 prediction distribution:")
    print(df["stage1_binary_label"].value_counts())

    y_true_binary = df["gold_binary_id"]
    y_pred_binary = df["stage1_binary_pred"]
    print_metrics(
        "Stage 1: ModernBERT political gate",
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=["non_political", "political"],
    )
    summary["stage1_metrics"] = metric_block(
        y_true_binary,
        y_pred_binary,
        labels=[0, 1],
        target_names=["non_political", "political"],
    )

    stage2_path = Path(args.stage2) if args.stage2 else None
    if stage2_path:
        if not stage2_path.exists():
            raise FileNotFoundError(f"Stage-2 prediction file not found: {stage2_path}")
        df = merge_predictions(df, stage2_path, suffix="_stage2")
        stage2_col = find_first(df, [
            "sft_political_label_id_reparsed",
            "sft_political_label_id",
            "our_4way_pred",
            "stage2_pred_id",
        ])
        if stage2_col is None:
            label_col = find_first(df, [
                "sft_political_label_reparsed",
                "sft_political_label",
                "our_4way_label",
                "stage2_pred_label",
            ])
            if label_col is None:
                raise ValueError("Could not find a 4-way stage-2 prediction column.")
            df["stage2_4way_pred"] = df[label_col].apply(normalize_fourway)
        else:
            df["stage2_4way_pred"] = df[stage2_col].apply(normalize_fourway)

        has_stage2 = df["stage2_4way_pred"].notna()
        df = df[has_stage2].copy()
        df["stage2_4way_pred"] = df["stage2_4way_pred"].astype(int)
        df["stage2_4way_label"] = df["stage2_4way_pred"].map(LABEL_NAMES)

        df["two_stage_pred"] = df["stage2_4way_pred"]
        df.loc[df["stage1_binary_pred"] == 0, "two_stage_pred"] = 0
        df["two_stage_label"] = df["two_stage_pred"].map(LABEL_NAMES)
        df["two_stage_correct"] = df["two_stage_pred"] == df["gold_4way_id"]

        print("\nStage-2 prediction distribution:")
        print(df["stage2_4way_label"].value_counts())
        print("\nFinal two-stage prediction distribution:")
        print(df["two_stage_label"].value_counts())

        y_true_4way = df["gold_4way_id"]
        y_pred_4way = df["two_stage_pred"]
        print_metrics(
            "Full pipeline: stage-1 gate + stage-2 4-way label",
            y_true_4way,
            y_pred_4way,
            labels=[0, 1, 2, 3],
            target_names=["non_political", "left", "center_neutral", "right"],
        )
        summary["two_stage_metrics"] = metric_block(
            y_true_4way,
            y_pred_4way,
            labels=[0, 1, 2, 3],
            target_names=["non_political", "left", "center_neutral", "right"],
        )
    else:
        print("\nNo --stage2 file supplied, so full two-stage final-label metrics were skipped.")

    predictions_path = output_dir / "two_stage_ood_predictions.csv"
    summary_path = output_dir / "two_stage_ood_summary.json"
    df.to_csv(predictions_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved predictions:", predictions_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
