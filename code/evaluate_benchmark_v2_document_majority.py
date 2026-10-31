import pandas as pd
from pathlib import Path

PRED_DIR = Path("outputs/original_classifier_benchmark_v2")
OUT_DIR = PRED_DIR / "document_level"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(PRED_DIR.glob("original_classifier__*.csv"))

summary_rows = []

for path in files:
    df = pd.read_csv(path)

    required = {"doc_id", "rile_label_3way", "parsed_prediction_label"}
    missing = required - set(df.columns)
    if missing:
        print(f"Skipping {path.name}, missing columns: {missing}")
        continue

    model_name = path.stem.replace("original_classifier__", "")

    doc_rows = []

    for doc_id, g in df.groupby("doc_id"):
        gold = g["rile_label_3way"].iloc[0]

        pred_counts = g["parsed_prediction_label"].value_counts()
        pred = pred_counts.index[0]

        doc_rows.append({
            "doc_id": doc_id,
            "gold_label": gold,
            "predicted_label": pred,
            "n_chunks": len(g),
            "pred_left_chunks": int((g["parsed_prediction_label"] == "left").sum()),
            "pred_center_chunks": int((g["parsed_prediction_label"] == "center").sum()),
            "pred_right_chunks": int((g["parsed_prediction_label"] == "right").sum()),
        })

    doc_df = pd.DataFrame(doc_rows)
    doc_df["correct"] = doc_df["gold_label"] == doc_df["predicted_label"]

    out_path = OUT_DIR / f"document_level__{model_name}.csv"
    doc_df.to_csv(out_path, index=False)

    acc = doc_df["correct"].mean()

    summary_rows.append({
        "model": model_name,
        "document_majority_accuracy": acc,
        "documents": len(doc_df),
        "doc_gold_left": int((doc_df["gold_label"] == "left").sum()),
        "doc_gold_center": int((doc_df["gold_label"] == "center").sum()),
        "doc_gold_right": int((doc_df["gold_label"] == "right").sum()),
        "doc_pred_left": int((doc_df["predicted_label"] == "left").sum()),
        "doc_pred_center": int((doc_df["predicted_label"] == "center").sum()),
        "doc_pred_right": int((doc_df["predicted_label"] == "right").sum()),
    })

    print("\nMODEL:", model_name)
    print("Documents:", len(doc_df))
    print("Document-majority accuracy:", round(acc, 4))
    print("\nDocument-level confusion:")
    print(pd.crosstab(doc_df["gold_label"], doc_df["predicted_label"], rownames=["gold"], colnames=["pred"]))

summary = pd.DataFrame(summary_rows).sort_values("document_majority_accuracy", ascending=False)
summary_path = OUT_DIR / "document_level_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSaved summary:", summary_path)
print("\nSummary:")
print(summary.to_string(index=False))
