from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")

df = pd.read_csv(IN, dtype={"manual_label": "string"})

pred_col = "sft_political_label_reparsed"
if pred_col not in df.columns:
    pred_col = "sft_political_label"

label_map = {
    "0": "non_political",
    "1": "left",
    "2": "center_neutral",
    "3": "right",
    0: "non_political",
    1: "left",
    2: "center_neutral",
    3: "right",
    "non_political": "non_political",
    "left": "left",
    "center": "center_neutral",
    "center_neutral": "center_neutral",
    "neutral": "center_neutral",
    "right": "right",
}

df["manual_label_norm"] = df["manual_label"].map(label_map)
df["pred_norm"] = df[pred_col].map(label_map)

eval_df = df[df["manual_label_norm"].notna()].copy()
eval_df = eval_df[eval_df["pred_norm"].notna()].copy()

labels = ["non_political", "center_neutral", "left", "right"]

print("Input file:", IN)
print("Total rows:", len(df))
print("Rows with manual labels excluding U/unmapped:", df["manual_label_norm"].notna().sum())
print("Rows evaluated after excluding NaN predictions:", len(eval_df))

print("\nManual label distribution:")
print(df["manual_label"].value_counts(dropna=False))

print("\nManual normalized distribution:")
print(df["manual_label_norm"].value_counts(dropna=False))

print("\nPrediction distribution:")
print(df[pred_col].value_counts(dropna=False))

print("\nPrediction normalized distribution:")
print(df["pred_norm"].value_counts(dropna=False))

print("\nAccuracy:")
print(accuracy_score(eval_df["manual_label_norm"], eval_df["pred_norm"]))

print("\nConfusion matrix:")
cm = pd.DataFrame(
    confusion_matrix(
        eval_df["manual_label_norm"],
        eval_df["pred_norm"],
        labels=labels,
    ),
    index=[f"gold_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels],
)
print(cm)

print("\nClassification report:")
print(classification_report(
    eval_df["manual_label_norm"],
    eval_df["pred_norm"],
    labels=labels,
    zero_division=0,
))

# Save error cases for inspection
out_dir = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_eval")
out_dir.mkdir(parents=True, exist_ok=True)

errors = eval_df[eval_df["manual_label_norm"] != eval_df["pred_norm"]].copy()
errors.to_csv(out_dir / "manual_eval_errors.csv", index=False)

print("\nSaved errors:")
print(out_dir / "manual_eval_errors.csv")
