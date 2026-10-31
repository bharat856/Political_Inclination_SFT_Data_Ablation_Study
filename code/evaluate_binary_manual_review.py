from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_binary.csv")

df = pd.read_csv(IN)

eval_df = df[df["manual_binary_label"].notna() & df["pred_binary_label"].notna()].copy()

labels = ["non_political", "political"]

print("Rows evaluated:", len(eval_df))
print("Accuracy:", accuracy_score(eval_df["manual_binary_label"], eval_df["pred_binary_label"]))

print("\nConfusion matrix:")
print(pd.DataFrame(
    confusion_matrix(eval_df["manual_binary_label"], eval_df["pred_binary_label"], labels=labels),
    index=[f"gold_{x}" for x in labels],
    columns=[f"pred_{x}" for x in labels],
))

print("\nClassification report:")
print(classification_report(
    eval_df["manual_binary_label"],
    eval_df["pred_binary_label"],
    labels=labels,
    zero_division=0,
))
