from pathlib import Path
import pandas as pd

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
OUT = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200_binary.csv")

df = pd.read_csv(IN, dtype={"manual_label": "string"})

def to_binary(x):
    x = str(x)
    if x == "0":
        return "non_political"
    if x in {"1", "2", "3"}:
        return "political"
    return None

df["manual_binary_label"] = df["manual_label"].map(to_binary)

# Convert Llama 4-way prediction to binary
def pred_to_binary(x):
    x = str(x)
    if x == "non_political":
        return "non_political"
    if x in {"left", "center_neutral", "right"}:
        return "political"
    return None

df["pred_binary_label"] = df["sft_political_label_reparsed"].map(pred_to_binary)

df.to_csv(OUT, index=False)

print("Saved:", OUT)
print()
print("Manual binary distribution:")
print(df["manual_binary_label"].value_counts(dropna=False))
print()
print("Pred binary distribution:")
print(df["pred_binary_label"].value_counts(dropna=False))
