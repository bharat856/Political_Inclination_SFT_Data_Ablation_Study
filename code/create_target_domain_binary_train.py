from pathlib import Path
import pandas as pd

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
OUT = Path("data/processed/tulu_olmo2_sft/target_domain_binary_manual_train.csv")

df = pd.read_csv(IN, dtype={"manual_label": "string"})

def to_binary_id(x):
    x = str(x)
    if x == "0":
        return 0
    if x in {"1", "2", "3"}:
        return 1
    return None

df["label"] = df["manual_label"].map(to_binary_id)

# Use text_preview for now because that is what manual labels were based on
if "text_preview" in df.columns:
    df["text"] = df["text_preview"]
elif "full_text" in df.columns:
    df["text"] = df["full_text"]
else:
    raise ValueError("No text column found")

out = df[df["label"].notna()][["row_id", "text", "label", "manual_label", "manual_notes"]].copy()
out["label"] = out["label"].astype(int)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(out))
print(out["label"].value_counts())
