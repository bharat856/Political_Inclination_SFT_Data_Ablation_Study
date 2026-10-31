from pathlib import Path
import pandas as pd

IN = Path("outputs/tulu_olmo2_sft_political_classification_sample/sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit__reparsed.csv")
OUT = Path("outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")

df = pd.read_csv(IN)

label_col = "sft_political_label_reparsed"

parts = []

# sample across predicted classes
for label, n in [
    ("non_political", 50),
    ("center_neutral", 50),
    ("left", 50),
    ("right", 40),
]:
    g = df[df[label_col] == label]
    if len(g) > 0:
        parts.append(g.sample(n=min(n, len(g)), random_state=42))

# add still unparsed
unparsed = df[df["parse_ok_reparsed"] == False]
if len(unparsed) > 0:
    parts.append(unparsed.sample(n=min(20, len(unparsed)), random_state=42))

review = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

review["manual_label"] = ""
review["manual_notes"] = ""
review["text_preview"] = review["full_text"].astype(str).str.replace("\n", " ", regex=False).str.slice(0, 1800)

keep = [
    "row_id",
    "id",
    "source",
    "strong_hits",
    "issue_hits",
    "context_hits",
    "sft_political_raw_output",
    "sft_political_label_reparsed",
    "manual_label",
    "manual_notes",
    "text_preview",
]

review[keep].to_csv(OUT, index=False)

print("Saved:", OUT)
print("Rows:", len(review))
print("\nPredicted label distribution:")
print(review["sft_political_label_reparsed"].value_counts(dropna=False))
