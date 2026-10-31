from pathlib import Path
import re
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path("outputs/validation_tanise/slant_testset_llama_3way_dmitry_prompt")
PRED = OUT_DIR / "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.csv"

df = pd.read_csv(PRED)

gold_map = {
    "left": "left",
    "neutral": "center_neutral",
    "right": "right",
}

num_to_label = {
    "1": "left",
    "2": "center_neutral",
    "3": "right",
}

def parse_raw(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    m = re.search(r"\b([1-3])\b", x)
    if not m:
        return None
    return num_to_label[m.group(1)]

df["gold_3way"] = df["label"].astype(str).str.strip().str.lower().map(gold_map)
df["pred_3way_dmitry"] = df["sft_political_raw_output"].apply(parse_raw)

labels = ["left", "center_neutral", "right"]

ok = df["pred_3way_dmitry"].isin(labels)
eval_df = df[ok].copy()

print("Rows:", len(df))
print("Parsed from raw:", len(eval_df), "/", len(df))

print("\nRaw output distribution:")
print(df["sft_political_raw_output"].value_counts(dropna=False).head(20))

print("\nPred distribution:")
print(df["pred_3way_dmitry"].value_counts(dropna=False))

print("\nGold distribution:")
print(df["gold_3way"].value_counts(dropna=False))

print("\nCrosstab gold vs pred:")
print(pd.crosstab(df["gold_3way"], df["pred_3way_dmitry"], dropna=False))

acc = accuracy_score(eval_df["gold_3way"], eval_df["pred_3way_dmitry"])
cm = confusion_matrix(eval_df["gold_3way"], eval_df["pred_3way_dmitry"], labels=labels)
report = classification_report(
    eval_df["gold_3way"],
    eval_df["pred_3way_dmitry"],
    labels=labels,
    digits=4,
    zero_division=0,
)

print("\nDmitry prompt 3-way accuracy:", acc)
print("\nConfusion matrix labels:", labels)
print(cm)
print("\nReport:")
print(report)

out_csv = OUT_DIR / "slant_testset_llama_3way_dmitry_prompt_evaluated_from_raw.csv"
df.to_csv(out_csv, index=False)

summary = {
    "rows": int(len(df)),
    "parsed_from_raw": int(len(eval_df)),
    "accuracy": float(acc),
    "labels": labels,
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "pred_distribution": df["pred_3way_dmitry"].value_counts(dropna=False).astype(int).to_dict(),
    "gold_distribution": df["gold_3way"].value_counts(dropna=False).astype(int).to_dict(),
    "output_csv": str(out_csv),
}

out_json = OUT_DIR / "slant_testset_llama_3way_dmitry_prompt_eval_summary.json"
out_json.write_text(json.dumps(summary, indent=2))

print("Saved:", out_csv)
print("Saved:", out_json)
