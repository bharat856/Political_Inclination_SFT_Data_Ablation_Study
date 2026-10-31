from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT_DIR = Path("outputs/validation_tanise/slant_testset_llama_4way")
PRED = OUT_DIR / "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit.csv"

df = pd.read_csv(PRED)

gold_map = {
    "left": "left",
    "neutral": "center_neutral",
    "right": "right",
}

df["gold_3way"] = df["label"].astype(str).str.strip().str.lower().map(gold_map)
df["pred_4way"] = df["sft_political_label"]

target_labels = ["left", "center_neutral", "right"]

# Main evaluation: count non_political as wrong, because every row has a left/neutral/right gold label.
df["pred_3way_strict"] = df["pred_4way"]
strict_acc = accuracy_score(df["gold_3way"], df["pred_3way_strict"])

strict_report = classification_report(
    df["gold_3way"],
    df["pred_3way_strict"],
    labels=target_labels + ["non_political"],
    digits=4,
    zero_division=0,
)

strict_cm = confusion_matrix(
    df["gold_3way"],
    df["pred_3way_strict"],
    labels=target_labels + ["non_political"],
)

# Parsed-target-only evaluation: ignore rows predicted non_political.
eval_df = df[df["pred_4way"].isin(target_labels)].copy()
parsed_acc = accuracy_score(eval_df["gold_3way"], eval_df["pred_4way"])

parsed_report = classification_report(
    eval_df["gold_3way"],
    eval_df["pred_4way"],
    labels=target_labels,
    digits=4,
    zero_division=0,
)

parsed_cm = confusion_matrix(
    eval_df["gold_3way"],
    eval_df["pred_4way"],
    labels=target_labels,
)

print("Rows:", len(df))
print("Pred distribution:")
print(df["pred_4way"].value_counts(dropna=False))

print("\nGold distribution:")
print(df["gold_3way"].value_counts(dropna=False))

print("\nCrosstab gold vs pred:")
print(pd.crosstab(df["gold_3way"], df["pred_4way"], dropna=False))

print("\nSTRICT 3-way evaluation, counting non_political as wrong")
print("Accuracy:", strict_acc)
print("Labels:", target_labels + ["non_political"])
print(strict_cm)
print(strict_report)

print("\nTarget-label-only evaluation, excluding non_political predictions")
print("Rows:", len(eval_df), "/", len(df))
print("Accuracy:", parsed_acc)
print("Labels:", target_labels)
print(parsed_cm)
print(parsed_report)

out_csv = OUT_DIR / "slant_testset_llama_4way_evaluated.csv"
df.to_csv(out_csv, index=False)

summary = {
    "rows": int(len(df)),
    "pred_distribution": df["pred_4way"].value_counts(dropna=False).astype(int).to_dict(),
    "gold_distribution": df["gold_3way"].value_counts(dropna=False).astype(int).to_dict(),
    "crosstab_gold_vs_pred": pd.crosstab(df["gold_3way"], df["pred_4way"], dropna=False).to_dict(),
    "strict_accuracy_counting_non_political_wrong": float(strict_acc),
    "strict_confusion_matrix_labels": target_labels + ["non_political"],
    "strict_confusion_matrix": strict_cm.tolist(),
    "strict_classification_report": strict_report,
    "target_label_only_rows": int(len(eval_df)),
    "target_label_only_accuracy": float(parsed_acc),
    "target_label_only_confusion_matrix_labels": target_labels,
    "target_label_only_confusion_matrix": parsed_cm.tolist(),
    "target_label_only_classification_report": parsed_report,
    "output_csv": str(out_csv),
}

out_json = OUT_DIR / "slant_testset_llama_4way_eval_summary.json"
out_json.write_text(json.dumps(summary, indent=2))

print("\nSaved:", out_csv)
print("Saved:", out_json)
