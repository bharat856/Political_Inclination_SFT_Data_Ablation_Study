import pandas as pd
from pathlib import Path

PRED_DIR = Path("outputs/original_classifier_benchmark_v2")
OUT_DIR = PRED_DIR / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LLAMA = PRED_DIR / "original_classifier__fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3.3-70B-Instruct-bnb-4bit.csv"
GEMMA = PRED_DIR / "original_classifier__cyankiwi_gemma-4-31B-it-AWQ-8bit.csv"
QWEN = PRED_DIR / "original_classifier__cyankiwi_Qwen3.5-27B-AWQ-BF16-INT8.csv"

def load(path, prefix):
    df = pd.read_csv(path)
    keep = [
        "benchmark_id", "chunk_id", "doc_id", "chunk_index", "text",
        "rile_label_3way", "rile", "countryname", "partyname", "date",
        "parsed_prediction_label", "raw_output"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df = df.rename(columns={
        "parsed_prediction_label": f"{prefix}_pred",
        "raw_output": f"{prefix}_raw_output"
    })
    return df

llama = load(LLAMA, "llama")
gemma = load(GEMMA, "gemma")
qwen = load(QWEN, "qwen")

df = llama.merge(
    gemma[["benchmark_id", "gemma_pred", "gemma_raw_output"]],
    on="benchmark_id",
    how="inner"
)

if QWEN.exists():
    df = df.merge(
        qwen[["benchmark_id", "qwen_pred", "qwen_raw_output"]],
        on="benchmark_id",
        how="left"
    )

gold = df["rile_label_3way"]

df["llama_correct"] = df["llama_pred"] == gold
df["gemma_correct"] = df["gemma_pred"] == gold
df["qwen_correct"] = df["qwen_pred"] == gold if "qwen_pred" in df.columns else False
df["llama_gemma_same"] = df["llama_pred"] == df["gemma_pred"]

# Main subsets
subsets = {
    "llama_correct_gemma_wrong": df[(df["llama_correct"]) & (~df["gemma_correct"])],
    "gemma_correct_llama_wrong": df[(df["gemma_correct"]) & (~df["llama_correct"])],
    "llama_gemma_both_wrong": df[(~df["llama_correct"]) & (~df["gemma_correct"])],
    "llama_gemma_disagree": df[df["llama_pred"] != df["gemma_pred"]],
    "llama_gemma_same_wrong": df[(df["llama_pred"] == df["gemma_pred"]) & (~df["llama_correct"])],
}

for name, sub in subsets.items():
    out = OUT_DIR / f"benchmark_v2__{name}.csv"
    sub.to_csv(out, index=False)
    print(f"Saved {name}: {len(sub)} rows -> {out}")

# Summary
summary_rows = []

summary_rows.append({
    "metric": "total_rows",
    "value": len(df)
})
summary_rows.append({
    "metric": "llama_correct",
    "value": int(df["llama_correct"].sum())
})
summary_rows.append({
    "metric": "gemma_correct",
    "value": int(df["gemma_correct"].sum())
})
summary_rows.append({
    "metric": "qwen_correct",
    "value": int(df["qwen_correct"].sum()) if "qwen_pred" in df.columns else None
})
summary_rows.append({
    "metric": "llama_gemma_same_prediction",
    "value": int(df["llama_gemma_same"].sum())
})
summary_rows.append({
    "metric": "llama_gemma_different_prediction",
    "value": int((~df["llama_gemma_same"]).sum())
})
summary_rows.append({
    "metric": "llama_correct_gemma_wrong",
    "value": len(subsets["llama_correct_gemma_wrong"])
})
summary_rows.append({
    "metric": "gemma_correct_llama_wrong",
    "value": len(subsets["gemma_correct_llama_wrong"])
})
summary_rows.append({
    "metric": "both_wrong",
    "value": len(subsets["llama_gemma_both_wrong"])
})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_DIR / "benchmark_v2_error_analysis_summary.csv", index=False)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nCorrectness comparison:")
print(pd.crosstab(df["gemma_correct"], df["llama_correct"], rownames=["Gemma correct"], colnames=["Llama correct"]))

print("\nLlama correct, Gemma wrong by gold:")
print(subsets["llama_correct_gemma_wrong"]["rile_label_3way"].value_counts())

print("\nGemma correct, Llama wrong by gold:")
print(subsets["gemma_correct_llama_wrong"]["rile_label_3way"].value_counts())

print("\nBoth wrong by gold:")
print(subsets["llama_gemma_both_wrong"]["rile_label_3way"].value_counts())

# Create small human-readable samples
sample_dir = OUT_DIR / "samples"
sample_dir.mkdir(exist_ok=True)

for name, sub in subsets.items():
    sample = sub.sample(n=min(20, len(sub)), random_state=42) if len(sub) else sub
    sample_out = sample_dir / f"{name}_sample_20.csv"
    sample.to_csv(sample_out, index=False)

print("\nSaved 20-row samples in:", sample_dir)
