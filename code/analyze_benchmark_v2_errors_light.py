import pandas as pd
from pathlib import Path

PRED_DIR = Path("outputs/original_classifier_benchmark_v2")
OUT_DIR = PRED_DIR / "analysis"
SAMPLE_DIR = OUT_DIR / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

LLAMA = PRED_DIR / "original_classifier__fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3.3-70B-Instruct-bnb-4bit.csv"
GEMMA = PRED_DIR / "original_classifier__cyankiwi_gemma-4-31B-it-AWQ-8bit.csv"
QWEN = PRED_DIR / "original_classifier__cyankiwi_Qwen3.5-27B-AWQ-BF16-INT8.csv"

def load(path, prefix):
    df = pd.read_csv(path)

    # Print columns once for debugging
    print(f"\n{prefix} columns:")
    print(df.columns.tolist())

    keep = [
        "benchmark_id", "chunk_id", "doc_id", "chunk_index",
        "rile_label_3way", "rile", "countryname", "partyname", "date",
        "text", "parsed_prediction_label"
    ]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].copy()
    df = df.rename(columns={"parsed_prediction_label": f"{prefix}_pred"})
    return df

llama = load(LLAMA, "llama")
gemma = load(GEMMA, "gemma")
qwen = load(QWEN, "qwen")

df = llama.merge(
    gemma[["benchmark_id", "gemma_pred"]],
    on="benchmark_id",
    how="inner"
)

df = df.merge(
    qwen[["benchmark_id", "qwen_pred"]],
    on="benchmark_id",
    how="left"
)

gold = df["rile_label_3way"]

df["llama_correct"] = df["llama_pred"] == gold
df["gemma_correct"] = df["gemma_pred"] == gold
df["qwen_correct"] = df["qwen_pred"] == gold
df["llama_gemma_same"] = df["llama_pred"] == df["gemma_pred"]

subsets = {
    "llama_correct_gemma_wrong": df[(df["llama_correct"]) & (~df["gemma_correct"])],
    "gemma_correct_llama_wrong": df[(df["gemma_correct"]) & (~df["llama_correct"])],
    "llama_gemma_both_wrong": df[(~df["llama_correct"]) & (~df["gemma_correct"])],
    "llama_gemma_disagree": df[df["llama_pred"] != df["gemma_pred"]],
    "llama_gemma_same_wrong": df[(df["llama_pred"] == df["gemma_pred"]) & (~df["llama_correct"])],
}

summary = pd.DataFrame([
    {"metric": "total_rows", "value": len(df)},
    {"metric": "llama_correct", "value": int(df["llama_correct"].sum())},
    {"metric": "gemma_correct", "value": int(df["gemma_correct"].sum())},
    {"metric": "qwen_correct", "value": int(df["qwen_correct"].sum())},
    {"metric": "llama_gemma_same_prediction", "value": int(df["llama_gemma_same"].sum())},
    {"metric": "llama_gemma_different_prediction", "value": int((~df["llama_gemma_same"]).sum())},
    {"metric": "llama_correct_gemma_wrong", "value": len(subsets["llama_correct_gemma_wrong"])},
    {"metric": "gemma_correct_llama_wrong", "value": len(subsets["gemma_correct_llama_wrong"])},
    {"metric": "both_wrong", "value": len(subsets["llama_gemma_both_wrong"])},
])

summary.to_csv(OUT_DIR / "benchmark_v2_error_analysis_summary.csv", index=False)

sample_cols = [
    "benchmark_id", "chunk_id", "doc_id", "chunk_index",
    "rile_label_3way", "rile", "countryname", "partyname", "date",
    "llama_pred", "gemma_pred", "qwen_pred",
    "llama_correct", "gemma_correct", "qwen_correct",
    "text"
]

for name, sub in subsets.items():
    sample = sub.sample(n=min(20, len(sub)), random_state=42) if len(sub) else sub
    sample = sample[[c for c in sample_cols if c in sample.columns]]
    sample.to_csv(SAMPLE_DIR / f"{name}_sample_20.csv", index=False)

pd.crosstab(
    df["gemma_correct"],
    df["llama_correct"],
    rownames=["gemma_correct"],
    colnames=["llama_correct"]
).to_csv(OUT_DIR / "llama_gemma_correctness_crosstab.csv")

for name, sub in subsets.items():
    sub["rile_label_3way"].value_counts().to_csv(OUT_DIR / f"{name}_by_gold.csv")

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

print("\nSaved summary:", OUT_DIR / "benchmark_v2_error_analysis_summary.csv")
print("Saved samples:", SAMPLE_DIR)
