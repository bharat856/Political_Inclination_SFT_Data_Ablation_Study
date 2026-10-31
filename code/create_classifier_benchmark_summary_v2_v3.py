import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs/classifier_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

def add_row(**kwargs):
    rows.append(kwargs)

# -----------------------------
# Benchmark v2: Manifesto chunks
# -----------------------------
# Known results from previous runs
add_row(
    benchmark="benchmark_v2_manifesto_chunks",
    task="left / center / right",
    dataset="Manifesto-derived chunks",
    label_quality="weak document-level RILE label copied to chunks",
    model="Llama-3.3-70B-bnb-4bit",
    model_type="prompted LLM",
    test_rows=900,
    accuracy=0.5156,
    macro_f1=None,
    notes="overpredicts center; affected by chunk/document label mismatch",
)

add_row(
    benchmark="benchmark_v2_manifesto_chunks",
    task="left / center / right",
    dataset="Manifesto-derived chunks",
    label_quality="weak document-level RILE label copied to chunks",
    model="Gemma-4-31B-AWQ",
    model_type="prompted LLM",
    test_rows=900,
    accuracy=0.5044,
    macro_f1=None,
    notes="overpredicts right; affected by chunk/document label mismatch",
)

add_row(
    benchmark="benchmark_v2_manifesto_chunks",
    task="left / center / right",
    dataset="Manifesto-derived chunks",
    label_quality="weak document-level RILE label copied to chunks",
    model="ModernBERT-large",
    model_type="fine-tuned encoder",
    test_rows=124,
    accuracy=0.4194,
    macro_f1=0.4190,
    notes="held-out document split; weak but not collapsed",
)

# -----------------------------
# Benchmark v3a: clean ideology statements
# -----------------------------
# Llama prompted
llama_v3a = Path("outputs/benchmark_v3a_binary/binary_classifier___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3_3-70B-Instruct-bnb-4bit__test.csv")
if llama_v3a.exists():
    df = pd.read_csv(llama_v3a)
    add_row(
        benchmark="benchmark_v3a_clean_ideology",
        task="left / right",
        dataset="JyotiNayak/political_ideologies",
        label_quality="clean text-level ideology label",
        model="Llama-3.3-70B-bnb-4bit",
        model_type="prompted LLM",
        test_rows=len(df),
        accuracy=round(float(df["is_correct"].mean()), 4),
        macro_f1=None,
        notes=f"parsed {int(df['parse_ok'].sum())}/{len(df)}",
    )
else:
    print("Missing:", llama_v3a)

# ModernBERT v3a
modernbert_v3a = Path("outputs/benchmark_v3a_binary/encoders/modernbert_large/test_metrics.json")
if modernbert_v3a.exists():
    m = json.loads(modernbert_v3a.read_text())
    add_row(
        benchmark="benchmark_v3a_clean_ideology",
        task="left / right",
        dataset="JyotiNayak/political_ideologies",
        label_quality="clean text-level ideology label",
        model="ModernBERT-large",
        model_type="fine-tuned encoder",
        test_rows=int(m["test_rows"]),
        accuracy=round(float(m["accuracy"]), 4),
        macro_f1=round(float(m["macro_f1"]), 4),
        notes="perfect on clean benchmark; likely comparatively easy",
    )
else:
    print("Missing:", modernbert_v3a)

# -----------------------------
# Benchmark v3b: senator tweets
# -----------------------------
# Llama prompted
llama_v3b = Path("outputs/benchmark_v3b_senator_tweets_binary/binary_classifier___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3_3-70B-Instruct-bnb-4bit__test.csv")
if llama_v3b.exists():
    df = pd.read_csv(llama_v3b)
    add_row(
        benchmark="benchmark_v3b_senator_tweets",
        task="Democrat(left) / Republican(right)",
        dataset="m-newhauser/senator-tweets",
        label_quality="real-world text-level party label",
        model="Llama-3.3-70B-bnb-4bit",
        model_type="prompted LLM",
        test_rows=len(df),
        accuracy=round(float(df["is_correct"].mean()), 4),
        macro_f1=None,
        notes=f"parsed {int(df['parse_ok'].sum())}/{len(df)}; overpredicts left",
    )
else:
    print("Missing:", llama_v3b)

# ModernBERT v3b
modernbert_v3b = Path("outputs/benchmark_v3b_senator_tweets_binary/encoders/modernbert_large/test_metrics.json")
if modernbert_v3b.exists():
    m = json.loads(modernbert_v3b.read_text())
    add_row(
        benchmark="benchmark_v3b_senator_tweets",
        task="Democrat(left) / Republican(right)",
        dataset="m-newhauser/senator-tweets",
        label_quality="real-world text-level party label",
        model="ModernBERT-large",
        model_type="fine-tuned encoder",
        test_rows=int(m["test_rows"]),
        accuracy=round(float(m["accuracy"]), 4),
        macro_f1=round(float(m["macro_f1"]), 4),
        notes="outperforms prompted Llama on harder real-world political text",
    )
else:
    print("Missing:", modernbert_v3b)

summary = pd.DataFrame(rows)

# Nice ordering
order = {
    "benchmark_v2_manifesto_chunks": 1,
    "benchmark_v3a_clean_ideology": 2,
    "benchmark_v3b_senator_tweets": 3,
}
summary["benchmark_order"] = summary["benchmark"].map(order)
summary = summary.sort_values(["benchmark_order", "accuracy"], ascending=[True, False])
summary = summary.drop(columns=["benchmark_order"])

csv_out = OUT_DIR / "classifier_benchmark_summary_v2_v3.csv"
md_out = OUT_DIR / "classifier_benchmark_summary_v2_v3.md"

summary.to_csv(csv_out, index=False)
md_out.write_text(summary.to_markdown(index=False), encoding="utf-8")

print(summary.to_string(index=False))
print("\nSaved:")
print(csv_out)
print(md_out)
