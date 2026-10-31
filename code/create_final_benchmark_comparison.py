import pandas as pd
from pathlib import Path

OUT_DIR = Path("outputs/original_classifier/final_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

# -------------------------
# Benchmark v1 known results
# -------------------------
v1_results = [
    {
        "benchmark": "benchmark_v1",
        "model_short": "Llama-3.3-70B-bnb-4bit",
        "chunk_accuracy": 0.6905,
        "document_majority_accuracy": 0.7407,
        "rows": 126,
        "documents": 27,
        "main_pattern": "best document-level result; stronger on left/center"
    },
    {
        "benchmark": "benchmark_v1",
        "model_short": "Gemma-4-31B-AWQ",
        "chunk_accuracy": 0.6905,
        "document_majority_accuracy": 0.6667,
        "rows": 126,
        "documents": 27,
        "main_pattern": "same chunk accuracy as Llama; stronger on right"
    },
    {
        "benchmark": "benchmark_v1",
        "model_short": "Qwen3.5-27B-AWQ",
        "chunk_accuracy": 0.3333,
        "document_majority_accuracy": 0.3333,
        "rows": 126,
        "documents": 27,
        "main_pattern": "collapsed to all-left predictions"
    },
]

rows.extend(v1_results)

# -------------------------
# Benchmark v2 chunk-level
# -------------------------
v2_pred_dir = Path("outputs/original_classifier_benchmark_v2")
v2_doc_summary_path = v2_pred_dir / "document_level" / "document_level_summary.csv"

name_map = {
    "fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3.3-70B-Instruct-bnb-4bit": "Llama-3.3-70B-bnb-4bit",
    "cyankiwi_gemma-4-31B-it-AWQ-8bit": "Gemma-4-31B-AWQ",
    "cyankiwi_Qwen3.5-27B-AWQ-BF16-INT8": "Qwen3.5-27B-AWQ",
}

pattern_map = {
    "Llama-3.3-70B-bnb-4bit": "overpredicts center",
    "Gemma-4-31B-AWQ": "overpredicts right",
    "Qwen3.5-27B-AWQ": "collapsed to all-left predictions",
}

doc_summary = pd.read_csv(v2_doc_summary_path)

for pred_file in sorted(v2_pred_dir.glob("original_classifier__*.csv")):
    pred = pd.read_csv(pred_file)

    model_raw = pred_file.stem.replace("original_classifier__", "")
    model_short = name_map.get(model_raw, model_raw)

    chunk_acc = (pred["rile_label_3way"] == pred["parsed_prediction_label"]).mean()

    doc_row = doc_summary[doc_summary["model"] == model_raw]
    if len(doc_row) == 1:
        doc_acc = float(doc_row["document_majority_accuracy"].iloc[0])
        documents = int(doc_row["documents"].iloc[0])
    else:
        doc_acc = None
        documents = pred["doc_id"].nunique()

    rows.append({
        "benchmark": "benchmark_v2",
        "model_short": model_short,
        "chunk_accuracy": chunk_acc,
        "document_majority_accuracy": doc_acc,
        "rows": len(pred),
        "documents": documents,
        "main_pattern": pattern_map.get(model_short, "")
    })

summary = pd.DataFrame(rows)

# Sort for readability
model_order = {
    "Llama-3.3-70B-bnb-4bit": 1,
    "Gemma-4-31B-AWQ": 2,
    "Qwen3.5-27B-AWQ": 3,
}

summary["model_order"] = summary["model_short"].map(model_order)
summary = summary.sort_values(["benchmark", "model_order"]).drop(columns=["model_order"])

csv_out = OUT_DIR / "v1_v2_benchmark_comparison.csv"
md_out = OUT_DIR / "v1_v2_benchmark_comparison.md"

summary.to_csv(csv_out, index=False)

md = []
md.append("# Original Classifier Benchmark Comparison\n")
md.append("## Summary table\n")
md.append(summary.to_markdown(index=False, floatfmt=".4f"))
md.append("\n\n## Key observations\n")
md.append("- Benchmark v2 is substantially larger and more diverse than Benchmark v1: 900 chunks from 193 documents compared with 126 chunks from 27 documents.\n")
md.append("- Llama performs best overall on both benchmarks, especially under document-majority aggregation.\n")
md.append("- Gemma is very close to Llama on Benchmark v2, but shows a stronger tendency to predict the right class.\n")
md.append("- Qwen3.5 collapses to all-left predictions on both benchmarks, making it unsuitable as the main classifier in this setup.\n")
md.append("- The drop from v1 to v2 suggests that the larger benchmark is more difficult and highlights the noise introduced by applying document-level RILE labels to individual chunks.\n")

md.append("\n## Thesis-ready paragraph\n")
md.append(
    "The original classifier was evaluated on two manifesto-based benchmarks. "
    "Benchmark v1 was a smaller pilot benchmark with 126 balanced chunks from 27 documents, while Benchmark v2 was a larger and more diverse benchmark with 900 balanced chunks from 193 documents. "
    "On Benchmark v1, Llama-3.3-70B and Gemma-4-31B both achieved 69.05% chunk-level accuracy, while Qwen3.5 remained at chance level after collapsing to left-only predictions. "
    "When chunk predictions were aggregated by document majority vote, Llama achieved the strongest result at 74.07%, followed by Gemma at 66.67%. "
    "On Benchmark v2, performance dropped substantially: Llama reached 51.56% chunk-level accuracy and 54.40% document-majority accuracy, while Gemma reached 50.44% chunk-level accuracy and 53.37% document-majority accuracy. "
    "Qwen3.5 again collapsed to predicting only the left class, reaching 33.33% chunk-level accuracy and 35.23% document-majority accuracy. "
    "These results suggest that Benchmark v2 is more realistic and difficult, and they also highlight the methodological limitation of evaluating chunk-level inputs using document-level RILE labels."
)

md_out.write_text("\n".join(md), encoding="utf-8")

print("Saved CSV:", csv_out)
print("Saved Markdown:", md_out)
print()
print(summary.to_string(index=False))
