import json
from pathlib import Path
import pandas as pd

rows = []

# Prompted LLM results, Benchmark v2 full 900-chunk benchmark
rows.extend([
    {
        "model": "Llama-3.3-70B-bnb-4bit",
        "type": "prompted LLM",
        "test_setting": "Benchmark v2 full set",
        "chunk_accuracy": 0.5156,
        "macro_f1": None,
        "document_majority_accuracy": 0.5440,
        "prediction_pattern": "overpredicts center",
    },
    {
        "model": "Gemma-4-31B-AWQ",
        "type": "prompted LLM",
        "test_setting": "Benchmark v2 full set",
        "chunk_accuracy": 0.5044,
        "macro_f1": None,
        "document_majority_accuracy": 0.5337,
        "prediction_pattern": "overpredicts right",
    },
    {
        "model": "Qwen3.5-27B-AWQ",
        "type": "prompted LLM",
        "test_setting": "Benchmark v2 full set",
        "chunk_accuracy": 0.3333,
        "macro_f1": None,
        "document_majority_accuracy": 0.3523,
        "prediction_pattern": "all-left collapse",
    },
])

# Encoder results, held-out document-level test split
encoder_base = Path("outputs/encoder_manifesto_benchmark_v2")

name_map = {
    "modernbert_large": "ModernBERT-large",
    "deberta_v3_large": "DeBERTa-v3-large",
    "mdeberta_v3_base": "mDeBERTa-v3-base",
}

pattern_map = {
    "modernbert_large": "weak, not collapsed",
    "deberta_v3_large": "all-right collapse",
    "mdeberta_v3_base": "all-center collapse",
}

for summary_path in sorted(encoder_base.glob("*/summary.json")):
    with open(summary_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    model_dir = summary_path.parent.name

    rows.append({
        "model": name_map.get(model_dir, model_dir),
        "type": "fine-tuned encoder",
        "test_setting": "Benchmark v2 held-out document split",
        "chunk_accuracy": d.get("chunk_accuracy"),
        "macro_f1": d.get("macro_f1"),
        "document_majority_accuracy": d.get("document_majority_accuracy"),
        "prediction_pattern": pattern_map.get(model_dir, ""),
    })

df = pd.DataFrame(rows)

df = df.sort_values(
    ["type", "chunk_accuracy"],
    ascending=[False, False],
)

out_dir = Path("outputs/classifier_comparison")
out_dir.mkdir(parents=True, exist_ok=True)

csv_out = out_dir / "all_classifier_comparison.csv"
md_out = out_dir / "all_classifier_comparison.md"

df.to_csv(csv_out, index=False)
md_out.write_text(df.to_markdown(index=False, floatfmt=".4f"), encoding="utf-8")

print(df.to_string(index=False))
print("\nSaved:", csv_out)
print("Saved:", md_out)
