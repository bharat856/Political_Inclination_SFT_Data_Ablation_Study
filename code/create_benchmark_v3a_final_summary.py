import json
from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs/benchmark_v3a_binary/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

# Prompted LLMs
prompted_files = [
    {
        "model": "Llama-3.3-70B-bnb-4bit",
        "type": "prompted LLM",
        "path": "outputs/benchmark_v3a_binary/binary_classifier___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3_3-70B-Instruct-bnb-4bit__test.csv",
        "pred_col": "parsed_prediction_label",
        "ok_col": "parse_ok",
        "correct_col": "is_correct",
        "notes": "strong; all outputs parsed",
    },
    {
        "model": "Qwen3.5-27B-AWQ",
        "type": "prompted LLM",
        "path": "outputs/benchmark_v3a_binary/binary_classifier_v2__cyankiwi_Qwen3_5-27B-AWQ-BF16-INT8__test.csv",
        "pred_col": "parsed_prediction_label",
        "ok_col": "parse_ok",
        "correct_col": "is_correct",
        "notes": "strong on parsed outputs; reasoning tags caused unparsed rows",
    },
    {
        "model": "Gemma-4-31B-AWQ",
        "type": "prompted LLM",
        "path": "outputs/benchmark_v3a_binary/binary_classifier__cyankiwi_gemma-4-31B-it-AWQ-8bit__test.csv",
        "pred_col": "parsed_prediction_label",
        "ok_col": "parse_ok",
        "correct_col": "is_correct",
        "notes": "unstable output format",
    },
]

for item in prompted_files:
    p = Path(item["path"])
    if not p.exists():
        print("Missing:", p)
        continue

    df = pd.read_csv(p)
    parsed = df[df[item["ok_col"]] == True]

    rows.append({
        "model": item["model"],
        "type": item["type"],
        "test_rows": len(df),
        "parsed_rows": int(df[item["ok_col"]].sum()),
        "parse_rate": round(float(df[item["ok_col"]].mean()), 4),
        "accuracy_all_rows": round(float(df[item["correct_col"]].mean()), 4),
        "accuracy_parsed_only": round(float(parsed[item["correct_col"]].mean()), 4) if len(parsed) else None,
        "macro_f1": None,
        "pred_left": int((df[item["pred_col"]] == "left").sum()),
        "pred_right": int((df[item["pred_col"]] == "right").sum()),
        "notes": item["notes"],
    })

# Encoder models
encoder_dirs = [
    {
        "model": "ModernBERT-large",
        "type": "fine-tuned encoder",
        "dir": "outputs/benchmark_v3a_binary/encoders/modernbert_large",
        "notes": "perfect on clean benchmark",
    },
    {
        "model": "DeBERTa-v3-large",
        "type": "fine-tuned encoder",
        "dir": "outputs/benchmark_v3a_binary/encoders/deberta_v3_large",
        "notes": "collapsed to all-left",
    },
    {
        "model": "mDeBERTa-v3-base",
        "type": "fine-tuned encoder",
        "dir": "outputs/benchmark_v3a_binary/encoders/mdeberta_v3_base",
        "notes": "collapsed to all-right",
    },
]

for item in encoder_dirs:
    d = Path(item["dir"])
    metrics_path = d / "test_metrics.json"
    preds_path = d / "test_predictions.csv"

    if not metrics_path.exists() or not preds_path.exists():
        print("Missing:", d)
        continue

    metrics = json.loads(metrics_path.read_text())
    preds = pd.read_csv(preds_path)

    rows.append({
        "model": item["model"],
        "type": item["type"],
        "test_rows": int(metrics["test_rows"]),
        "parsed_rows": int(metrics["test_rows"]),
        "parse_rate": 1.0,
        "accuracy_all_rows": round(float(metrics["accuracy"]), 4),
        "accuracy_parsed_only": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "pred_left": int((preds["pred_label"] == "left").sum()),
        "pred_right": int((preds["pred_label"] == "right").sum()),
        "notes": item["notes"],
    })

summary = pd.DataFrame(rows)
summary = summary.sort_values(["accuracy_all_rows", "parse_rate"], ascending=False)

csv_out = OUT_DIR / "benchmark_v3a_final_summary.csv"
md_out = OUT_DIR / "benchmark_v3a_final_summary.md"

summary.to_csv(csv_out, index=False)
md_out.write_text(summary.to_markdown(index=False), encoding="utf-8")

print(summary.to_string(index=False))
print("\nSaved:")
print(csv_out)
print(md_out)
