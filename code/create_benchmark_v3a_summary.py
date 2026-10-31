from pathlib import Path
import pandas as pd

OUT_DIR = Path("outputs/benchmark_v3a_binary/summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "Llama-3.3-70B-bnb-4bit": "outputs/benchmark_v3a_binary/binary_classifier___fs_scratch_users_bharatpi_causal_political_content_hf_cache_unsloth_Llama-3_3-70B-Instruct-bnb-4bit__test.csv",
    "Gemma-4-31B-AWQ": "outputs/benchmark_v3a_binary/binary_classifier__cyankiwi_gemma-4-31B-it-AWQ-8bit__test__reparsed.csv",
    "Gemma-4-31B-AWQ-v2prompt": "outputs/benchmark_v3a_binary/binary_classifier_v2__cyankiwi_gemma-4-31B-it-AWQ-8bit__test__reparsed.csv",
    "Qwen3.5-27B-AWQ": "outputs/benchmark_v3a_binary/binary_classifier_v2__cyankiwi_Qwen3_5-27B-AWQ-BF16-INT8__test__reparsed.csv",
}

rows = []

for model, path in FILES.items():
    p = Path(path)
    if not p.exists():
        print("Missing:", p)
        continue

    df = pd.read_csv(p)

    if "reparsed_prediction_label" in df.columns:
        pred_col = "reparsed_prediction_label"
        ok_col = "reparse_ok"
        correct_col = "reparse_correct"
    else:
        pred_col = "parsed_prediction_label"
        ok_col = "parse_ok"
        correct_col = "is_correct"

    parsed = df[df[ok_col] == True].copy()

    rows.append({
        "model": model,
        "rows": len(df),
        "parsed": int(df[ok_col].sum()),
        "parse_rate": round(float(df[ok_col].mean()), 4),
        "accuracy_all_rows": round(float(df[correct_col].mean()), 4),
        "accuracy_parsed_only": round(float(parsed[correct_col].mean()), 4) if len(parsed) else None,
        "pred_left": int((df[pred_col] == "left").sum()),
        "pred_right": int((df[pred_col] == "right").sum()),
        "unparsed": int(df[pred_col].isna().sum()),
    })

summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "benchmark_v3a_summary.csv", index=False)

md = summary.to_markdown(index=False)
(OUT_DIR / "benchmark_v3a_summary.md").write_text(md, encoding="utf-8")

print(summary.to_string(index=False))
print("\nSaved:")
print(OUT_DIR / "benchmark_v3a_summary.csv")
print(OUT_DIR / "benchmark_v3a_summary.md")
