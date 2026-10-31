from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="outputs/tulu_olmo2_sft_political_classification_sample/manual_review_sample_200.csv")
parser.add_argument("--output", default="outputs/tulu_olmo2_sft_political_classification_sample/manual_review_batch.txt")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--n", type=int, default=25)
args = parser.parse_args()

inp = Path(args.input)
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(inp)

# choose best available text column
if "text_preview" in df.columns:
    text_col = "text_preview"
elif "full_text" in df.columns:
    text_col = "full_text"
elif "text" in df.columns:
    text_col = "text"
else:
    raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")

pred_col = "sft_political_label_reparsed" if "sft_political_label_reparsed" in df.columns else "sft_political_label"

batch = df.iloc[args.start: args.start + args.n].copy()

lines = []
lines.append("Please relabel these SFT examples using:")
lines.append("0 = non_political")
lines.append("1 = left")
lines.append("2 = center_neutral")
lines.append("3 = right")
lines.append("U = unclear")
lines.append("")
lines.append("Return only this format:")
lines.append("row_id<TAB>manual_label<TAB>short_note")
lines.append("")
lines.append("=" * 100)

for _, r in batch.iterrows():
    row_id = r.get("row_id", r.name)
    pred = r.get(pred_col, "")
    source = r.get("source", "")
    raw = r.get("sft_political_raw_output", "")
    text = str(r.get(text_col, "")).replace("\r", " ").strip()

    lines.append(f"\nROW_ID: {row_id}")
    lines.append(f"PREDICTED_LABEL: {pred}")
    lines.append(f"SOURCE: {source}")
    lines.append(f"RAW_MODEL_OUTPUT: {raw}")
    lines.append("TEXT:")
    lines.append(text)
    lines.append("-" * 100)

out.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {out}")
print(f"Rows exported: {len(batch)}")
print(f"Start: {args.start}")
print(f"End exclusive: {args.start + len(batch)}")
