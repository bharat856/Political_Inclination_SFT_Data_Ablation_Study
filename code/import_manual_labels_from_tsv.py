from pathlib import Path
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--labels_tsv", required=True)
parser.add_argument("--output_csv", required=True)
args = parser.parse_args()

input_csv = Path(args.input_csv)
labels_tsv = Path(args.labels_tsv)
output_csv = Path(args.output_csv)

df = pd.read_csv(input_csv)

if "manual_label" not in df.columns:
    df["manual_label"] = ""
if "manual_notes" not in df.columns:
    df["manual_notes"] = ""

# Force object/string dtype because labels include "U"
df["manual_label"] = df["manual_label"].fillna("").astype("object")
df["manual_notes"] = df["manual_notes"].fillna("").astype("object")

labels = []

with open(labels_tsv, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.rstrip("\n")
        if not line.strip():
            continue

        # Skip accidental header lines
        if line.lower().startswith("row_id"):
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            print(f"Skipping malformed line {line_no}: {line}")
            continue

        row_id = parts[0].strip()
        manual_label = parts[1].strip()
        manual_note = parts[2].strip() if len(parts) >= 3 else ""

        if manual_label not in {"0", "1", "2", "3", "U"}:
            print(f"Skipping invalid label on line {line_no}: {line}")
            continue

        labels.append({
            "row_id": row_id,
            "manual_label_new": manual_label,
            "manual_notes_new": manual_note,
        })

lab = pd.DataFrame(labels)

if lab.empty:
    raise ValueError("No valid labels found in labels_tsv.")

df["row_id_str"] = df["row_id"].astype(str)
lab["row_id_str"] = lab["row_id"].astype(str)

merged = df.merge(
    lab[["row_id_str", "manual_label_new", "manual_notes_new"]],
    on="row_id_str",
    how="left"
)

mask = merged["manual_label_new"].notna()

merged.loc[mask, "manual_label"] = merged.loc[mask, "manual_label_new"]
merged.loc[mask, "manual_notes"] = merged.loc[mask, "manual_notes_new"]

merged = merged.drop(columns=["row_id_str", "manual_label_new", "manual_notes_new"])

output_csv.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(output_csv, index=False)

print("Saved:", output_csv)
print("Valid labels in TSV:", len(lab))
print("Imported labels:", int(mask.sum()))
print()
print("Manual label counts:")
print(merged["manual_label"].fillna("").replace("", "unlabeled").value_counts(dropna=False))
