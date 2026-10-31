import json
from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data/processed/benchmark_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "benchmark_v3a_political_ideologies_binary.csv"
META = OUT_DIR / "benchmark_v3a_political_ideologies_binary.metadata.json"

DATASET = "JyotiNayak/political_ideologies"

# Based on inspected examples:
# 0 = conservative/right, 1 = liberal/left
LABEL_MAP = {
    0: "right",
    1: "left",
}

ds = load_dataset(DATASET)

rows = []
for split in ["train", "validation", "test"]:
    df = pd.DataFrame(ds[split])
    for i, row in df.iterrows():
        label_num = int(row["label"])
        if label_num not in LABEL_MAP:
            continue

        rows.append({
            "benchmark_id": f"v3a_{split}_{i:05d}",
            "source_dataset": DATASET,
            "source_split": split,
            "text": row["statement"],
            "label": LABEL_MAP[label_num],
            "label_id": 0 if LABEL_MAP[label_num] == "left" else 1,
            "original_label": label_num,
            "issue_type": row.get("issue_type", None),
        })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

meta = {
    "dataset": DATASET,
    "task": "binary political ideology classification",
    "label_mapping": LABEL_MAP,
    "rows": len(out),
    "splits": out["source_split"].value_counts().to_dict(),
    "labels": out["label"].value_counts().to_dict(),
    "output": str(OUT),
}

META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Saved:", OUT)
print("Rows:", len(out))
print("\nSplit distribution:")
print(out["source_split"].value_counts())
print("\nLabel distribution:")
print(pd.crosstab(out["source_split"], out["label"]))
