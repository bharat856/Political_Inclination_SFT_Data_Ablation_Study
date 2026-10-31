import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "m-newhauser/senator-tweets"

OUT_DIR = Path("data/processed/benchmark_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "benchmark_v3b_senator_tweets_binary.csv"
META = OUT_DIR / "benchmark_v3b_senator_tweets_binary.metadata.json"

SEED = 42
PER_CLASS_TRAIN = 1280
PER_CLASS_VAL = 160
PER_CLASS_TEST = 160

LABEL_MAP = {
    "Democrat": "left",
    "Republican": "right",
}

LABEL_ID = {
    "left": 0,
    "right": 1,
}

print("Loading:", DATASET)
ds = load_dataset(DATASET)

frames = []
for split in ["train", "test"]:
    df = pd.DataFrame(ds[split])
    df["source_split_original"] = split
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

raw["party"] = raw["party"].astype(str)
raw["label"] = raw["party"].map(LABEL_MAP)
raw["text"] = raw["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

clean = raw[
    raw["label"].isin(["left", "right"])
    & raw["text"].notna()
    & (raw["text"].str.len() >= 40)
].copy()

print("Clean rows:", len(clean))
print("Clean label distribution:")
print(clean["label"].value_counts())

# Shuffle within each class
left = clean[clean["label"] == "left"].sample(frac=1, random_state=SEED).reset_index(drop=True)
right = clean[clean["label"] == "right"].sample(frac=1, random_state=SEED).reset_index(drop=True)

needed = PER_CLASS_TRAIN + PER_CLASS_VAL + PER_CLASS_TEST
if len(left) < needed or len(right) < needed:
    raise RuntimeError(f"Not enough rows. Need {needed} per class. Have left={len(left)}, right={len(right)}")

def assign_split(df):
    train = df.iloc[:PER_CLASS_TRAIN].copy()
    val = df.iloc[PER_CLASS_TRAIN:PER_CLASS_TRAIN + PER_CLASS_VAL].copy()
    test = df.iloc[PER_CLASS_TRAIN + PER_CLASS_VAL:PER_CLASS_TRAIN + PER_CLASS_VAL + PER_CLASS_TEST].copy()

    train["source_split"] = "train"
    val["source_split"] = "validation"
    test["source_split"] = "test"

    return pd.concat([train, val, test], ignore_index=True)

out = pd.concat([assign_split(left), assign_split(right)], ignore_index=True)
out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

out.insert(0, "benchmark_id", [f"v3b_tweet_{i:05d}" for i in range(len(out))])
out["source_dataset"] = DATASET
out["label_id"] = out["label"].map(LABEL_ID)
out["original_label"] = out["party"]

keep = [
    "benchmark_id",
    "source_dataset",
    "source_split",
    "source_split_original",
    "text",
    "label",
    "label_id",
    "original_label",
    "username",
    "date",
]

out = out[keep]
out.to_csv(OUT, index=False)

meta = {
    "dataset": DATASET,
    "task": "binary real-world political text classification",
    "note": "Democrat mapped to left, Republican mapped to right; no center class.",
    "rows": int(len(out)),
    "splits": out["source_split"].value_counts().to_dict(),
    "label_distribution_total": out["label"].value_counts().to_dict(),
    "label_distribution_by_split": {
        split: out[out["source_split"] == split]["label"].value_counts().to_dict()
        for split in ["train", "validation", "test"]
    },
    "label_id": LABEL_ID,
    "output": str(OUT),
}

META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSaved:", OUT)
print("Rows:", len(out))
print("\nBy split:")
print(pd.crosstab(out["source_split"], out["label"]))
print("\nSample:")
print(out.head(5).to_string(index=False))
