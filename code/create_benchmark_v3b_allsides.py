import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "valurank/PoliticalBias_AllSides_Txt"

OUT_DIR = Path("data/processed/benchmark_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT = OUT_DIR / "benchmark_v3b_allsides_lcr.csv"
META = OUT_DIR / "benchmark_v3b_allsides_lcr.metadata.json"

SEED = 42
PER_CLASS = 300

LABEL_ID = {
    "left": 0,
    "center": 1,
    "right": 2,
}

def normalize_label(x):
    s = str(x).strip().lower()

    # Common AllSides-style labels
    if s in {"left", "liberal"}:
        return "left"
    if s in {"right", "conservative"}:
        return "right"
    if s in {"center", "centre", "neutral", "middle"}:
        return "center"

    # Leaning variants
    if s in {"lean left", "leans left", "left-center", "center-left", "left_center", "lean_left"}:
        return "left"
    if s in {"lean right", "leans right", "right-center", "center-right", "right_center", "lean_right"}:
        return "right"

    # Numeric fallbacks, only if dataset uses simple coding
    if s in {"0"}:
        return "left"
    if s in {"1"}:
        return "center"
    if s in {"2"}:
        return "right"

    return None

def choose_text_col(df):
    candidates = [
        "text", "article", "content", "body", "article_text",
        "full_text", "txt", "sentence", "statement"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: longest object/string column
    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise RuntimeError("Could not infer text column.")

    lengths = {}
    for c in object_cols:
        lengths[c] = df[c].astype(str).str.len().mean()

    return max(lengths, key=lengths.get)

def choose_label_col(df):
    candidates = [
        "label", "bias", "bias_label", "political_bias",
        "allsides_bias", "leaning", "ideology", "source_bias"
    ]
    for c in candidates:
        if c in df.columns:
            mapped = df[c].apply(normalize_label)
            if mapped.notna().sum() > 0:
                return c

    # fallback: any column with mappable labels
    for c in df.columns:
        mapped = df[c].apply(normalize_label)
        if mapped.notna().sum() > 0:
            return c

    raise RuntimeError("Could not infer label column.")

print("Loading:", DATASET)
ds = load_dataset(DATASET)

frames = []
for split in ds:
    df = pd.DataFrame(ds[split])
    df["source_split_original"] = split
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)

print("Raw rows:", len(raw))
print("Columns:", raw.columns.tolist())

text_col = choose_text_col(raw)
label_col = choose_label_col(raw)

print("Using text column:", text_col)
print("Using label column:", label_col)

raw["label"] = raw[label_col].apply(normalize_label)
raw["text"] = raw[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

clean = raw[
    raw["label"].isin(["left", "center", "right"])
    & raw["text"].notna()
    & (raw["text"].str.len() >= 200)
].copy()

print("\nClean label distribution:")
print(clean["label"].value_counts())

available = clean["label"].value_counts().to_dict()
missing = [lab for lab in ["left", "center", "right"] if available.get(lab, 0) < PER_CLASS]
if missing:
    raise RuntimeError(f"Not enough examples for labels {missing}. Available: {available}")

sampled = (
    clean.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(n=PER_CLASS, random_state=SEED))
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)

sampled.insert(0, "benchmark_id", [f"v3b_{i:04d}" for i in range(len(sampled))])
sampled["source_dataset"] = DATASET
sampled["label_id"] = sampled["label"].map(LABEL_ID)
sampled["original_label"] = sampled[label_col].astype(str)

keep = [
    "benchmark_id",
    "source_dataset",
    "source_split_original",
    "text",
    "label",
    "label_id",
    "original_label",
]

out = sampled[keep].copy()
out.to_csv(OUT, index=False)

meta = {
    "dataset": DATASET,
    "task": "three-way text-level political bias classification",
    "text_column": text_col,
    "label_column": label_col,
    "rows": int(len(out)),
    "per_class": PER_CLASS,
    "seed": SEED,
    "label_distribution": out["label"].value_counts().to_dict(),
    "label_id": LABEL_ID,
    "output": str(OUT),
}

META.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

print("\nSaved:", OUT)
print("Rows:", len(out))
print("\nFinal label distribution:")
print(out["label"].value_counts())
print("\nSample:")
print(out.head(5).to_string(index=False))
