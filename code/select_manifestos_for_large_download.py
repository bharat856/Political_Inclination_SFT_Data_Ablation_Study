import pandas as pd
from pathlib import Path

INPUT = Path("data/raw/manifesto/MPDataset_MPDS2025a.csv")
OUT_DIR = Path("data/raw/manifesto")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_CLASS = 100
CENTER_MARGIN = 5.0
SEED = 42

df = pd.read_csv(INPUT)

def label_from_rile(x):
    if pd.isna(x):
        return None
    x = float(x)
    if x < -CENTER_MARGIN:
        return "left"
    if x > CENTER_MARGIN:
        return "right"
    return "center"

df = df.copy()
df["rile_label_3way"] = df["rile"].apply(label_from_rile)
df = df[df["rile_label_3way"].isin(["left", "center", "right"])].copy()

# Try to keep only rows likely to have downloadable corpus text
if "has_eu_code" in df.columns:
    pass

print("Available labels:")
print(df["rile_label_3way"].value_counts())

sampled = []
for label in ["left", "center", "right"]:
    sub = df[df["rile_label_3way"] == label]
    n = min(PER_CLASS, len(sub))
    sampled.append(sub.sample(n=n, random_state=SEED))

out = pd.concat(sampled, ignore_index=True)
out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

out_file = OUT_DIR / "manifesto_download_large_300.csv"
out.to_csv(out_file, index=False)

print("\nSaved:", out_file)
print("Rows:", len(out))
print("Label distribution:")
print(out["rile_label_3way"].value_counts())

print("\nColumns:")
print(out.columns.tolist())
