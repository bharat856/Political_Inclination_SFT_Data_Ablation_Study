from pathlib import Path
import pandas as pd
from datasets import load_dataset

OUT_DIR = Path("data/validation_external")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("SetFit/bbc-news")

print(ds)

# Usually this dataset has a single train split
split_name = list(ds.keys())[0]
df = ds[split_name].to_pandas()

print("Columns:", df.columns.tolist())
print(df.head())

# Normalize likely column names
# Expected columns are often: text, label, label_text or similar.
print("\nValue counts for object/category columns:")
for c in df.columns:
    if df[c].dtype == "object" or "label" in c.lower() or "category" in c.lower():
        print("\n", c)
        print(df[c].value_counts(dropna=False).head(20))

out = OUT_DIR / "bbc_news.csv"
df.to_csv(out, index=False)

print("\nSaved:", out)
