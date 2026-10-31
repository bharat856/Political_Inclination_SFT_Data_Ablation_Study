from datasets import load_dataset
import pandas as pd

name = "newsmediabias/news-bias-full-data"

print("Loading:", name)
ds = load_dataset(name)

print(ds)

for split in ds:
    print("\n" + "="*100)
    print("SPLIT:", split)

    df = pd.DataFrame(ds[split].select(range(min(20, len(ds[split])))))
    print("Rows sample:", len(df))
    print("Columns:", df.columns.tolist())

    for col in df.columns:
        vals = df[col].astype(str)
        if vals.str.len().mean() < 100:
            print("\nSHORT COLUMN:", col)
            print(vals.value_counts(dropna=False).head(20))

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
