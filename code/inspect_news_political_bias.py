from datasets import load_dataset
import pandas as pd

name = "rijulpaul/news-political-bias"

print("Loading:", name)
ds = load_dataset(name)

print(ds)

for split in ds:
    print("\n" + "="*100)
    print("SPLIT:", split)

    n = min(30, len(ds[split]))
    df = pd.DataFrame(ds[split].select(range(n)))

    print("Rows sample:", len(df))
    print("Columns:", df.columns.tolist())

    for col in df.columns:
        vals = df[col].astype(str)
        if vals.str.len().mean() < 200:
            print("\nSHORT COLUMN:", col)
            print(vals.value_counts(dropna=False).head(30))

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))
