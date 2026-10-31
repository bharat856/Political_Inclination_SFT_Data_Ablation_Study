from datasets import load_dataset
import pandas as pd

candidates = [
    "rijulpaul/news-political-bias",
    "mediabiasgroup/BABE",
    "mediabiasgroup/mbib-base",
    "mediabiasgroup/mbib-aggregated",
    "R42B/LFQA-Political-Bias",
    "m-newhauser/senator-tweets",
    "bucketresearch/politicalBias",
    "cajcodes/political-bias",
    "JyotiNayak/political_ideologies",
]

for name in candidates:
    print("\n" + "="*120)
    print("TRY:", name)
    try:
        ds = load_dataset(name)
        print(ds)

        split = list(ds.keys())[0]
        n = min(5, len(ds[split]))
        df = pd.DataFrame(ds[split].select(range(n)))

        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string(index=False))

        for col in df.columns:
            vals = df[col].astype(str)
            if vals.str.len().mean() < 200:
                print("\nShort column:", col)
                print(vals.value_counts(dropna=False).head(20))

    except Exception as e:
        print("FAILED:", repr(e))
