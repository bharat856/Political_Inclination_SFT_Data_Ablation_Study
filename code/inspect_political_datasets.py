from datasets import load_dataset

candidates = [
    "cajcodes/political-bias",
    "JyotiNayak/political_ideologies",
    "mlburnham/dem_rep_party_platform_topics",
    "newsmediabias/news-bias-full-data",
]

for name in candidates:
    print("\n" + "="*100)
    print("DATASET:", name)
    print("="*100)

    try:
        ds = load_dataset(name)
        print(ds)

        split = list(ds.keys())[0]
        print("Using split:", split)

        for i in range(min(3, len(ds[split]))):
            print("\nExample", i)
            ex = ds[split][i]
            print(ex)

    except Exception as e:
        print("FAILED:", repr(e))
