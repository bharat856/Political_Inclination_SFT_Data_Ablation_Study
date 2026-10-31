from datasets import load_dataset
from collections import Counter

name = "valurank/PoliticalBias_AllSides_Txt"

print("Loading streaming dataset:", name)
ds = load_dataset(name, split="train", streaming=True)

rows = []
for i, ex in enumerate(ds):
    if i >= 20:
        break
    rows.append(ex)

print("Collected rows:", len(rows))

if rows:
    print("Keys:", list(rows[0].keys()))

    for i, ex in enumerate(rows[:5]):
        print("\n" + "="*80)
        print("Example", i)
        for k, v in ex.items():
            s = str(v)
            print(f"{k}: {s[:500]}")

    counters = {}
    for k in rows[0].keys():
        vals = [str(r.get(k, "")) for r in rows]
        short_vals = [v for v in vals if len(v) < 100]
        if short_vals:
            counters[k] = Counter(short_vals).most_common(10)

    print("\nPossible short-value columns:")
    for k, c in counters.items():
        print("\n", k)
        print(c)
