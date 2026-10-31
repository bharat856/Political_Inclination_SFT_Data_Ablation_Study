from datasets import load_dataset
from collections import Counter
import json

DATASET = "allenai/tulu-3-sft-olmo-2-mixture"

ds = load_dataset(DATASET, split="train", streaming=True)

sources = Counter()
datasets = Counter()
role_patterns = Counter()

examples = []

for i, ex in enumerate(ds):
    if i >= 1000:
        break

    sources[str(ex.get("source"))] += 1
    datasets[str(ex.get("dataset"))] += 1

    messages = ex.get("messages", [])
    roles = tuple(m.get("role", "") for m in messages if isinstance(m, dict))
    role_patterns[roles] += 1

    if len(examples) < 5:
        examples.append(ex)

print("Top sources:")
print(sources.most_common(30))

print("\nTop dataset field values:")
print(datasets.most_common(30))

print("\nRole patterns:")
print(role_patterns.most_common(20))

print("\nSample examples:")
for ex in examples:
    print("=" * 100)
    print(json.dumps(ex, ensure_ascii=False, indent=2)[:4000])
