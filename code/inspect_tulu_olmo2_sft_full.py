from datasets import load_dataset
from collections import Counter
import pandas as pd
import json
from pathlib import Path

DATASET = "allenai/tulu-3-sft-olmo-2-mixture"

OUT_DIR = Path("data/processed/tulu_olmo2_sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading full dataset:", DATASET)
ds = load_dataset(DATASET, split="train")

print(ds)
print("Rows:", len(ds))
print("Columns:", ds.column_names)

sources = Counter()
datasets = Counter()
role_patterns = Counter()
num_turns = Counter()

for ex in ds:
    sources[str(ex.get("source"))] += 1
    datasets[str(ex.get("dataset"))] += 1

    messages = ex.get("messages", [])
    roles = tuple(m.get("role", "") for m in messages if isinstance(m, dict))
    role_patterns[roles] += 1
    num_turns[len(messages)] += 1

source_df = pd.DataFrame(sources.most_common(), columns=["source", "count"])
source_df["percent"] = source_df["count"] / len(ds) * 100

dataset_df = pd.DataFrame(datasets.most_common(), columns=["dataset", "count"])
dataset_df["percent"] = dataset_df["count"] / len(ds) * 100

role_df = pd.DataFrame(
    [(str(k), v) for k, v in role_patterns.most_common()],
    columns=["role_pattern", "count"]
)
role_df["percent"] = role_df["count"] / len(ds) * 100

turn_df = pd.DataFrame(num_turns.most_common(), columns=["num_messages", "count"])
turn_df["percent"] = turn_df["count"] / len(ds) * 100

source_df.to_csv(OUT_DIR / "source_distribution.csv", index=False)
dataset_df.to_csv(OUT_DIR / "dataset_distribution.csv", index=False)
role_df.to_csv(OUT_DIR / "role_pattern_distribution.csv", index=False)
turn_df.to_csv(OUT_DIR / "num_messages_distribution.csv", index=False)

meta = {
    "dataset": DATASET,
    "rows": len(ds),
    "columns": ds.column_names,
    "top_sources": source_df.head(30).to_dict(orient="records"),
    "top_dataset_values": dataset_df.head(30).to_dict(orient="records"),
    "top_role_patterns": role_df.head(20).to_dict(orient="records"),
    "num_messages_distribution": turn_df.head(20).to_dict(orient="records"),
}

(OUT_DIR / "inspection_summary.json").write_text(
    json.dumps(meta, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\nTop sources:")
print(source_df.head(30).to_string(index=False))

print("\nTop dataset values:")
print(dataset_df.head(30).to_string(index=False))

print("\nRole patterns:")
print(role_df.head(20).to_string(index=False))

print("\nNumber of messages:")
print(turn_df.head(20).to_string(index=False))

print("\nSaved files in:", OUT_DIR)
