import json
from pathlib import Path
import pandas as pd

IN = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_political_candidates_highconf.jsonl")
OUT_DIR = Path("data/processed/tulu_olmo2_sft")
OUT_CSV = OUT_DIR / "tulu_olmo2_sft_highconf_classification_sample_2000.csv"
OUT_JSONL = OUT_DIR / "tulu_olmo2_sft_highconf_classification_sample_2000.jsonl"

SEED = 42
N_TOTAL = 2000

rows = []
with IN.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        rows.append({
            "row_id": row["row_id"],
            "id": row["id"],
            "source": row["source"],
            "dataset": row.get("dataset"),
            "num_messages": row["num_messages"],
            "char_len_full": row["char_len_full"],
            "full_text": row["full_text"],
            "user_text": row["user_text"],
            "assistant_text": row["assistant_text"],
            "strong_hits": ", ".join(row.get("highconf_strong_hits", [])),
            "issue_hits": ", ".join(row.get("highconf_issue_hits", [])),
            "context_hits": ", ".join(row.get("highconf_context_hits", [])),
        })

df = pd.DataFrame(rows)
print("High-confidence rows:", len(df))
print("\nSource distribution:")
print(df["source"].value_counts().head(30))

# Balanced-ish source sampling so one source does not dominate.
sample_parts = []
per_source_cap = max(50, N_TOTAL // 20)

for source, g in df.groupby("source"):
    n = min(len(g), per_source_cap)
    sample_parts.append(g.sample(n=n, random_state=SEED))

sample = pd.concat(sample_parts, ignore_index=True)

if len(sample) > N_TOTAL:
    sample = sample.sample(n=N_TOTAL, random_state=SEED)
elif len(sample) < N_TOTAL:
    remaining = df.drop(index=sample.index, errors="ignore")
    need = min(N_TOTAL - len(sample), len(remaining))
    if need > 0:
        sample = pd.concat([sample, remaining.sample(n=need, random_state=SEED)], ignore_index=True)

sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)

sample.to_csv(OUT_CSV, index=False)

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for _, r in sample.iterrows():
        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

print("\nSaved:")
print(OUT_CSV)
print(OUT_JSONL)
print("\nSample rows:", len(sample))
print("\nSample source distribution:")
print(sample["source"].value_counts().head(30))
