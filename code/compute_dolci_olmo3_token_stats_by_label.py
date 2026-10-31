from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

CLASSIFIED = Path("outputs/dolci_olmo3_sft_modernbert_combined/4way_outputs/combined/dolci_olmo3_sft_binary_candidates_4way_combined.csv")

FLAT_FILES = {
    "instruct": Path("data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl"),
    "think": Path("data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl"),
}

MODEL = "/fs/scratch/users/bharatpi/causal_political_content/hf_cache/unsloth_Llama-3.3-70B-Instruct-bnb-4bit"

OUT = Path("data/interventions/dolci_olmo3_sft/token_stats_by_label.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading tokenizer:", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

print("Loading classification file:", CLASSIFIED)
cls = pd.read_csv(CLASSIFIED)

cls["label"] = cls["final_political_4way_status"].fillna("unparsed")
label_by_key = {
    (str(r["dolci_split"]), str(r["row_id"])): str(r["label"])
    for _, r in cls.iterrows()
}

stats = {}

def add(split, label, n_tokens):
    key = f"{split}::{label}"
    if key not in stats:
        stats[key] = {
            "dolci_split": split,
            "label": label,
            "rows": 0,
            "tokens": 0,
        }
    stats[key]["rows"] += 1
    stats[key]["tokens"] += int(n_tokens)

for split, path in FLAT_FILES.items():
    print("Processing:", split, path)

    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=split):
            obj = json.loads(line)
            row_id = str(obj.get("row_id"))
            text = obj.get("text", "")

            label = label_by_key.get((split, row_id), "not_candidate")
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            add(split, label, n_tokens)

rows = list(stats.values())

# Add totals and percentages
total_tokens = sum(r["tokens"] for r in rows)
total_rows = sum(r["rows"] for r in rows)

for r in rows:
    r["row_pct"] = r["rows"] / total_rows
    r["token_pct"] = r["tokens"] / total_tokens
    r["avg_tokens_per_row"] = r["tokens"] / max(1, r["rows"])

summary = {
    "total_rows": total_rows,
    "total_tokens": total_tokens,
    "by_split_and_label": rows,
}

OUT.write_text(json.dumps(summary, indent=2))
print("Saved:", OUT)

df = pd.DataFrame(rows)
print("\nBy label overall:")
overall = df.groupby("label")[["rows", "tokens"]].sum()
overall["row_pct"] = overall["rows"] / overall["rows"].sum()
overall["token_pct"] = overall["tokens"] / overall["tokens"].sum()
overall["avg_tokens_per_row"] = overall["tokens"] / overall["rows"]
print(overall.sort_values("tokens", ascending=False))

print("\nBy split and label:")
print(df.sort_values(["dolci_split", "tokens"], ascending=[True, False]))
