from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

MODEL = "/fs/scratch/users/bharatpi/causal_political_content/hf_cache/unsloth_Llama-3.3-70B-Instruct-bnb-4bit"

FLAT_FILES = {
    "instruct": Path("data/processed/dolci_olmo3_sft/dolci_instruct_sft_7b_flattened.jsonl"),
    "think": Path("data/processed/dolci_olmo3_sft/dolci_think_sft_7b_flattened.jsonl"),
}

ROW_LISTS = {
    "no_pol": Path("data/interventions/dolci_olmo3_sft/remove_no_pol_row_ids.csv"),
    "no_ideology": Path("data/interventions/dolci_olmo3_sft/remove_no_ideology_row_ids.csv"),
    "no_left": Path("data/interventions/dolci_olmo3_sft/remove_left_row_ids.csv"),
    "no_right": Path("data/interventions/dolci_olmo3_sft/remove_right_row_ids.csv"),
    "lr_balanced": Path("data/interventions/dolci_olmo3_sft/remove_extra_left_for_lr_balance_row_ids.csv"),
}

OUT = Path("data/interventions/dolci_olmo3_sft/intervention_token_removal_stats.json")

print("Loading tokenizer:", MODEL)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

remove_maps = {}
for variant, path in ROW_LISTS.items():
    df = pd.read_csv(path)
    remove_maps[variant] = {}
    for split, sub in df.groupby("dolci_split"):
        remove_maps[variant][str(split)] = set(sub["row_id"].astype(str))

stats = {
    v: {
        "removed_rows": 0,
        "removed_tokens": 0,
        "by_split": {},
    }
    for v in ROW_LISTS
}

total_rows = 0
total_tokens = 0

for split, path in FLAT_FILES.items():
    print("Processing:", split)

    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=split):
            total_rows += 1
            obj = json.loads(line)
            row_id = str(obj.get("row_id"))
            text = obj.get("text", "")
            n_tokens = len(tok.encode(text, add_special_tokens=False))
            total_tokens += n_tokens

            for variant, by_split in remove_maps.items():
                if row_id in by_split.get(split, set()):
                    stats[variant]["removed_rows"] += 1
                    stats[variant]["removed_tokens"] += n_tokens

                    if split not in stats[variant]["by_split"]:
                        stats[variant]["by_split"][split] = {
                            "removed_rows": 0,
                            "removed_tokens": 0,
                        }

                    stats[variant]["by_split"][split]["removed_rows"] += 1
                    stats[variant]["by_split"][split]["removed_tokens"] += n_tokens

for variant, s in stats.items():
    s["removed_row_pct"] = s["removed_rows"] / total_rows
    s["removed_token_pct"] = s["removed_tokens"] / total_tokens
    s["avg_tokens_removed_row"] = s["removed_tokens"] / max(1, s["removed_rows"])

summary = {
    "total_rows": total_rows,
    "total_tokens": total_tokens,
    "variants": stats,
}

OUT.write_text(json.dumps(summary, indent=2))
print("Saved:", OUT)
print(json.dumps(summary, indent=2))
