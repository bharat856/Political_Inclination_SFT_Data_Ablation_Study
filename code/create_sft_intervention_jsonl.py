from pathlib import Path
import json
import pandas as pd
from collections import Counter

BASE = Path("data/processed/tulu_olmo2_sft/tulu_olmo2_sft_flattened.jsonl")

INTERVENTION_DIR = Path("outputs/target_domain_binary_modernbert/intervention_sets")
OUT_DIR = Path("data/processed/tulu_olmo2_sft/intervention_jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_POL = INTERVENTION_DIR / "original_clean_political_position.csv"
DOWNSAMPLED = INTERVENTION_DIR / "balanced_downsample_left.csv"
UPSAMPLED = INTERVENTION_DIR / "balanced_upsample_right.csv"

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                r["_row_id"] = i
                rows.append(r)
    return rows

def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            r = dict(r)
            r.pop("_row_id", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Saved:", path, "rows:", len(rows))

print("Loading base SFT...")
base_rows = load_rows(BASE)
print("Base rows:", len(base_rows))

original_pol = pd.read_csv(ORIGINAL_POL)
downsampled = pd.read_csv(DOWNSAMPLED)
upsampled = pd.read_csv(UPSAMPLED)

original_ids = set(original_pol["row_id"].astype(int))
down_ids = set(downsampled["row_id"].astype(int))
up_ids = list(upsampled["row_id"].astype(int))  # keep duplicates for upsampling

print("\nPolitical-position row counts:")
print("original political-position:", len(original_ids))
print("downsampled unique:", len(down_ids))
print("upsampled rows including duplicates:", len(up_ids))
print("upsampled duplicate count:", len(up_ids) - len(set(up_ids)))

# 1. Baseline original full SFT
write_jsonl(
    base_rows,
    OUT_DIR / "baseline_original_full.jsonl"
)

# 2. Remove all detected political-position examples
remove_all = [r for r in base_rows if int(r["_row_id"]) not in original_ids]
write_jsonl(
    remove_all,
    OUT_DIR / "remove_all_political_position.jsonl"
)

# 3. Downsample left: remove all original political-position rows, then add downsampled political rows
base_without_original_pol = [r for r in base_rows if int(r["_row_id"]) not in original_ids]
row_map = {int(r["_row_id"]): r for r in base_rows}

down_rows = [row_map[i] for i in down_ids if i in row_map]
down_full = base_without_original_pol + down_rows
write_jsonl(
    down_full,
    OUT_DIR / "balanced_downsample_left_full.jsonl"
)

# 4. Upsample right: remove all original political-position rows, then add upsampled political rows including duplicates
up_rows = [row_map[i] for i in up_ids if i in row_map]
up_full = base_without_original_pol + up_rows
write_jsonl(
    up_full,
    OUT_DIR / "balanced_upsample_right_full.jsonl"
)

print("\nFinal row counts:")
for p in sorted(OUT_DIR.glob("*.jsonl")):
    n = sum(1 for _ in open(p, "r", encoding="utf-8"))
    print(p.name, n)

print("\nExpected approximate changes:")
print("baseline:", len(base_rows))
print("remove_all:", len(base_rows) - len(original_ids))
print("downsample:", len(base_rows) - len(original_ids) + len(down_ids))
print("upsample:", len(base_rows) - len(original_ids) + len(up_ids))
