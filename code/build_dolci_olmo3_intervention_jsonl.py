from pathlib import Path
import json
import pandas as pd
import argparse
from tqdm import tqdm

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

OUT_BASE = Path("data/interventions/dolci_olmo3_sft/jsonl")


def load_remove_ids(path):
    df = pd.read_csv(path)
    remove = {}
    for split, sub in df.groupby("dolci_split"):
        remove[str(split)] = set(sub["row_id"].astype(str))
    return remove


def stream_filter(infile, outfile, remove_ids):
    total = 0
    kept = 0
    removed = 0

    outfile.parent.mkdir(parents=True, exist_ok=True)

    with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=str(infile.name)):
            total += 1
            obj = json.loads(line)
            rid = str(obj.get("row_id"))

            if rid in remove_ids:
                removed += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return {"total": total, "kept": kept, "removed": removed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(ROW_LISTS.keys()))
    args = ap.parse_args()

    remove_by_split = load_remove_ids(ROW_LISTS[args.variant])

    summary = {}

    for split, infile in FLAT_FILES.items():
        remove_ids = remove_by_split.get(split, set())
        outfile = OUT_BASE / args.variant / f"dolci_{split}_sft_7b_{args.variant}.jsonl"

        stats = stream_filter(infile, outfile, remove_ids)
        stats["outfile"] = str(outfile)
        summary[split] = stats

    summary_out = OUT_BASE / args.variant / f"{args.variant}_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("Saved summary:", summary_out)


if __name__ == "__main__":
    main()
