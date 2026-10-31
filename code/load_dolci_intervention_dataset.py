from pathlib import Path
import argparse
import pandas as pd
from datasets import load_dataset, concatenate_datasets


FLAT_FILES = {
    "instruct": (
        "data/processed/dolci_olmo3_sft/"
        "dolci_instruct_sft_7b_flattened.jsonl"
    ),
    "think": (
        "data/processed/dolci_olmo3_sft/"
        "dolci_think_sft_7b_flattened.jsonl"
    ),
}

ROW_LISTS = {
    "no_pol": (
        "data/interventions/dolci_olmo3_sft_union_t04/"
        "row_lists/remove_no_pol_row_ids.csv"
    ),
    "no_ideology": (
        "data/interventions/dolci_olmo3_sft_union_t04/"
        "row_lists/remove_no_ideology_row_ids.csv"
    ),
    "no_left": (
        "data/interventions/dolci_olmo3_sft_union_t04/"
        "row_lists/remove_no_left_row_ids.csv"
    ),
    "no_right": (
        "data/interventions/dolci_olmo3_sft_union_t04/"
        "row_lists/remove_no_right_row_ids.csv"
    ),
    "lr_balanced": (
        "data/interventions/dolci_olmo3_sft_union_t04/"
        "row_lists/remove_lr_balanced_row_ids.csv"
    ),
}


def load_intervention(condition: str):
    if condition not in {"full", *ROW_LISTS.keys()}:
        raise ValueError(
            f"Unknown condition: {condition}. "
            f"Choose from full, {', '.join(ROW_LISTS)}"
        )

    remove_by_split = {
        "instruct": set(),
        "think": set(),
    }

    if condition != "full":
        removal_df = pd.read_csv(ROW_LISTS[condition])

        for split in remove_by_split:
            remove_by_split[split] = set(
                removal_df.loc[
                    removal_df["split"] == split,
                    "row_id",
                ].astype(int)
            )

    datasets = []

    for split, path in FLAT_FILES.items():
        print(f"Loading {split}: {path}")

        ds = load_dataset(
            "json",
            data_files=path,
            split="train",
        )

        before = len(ds)
        remove_ids = remove_by_split[split]

        if remove_ids:
            ds = ds.filter(
                lambda row: int(row["row_id"]) not in remove_ids,
                desc=f"Filtering {condition}/{split}",
                num_proc=1,
            )

        after = len(ds)

        print(
            f"{split}: before={before:,}, "
            f"removed={before-after:,}, after={after:,}"
        )

        datasets.append(ds)

    combined = concatenate_datasets(datasets)

    print(f"Combined rows for {condition}: {len(combined):,}")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        required=True,
        choices=[
            "full",
            "no_pol",
            "no_ideology",
            "no_left",
            "no_right",
            "lr_balanced",
        ],
    )
    parser.add_argument(
        "--save-to-disk",
        default=None,
        help="Optional output path. Omit this during training.",
    )
    args = parser.parse_args()

    ds = load_intervention(args.condition)

    if args.save_to_disk:
        out = Path(args.save_to_disk)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out))
        print("Saved:", out)


if __name__ == "__main__":
    main()
