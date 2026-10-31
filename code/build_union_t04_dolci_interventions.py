from pathlib import Path
import argparse
import json
import os
import random

import pandas as pd
from tqdm import tqdm


INSTRUCT_LABELS = Path(
    "outputs/dolci_olmo3_sft_union_embedding_t04/4way_outputs/full_instruct/"
    "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_"
    "hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit__reparsed.csv"
)

THINK_LABELS = Path(
    "outputs/dolci_olmo3_sft_union_embedding_t04/4way_outputs/full_think/"
    "sft_political_4way___fs_scratch_users_bharatpi_causal_political_content_"
    "hf_cache_unsloth_Llama_3_3_70B_Instruct_bnb_4bit__reparsed.csv"
)

FLAT_FILES = {
    "instruct": Path(
        "data/processed/dolci_olmo3_sft/"
        "dolci_instruct_sft_7b_flattened.jsonl"
    ),
    "think": Path(
        "data/processed/dolci_olmo3_sft/"
        "dolci_think_sft_7b_flattened.jsonl"
    ),
}

OUTPUT_BASE = Path(
    "data/interventions/dolci_olmo3_sft_union_t04"
)

LABEL_COLUMN = "sft_political_label_reparsed"
SEED = 42


def load_labels(path: Path, split: str) -> pd.DataFrame:
    print(f"Reading {split} labels: {path}", flush=True)

    df = pd.read_csv(
        path,
        usecols=["row_id", LABEL_COLUMN],
    )

    df["split"] = split
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)

    duplicate_count = int(
        df.duplicated(subset=["split", "row_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"{split}: found {duplicate_count} duplicate row IDs"
        )

    print(f"{split} rows: {len(df):,}", flush=True)
    print(
        df[LABEL_COLUMN].value_counts(dropna=False),
        flush=True,
    )

    return df


def save_row_list(
    df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["split", "row_id", LABEL_COLUMN]].to_csv(
        path,
        index=False,
    )
    print(f"Saved row list: {path} ({len(df):,} rows)", flush=True)


def create_row_lists(labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    political = labels[
        labels[LABEL_COLUMN].isin(
            ["center_neutral", "left", "right"]
        )
    ].copy()

    ideological = labels[
        labels[LABEL_COLUMN].isin(["left", "right"])
    ].copy()

    left = labels[
        labels[LABEL_COLUMN] == "left"
    ].copy()

    right = labels[
        labels[LABEL_COLUMN] == "right"
    ].copy()

    left_count = len(left)
    right_count = len(right)

    if left_count < right_count:
        raise ValueError(
            "There are fewer left rows than right rows; "
            "the current balancing rule assumes left > right."
        )

    excess_left_count = left_count - right_count

    rng = random.Random(SEED)
    left_indices = left.index.tolist()
    remove_indices = set(
        rng.sample(left_indices, excess_left_count)
    )

    extra_left = left.loc[
        left.index.isin(remove_indices)
    ].copy()

    row_lists = {
        "no_pol": political,
        "no_ideology": ideological,
        "no_left": left,
        "no_right": right,
        "lr_balanced": extra_left,
    }

    return row_lists


def row_sets_by_split(
    removal_df: pd.DataFrame,
) -> dict[str, set[int]]:
    return {
        split: set(
            removal_df.loc[
                removal_df["split"] == split,
                "row_id",
            ].astype(int)
        )
        for split in FLAT_FILES
    }


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def build_filtered_jsonl(
    input_path: Path,
    output_path: Path,
    remove_ids: set[int],
    split: str,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    removed = 0
    kept = 0

    with input_path.open("r", encoding="utf-8") as src, \
         output_path.open("w", encoding="utf-8") as dst:

        for line in tqdm(src, desc=f"{output_path.parent.name}/{split}"):
            if not line.strip():
                continue

            obj = json.loads(line)
            row_id = int(obj["row_id"])

            total += 1

            if row_id in remove_ids:
                removed += 1
                continue

            dst.write(line)
            kept += 1

    if removed != len(remove_ids):
        raise ValueError(
            f"{split}: expected to remove {len(remove_ids):,} rows "
            f"but removed {removed:,}"
        )

    return {
        "input_rows": total,
        "removed_rows": removed,
        "output_rows": kept,
        "output_path": str(output_path),
    }


def create_full_symlinks(jsonl_base: Path) -> dict:
    full_dir = jsonl_base / "full"
    full_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    for split, source in FLAT_FILES.items():
        destination = (
            full_dir /
            f"dolci_{split}_sft_7b_full.jsonl"
        )

        if destination.exists() or destination.is_symlink():
            destination.unlink()

        relative_target = os.path.relpath(
            source.resolve(),
            start=destination.parent.resolve(),
        )

        destination.symlink_to(relative_target)

        outputs[split] = {
            "source": str(source),
            "symlink": str(destination),
        }

        print(
            f"Created Full symlink: {destination} -> {relative_target}",
            flush=True,
        )

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--row-lists-only",
        action="store_true",
        help="Create removal row lists and report, but do not write JSONLs.",
    )
    args = parser.parse_args()

    row_list_dir = OUTPUT_BASE / "row_lists"
    jsonl_base = OUTPUT_BASE / "jsonl"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    instruct = load_labels(INSTRUCT_LABELS, "instruct")
    think = load_labels(THINK_LABELS, "think")

    labels = pd.concat(
        [instruct, think],
        ignore_index=True,
    )

    print("\nCombined reparsed label distribution:", flush=True)
    print(
        labels[LABEL_COLUMN].value_counts(dropna=False),
        flush=True,
    )

    row_lists = create_row_lists(labels)

    row_list_paths = {}

    for condition, removal_df in row_lists.items():
        path = row_list_dir / f"remove_{condition}_row_ids.csv"
        save_row_list(removal_df, path)
        row_list_paths[condition] = str(path)

    left_total = int(
        (labels[LABEL_COLUMN] == "left").sum()
    )
    right_total = int(
        (labels[LABEL_COLUMN] == "right").sum()
    )

    report = {
        "pipeline": "ModernBERT OR embedding_t0.4 -> Llama 4-way",
        "label_column": LABEL_COLUMN,
        "seed": SEED,
        "candidate_rows": int(len(labels)),
        "label_distribution": {
            str(k): int(v)
            for k, v in labels[
                LABEL_COLUMN
            ].value_counts(dropna=False).items()
        },
        "confirmed_political_rows": int(
            labels[LABEL_COLUMN]
            .isin(["center_neutral", "left", "right"])
            .sum()
        ),
        "ideological_rows": int(
            labels[LABEL_COLUMN]
            .isin(["left", "right"])
            .sum()
        ),
        "original_left_rows": left_total,
        "original_right_rows": right_total,
        "original_left_right_ratio": (
            left_total / right_total if right_total else None
        ),
        "lr_balanced": {
            "left_rows_removed": int(
                len(row_lists["lr_balanced"])
            ),
            "left_rows_remaining": right_total,
            "right_rows_remaining": right_total,
        },
        "removal_counts": {
            condition: {
                "combined": int(len(df)),
                "instruct": int(
                    (df["split"] == "instruct").sum()
                ),
                "think": int(
                    (df["split"] == "think").sum()
                ),
            }
            for condition, df in row_lists.items()
        },
        "row_list_paths": row_list_paths,
        "jsonl_outputs": {},
    }

    if args.row_lists_only:
        report_path = OUTPUT_BASE / "intervention_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"Saved: {report_path}")
        return

    report["jsonl_outputs"]["full"] = create_full_symlinks(
        jsonl_base
    )

    for condition, removal_df in row_lists.items():
        print("\n" + "=" * 100, flush=True)
        print(f"Building intervention: {condition}", flush=True)
        print("=" * 100, flush=True)

        remove_by_split = row_sets_by_split(removal_df)
        condition_outputs = {}

        for split, input_path in FLAT_FILES.items():
            output_path = (
                jsonl_base /
                condition /
                f"dolci_{split}_sft_7b_{condition}.jsonl"
            )

            condition_outputs[split] = build_filtered_jsonl(
                input_path=input_path,
                output_path=output_path,
                remove_ids=remove_by_split[split],
                split=split,
            )

        report["jsonl_outputs"][condition] = condition_outputs

    report_path = OUTPUT_BASE / "intervention_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 100, flush=True)
    print("FINAL REPORT", flush=True)
    print("=" * 100, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved: {report_path}", flush=True)


if __name__ == "__main__":
    main()
