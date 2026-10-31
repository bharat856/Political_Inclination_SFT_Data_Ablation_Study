from pathlib import Path
from typing import Dict, Iterator, Set
import json

import pandas as pd
from datasets import Features, IterableDataset, Value
from datasets.distributed import split_dataset_by_node


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FLAT_FILES = {
    "instruct": PROJECT_ROOT / (
        "data/processed/dolci_olmo3_sft/"
        "dolci_instruct_sft_7b_flattened.jsonl"
    ),
    "think": PROJECT_ROOT / (
        "data/processed/dolci_olmo3_sft/"
        "dolci_think_sft_7b_flattened.jsonl"
    ),
}

ROW_LISTS = {
    "no_pol": PROJECT_ROOT / (
        "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
        "row_lists/remove_no_pol_row_ids.csv"
    ),
    "no_ideology": PROJECT_ROOT / (
        "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
        "row_lists/remove_no_ideology_row_ids.csv"
    ),
    "no_left": PROJECT_ROOT / (
        "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
        "row_lists/remove_no_left_row_ids.csv"
    ),
    "no_right": PROJECT_ROOT / (
        "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
        "row_lists/remove_no_right_row_ids.csv"
    ),
    "lr_balanced": PROJECT_ROOT / (
        "data/interventions/dolci_olmo3_sft_union_mpnet_t04/"
        "row_lists/remove_lr_balanced_row_ids.csv"
    ),
}

VALID_CONDITIONS = [
    "full",
    "no_pol",
    "no_ideology",
    "no_left",
    "no_right",
    "lr_balanced",
]

EXPECTED_ROWS = {
    "full": 4_420_290,
    "no_pol": 3_726_651,
    "no_ideology": 4_389_620,
    "no_left": 4_393_578,
    "no_right": 4_416_332,
    "lr_balanced": 4_397_536,
}


def validate_inputs(condition: str) -> None:
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Choose from: {', '.join(VALID_CONDITIONS)}"
        )

    for split, path in FLAT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing flattened file for {split}: {path}"
            )

    if condition != "full" and not ROW_LISTS[condition].exists():
        raise FileNotFoundError(
            f"Missing intervention row list: {ROW_LISTS[condition]}"
        )


def load_removal_ids(condition: str) -> Dict[str, Set[int]]:
    remove_by_split: Dict[str, Set[int]] = {
        "instruct": set(),
        "think": set(),
    }

    if condition == "full":
        return remove_by_split

    df = pd.read_csv(
        ROW_LISTS[condition],
        usecols=["split", "row_id"],
    )

    df["row_id"] = pd.to_numeric(
        df["row_id"],
        errors="raise",
    ).astype(int)

    unknown = set(df["split"].dropna().unique()) - set(FLAT_FILES)
    if unknown:
        raise ValueError(f"Unexpected split values: {unknown}")

    for split in FLAT_FILES:
        remove_by_split[split] = set(
            df.loc[df["split"] == split, "row_id"].tolist()
        )

    return remove_by_split


def iter_split(
    split: str,
    path: Path,
    remove_ids: Set[int],
) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            row_id = int(obj["row_id"])

            if row_id in remove_ids:
                continue

            yield {
                "row_id": row_id,
                "text": str(obj.get("text", "")),
                "split": split,
            }


def load_dolci_intervention(
    condition: str,
    *,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
) -> IterableDataset:
    validate_inputs(condition)
    remove_by_split = load_removal_ids(condition)

    print("=" * 100)
    print(f"Loading streaming DOLCI intervention: {condition}")
    print("=" * 100)

    for split in FLAT_FILES:
        print(
            f"{split}: remove {len(remove_by_split[split]):,} rows "
            f"from {FLAT_FILES[split]}"
        )

    def combined_generator():
        for split, path in FLAT_FILES.items():
            yield from iter_split(
                split=split,
                path=path,
                remove_ids=remove_by_split[split],
            )

    features = Features({
        "row_id": Value("int64"),
        "text": Value("string"),
        "split": Value("string"),
    })

    dataset = IterableDataset.from_generator(
        combined_generator,
        features=features,
    )

    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )

    if world_size > 1:
        dataset = split_dataset_by_node(
            dataset,
            rank=rank,
            world_size=world_size,
        )

    return dataset


def expected_row_count(condition: str) -> int:
    validate_inputs(condition)
    return EXPECTED_ROWS[condition]
