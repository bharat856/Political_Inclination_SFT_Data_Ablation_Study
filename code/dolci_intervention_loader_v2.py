from pathlib import Path
from typing import Dict, Iterator, Set
import json
import time

import pandas as pd
from datasets import Features, IterableDataset, Value
from datasets.distributed import split_dataset_by_node
from tqdm.auto import tqdm


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


def log_rank(message: str, rank: int) -> None:
    print(f"[dataset rank={rank}] {message}", flush=True)


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


def load_removal_ids(
    condition: str,
    rank: int,
) -> Dict[str, Set[int]]:
    remove_by_split: Dict[str, Set[int]] = {
        "instruct": set(),
        "think": set(),
    }

    if condition == "full":
        log_rank(
            "Full condition: no removal CSV needs to be loaded.",
            rank,
        )
        return remove_by_split

    path = ROW_LISTS[condition]

    log_rank(f"Reading removal IDs from {path}", rank)
    started = time.time()

    df = pd.read_csv(
        path,
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

    elapsed = time.time() - started

    log_rank(
        (
            f"Removal IDs loaded in {elapsed:.2f}s | "
            f"instruct={len(remove_by_split['instruct']):,} "
            f"think={len(remove_by_split['think']):,}"
        ),
        rank,
    )

    return remove_by_split


def iter_split(
    split: str,
    path: Path,
    remove_ids: Set[int],
    rank: int,
) -> Iterator[dict]:
    log_rank(f"Opening {split} JSONL: {path}", rank)

    seen = 0
    yielded = 0
    removed = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            seen += 1
            obj = json.loads(line)
            row_id = int(obj["row_id"])

            if row_id in remove_ids:
                removed += 1
                continue

            yielded += 1

            if seen % 250_000 == 0:
                log_rank(
                    (
                        f"{split}: raw_seen={seen:,} "
                        f"yielded={yielded:,} "
                        f"removed={removed:,}"
                    ),
                    rank,
                )

            yield {
                "row_id": row_id,
                "text": str(obj.get("text", "")),
                "split": split,
            }

    log_rank(
        (
            f"Finished {split} | "
            f"raw_seen={seen:,} "
            f"yielded={yielded:,} "
            f"removed={removed:,}"
        ),
        rank,
    )


def load_dolci_intervention(
    condition: str,
    *,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    rank: int = 0,
    world_size: int = 1,
    resume_skip_samples: int = 0,
) -> IterableDataset:
    validate_inputs(condition)

    log_rank(
        f"Validated input files for condition={condition}",
        rank,
    )

    remove_by_split = load_removal_ids(condition, rank)

    log_rank(
        (
            f"Creating streaming DOLCI intervention | "
            f"condition={condition}"
        ),
        rank,
    )

    for split in FLAT_FILES:
        log_rank(
            (
                f"{split}: remove "
                f"{len(remove_by_split[split]):,} rows "
                f"from {FLAT_FILES[split]}"
            ),
            rank,
        )

    def combined_generator():
        log_rank("combined_generator() started", rank)

        for split, path in FLAT_FILES.items():
            yield from iter_split(
                split=split,
                path=path,
                remove_ids=remove_by_split[split],
                rank=rank,
            )

        log_rank("combined_generator() exhausted", rank)

    features = Features({
        "row_id": Value("int64"),
        "text": Value("string"),
        "split": Value("string"),
    })

    dataset = IterableDataset.from_generator(
        combined_generator,
        features=features,
    )

    log_rank("HF IterableDataset created.", rank)

    if shuffle_buffer_size > 0:
        log_rank(
            (
                f"Configuring streaming shuffle | "
                f"buffer={shuffle_buffer_size:,} "
                f"seed={seed}"
            ),
            rank,
        )

        dataset = dataset.shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )

        log_rank("Streaming shuffle configured.", rank)

    if world_size > 1:
        log_rank(
            (
                f"Configuring split_dataset_by_node | "
                f"rank={rank}/{world_size}"
            ),
            rank,
        )

        dataset = split_dataset_by_node(
            dataset,
            rank=rank,
            world_size=world_size,
        )

        log_rank("Distributed dataset split configured.", rank)

    if resume_skip_samples > 0:
        base_dataset = dataset

        def progress_generator():
            started = time.time()
            count = 0
            milestone_logged = False

            bar = None

            if rank == 0:
                bar = tqdm(
                    total=resume_skip_samples,
                    desc="Resume replay rank0",
                    unit="samples",
                    dynamic_ncols=True,
                )

            log_rank(
                (
                    "Resume replay instrumentation active | "
                    f"expected_samples={resume_skip_samples:,}"
                ),
                rank,
            )

            for example in base_dataset:
                count += 1

                if bar is not None and count <= resume_skip_samples:
                    bar.update(1)

                    if count % 10_000 == 0:
                        elapsed = time.time() - started
                        rate = count / elapsed if elapsed else 0.0

                        bar.set_postfix(
                            samples_per_s=f"{rate:.2f}",
                        )

                if (
                    not milestone_logged
                    and count >= resume_skip_samples
                ):
                    milestone_logged = True
                    elapsed = time.time() - started

                    if bar is not None:
                        bar.close()

                    log_rank(
                        (
                            "Expected checkpoint replay count reached | "
                            f"samples={count:,} "
                            f"elapsed={elapsed / 3600:.2f}h"
                        ),
                        rank,
                    )

                yield example

        dataset = IterableDataset.from_generator(
            progress_generator,
            features=features,
        )

        log_rank(
            "Resume replay progress wrapper configured.",
            rank,
        )

    return dataset


def expected_row_count(condition: str) -> int:
    validate_inputs(condition)
    return EXPECTED_ROWS[condition]
