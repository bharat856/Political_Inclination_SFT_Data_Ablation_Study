#!/usr/bin/env python3
"""Build the first reusable Manifesto chunk benchmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from benchmark_utils import LABEL_TO_NUM, normalize_label, resolve_project_path, timestamp_utc, write_json


DEFAULT_INPUT = "data/processed/manifesto/chunks/sample_30_manifesto_chunks_strict_balanced.csv"
DEFAULT_OUTPUT = "data/processed/benchmark_v1/benchmark_v1.csv"
REQUIRED_COLUMNS = [
    "doc_id",
    "chunk_id",
    "chunk_index",
    "countryname",
    "date",
    "party",
    "partyname",
    "rile",
    "label_group",
    "rile_label_3way",
    "text",
    "word_count",
]
MANDATORY_COLUMNS = ["text", "rile_label_3way"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark_v1.csv from Manifesto chunks.")
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--make-splits", action="store_true")
    parser.add_argument("--balance", choices=["none", "undersample"], default="none")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_mandatory = [col for col in MANDATORY_COLUMNS if col not in df.columns]
    if missing_mandatory:
        raise ValueError(f"Missing mandatory columns: {missing_mandatory}")

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            print(f"Warning: missing metadata column {column!r}; filling with empty values.", flush=True)
            df[column] = ""
    return df


def add_split(df: pd.DataFrame, make_splits: bool, random_seed: int) -> pd.Series:
    if not make_splits or len(df) < 30:
        return pd.Series(["test"] * len(df), index=df.index)

    shuffled = df.sample(frac=1.0, random_state=random_seed).index.tolist()
    split_values = pd.Series(["test"] * len(df), index=df.index)
    train_end = int(len(shuffled) * 0.7)
    dev_end = int(len(shuffled) * 0.85)
    split_values.loc[shuffled[:train_end]] = "train"
    split_values.loc[shuffled[train_end:dev_end]] = "dev"
    split_values.loc[shuffled[dev_end:]] = "test"
    return split_values


def apply_balance(df: pd.DataFrame, balance: str, random_seed: int) -> pd.DataFrame:
    if balance == "none" or df.empty:
        return df

    min_count = int(df["gold_label"].value_counts().min())
    pieces = [
        group.sample(n=min_count, random_state=random_seed)
        for _, group in df.groupby("gold_label", sort=True)
    ]
    return pd.concat(pieces).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def build_benchmark(args: argparse.Namespace) -> pd.DataFrame:
    input_path = resolve_project_path(args.input)
    output_path = resolve_project_path(args.output)
    print(f"Reading input CSV: {input_path}", flush=True)
    df = pd.read_csv(input_path)
    df = ensure_columns(df)

    df["gold_label"] = df["rile_label_3way"].map(normalize_label)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    before = len(df)
    df = df[(df["text"] != "") & df["gold_label"].notna()].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with empty text or invalid labels.", flush=True)

    df = apply_balance(df, args.balance, args.random_seed)
    df["benchmark_id"] = [f"benchmark_v1_{i:06d}" for i in range(1, len(df) + 1)]
    df["gold_label_num"] = df["gold_label"].map(LABEL_TO_NUM).astype(int)
    df["text_length_chars"] = df["text"].str.len()
    df["prompt_version"] = args.prompt_version
    df["split"] = add_split(df, args.make_splits, args.random_seed)

    leading = [
        "benchmark_id",
        "gold_label",
        "gold_label_num",
        "text_length_chars",
        "prompt_version",
        "split",
    ]
    ordered_columns = leading + [col for col in df.columns if col not in leading]
    df = df[ordered_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved benchmark CSV: {output_path}", flush=True)

    metadata = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "number_of_rows": int(len(df)),
        "label_distribution": df["gold_label"].value_counts().sort_index().to_dict(),
        "country_distribution": df["countryname"].fillna("").astype(str).value_counts().head(50).to_dict(),
        "creation_timestamp": timestamp_utc(),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    write_json(metadata_path, metadata)
    print(f"Saved metadata JSON: {metadata_path}", flush=True)
    print(f"Label distribution: {metadata['label_distribution']}", flush=True)
    return df


def main() -> None:
    build_benchmark(parse_args())


if __name__ == "__main__":
    main()
