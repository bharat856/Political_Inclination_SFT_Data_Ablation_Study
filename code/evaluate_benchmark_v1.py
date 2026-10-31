#!/usr/bin/env python3
"""Evaluate benchmark_v1 prediction CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark_utils import LABELS, resolve_project_path


DEFAULT_PREDICTIONS_DIR = "outputs/benchmark_v1/predictions"
DEFAULT_OUTPUT_DIR = "outputs/benchmark_v1/metrics"
SUMMARY_COLUMNS = [
    "model_name",
    "backend",
    "prompt_version",
    "prediction_file",
    "n_rows",
    "parsed_rows",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "gold_distribution",
    "prediction_distribution",
    "warning",
]
PER_CLASS_COLUMNS = [
    "model_name",
    "backend",
    "prompt_version",
    "class_label",
    "precision",
    "recall",
    "f1",
    "support",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark_v1 predictions.")
    parser.add_argument("--predictions-dir", type=Path, default=Path(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def simple_confusion(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_index = {label: i for i, label in enumerate(LABELS)}
    for true, pred in zip(y_true, y_pred):
        if true in label_index and pred in label_index:
            matrix[label_index[true]][label_index[pred]] += 1
    return matrix


def simple_prf(matrix: list[list[int]]) -> list[dict[str, Any]]:
    rows = []
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(LABELS)) if r != i)
        fn = sum(matrix[i][c] for c in range(len(LABELS)) if c != i)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class_label": label, "precision": precision, "recall": recall, "f1": f1})
    return rows


def evaluate_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(path)
    model_name = str(df["model_name"].iloc[0]) if "model_name" in df.columns and len(df) else path.stem
    prompt_version = str(df["prompt_version"].iloc[0]) if "prompt_version" in df.columns and len(df) else ""
    backend = str(df["backend"].iloc[0]) if "backend" in df.columns and len(df) else ""
    parsed_df = df[df["parse_ok"].astype(bool)].copy() if "parse_ok" in df.columns else df.copy()
    y_true = parsed_df["gold_label"].astype(str).tolist()
    y_pred = parsed_df["parsed_prediction_label"].astype(str).tolist()
    n_rows = len(df)
    parsed_rows = len(parsed_df)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / parsed_rows if parsed_rows else 0.0

    try:
        from sklearn.metrics import classification_report, confusion_matrix, f1_score

        macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if parsed_rows else 0.0
        weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0) if parsed_rows else 0.0
        matrix = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        per_class = [
            {
                "model_name": model_name,
                "backend": backend,
                "prompt_version": prompt_version,
                "class_label": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABELS
        ]
        f1_warning = ""
    except Exception as exc:
        print(f"Warning: sklearn unavailable or failed ({exc}); using simple metrics.", flush=True)
        matrix = simple_confusion(y_true, y_pred)
        prf_rows = simple_prf(matrix)
        macro_f1 = sum(row["f1"] for row in prf_rows) / len(prf_rows)
        supports = [sum(row) for row in matrix]
        total_support = sum(supports)
        weighted_f1 = (
            sum(row["f1"] * support for row, support in zip(prf_rows, supports)) / total_support
            if total_support
            else 0.0
        )
        per_class = [
            {
                "model_name": model_name,
                "backend": backend,
                "prompt_version": prompt_version,
                "support": supports[i],
                **row,
            }
            for i, row in enumerate(prf_rows)
        ]
        f1_warning = "sklearn was unavailable; F1 was computed with a simple fallback."

    summary = {
        "model_name": model_name,
        "backend": backend,
        "prompt_version": prompt_version,
        "prediction_file": str(path),
        "n_rows": n_rows,
        "parsed_rows": parsed_rows,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "gold_distribution": dict(sorted(df["gold_label"].astype(str).value_counts().to_dict().items())),
        "prediction_distribution": dict(
            sorted(df["parsed_prediction_label"].fillna("unparsed").astype(str).value_counts().to_dict().items())
        ),
        "warning": f1_warning,
    }
    confusion = {
        "model_name": model_name,
        "labels": LABELS,
        "matrix": matrix,
    }
    return summary, per_class, confusion


def write_report(summary_df: pd.DataFrame, output_path: Path, dataset_size: int, warnings: list[str]) -> None:
    table_columns = ["model_name", "backend", "prompt_version", "n_rows", "parsed_rows", "accuracy", "macro_f1", "weighted_f1"]
    report = [
        "# Benchmark v1 Report",
        "",
        f"Dataset size: {dataset_size} rows.",
        "",
        "Important warning: chunk-level labels are approximations when gold labels come from manifesto-level RILE or party/document-level ideology. Some apparent model errors may reflect label mismatch rather than a classification mistake.",
        "",
        "## Model Comparison",
        "",
        summary_df[table_columns].to_markdown(index=False) if not summary_df.empty else "No prediction files found.",
        "",
        "## Common Failure Modes",
        "",
        "- Center confusion: moderate or mixed policy text can be hard to separate from weak left/right signals.",
        "- Local text vs party-level ideology mismatch: a right party can write a locally left-sounding welfare passage, and vice versa.",
        "- Multilingual difficulty: manifesto chunks span languages and may include OCR or formatting artifacts.",
    ]
    if warnings:
        report.extend(["", "## Warnings", ""])
        report.extend(f"- {warning}" for warning in warnings if warning)
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions_dir = resolve_project_path(args.predictions_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(predictions_dir.glob("*.csv"))
    print(f"Found {len(files)} prediction CSVs in {predictions_dir}", flush=True)
    summaries = []
    per_class_rows = []
    confusion_matrices = {}
    warnings = []

    for path in files:
        print(f"Evaluating {path}", flush=True)
        summary, per_class, confusion = evaluate_file(path)
        summaries.append(summary)
        per_class_rows.extend(per_class)
        confusion_matrices[summary["model_name"]] = confusion
        if summary.get("warning"):
            warnings.append(summary["warning"])

    summary_df = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    per_class_df = pd.DataFrame(per_class_rows, columns=PER_CLASS_COLUMNS)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    (output_dir / "confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset_size = int(summary_df["n_rows"].max()) if not summary_df.empty else 0
    write_report(summary_df, output_dir / "benchmark_report.md", dataset_size, warnings)
    print(f"Saved metrics to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
