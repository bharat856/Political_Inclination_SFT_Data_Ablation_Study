"""Selection rules taken directly from the authoritative ProbVAA CSV schema.

This module deliberately implements only the supervisor-authorized template
filter.  It does not supply the missing upstream prompt formatter, parser, or
generation behavior.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_TEMPLATE_CSV = (
    REPOSITORY_ROOT
    / "references"
    / "political_content_training_data_fresh"
    / "model_behavior"
    / "data"
    / "en_clf_templates_final.csv"
)
PROMPT_TYPE_COLUMN = "prompt_type"
PERSONAL_PROMPT_TYPE = "personal"
EXPECTED_PERSONAL_TEMPLATES = 6


def load_personal_templates(
    csv_path: Path = AUTHORITATIVE_TEMPLATE_CSV,
) -> list[dict[str, str]]:
    """Return personal templates in their original CSV order.

    Selection is exclusively by the CSV's ``prompt_type`` value.  Template
    IDs, row numbers, and positional slices are intentionally not used.
    """

    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or PROMPT_TYPE_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Authoritative template CSV lacks {PROMPT_TYPE_COLUMN!r}: {csv_path}"
            )
        selected = [
            dict(row)
            for row in reader
            if row[PROMPT_TYPE_COLUMN] == PERSONAL_PROMPT_TYPE
        ]

    if len(selected) != EXPECTED_PERSONAL_TEMPLATES:
        raise ValueError(
            "Expected exactly "
            f"{EXPECTED_PERSONAL_TEMPLATES} personal templates, found {len(selected)}"
        )
    if any(row[PROMPT_TYPE_COLUMN] != PERSONAL_PROMPT_TYPE for row in selected):
        raise AssertionError("Impersonal template survived the prompt_type filter")
    return selected


def expected_generation_count(
    statement_rows: int,
    iterations: int,
    templates: Iterable[dict[str, str]],
) -> int:
    """Calculate the exact generation count without changing source data."""

    return statement_rows * len(list(templates)) * iterations
