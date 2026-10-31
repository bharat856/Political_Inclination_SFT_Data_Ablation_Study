#!/usr/bin/env python3
"""Shared helpers for Manifesto benchmark v1."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/fs/scratch/users/bharatpi/causal_political_content")
LABEL_TO_NUM = {"left": 1, "center": 2, "right": 3}
NUM_TO_LABEL = {value: key for key, value in LABEL_TO_NUM.items()}
LABELS = ["left", "center", "right"]


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    aliases = {
        "1": "left",
        "l": "left",
        "left-wing": "left",
        "left wing": "left",
        "2": "center",
        "centre": "center",
        "centrist": "center",
        "neutral": "center",
        "moderate": "center",
        "3": "right",
        "r": "right",
        "right-wing": "right",
        "right wing": "right",
    }
    label = aliases.get(label, label)
    if label in LABEL_TO_NUM:
        return label
    return None


def safe_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name.strip())
    return cleaned.strip("_") or "model"


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_prediction(raw_output: Any) -> tuple[int | None, str | None, bool]:
    text = "" if raw_output is None else str(raw_output).strip()
    if text in {"1", "2", "3"}:
        num = int(text)
        return num, NUM_TO_LABEL[num], True

    matches = re.findall(r"(?<!\d)([123])(?!\d)", text)
    if len(set(matches)) == 1:
        num = int(matches[0])
        return num, NUM_TO_LABEL[num], True

    return None, None, False
