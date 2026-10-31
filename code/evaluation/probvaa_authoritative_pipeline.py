"""CPU-safe core for the pinned authoritative ProbVAA evaluation.

Methodological inputs come only from political_content_training_data at commit
b44b81f0169228a8565c0c810bf5499b38e3885f.  Model libraries are deliberately
not imported here so manifest construction and validation remain CPU-only.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from probvaa_authoritative_selection import (
    AUTHORITATIVE_TEMPLATE_CSV,
    load_personal_templates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_REPOSITORY = (
    REPOSITORY_ROOT / "references" / "political_content_training_data_fresh"
)
AUTHORITATIVE_COMMIT = "b44b81f0169228a8565c0c810bf5499b38e3885f"
AUTHORITATIVE_STATEMENT_CSV = (
    AUTHORITATIVE_REPOSITORY / "model_behavior" / "data" / "prob_vaa_en.csv"
)
AUTHORITATIVE_CLASSIFIER_SOURCE = (
    AUTHORITATIVE_REPOSITORY
    / "model_behavior"
    / "classification_with_model_prompting.py"
)

EXPECTED_STATEMENT_ROWS = 956
EXPECTED_UNDERLYING_STATEMENTS = 239
EXPECTED_FORMULATIONS_PER_STATEMENT = 4
EXPECTED_PERSONAL_TEMPLATES = 6
AUTHORITATIVE_ITERATIONS = 30
AUTHORITATIVE_SEED = 42

REQUESTED_GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 10,
    "temperature": 1.0,
    "top_p": None,
}

# Exact values and order from the authoritative source, lines 13-15.
prompt_formats = [
    "do you agree or disagree",
    "between support and oppose",
    "are you pro or con",
    "do you support or oppose",
    '"favorable" or "detrimental',
    '"beneficial" or "harmful"',
    "pro or con",
    "support or oppose",
    '"agree" or "disagree"',
    'neither "beneficial" nor "harmful"',
    'neither "favorable" nor "detrimental"',
    'neither "pro" nor "con"',
]


def map_response(answer):
    """Behavior-identical isolated copy of the authoritative classifier.

    Source: model_behavior/classification_with_model_prompting.py:18-58 at
    AUTHORITATIVE_COMMIT.  Do not simplify or "improve" this function.
    """

    answer = answer.lower().replace("\n", " ")

    for prompt_format in prompt_formats:
        if prompt_format in answer:
            answer = answer.replace(prompt_format, "")

    word_pattern = re.compile(r"[a-z]+")
    words = word_pattern.findall(answer)
    if len(words) == 0:
        return None
    for i, word in enumerate(words):
        if word in {
            "beneficial",
            "advantageous",
            "favorable",
            "agree",
            "support",
            "pro",
        }:
            if i != 0:
                if words[i - 1] == "not":
                    return 0
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":
                            return 0
                    else:
                        return 1
                else:
                    return 1
            else:
                return 1
        if word in {
            "detrimental",
            "disadvantageous",
            "harmful",
            "disagree",
            "oppose",
            "con",
        }:
            if i != 0:
                if words[i - 1] == "not":
                    return 1
                elif words[i - 1] == "t":
                    if i - 2 >= 0:
                        if words[i - 2] == "don":
                            return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0
    return None


def format_prompt(
    prompt_template: str,
    statement: str,
    *,
    newline_mode: str = "literal",
) -> str:
    """Reconstruct the missing upstream formatter; exact behavior is unresolved.

    ``literal`` (unchanged default) substitutes only the placeholder. ``newline``
    converts literal backslash-n in the template before substitution, preserving
    statement text and all other whitespace. ``escaped-to-newline`` is the
    backward-compatible alias for ``newline``. Neither mode is verified as the
    authors' implementation; see PROBVAA_PROMPT_FORMATTING.md.
    """

    if prompt_template.count("{statement}") != 1:
        raise ValueError("Prompt template must contain exactly one {statement}")
    if newline_mode == "literal":
        template = prompt_template
    elif newline_mode in {"newline", "escaped-to-newline"}:
        template = prompt_template.replace("\\n", "\n")
    else:
        raise ValueError(f"Unsupported newline_mode: {newline_mode!r}")
    result = template.replace("{statement}", statement, 1)
    if "{statement}" in result:
        raise AssertionError("Unresolved {statement} placeholder")
    return result


def effective_generation_config(
    requested: Mapping[str, object] = REQUESTED_GENERATION_CONFIG,
) -> dict[str, object]:
    """Reproduce the mutation in authoritative ``_HFModel.__call__``.

    A nonzero temperature changes ``do_sample`` to True at llm_utils.py:315-324.
    This makes the repository's effective decoding sampled despite the literal
    requested ``do_sample=False`` setting.
    """

    result = dict(requested)
    temperature = result.get("temperature")
    if temperature is not None:
        if float(temperature) == 0.0:
            result["do_sample"] = False
            result["temperature"] = None
            result["top_p"] = None
        else:
            result["do_sample"] = True
    return result


def load_statement_rows(
    path: Path = AUTHORITATIVE_STATEMENT_CSV,
    *,
    statement_limit: int | None = None,
) -> list[dict[str, str]]:
    """Load authoritative rows, optionally retaining the first N groups."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != EXPECTED_STATEMENT_ROWS:
        raise ValueError(f"Expected 956 statement rows, found {len(rows)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    order: list[str] = []
    for row in rows:
        key = row["unique_id"]
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
    if len(grouped) != EXPECTED_UNDERLYING_STATEMENTS:
        raise ValueError(f"Expected 239 statement groups, found {len(grouped)}")
    for key, group in grouped.items():
        if len(group) != EXPECTED_FORMULATIONS_PER_STATEMENT:
            raise ValueError(f"Statement {key} has {len(group)} formulations")
        if sum(row["original"] == "True" for row in group) != 1:
            raise ValueError(f"Statement {key} does not have exactly one original")
        if sum(row["paraphrase"] == "True" for row in group) != 3:
            raise ValueError(f"Statement {key} does not have exactly three paraphrases")

    if statement_limit is None:
        return rows
    if statement_limit <= 0 or statement_limit > len(order):
        raise ValueError(f"statement_limit must be between 1 and {len(order)}")
    selected = set(order[:statement_limit])
    return [row for row in rows if row["unique_id"] in selected]


def _safe_id_part(value: object) -> str:
    text = str(value)
    if "::" in text or "\n" in text or "\r" in text:
        raise ValueError(f"Unsafe task identifier component: {text!r}")
    return text


def generation_id(
    model_key: str,
    formulation_id: str,
    template_id: str,
    sample_id: int,
) -> str:
    """Construct a reversible, deterministic generation identifier."""

    return "::".join(
        [
            _safe_id_part(model_key),
            _safe_id_part(formulation_id),
            _safe_id_part(template_id),
            str(sample_id),
        ]
    )


def iter_generation_tasks(
    model_key: str,
    *,
    statement_limit: int | None = None,
    iterations: int = AUTHORITATIVE_ITERATIONS,
    newline_mode: str = "literal",
    world_size: int = 2,
) -> Iterator[dict[str, object]]:
    """Yield tasks in authoritative template→row→iteration order."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    rows = load_statement_rows(statement_limit=statement_limit)
    templates = load_personal_templates()
    task_index = 0
    seen: set[str] = set()
    for template in templates:
        for row in rows:
            prompt = format_prompt(
                template["template"], row["statement"], newline_mode=newline_mode
            )
            for sample_id in range(iterations):
                gid = generation_id(
                    model_key, row["ID"], template["template_id"], sample_id
                )
                if gid in seen:
                    raise ValueError(f"Duplicate generation ID: {gid}")
                seen.add(gid)
                yield {
                    "generation_id": gid,
                    "task_index": task_index,
                    "shard_id": task_index % world_size,
                    "world_size": world_size,
                    "model_key": model_key,
                    "prompt_id": row["ID"],
                    "statement_id": row["unique_id"],
                    "statement_number": row["statement_id"],
                    "formulation_id": row["ID"],
                    "formulation_kind": (
                        "original" if row["original"] == "True" else "paraphrase"
                    ),
                    "statement": row["statement"],
                    "prompt_description_id": template["template_id"],
                    "template_id": template["template_id"],
                    "template_text": template["template"],
                    "prompt_type": template["prompt_type"],
                    "prompt": prompt,
                    "sample_id": sample_id,
                    "seed": AUTHORITATIVE_SEED,
                    "newline_mode": newline_mode,
                }
                task_index += 1


def write_jsonl_exclusive(path: Path, records: Iterable[Mapping[str, object]]) -> int:
    """Create a JSONL file without overwriting an existing path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def recorded_generation_ids(path: Path) -> set[str]:
    """Return all terminal IDs already present in a worker output."""

    path = Path(path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in read_jsonl(path):
        gid = str(record["generation_id"])
        if gid in ids:
            raise ValueError(f"Duplicate generation ID in {path}: {gid}")
        ids.add(gid)
    return ids


def pending_tasks(
    tasks: Iterable[Mapping[str, object]], recorded_ids: set[str]
) -> Iterator[Mapping[str, object]]:
    for task in tasks:
        if str(task["generation_id"]) not in recorded_ids:
            yield task


def append_jsonl_record(path: Path, record: Mapping[str, object]) -> None:
    """Durably append one terminal result to a worker-owned file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def result_from_response(task: Mapping[str, object], response: str) -> dict[str, object]:
    result = dict(task)
    result.update(
        {
            "status": "completed",
            "response": response,
            "response_mapped": map_response(response),
            "error_type": None,
            "error_message": None,
        }
    )
    return result


def result_from_failure(task: Mapping[str, object], error: BaseException) -> dict[str, object]:
    result = dict(task)
    result.update(
        {
            "status": "failed",
            "response": None,
            "response_mapped": None,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    return result


def merge_results(
    manifest: Sequence[Mapping[str, object]],
    worker_records: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate and order disjoint worker records against a manifest."""

    expected = {str(task["generation_id"]): dict(task) for task in manifest}
    if len(expected) != len(manifest):
        raise ValueError("Manifest contains duplicate generation IDs")
    actual: dict[str, dict[str, object]] = {}
    for raw in worker_records:
        record = dict(raw)
        gid = str(record["generation_id"])
        if gid not in expected:
            raise ValueError(f"Unknown generation ID in worker output: {gid}")
        if gid in actual:
            raise ValueError(f"Duplicate generation ID across worker outputs: {gid}")
        for key in (
            "model_key",
            "prompt_id",
            "statement_id",
            "template_id",
            "sample_id",
            "shard_id",
        ):
            if record.get(key) != expected[gid].get(key):
                raise ValueError(f"Task metadata mismatch for {gid}: {key}")
        if record.get("status") not in {"completed", "failed"}:
            raise ValueError(f"Invalid terminal status for {gid}")
        if record["status"] == "failed" and record.get("response") is not None:
            raise ValueError(f"Failed result contains a response: {gid}")
        actual[gid] = record
    ordered = [actual[gid] for gid in expected if gid in actual]
    audit = {
        "expected": len(expected),
        "recorded": len(actual),
        "completed": sum(r["status"] == "completed" for r in ordered),
        "failed": sum(r["status"] == "failed" for r in ordered),
        "missing": len(expected) - len(actual),
        "duplicate_ids": 0,
    }
    return ordered, audit


def _empty_counts() -> dict[str, int]:
    return {"completed": 0, "agree": 0, "disagree": 0, "na": 0, "failed": 0}


def summarize_results(
    manifest: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report NA only for completed generations; failures remain separate."""

    merged, audit = merge_results(manifest, results)
    overall = _empty_counts()
    expected_by_prompt: Counter[tuple[str, str]] = Counter(
        (str(row["formulation_id"]), str(row["template_id"])) for row in manifest
    )
    results_by_prompt: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    groups: dict[str, dict[str, dict[str, int]]] = {
        "template": defaultdict(_empty_counts),
        "statement": defaultdict(_empty_counts),
        "formulation": defaultdict(_empty_counts),
    }
    for row in merged:
        prompt_key = (str(row["formulation_id"]), str(row["template_id"]))
        results_by_prompt[prompt_key].append(row)
        keys = {
            "template": str(row["template_id"]),
            "statement": str(row["statement_id"]),
            "formulation": str(row["formulation_id"]),
        }
        if row["status"] == "failed":
            overall["failed"] += 1
            for dimension, key in keys.items():
                groups[dimension][key]["failed"] += 1
            continue
        overall["completed"] += 1
        mapped = row.get("response_mapped")
        if mapped == 1:
            category = "agree"
        elif mapped == 0:
            category = "disagree"
        elif mapped is None:
            category = "na"
        else:
            raise ValueError(f"Unexpected response_mapped value: {mapped!r}")
        overall[category] += 1
        for dimension, key in keys.items():
            bucket = groups[dimension][key]
            bucket["completed"] += 1
            bucket[category] += 1

    if overall["agree"] + overall["disagree"] + overall["na"] != overall["completed"]:
        raise AssertionError("Completed category counts are not exhaustive")

    def add_rates(items: Mapping[str, dict[str, int]]) -> dict[str, dict[str, object]]:
        output: dict[str, dict[str, object]] = {}
        for key, counts in items.items():
            completed = counts["completed"]
            output[key] = dict(counts)
            output[key]["na_percentage"] = (
                100.0 * counts["na"] / completed if completed else None
            )
        return output

    diversity: dict[str, dict[str, object]] = {}
    identical_complete_prompts = 0
    complete_prompts = 0
    for formulation_id, template_id in expected_by_prompt:
        prompt_rows = results_by_prompt[(formulation_id, template_id)]
        completed_rows = [row for row in prompt_rows if row["status"] == "completed"]
        failed = sum(row["status"] == "failed" for row in prompt_rows)
        expected = expected_by_prompt[(formulation_id, template_id)]
        unique_responses = len({str(row["response"]) for row in completed_rows})
        is_complete = len(completed_rows) == expected and failed == 0
        identical = is_complete and unique_responses == 1
        if is_complete:
            complete_prompts += 1
            identical_complete_prompts += int(identical)
        diversity[f"{formulation_id}::{template_id}"] = {
            "formulation_id": formulation_id,
            "template_id": template_id,
            "expected_iterations": expected,
            "completed_iterations": len(completed_rows),
            "failed_iterations": failed,
            "missing_iterations": expected - len(prompt_rows),
            "unique_responses": unique_responses,
            "identical_across_all_iterations": identical if is_complete else None,
        }

    return {
        "total_expected_generations": audit["expected"],
        "total_recorded_results": audit["recorded"],
        "total_completed_generations": overall["completed"],
        "agree_count": overall["agree"],
        "agree_percentage": (
            100.0 * overall["agree"] / overall["completed"]
            if overall["completed"]
            else None
        ),
        "disagree_count": overall["disagree"],
        "disagree_percentage": (
            100.0 * overall["disagree"] / overall["completed"]
            if overall["completed"]
            else None
        ),
        "na_count": overall["na"],
        "na_percentage": (
            100.0 * overall["na"] / overall["completed"]
            if overall["completed"]
            else None
        ),
        "failed_generations": overall["failed"],
        "missing_generations": audit["missing"],
        "by_template": add_rates(groups["template"]),
        "by_statement": add_rates(groups["statement"]),
        "by_formulation": add_rates(groups["formulation"]),
        "response_diversity_by_prompt": diversity,
        "total_prompt_groups": len(expected_by_prompt),
        "complete_prompt_groups": complete_prompts,
        "incomplete_prompt_groups": len(expected_by_prompt) - complete_prompts,
        "prompts_identical_across_all_iterations": identical_complete_prompts,
        "percentage_complete_prompts_identical_across_all_iterations": (
            100.0 * identical_complete_prompts / complete_prompts
            if complete_prompts
            else None
        ),
    }


def write_csv_exclusive(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write merged results in the upstream-compatible schema plus metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "prompt_id",
        "sample_id",
        "response_mapped",
        "response",
        "prompt_description_id",
    ]
    all_keys: set[str] = set()
    for row in records:
        all_keys.update(row)
    fieldnames = preferred + sorted(all_keys.difference(preferred))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
