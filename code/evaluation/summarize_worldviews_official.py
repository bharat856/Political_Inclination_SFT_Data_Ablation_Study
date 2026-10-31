"""CPU-only Agree/Disagree/NA summary for isolated OfficialSFT runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import worldviews_probvaa as probvaa


def summarize(input_dir):
    input_dir = Path(input_dir).resolve()
    config_path = input_dir / "configuration.json"
    runtime_path = input_dir / "runtime.json"
    complete_path = input_dir / "COMPLETE"
    for path in (config_path, runtime_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete official run: missing {path}")
    config = json.loads(config_path.read_text())
    if config.get("model") != probvaa.OFFICIAL_MODEL:
        raise ValueError("Input is not an isolated OfficialSFT run")
    segments = sorted(input_dir.glob("segment_*.jsonl"))
    if not segments:
        raise FileNotFoundError("Official run has no generation segment")
    _, _, parser, _ = probvaa.reference_modules()
    counts = Counter()
    diagnostics = Counter()
    prompt_blocks = 0
    response_ids = set()
    final_template_ids = set()
    statement_ids = set()
    source_statement_ids = set()
    for segment in segments:
        with segment.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                prompt_blocks += 1
                completions = row["generated_completions"]
                indices = row["response_indices"]
                parsed = row["parsed"]
                row_diagnostics = row["parsing_diagnostics"]
                expected = probvaa.GENERATION["num_return_sequences"]
                if not all(len(values) == expected for values in
                           (completions, indices, parsed, row_diagnostics)):
                    raise ValueError(f"Incomplete response block: {segment}:{line_number}")
                if indices != list(range(expected)):
                    raise ValueError(f"Changed response indices: {segment}:{line_number}")
                reparsed = [parser(completion) for completion in completions]
                stored = [None if value == "not_applied" else value for value in parsed]
                if reparsed != stored:
                    raise ValueError(f"Saved parser output differs: {segment}:{line_number}")
                for response_index, (value, diagnostic) in enumerate(zip(reparsed, row_diagnostics)):
                    response_id = (row["prompt_index"], response_index)
                    if response_id in response_ids:
                        raise ValueError(f"Duplicate response ID: {response_id}")
                    response_ids.add(response_id)
                    counts["NA" if value is None else "Agree" if value == 1 else "Disagree"] += 1
                    diagnostics["ambiguous_both_polarities"] += int(
                        diagnostic["contains_both_label_polarities"])
                    diagnostics["hit_max_new_tokens"] += int(diagnostic["hit_max_new_tokens"])
                final_template_ids.add(row["final_template_id"])
                statement_ids.add(row["statement_id"])
                source_statement_ids.add(probvaa.source_statement_id(row["statement_id"]))
    if prompt_blocks != config["prompt_count"]:
        raise ValueError(f"Prompt count mismatch: {prompt_blocks}/{config['prompt_count']}")
    total = sum(counts.values())
    parsed_denominator = counts["Agree"] + counts["Disagree"]
    if total != prompt_blocks * probvaa.GENERATION["num_return_sequences"]:
        raise ValueError("Response-count arithmetic mismatch")
    return {
        "model": probvaa.OFFICIAL_MODEL,
        "counts": {label: counts[label] for label in ("Agree", "Disagree", "NA")},
        "denominators": {"all_responses": total, "parsed_responses": parsed_denominator},
        "proportions_all_responses": {
            label: counts[label] / total for label in ("Agree", "Disagree", "NA")},
        "proportions_among_parsed": {
            "Agree": (counts["Agree"] / parsed_denominator if parsed_denominator else None),
            "Disagree": (counts["Disagree"] / parsed_denominator if parsed_denominator else None),
        },
        "diagnostics": dict(diagnostics),
        "coverage": {
            "prompt_blocks": prompt_blocks, "responses": total,
            "source_statements": len(source_statement_ids),
            "formulations": len(statement_ids),
            "final_template_ids": sorted(final_template_ids),
        },
        "na_convention": "NA is reported in the all-response denominator and excluded only from proportions_among_parsed.",
        "input": {
            "directory": str(input_dir),
            "configuration_sha256": probvaa.digest(config_path),
            "runtime_sha256": probvaa.digest(runtime_path),
            "complete_sha256": probvaa.digest(complete_path),
            "segments": {path.name: probvaa.digest(path) for path in segments},
        },
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input-dir", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    args = cli.parse_args()
    result = summarize(args.input_dir)
    output = probvaa.create_new_output_directory(args.output_dir)
    (output / "response_distribution.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
