"""CPU-only validation for the pinned authoritative ProbVAA pipeline."""

import ast
import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from probvaa_authoritative_pipeline import (
    AUTHORITATIVE_CLASSIFIER_SOURCE,
    AUTHORITATIVE_ITERATIONS,
    REQUESTED_GENERATION_CONFIG,
    effective_generation_config,
    format_prompt,
    iter_generation_tasks,
    load_statement_rows,
    map_response,
    merge_results,
    pending_tasks,
    recorded_generation_ids,
    result_from_failure,
    result_from_response,
    summarize_results,
    write_jsonl_exclusive,
)
from probvaa_authoritative_selection import AUTHORITATIVE_TEMPLATE_CSV, load_personal_templates
from run_probvaa_authoritative import LocalHFGenerator, SeedController, parser, validate_resume_prefix


def authoritative_map_response():
    """Extract the original function without importing its ML dependencies."""

    tree = ast.parse(AUTHORITATIVE_CLASSIFIER_SOURCE.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "prompt_formats"
            for target in node.targets
        )
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "map_response"
    )
    namespace = {"re": re, "prompt_formats": ast.literal_eval(assignment.value)}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(AUTHORITATIVE_CLASSIFIER_SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["map_response"]


class AuthoritativePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_statement_rows()
        cls.templates = load_personal_templates()

    def test_authoritative_data_counts(self):
        self.assertEqual(len(self.rows), 956)
        self.assertEqual(len({row["unique_id"] for row in self.rows}), 239)
        self.assertEqual(len(self.templates), 6)
        self.assertTrue(all(t["prompt_type"] == "personal" for t in self.templates))

    def test_prompt_formatting_preserves_template(self):
        statement = "A deterministic test statement."
        for template in self.templates:
            raw = template["template"]
            prompt = format_prompt(raw, statement)
            self.assertNotIn("{statement}", prompt)
            self.assertIn(statement, prompt)
            self.assertEqual(prompt, raw.replace("{statement}", statement, 1))
            self.assertEqual(prompt, format_prompt(raw, statement))

    def test_optional_newline_mode_only_changes_template_escapes(self):
        raw = self.templates[0]["template"]
        statement = r"statement with literal \n content"
        literal = format_prompt(raw, statement, newline_mode="literal")
        converted = format_prompt(raw, statement, newline_mode="escaped-to-newline")
        self.assertIn(r"\n", literal)
        self.assertIn(statement, converted)
        self.assertNotEqual(literal, converted)

    def test_actual_csv_prompts_and_repr_in_both_modes(self):
        with AUTHORITATIVE_TEMPLATE_CSV.open(newline="", encoding="utf-8") as handle:
            templates = list(csv.DictReader(handle))
        self.assertEqual(len(templates), 12)
        self.assertEqual(sum(t["template"].count(r"\n") for t in templates), 25)
        for template in templates:
            raw = template["template"]
            self.assertNotIn("\n", raw)
            for row in self.rows:
                statement = row["statement"]
                literal = format_prompt(raw, statement)
                newline = format_prompt(raw, statement, newline_mode="newline")
                self.assertEqual(literal, raw.replace("{statement}", statement))
                self.assertEqual(newline, raw.replace(r"\n", "\n").replace("{statement}", statement))
                self.assertEqual(ast.literal_eval(repr(literal)), literal)
                self.assertEqual(ast.literal_eval(repr(newline)), newline)
                self.assertIn(r"\\nStatement:", repr(literal))
                self.assertIn(r"\nStatement:", repr(newline))
                self.assertNotEqual(repr(literal), repr(newline))

    def test_newline_alias_preserves_statement_and_other_whitespace(self):
        raw = " \t" + self.templates[0]["template"] + "\n "
        statement = r"A statement with \n and {other} text."
        expected = raw.replace(r"\n", "\n").replace("{statement}", statement)
        for mode in ("newline", "escaped-to-newline"):
            self.assertEqual(format_prompt(raw, statement, newline_mode=mode), expected)
        with self.assertRaises(ValueError):
            format_prompt(raw, statement, newline_mode="unknown")

    def test_cli_and_manifest_preserve_literal_default_and_accept_newline(self):
        for mode in (None, "literal", "newline", "escaped-to-newline"):
            argv = ["prepare", "--output-dir", "unused"]
            if mode is not None:
                argv.extend(["--newline-mode", mode])
            parsed = parser().parse_args(argv)
            self.assertEqual(parsed.newline_mode, mode or "literal")
            task = next(iter_generation_tasks(
                "official", statement_limit=1, iterations=1,
                newline_mode=parsed.newline_mode,
            ))
            self.assertEqual(task["newline_mode"], mode or "literal")
            self.assertEqual(task["prompt"], format_prompt(
                task["template_text"], task["statement"], newline_mode=mode or "literal",
            ))

    def test_tokenization_boundary_preserves_formatted_content(self):
        # No tokenizer/model import: observe exactly what our wrapper passes on.
        from unittest.mock import Mock

        generator = LocalHFGenerator.__new__(LocalHFGenerator)
        for mode in ("literal", "newline"):
            prompt = format_prompt(self.templates[0]["template"], self.rows[0]["statement"], newline_mode=mode)
            for prompt_mode in ("raw", "chat_template"):
                generator.spec = {"prompt_mode": prompt_mode}
                generator.tokenizer = Mock()
                generator._tokenize(prompt)
                if prompt_mode == "raw":
                    self.assertEqual(generator.tokenizer.call_args.args, (prompt,))
                else:
                    call = generator.tokenizer.apply_chat_template.call_args
                    self.assertEqual(call.args, ([{"role": "user", "content": prompt}],))
                    self.assertTrue(call.kwargs["tokenize"])
                    self.assertTrue(call.kwargs["add_generation_prompt"])

    def test_full_manifest_counts_ids_prompts_and_shards(self):
        count = 0
        prompt_ids = set()
        generation_ids = set()
        shards = [0, 0]
        for task in iter_generation_tasks("official"):
            count += 1
            prompt_ids.add((task["formulation_id"], task["template_id"]))
            generation_ids.add(task["generation_id"])
            shards[task["shard_id"]] += 1
            self.assertEqual(task["prompt_type"], "personal")
            self.assertIn(task["statement"], task["prompt"])
        self.assertEqual(count, 172_080)
        self.assertEqual(len(prompt_ids), 5_736)
        self.assertEqual(len(generation_ids), 172_080)
        self.assertEqual(shards, [86_040, 86_040])

    def test_seven_statement_pilot_is_balanced_and_complete(self):
        tasks = list(iter_generation_tasks("official", statement_limit=7))
        self.assertEqual(len(tasks), 5_040)
        self.assertEqual(len({task["statement_id"] for task in tasks}), 7)
        shard_zero = {t["generation_id"] for t in tasks if t["shard_id"] == 0}
        shard_one = {t["generation_id"] for t in tasks if t["shard_id"] == 1}
        self.assertEqual((len(shard_zero), len(shard_one)), (2_520, 2_520))
        self.assertFalse(shard_zero & shard_one)
        self.assertEqual(shard_zero | shard_one, {t["generation_id"] for t in tasks})

    def test_effective_generation_behavior_matches_source(self):
        self.assertEqual(
            REQUESTED_GENERATION_CONFIG,
            {
                "do_sample": False,
                "max_new_tokens": 10,
                "temperature": 1.0,
                "top_p": None,
            },
        )
        effective = effective_generation_config()
        self.assertTrue(effective["do_sample"])
        self.assertEqual(effective["temperature"], 1.0)
        self.assertIsNone(effective["top_p"])

    def test_reset_each_generation_is_default_and_reseeds_every_call(self):
        parsed = parser().parse_args(
            [
                "worker",
                "--manifest",
                "manifest.jsonl",
                "--model-key",
                "official",
                "--worker-id",
                "0",
                "--output",
                "worker.jsonl",
            ]
        )
        self.assertEqual(parsed.seed_mode, "reset_each_generation")
        self.assertEqual(parsed.worker_seed, 42)
        calls = []
        controller = SeedController("reset_each_generation", 42, calls.append)
        self.assertEqual(calls, [])
        controller.before_generation(42)
        controller.before_generation(42)
        self.assertEqual(calls, [42, 42])

    def test_seed_once_seeds_worker_once_and_never_reseeds_per_generation(self):
        calls = []
        controller = SeedController("seed_once", 42, calls.append)
        self.assertEqual(calls, [42])
        for _ in range(30):
            controller.before_generation(42)
        self.assertEqual(calls, [42])

    def test_seed_modes_do_not_change_manifest_or_task_counts(self):
        tasks = list(iter_generation_tasks("official", statement_limit=7))
        ids = [task["generation_id"] for task in tasks]
        self.assertEqual(len(tasks), 5_040)
        self.assertEqual(len(set(ids)), 5_040)
        self.assertEqual(sum(task["shard_id"] == 0 for task in tasks), 2_520)
        self.assertEqual(sum(task["shard_id"] == 1 for task in tasks), 2_520)

    def test_resume_requires_exact_ordered_shard_prefix(self):
        tasks = list(
            iter_generation_tasks(
                "official", statement_limit=1, iterations=2, world_size=2
            )
        )
        assigned = [task for task in tasks if task["shard_id"] == 0]
        prefix = [result_from_response(task, "Agree") for task in assigned[:3]]
        validate_resume_prefix(assigned, prefix)
        with self.assertRaises(ValueError):
            validate_resume_prefix(assigned, list(reversed(prefix)))
        with self.assertRaises(ValueError):
            validate_resume_prefix(assigned, prefix + [dict(prefix[-1])])

    def test_isolated_classifier_matches_authoritative_function(self):
        reference = authoritative_map_response()
        examples = {
            "Agree": 1,
            "Disagree": 0,
            "Support": 1,
            "Oppose": 0,
            "Beneficial": 1,
            "Harmful": 0,
            "Favorable": 1,
            "Detrimental": 0,
            "I do not agree": 0,
            "I do not disagree": 1,
            "Agree and disagree": 1,
            "Disagree and agree": 0,
            "Maybe": None,
            "": None,
            "aGrEe": 1,
            '"Agree"': 1,
            "Disagree!": 0,
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(reference(text), expected)
                self.assertEqual(map_response(text), expected)
                self.assertEqual(map_response(text), reference(text))

    def test_merge_resume_and_failure_na_separation(self):
        tasks = list(
            iter_generation_tasks(
                "official", statement_limit=1, iterations=2, world_size=2
            )
        )
        records = [
            result_from_response(tasks[0], "Agree"),
            result_from_response(tasks[1], "Maybe"),
            result_from_failure(tasks[2], RuntimeError("synthetic failure")),
        ]
        merged, audit = merge_results(tasks, records)
        self.assertEqual(audit["completed"], 2)
        self.assertEqual(audit["failed"], 1)
        self.assertEqual(audit["missing"], len(tasks) - 3)
        summary = summarize_results(tasks, merged)
        self.assertEqual(summary["agree_count"], 1)
        self.assertEqual(summary["disagree_count"], 0)
        self.assertEqual(summary["na_count"], 1)
        self.assertEqual(summary["na_percentage"], 50.0)
        self.assertEqual(summary["failed_generations"], 1)
        self.assertEqual(summary["missing_generations"], len(tasks) - 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.jsonl"
            write_jsonl_exclusive(path, records)
            recorded = recorded_generation_ids(path)
            remaining = list(pending_tasks(tasks, recorded))
            self.assertEqual(len(remaining), len(tasks) - 3)
            self.assertFalse(recorded & {r["generation_id"] for r in remaining})

    def test_merge_rejects_duplicate_worker_ids(self):
        tasks = list(
            iter_generation_tasks(
                "official", statement_limit=1, iterations=1, world_size=2
            )
        )
        record = result_from_response(tasks[0], "Agree")
        with self.assertRaises(ValueError):
            merge_results(tasks, [record, dict(record)])

    def test_response_diversity_and_class_percentages(self):
        tasks = list(
            iter_generation_tasks(
                "official", statement_limit=1, iterations=2, world_size=2
            )
        )
        first_prompt = (tasks[0]["formulation_id"], tasks[0]["template_id"])
        records = []
        for task in tasks:
            prompt_key = (task["formulation_id"], task["template_id"])
            response = (
                "Disagree"
                if prompt_key == first_prompt and task["sample_id"] == 1
                else "Agree"
            )
            records.append(result_from_response(task, response))
        summary = summarize_results(tasks, records)
        self.assertEqual(summary["total_prompt_groups"], 24)
        self.assertEqual(summary["complete_prompt_groups"], 24)
        self.assertEqual(summary["incomplete_prompt_groups"], 0)
        self.assertEqual(summary["prompts_identical_across_all_iterations"], 23)
        self.assertAlmostEqual(
            summary["percentage_complete_prompts_identical_across_all_iterations"],
            100 * 23 / 24,
        )
        diversity = summary["response_diversity_by_prompt"]
        first_key = f"{first_prompt[0]}::{first_prompt[1]}"
        self.assertEqual(diversity[first_key]["unique_responses"], 2)
        self.assertFalse(diversity[first_key]["identical_across_all_iterations"])
        self.assertEqual(summary["agree_count"], len(tasks) - 1)
        self.assertEqual(summary["disagree_count"], 1)
        self.assertAlmostEqual(summary["agree_percentage"], 100 * (len(tasks) - 1) / len(tasks))
        self.assertAlmostEqual(summary["disagree_percentage"], 100 / len(tasks))


if __name__ == "__main__":
    unittest.main()
