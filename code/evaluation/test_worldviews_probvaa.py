"""CPU-only checks of the OLMo adapter against the pinned author implementation."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import worldviews_probvaa as replica
import summarize_worldviews_official as official_summary


class OriginalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = replica.build_jobs()

    def test_complete_original_prompt_grid(self):
        jobs = self.jobs
        self.assertEqual(len(jobs), 12 * 1434)
        self.assertEqual(len({x["statement_id"] for x in jobs}), 1434)
        self.assertEqual(len({x["final_template_id"] for x in jobs}), 12)
        self.assertEqual(len({x["template_id"] for x in jobs}), 6)
        self.assertEqual({x["operational_inverted"] for x in jobs}, {0, 1})
        self.assertTrue(all(x["statement_id"].endswith("_en") for x in jobs))
        self.assertEqual([x["prompt_index"] for x in jobs], list(range(len(jobs))))
        selected = [jobs[0], jobs[1434], jobs[6 * 1434], jobs[-1]]
        templates = pd.read_csv(replica.TEMPLATES).set_index("final_template_id")
        statements = pd.read_csv(replica.DATA).set_index("ID")
        for job in selected:
            self.assertEqual(job["prompt"], templates.loc[job["final_template_id"], "template"]
                             % statements.loc[job["statement_id"], "statement"])
        for family in {x["template_id"] for x in jobs}:
            first = [x for x in jobs if x["template_id"] == family and
                     x["statement_id"] == jobs[0]["statement_id"]]
            self.assertEqual(len(first), 2)
            self.assertEqual({x["operational_inverted"] for x in first}, {0, 1})

    def test_olmo_tokenizer_preserves_original_prompts(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(replica.BASE, local_files_only=True,
                                                  trust_remote_code=True)
        longest = 0
        for job in self.jobs:
            ids = tokenizer(job["prompt"])["input_ids"]
            self.assertEqual(tokenizer.decode(ids, skip_special_tokens=True), job["prompt"])
            longest = max(longest, len(ids))
        context = json.loads((replica.BASE / "config.json").read_text())["max_position_embeddings"]
        self.assertLess(longest + replica.GENERATION["max_new_tokens"], context)

    def test_six_local_model_configurations(self):
        shared_training = None
        shared_adapter = None
        for model, directory in replica.ADAPTERS.items():
            path = replica.ROOT / "outputs/sft_runs" / directory
            self.assertTrue((path / "adapter_model.safetensors").is_file())
            adapter = json.loads((path / "adapter_config.json").read_text())
            self.assertEqual(Path(adapter["base_model_name_or_path"]).resolve(), replica.BASE.resolve())
            training = json.loads((path / "run_config_v3.json").read_text())
            self.assertEqual(training["model"], str(replica.BASE))
            comparable_training = {key: value for key, value in training.items() if key not in {
                "condition", "dataset_rows", "max_steps", "rows_consumed",
                "rows_omitted_by_batch_divisibility"}}
            comparable_adapter = adapter | {"target_modules": sorted(adapter["target_modules"])}
            if shared_training is None:
                shared_training, shared_adapter = comparable_training, comparable_adapter
            else:
                self.assertEqual(comparable_training, shared_training, model)
                self.assertEqual(comparable_adapter, shared_adapter, model)

    def test_original_response_parser_and_aggregation(self):
        _, _, parser, binarize = replica.reference_modules()
        from reliability.aggregate import get_bootstrapped_mean_questionnaire
        from reliability.utils import add_variante_type_info
        from reliability.evaluation import (compute_all_results,
                                            create_aggregated_questionnaire_cross_template)
        base_id = "_".join(self.jobs[0]["statement_id"].split("_")[:2])
        selected = [j for j in self.jobs if "_".join(j["statement_id"].split("_")[:2]) == base_id]
        self.assertEqual(len(selected), 72)
        column = "Full-generation_response"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.csv"
            rows = []
            for job in selected:
                rows.extend(dict(template_id=job["template_id"], statement_id=job["statement_id"],
                                 prompt=job["prompt"], **{column: job["prompt"] + " agree"})
                            for _ in range(30))
            pd.DataFrame(rows).to_csv(path, index=False)
            import os
            previous = Path.cwd()
            try:
                os.chdir(replica.SOURCE / "response_postprocessing")
                with contextlib.redirect_stdout(io.StringIO()):
                    parsed = binarize([str(path)])
            finally:
                os.chdir(previous)
            self.assertEqual(len(parsed), 2160)
            self.assertEqual(set(parsed.binary_answer), {parser(" agree")})
            self.assertEqual(parsed.groupby(["template_id", "statement_id", "inverted"])
                             .size().unique().tolist(), [30])
            aggregated = get_bootstrapped_mean_questionnaire(parsed, 30)
            self.assertEqual(len(aggregated), 72)
            annotated = add_variante_type_info(aggregated)
            reliability_dir = Path(tmp) / "reliability"
            reliability_dir.mkdir()
            compute_all_results(annotated, str(reliability_dir))
            self.assertEqual(len(list(reliability_dir.glob("*_statements_reliability_tests.csv"))), 6)
            cross = reliability_dir / "crosstemplate"
            cross.mkdir()
            create_aggregated_questionnaire_cross_template(str(reliability_dir), str(cross))
            self.assertTrue((cross / "number_of_statements.csv").is_file())
            reliability = pd.read_csv(reliability_dir /
                                      f"Full-generation_{selected[0]['template_id']}_statements_reliability_tests.csv")
            self.assertEqual(len(reliability), 1)
            self.assertEqual(reliability.iloc[0]["string_answers"].split(", ")[0], "original: 1")

    def test_original_policy_arithmetic_and_conditions(self):
        scope = replica.original_policy_functions()
        conditions = replica.original_policy_conditions()
        self.assertIn(False, conditions)
        self.assertIn("hard_pass", conditions)
        self.assertEqual(len(scope["policy2idx"]), 8)
        annotation = pd.read_csv(replica.ANNOTATIONS).fillna(0)
        chosen = annotation.loc[(annotation.iloc[:, 3:] == 1).any(axis=1)].iloc[0]
        unique_id = "".join(chosen["ID"].split("_")[:2])
        with tempfile.TemporaryDirectory() as tmp:
            families = sorted({j["template_id"] for j in self.jobs})
            for family in families:
                answer = 0 if family == families[0] else 1
                pd.DataFrame([dict(unique_id=unique_id, string_answers=f"original: {answer}, inverted: {answer}",
                                   hard_pass=True)]).to_csv(
                    Path(tmp) / f"Full-generation_{family}_statements_reliability_tests.csv", index=False)
            result = pd.DataFrame(replica.policy_scores(tmp, "Full", False))
            for policy in scope["policy2idx"]:
                positive_annotations = int((annotation[policy] == 1).sum())
                negative_annotations = int((annotation[policy] == -1).sum())
                if chosen[policy] == 1:
                    template_values = [0] + [1 / positive_annotations] * 5
                elif chosen[policy] == -1:
                    template_values = [-1 / negative_annotations] + [0] * 5
                else:
                    template_values = [0] * 6
                expected = pd.Series(template_values).mean()
                actual = result[(result.policy == policy) & (result.template_id == "mean")]
                self.assertAlmostEqual(actual.score.iloc[0], expected)
                self.assertAlmostEqual(actual.template_std.iloc[0],
                                       pd.Series(template_values + [expected]).std())

    def test_resume_rejects_overlap_and_torn_tail(self):
        _, _, parser, _ = replica.reference_modules()
        job = self.jobs[0]
        row = job | dict(outputs=[job["prompt"] + " agree"] * 30, parsed=[1] * 30,
                         rng_after={"cpu": [], "cuda": []})
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "segment_000000.jsonl").write_text(json.dumps(row) + "\n")
            (directory / "segment_000001.jsonl").write_text('{"unfinished":')
            self.assertEqual(len(replica.read_segments(directory, self.jobs, parser)), 1)
            next_job = self.jobs[1]
            next_row = next_job | dict(outputs=[next_job["prompt"] + " agree"] * 30,
                                      parsed=[1] * 30, rng_after={"cpu": [], "cuda": []})
            (directory / "segment_000002.jsonl").write_text(json.dumps(next_row) + "\n")
            self.assertEqual(len(replica.read_segments(directory, self.jobs, parser)), 2)
            (directory / "segment_000003.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                replica.read_segments(directory, self.jobs, parser)

    def test_comparison_checks_provenance_and_never_overwrites(self):
        policies = replica.original_policy_functions()["policy2idx"]
        conditions = ["all" if not item else item for item in replica.original_policy_conditions()]
        families = sorted({job["template_id"] for job in self.jobs})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for offset, model in enumerate(replica.ADAPTERS):
                location = root / f"scores_{model}"
                location.mkdir()
                rows = [dict(model=model, policy=policy, passed_tests=condition,
                             template_id=family, score=float(offset))
                        for policy in policies for condition in conditions
                        for family in [*families, "mean"]]
                scores = location / "policy_scores.csv"
                pd.DataFrame(rows).to_csv(scores, index=False)
                config = dict(model=model, prompt_count=17208,
                              reference_commit=replica.REFERENCE_COMMIT,
                              source_sha256={"source": "same"}, base="same",
                              base_config_sha256="same", tokenizer_sha256="same",
                              prompt_manifest_sha256="same", generation=replica.GENERATION,
                              seed=42, adapter=f"adapter_{model}", adapter_sha256=f"sha_{model}")
                metadata = dict(generation_config=config, response_count=516240,
                                outputs={"policy_scores.csv": replica.digest(scores)})
                (location / "metadata.json").write_text(json.dumps(metadata))
            target = root / "comparison"
            replica.compare_scores(root, "scores_{model}", target)
            differences = pd.read_csv(target / "differences_from_full.csv")
            self.assertEqual(len(differences), len(replica.ADAPTERS) * len(policies) * len(conditions))
            self.assertEqual(set(differences[differences.model == "Full"].difference_from_full), {0.0})
            self.assertEqual(set(differences[differences.model == "NoPolitical"].difference_from_full), {5.0})
            with self.assertRaises(FileExistsError):
                replica.compare_scores(root, "scores_{model}", target)

    def test_official_uses_complete_reference_chat_template_grid(self):
        self.assertNotIn(replica.OFFICIAL_MODEL, replica.ADAPTERS)
        self.assertEqual(len(replica.ADAPTERS), 6)
        jobs = replica.build_jobs("chat")
        self.assertEqual(len(jobs), 12 * 1434)
        self.assertEqual({job["final_template_id"] for job in jobs},
                         {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20})
        self.assertEqual(len({job["statement_id"] for job in jobs}), 1434)
        self.assertEqual({job["operational_inverted"] for job in jobs}, {0, 1})
        templates = pd.read_csv(replica.TEMPLATES).set_index("final_template_id")
        statements = pd.read_csv(replica.DATA).set_index("ID")
        for job in (jobs[0], jobs[6 * 1434], jobs[-1]):
            self.assertEqual(job["prompt"], templates.loc[job["final_template_id"], "template"]
                             % statements.loc[job["statement_id"], "statement"])

    def test_balanced_official_pilot_keeps_all_forms_and_answer_orders(self):
        selected, source_ids = replica.balanced_pilot_jobs(replica.build_jobs("chat"), 7)
        self.assertEqual(len(source_ids), 7)
        self.assertEqual({source.split("_", 1)[0] for source in source_ids},
                         {"ch", "de", "es", "hu", "it", "nl", "pl"})
        self.assertEqual(len(selected), 7 * 6 * 12)
        for source_id in source_ids:
            group = [job for job in selected
                     if replica.source_statement_id(job["statement_id"]) == source_id]
            self.assertEqual(len({job["statement_id"] for job in group}), 6)
            for family in {job["template_id"] for job in group}:
                self.assertEqual({job["operational_inverted"] for job in group
                                  if job["template_id"] == family}, {0, 1})

    def test_official_tokenizer_native_chat_rendering_and_eos(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            replica.OFFICIAL_CHECKPOINT, local_files_only=True, trust_remote_code=True)
        self.assertTrue(tokenizer.chat_template)
        prompt = replica.build_jobs("chat")[0]["prompt"]
        tokens, messages = replica.render_official_prompt(tokenizer, prompt)
        rendered = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=False)
        self.assertEqual(messages, [{"role": "user", "content": prompt}])
        self.assertIn("<|im_start|>system\n", rendered)
        self.assertIn("<|im_start|>user\n" + prompt + "<|im_end|>", rendered)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))
        generation = json.loads((replica.OFFICIAL_CHECKPOINT / "generation_config.json").read_text())
        self.assertEqual(generation["eos_token_id"], replica.OFFICIAL_EOS_TOKEN_IDS)
        self.assertEqual(replica.OFFICIAL_EOS_TOKEN_IDS, [100265, 100257])
        effective = replica.official_generation_settings(tokenizer)
        self.assertEqual(effective["eos_token_id"], [100265, 100257])
        self.assertEqual(effective["pad_token_id"], 100277)
        for key, value in replica.GENERATION.items():
            self.assertEqual(effective[key], value)

    def test_completion_parser_diagnostics_preserve_reference_behavior(self):
        _, _, parser, _ = replica.reference_modules()
        cases = {
            "AGREE": 1, '"Disagree"': 0, "“Agree”": 1,
            "NA": None, "N/A": None, "disagree or agree": 0,
            "agree or disagree": 1, "not agree": 0, "not disagree": 1,
        }
        for completion, expected in cases.items():
            self.assertEqual(parser(completion), expected, completion)
        diagnostic = replica.completion_diagnostic("Disagree, not Agree", [1] * 20)
        self.assertTrue(diagnostic["contains_both_label_polarities"])
        self.assertTrue(diagnostic["hit_max_new_tokens"])
        stopped = replica.completion_diagnostic("Agree", [5, 100265, 100277])
        self.assertEqual(stopped["eos_token_id"], 100265)
        self.assertFalse(stopped["hit_max_new_tokens"])

    def test_official_output_directory_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "official-pilot"
            self.assertEqual(replica.create_new_output_directory(target), target.resolve())
            with self.assertRaises(FileExistsError):
                replica.create_new_output_directory(target)

    def test_official_distribution_summary_counts_and_denominators(self):
        _, _, parser, _ = replica.reference_modules()
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            (run / "configuration.json").write_text(json.dumps({
                "model": replica.OFFICIAL_MODEL, "prompt_count": 1}))
            (run / "runtime.json").write_text("{}")
            (run / "COMPLETE").write_text("{}")
            completions = ["Agree"] * 10 + ['"DISAGREE"'] * 10 + ["N/A"] * 10
            parsed = [parser(value) for value in completions]
            diagnostics = []
            for index, value in enumerate(completions):
                item = replica.completion_diagnostic(value, [100265])
                item.update(response_index=index,
                            parsed_answer="NA" if parsed[index] is None else
                            "Agree" if parsed[index] == 1 else "Disagree")
                diagnostics.append(item)
            row = dict(prompt_index=17, final_template_id=0,
                       statement_id="nl_1_1000001_en",
                       generated_completions=completions,
                       response_indices=list(range(30)),
                       parsed=["not_applied" if value is None else value for value in parsed],
                       parsing_diagnostics=diagnostics)
            (run / "segment_000000.jsonl").write_text(json.dumps(row) + "\n")
            result = official_summary.summarize(run)
            self.assertEqual(result["counts"], {"Agree": 10, "Disagree": 10, "NA": 10})
            self.assertEqual(result["denominators"],
                             {"all_responses": 30, "parsed_responses": 20})
            self.assertEqual(result["proportions_among_parsed"],
                             {"Agree": 0.5, "Disagree": 0.5})


if __name__ == "__main__":
    unittest.main()
