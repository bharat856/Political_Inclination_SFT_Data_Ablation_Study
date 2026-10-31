"""CPU-only fidelity tests. Reference functions extracted without ML imports."""
import ast
import itertools
import json
import math
from pathlib import Path
import sys
import unittest
from probvaa_config import PRIMARY, WORLDVIEWS, DIMENSIONS, EFFECTIVE_GENERATION, model_spec, MissingReferenceDependency
from probvaa_data import load_formulations, load_annotations, asset_report, load_templates, read_csv
from probvaa_parsing import map_response, normalize
from probvaa_prompting import get_formatter, tokenize_prompt
from probvaa_ci import policy_estimate
from probvaa_scoring import aggregate_statements, aggregate_policies
from run_probvaa import prepare


def reference_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class FidelityTests(unittest.TestCase):
    def test_dataset_counts_and_identity(self):
        r = asset_report()
        self.assertEqual((r['statements'], r['formulations'], r['distinct_texts']), (239, 1434, 1432))
        self.assertEqual(r['countries'], {'nl': 30, 'de': 38, 'es': 24, 'pl': 20, 'hu': 37, 'ch': 60, 'it': 30})
        self.assertEqual(set(r['formulation_types'].values()), {239})
        self.assertEqual(sum(r['dimensions'].values()), 258)
        self.assertEqual(r['annotation_memberships'][0], 46)

    def test_parser_matches_actual_function(self):
        import re
        p = PRIMARY / 'model_behavior/classification_with_model_prompting.py'
        tree = ast.parse(p.read_text())
        formats = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'prompt_formats' for t in n.targets))
        ns = {'re': re, 'prompt_formats': ast.literal_eval(formats.value)}
        ref = reference_function(p, 'map_response', ns)
        pieces = ['agree', 'disagree', 'support', 'oppose', 'not', "don't", "isn't", 't', 'yes', 'no', 'NA', 'beneficial', 'harmful', 'pro', 'con', '\n', '123', '']
        for words in itertools.product(pieces, repeat=3):
            raw = ' '.join(words)
            self.assertEqual(map_response(raw), ref(raw), repr(raw))

    def test_parser_examples(self):
        examples = {'Agree': 1, 'disagree': 0, 'YES': None, 'no': None, '': None,
                    'support then oppose': 1, 'oppose then support': 0,
                    'not agree': 0, "don't disagree": 1, "isn't agree": None,
                    'do you agree or disagree': None, 'Statement: support. Answer: disagree': 1,
                    'neither pro nor con': 1}
        for raw, expected in examples.items():
            self.assertEqual(map_response(raw), expected, raw)

    def test_normalization(self):
        for value in (None, 0, 1):
            self.assertEqual(normalize(value), value)
            for flags in [(True, False), (False, True), (True, True)]:
                self.assertEqual(normalize(value, *flags), None if value is None else 1-value)

    def test_na_denominator_and_dedup(self):
        base = {'model': 'dummy', 'statement_id': 'nl1', 'country_code': 'nl', 'template_id': '1',
                'negation': False, 'opposite': False, 'formulation_id': 'nl_1_1000001_en'}
        rows = [base | {'sample_id': i, 'raw_output': v} for i, v in enumerate(['agree','NA','NA'])]
        self.assertAlmostEqual(aggregate_statements(rows)[0]['score'], 1/3)
        rows.append(base | {'sample_id': 0, 'raw_output': 'disagree'})
        self.assertAlmostEqual(aggregate_statements(rows)[0]['score'], -1/3)
        self.assertEqual(aggregate_statements([base | {'sample_id': 0, 'raw_output': 'NA'}])[0]['score'], 0)

    def test_statement_aggregation_against_pandas_reference(self):
        import pandas as pd
        values = [1, 0, 'na', 'na', 1]
        expected = (pd.Series(values) == 1).mean() - (pd.Series(values) == 0).mean()
        rows = [{'model': 'dummy', 'statement_id': 'nl1', 'country_code': 'nl', 'template_id': '1',
                 'formulation_id': 'nl_1_1000001_en', 'sample_id': i,
                 'raw_output': 'agree' if v == 1 else 'disagree' if v == 0 else 'NA',
                 'negation': False, 'opposite': False} for i, v in enumerate(values)]
        self.assertAlmostEqual(aggregate_statements(rows)[0]['score'], expected)

    def test_final_notebook_policy_function(self):
        import numpy as np
        import pandas as pd
        from collections import defaultdict
        nb = json.loads((PRIMARY / 'model_behavior/results_model_behavior.ipynb').read_text())
        utils = ast.parse((PRIMARY / 'model_behavior/utils.py').read_text())
        maps = {}
        for n in utils.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('id2idx', 'policy2idx'):
                maps[n.targets[0].id] = ast.literal_eval(n.value)
        class LocalPandas:
            @staticmethod
            def read_csv(_):
                return pd.read_csv(PRIMARY / 'model_behavior/data/annotations_spiderweb_gold.csv')
        ns = {'pd': LocalPandas, 'np': np, 'defaultdict': defaultdict, **maps}
        exec(''.join(nb['cells'][11]['source']), ns)
        exec(''.join(nb['cells'][15]['source']), ns)
        annotations = load_annotations()
        # Include finite zero (NA-only) and missing scores. Uses original annotation order.
        finite = {uid: ((i % 7)-3)/3 for i, uid in enumerate(annotations) if i % 11 != 0}
        frame = pd.DataFrame([{'model_name': 'dummy', 'unique_id': uid, 'answer': score} for uid, score in finite.items()])
        statements = [{'model': 'dummy', 'statement_id': uid, 'score': score} for uid, score in finite.items()]
        for weighted in (False, True):
            ref, cov, ci = ns['positioning_per_model'](frame, weighted=weighted)
            ours = aggregate_policies(statements, annotations, weighted)
            for row in ours:
                d = row['dimension']
                self.assertEqual(row['coverage'], cov['dummy'][d])
                for own, expected in [(row['score'], ref['dummy'][d]), (row['ci_halfwidth'], ci['dummy'][d])]:
                    if np.isnan(expected):
                        self.assertIsNone(own)
                    else:
                        self.assertAlmostEqual(own, expected, places=13)

    def test_ci_edge_cases(self):
        self.assertIsNone(policy_estimate([])['score'])
        self.assertEqual(policy_estimate([(1,0)])['score'], 0)
        self.assertIsNone(policy_estimate([(1,1)])['ci_halfwidth'])
        self.assertIsNone(policy_estimate([(1,0),(-1,0)], True)['score'])

    def test_generation_override_without_torch(self):
        from types import SimpleNamespace
        path = PRIMARY / 'model_behavior/llm_utils.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_HFModel')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
        ns = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        class Tokens:
            shape = (1, 2)
            def to(self, device): return self
        class Tokenizer:
            eos_token_id = 9
            def __call__(self, *a, **kw): return {'input_ids': Tokens()}
            def decode(self, ids, **kw): return 'agree'
        captured = {}
        def generate(**kw):
            captured.update(kw)
            return [[1,2,3]]
        wrapper = SimpleNamespace(update_seed=lambda s: captured.update(seed=s), tokenizer=Tokenizer(),
                                  model=SimpleNamespace(device='cpu', generate=generate),
                                  pad_token_id=8, get_temperature=lambda t:t)
        ns['__call__'](wrapper, 'hello', 'base', dict(do_sample=False, temperature=1.0, max_new_tokens=10, top_p=None), seed=42)
        for k, v in EFFECTIVE_GENERATION.items(): self.assertEqual(captured[k], v)
        self.assertEqual(captured['seed'], 42)

    def test_missing_dependencies_fail_closed(self):
        with self.assertRaises(MissingReferenceDependency): load_templates(model_spec('official'))
        with self.assertRaises(MissingReferenceDependency): get_formatter()
        class NoChat:
            chat_template = None
        with self.assertRaises(MissingReferenceDependency): tokenize_prompt(NoChat(), 'hello', model_spec('official'))

    def test_upstream_manifest_cpu_only(self):
        # This is an internal CPU check, not execution of the manual CLI commands.
        for preset in ('full','official'):
            metadata, jobs = prepare(model_spec(preset), template_source='worldviews', format_mode='worldviews-percent')
            self.assertEqual((metadata['templates'], metadata['expected_total_generations']), (12,516240))
            first = next(jobs())
            self.assertEqual(first['raw_prompt'], first['template_text'] % first['statement_text'])
            self.assertIn('NOT exact', metadata['replication_status'])
            m, _ = prepare(model_spec(preset), samples=2, statement_limit=5, template_source='worldviews', format_mode='worldviews-percent')
            self.assertEqual(m['expected_total_generations'], 720)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)


if __name__ == '__main__':
    unittest.main()
