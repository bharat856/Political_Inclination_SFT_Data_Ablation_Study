import copy
import unittest
from pathlib import Path
from ifeval_artifacts import read_jsonl
from prepare_probvaa_candidate import candidate_jobs
from audit_probvaa_candidate import audit


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts=list(candidate_jobs())
        repo=Path(__file__).resolve().parents[2]
        cls.records=read_jsonl(repo/'outputs/evaluation_stage_20261011/Full/probvaa_pilot/generations.jsonl')

    def test_all_variant_and_seed_blocks_are_unique(self):
        self.assertEqual(len(self.prompts),17208)
        self.assertEqual(len({(r['formulation_id'],r['template_variant_id']) for r in self.prompts}),17208)
        self.assertEqual(len({r['seed_base'] for r in self.prompts}),17208)

    def test_real_pilot_reference_compatible_scoring(self):
        result,inputs=audit(self.records,self.prompts,2,True)
        self.assertEqual(result['total'],288)
        self.assertEqual(result['distribution'],dict(Agree=146,Disagree=118,NA=24))
        self.assertEqual(sum(r['total'] for r in result['statements']),288)
        self.assertEqual(set(result['per_variant'].values()),{24})

    def test_pilot_cannot_be_reported_as_complete_benchmark(self):
        with self.assertRaises(ValueError):audit(self.records,self.prompts,2,False)

    def test_duplicate_and_missing_whole_variant_rejected(self):
        with self.assertRaises(ValueError):audit(self.records+[self.records[0]],self.prompts,2,True)
        missing=[r for r in self.records if r['template_variant_id']!='3']
        with self.assertRaises(ValueError):audit(missing,self.prompts,2,True)

    def test_changed_seed_and_prompt_rejected(self):
        for key,value in [('seed',42),('training_format_prompt','changed')]:
            rows=copy.deepcopy(self.records);rows[0][key]=value
            with self.assertRaises(ValueError):audit(rows,self.prompts,2,True)


if __name__=='__main__':unittest.main()
