import copy,unittest
from ifeval_artifacts import read_jsonl
from pathlib import Path
from prepare_probvaa_candidate import candidate_jobs
from resume_probvaa_candidate import validate_saved
from paired_ifeval import compare

class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts=list(candidate_jobs())
        cls.rows=read_jsonl(Path(__file__).resolve().parents[2]/'outputs/evaluation_stage_20261011/Full/probvaa_pilot/generations.jsonl')
        cls.params=cls.rows[0]['generation_parameters']
    def test_existing_pilot_can_be_validated_without_regeneration(self):
        self.assertEqual(len(validate_saved(self.rows,self.prompts,2,'Full',self.params)),288)
    def test_duplicate_stops(self):
        with self.assertRaises(ValueError):validate_saved(self.rows+[self.rows[0]],self.prompts,2,'Full',self.params)
    def test_seed_and_config_corruption_stops(self):
        for field,value in [('seed',0),('generation_parameters',{}),('raw_output','disagree')]:
            rows=copy.deepcopy(self.rows[:1]);rows[0][field]=value
            # first original label must differ for parser corruption check
            if field=='raw_output':rows[0]['parsed_answer']='Agree'
            with self.assertRaises(ValueError):validate_saved(rows,self.prompts,2,'Full',self.params)
    def test_identity_pair_has_zero_uncertainty(self):
        rows=[dict(key=1,prompt='a',instruction_ids=['x'],strict=[True],loose=[True]),dict(key=2,prompt='b',instruction_ids=['x','y'],strict=[False,True],loose=[True,True])]
        for r in compare(rows,rows,draws=100).values():
            self.assertEqual(r['difference_percentage_points'],0)
            self.assertEqual(r['ci95_percentage_points'],[0,0])
    def test_unpaired_rejected(self):
        with self.assertRaises(ValueError):compare([dict(key=1)], [dict(key=2)],draws=1)

if __name__=='__main__':unittest.main()
