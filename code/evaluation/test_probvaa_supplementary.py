import unittest
from probvaa_supplementary import supplementary
from probvaa_scoring import aggregate_statements
class SupplementTests(unittest.TestCase):
    def rows(self):
        return [dict(model='X',statement_id='es7',country_code='es',formulation_id='x',sample_id=i,template_id=3,template_variant_id=3,negation=False,opposite=False,raw_output=raw,output_token_ids=[1]*64,generation_parameters={'max_new_tokens':64,'eos_token_id':2}) for i,raw in enumerate(['Agree','Disagree','Agree','Unknown'])]
    def test_NA_denominator_and_conditional(self):
        rows=self.rows();before=aggregate_statements(rows);r=supplementary(rows)['models']['X'];s=r['statement_scores'][0]
        self.assertEqual(before,aggregate_statements(rows));self.assertEqual(s['primary_score'],.25);self.assertAlmostEqual(s['conditional_score'],1/3);self.assertAlmostEqual(s['conditional_minus_primary'],1/12)
        self.assertEqual(r['diagnostics']['parsing_success'],.75);self.assertEqual(r['by_template_variant']['3']['at_token_cap'],4)
    def test_all_NA_null_conditional(self):
        rows=self.rows()
        for r in rows:r['raw_output']='Unknown'
        s=supplementary(rows)['models']['X']['statement_scores'][0]
        self.assertEqual(s['primary_score'],0);self.assertIsNone(s['conditional_score'])
    def test_exact_cap_with_EOS_is_separate(self):
        rows=self.rows();rows[0]['output_token_ids'][-1]=2
        d=supplementary(rows)['models']['X']['diagnostics'];self.assertEqual(d['at_token_cap'],4);self.assertEqual(d['at_cap_without_eos'],3)
if __name__=='__main__':unittest.main()
