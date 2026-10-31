import copy
import json
import tempfile
import unittest
from pathlib import Path

from ifeval_artifacts import digest, merge_segments, read_jsonl, summarize_scores, validate_segment


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = [dict(key=i, prompt=f'prompt {i}', instruction_id_list=['one','two'], kwargs=[{},{}]) for i in range(3)]
        self.data_path = self.root / 'data.jsonl'
        self.write(self.data_path, self.data)
        self.raw = [row | dict(raw_output='answer', rendered_model_input='USER:\n'+row['prompt']+'\n\nASSISTANT:\n', output_token_ids=[1,2]) for row in self.data]
        self.responses = [dict(prompt=row['prompt'], response='answer') for row in self.data]

    def write(self, path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def segment(self, name, start, stop):
        path = self.root / name
        path.mkdir()
        metadata = dict(start_index=start, model={'model':'Full'}, runtime={'a':1}, seed=42,
                        prompt_format='same', input_sha256=digest(self.data_path))
        (path/'metadata.json').write_text(json.dumps(metadata))
        self.write(path/'generations.jsonl', self.raw[start:stop])
        self.write(path/'responses.jsonl', self.responses[start:stop])
        return path

    def score_files(self):
        strict = [dict(prompt=r['prompt'],response='answer',instruction_id_list=['one','two'],follow_instruction_list=[True,False],follow_all_instructions=False) for r in self.data]
        loose = [row | dict(follow_instruction_list=[True,True],follow_all_instructions=True) for row in strict]
        paths = [self.root/'raw.jsonl',self.root/'strict.jsonl',self.root/'loose.jsonl']
        for path, rows in zip(paths, [self.raw,strict,loose]): self.write(path,rows)
        return paths

    def test_valid_prefix_and_suffix(self):
        self.assertEqual(validate_segment(self.data,self.raw[:1],self.responses[:1]),1)
        self.assertEqual(validate_segment(self.data,self.raw[1:],self.responses[1:],1),3)

    def test_reject_changed_prompt_wrapper_and_response(self):
        for key,value in [('prompt','changed'),('rendered_model_input','changed'),('raw_output','changed')]:
            rows=copy.deepcopy(self.raw);rows[0][key]=value
            with self.assertRaises(ValueError):validate_segment(self.data,rows,self.responses)

    def test_reject_mismatched_or_duplicate_records(self):
        with self.assertRaises(ValueError):validate_segment(self.data,self.raw,self.responses[:-1])
        with self.assertRaises(ValueError):validate_segment(self.data,[self.raw[0],self.raw[0]],self.responses[:2])

    def test_merge_preserves_source_and_requires_new_output(self):
        prefix=self.segment('prefix',0,1);suffix=self.segment('suffix',1,3)
        original=digest(prefix/'generations.jsonl');out=self.root/'merged'
        metadata=merge_segments(self.data_path,[prefix,suffix],out)
        self.assertEqual(read_jsonl(out/'generations.jsonl'),self.raw)
        self.assertEqual(digest(prefix/'generations.jsonl'),original)
        self.assertIsNone(metadata['peak_allocated_gib'])
        with self.assertRaises(FileExistsError):merge_segments(self.data_path,[prefix,suffix],out)

    def test_reject_gap_overlap_and_incomplete_before_writing(self):
        prefix=self.segment('prefix',0,1)
        for name,start,stop in [('gap',2,3),('overlap',0,2),('short',1,2)]:
            suffix=self.segment(name,start,stop);out=self.root/(name+'out')
            with self.assertRaises(ValueError):merge_segments(self.data_path,[prefix,suffix],out)
            self.assertFalse(out.exists())

    def test_reject_changed_segment_settings(self):
        prefix=self.segment('prefix',0,1);suffix=self.segment('suffix',1,3)
        p=suffix/'metadata.json';m=json.loads(p.read_text());m['seed']=1;p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):merge_segments(self.data_path,[prefix,suffix],self.root/'out')

    def test_score_denominators_and_per_example_join(self):
        paths=self.score_files();scores,joined=summarize_scores(self.data_path,*paths)
        self.assertEqual(scores['strict']['prompt_accuracy'],0)
        self.assertEqual(scores['strict']['instruction_accuracy'],0.5)
        self.assertEqual(scores['loose']['prompt_accuracy'],1)
        self.assertEqual(scores['loose']['instruction_total'],6)
        self.assertEqual([r['key'] for r in joined],[0,1,2])

    def test_reject_incomplete_or_misaligned_scoring(self):
        paths=self.score_files();rows=read_jsonl(paths[1]);self.write(paths[1],rows[:-1])
        with self.assertRaises(ValueError):summarize_scores(self.data_path,*paths)
        rows[0]['response']='changed';self.write(paths[1],rows)
        with self.assertRaises(ValueError):summarize_scores(self.data_path,*paths)

    def test_preserve_and_flag_official_loose_strict_regression(self):
        paths=self.score_files();rows=read_jsonl(paths[2]);rows[0].update(follow_instruction_list=[False,True],follow_all_instructions=False);self.write(paths[2],rows)
        scores,joined=summarize_scores(self.data_path,*paths)
        self.assertEqual(joined[0]['strict_pass_loose_fail_indices'],[0])
        self.assertEqual(scores['loose']['instruction_correct'],5)

    def test_malformed_json_rejected(self):
        path=self.root/'bad.jsonl';path.write_text('{"partial":')
        with self.assertRaises(ValueError):read_jsonl(path)


if __name__ == '__main__':unittest.main()
