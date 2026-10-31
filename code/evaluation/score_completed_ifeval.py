"""Guard full coverage and new output paths before invoking official CPU scorer."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from ifeval_artifacts import digest, read_jsonl, validate_segment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    args=parser.parse_args()
    examples=read_jsonl(args.data)
    raw=read_jsonl(args.input_dir/'generations.jsonl')
    responses=read_jsonl(args.input_dir/'responses.jsonl')
    validate_segment(examples,raw,responses)
    if len(examples)!=541 or len(raw)!=541:
        raise ValueError('Refuse official full scoring without all 541 raw responses')
    metadata=json.loads((args.input_dir/'metadata.json').read_text())
    if metadata['input_sha256']!=digest(args.data):
        raise ValueError('Benchmark data hash differs')
    provenance=json.loads(args.provenance.read_text())
    for item in provenance['ifeval_provenance']['files']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Pinned official scorer/data changed')
    repo=Path(__file__).resolve().parents[2]
    diag=repo/'outputs/evaluation_diagnostics_20261011'
    target=args.input_dir/'scoring'
    target.mkdir(exist_ok=False)
    env=os.environ.copy();env['NLTK_DATA']=str(diag/'nltk_data')
    command=[sys.executable,str(diag/'score_ifeval.py'),'--input_data',str(args.data),
             '--input_response_data',str(args.input_dir/'responses.jsonl'),'--output_dir',str(target)]
    (target/'invocation.json').write_text(json.dumps(dict(command=command,scorer_version=provenance['ifeval_provenance']),indent=2))
    with (target/'official_scorer.log').open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
