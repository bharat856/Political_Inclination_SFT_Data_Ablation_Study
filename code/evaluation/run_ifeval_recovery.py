"""Persistent per-model resume -> merge -> official CPU scoring -> analysis."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from ifeval_artifacts import digest
from resume_ifeval import prepare, REPO, STAGE, DATA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['Full','NoLeft','NoRight','LRBalanced'],required=True)
    args=parser.parse_args()
    root=REPO/'outputs/evaluation_recovery_20261011'
    out=root/args.model
    expected=json.loads((root/'approved_launch_preflight.json').read_text())[args.model]
    plan,_,_=prepare(args.model)
    for key in ['completed','original_artifact_hashes','adapter_sha256','runtime','next_official_key']:
        if plan[key]!=expected[key]:raise ValueError(f'Preflight changed: {key}')
    state=dict(model=args.model,pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
               completed_prefix=plan['completed'],planned_missing=plan['remaining'],
               status='running',phase='resume',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary=out/'recovery_state.tmp';temporary.write_text(json.dumps(state,indent=2))
        temporary.replace(out/'recovery_state.json')
    if (out/'recovery_state.json').exists():raise FileExistsError('Do not duplicate a recovery worker')
    save()
    def run(phase,arguments):
        state['phase']=phase;save();started=time.perf_counter()
        command=[sys.executable,*map(str,arguments)]
        with (out/(phase+'_command.json')).open('x') as handle:json.dump(command,handle,indent=2)
        with (out/(phase+'.log')).open('x') as log:
            subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
        state[phase+'_seconds']=time.perf_counter()-started;save()
    code=REPO/'code/evaluation';complete=out/'complete';provenance=STAGE/'provenance.json'
    try:
        run('resume',[code/'resume_ifeval.py','--model',args.model,'--output-dir',out/'resumed','--execute'])
        for name,value in expected['original_artifact_hashes'].items():
            if digest(STAGE/args.model/'ifeval'/name)!=value:
                raise ValueError(f'Original responses changed: {name}')
        run('merge',[code/'ifeval_artifacts.py','--data',DATA,'--segments',STAGE/args.model/'ifeval',out/'resumed','--output-dir',complete])
        run('official_score',[code/'score_completed_ifeval.py','--data',DATA,'--input-dir',complete,'--provenance',provenance])
        run('analyze',[code/'summarize_ifeval.py','--data',DATA,'--input-dir',complete,'--provenance',provenance,'--output-dir',out/'analysis'])
        state.update(status='complete',phase='complete',completed_overall=541,original_files_unchanged=True)
        save()
    except Exception as error:
        state.update(status='failed',error=repr(error));save();raise


if __name__=='__main__':main()
