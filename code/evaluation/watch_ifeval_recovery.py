"""CPU-only persistent progress report and final comparison after all jobs finish."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'outputs/evaluation_recovery_20261011'
MODELS=['Full','NoLeft','NoRight','LRBalanced']


def snapshot():
    rows=[]
    for model in MODELS:
        directory=ROOT/model;state_path=directory/'recovery_state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(status='starting',phase='preflight')
        raw=directory/'resumed/generations.jsonl';prefix={'Full':137,'NoLeft':132,'NoRight':145,'LRBalanced':136}[model]
        # Count only complete valid records, even during a concurrent write.
        generated=0
        if raw.exists():
            for line in raw.read_text().splitlines():
                try:json.loads(line);generated+=1
                except json.JSONDecodeError:break
        rows.append(dict(state, model=model,completed=prefix+generated,missing=541-prefix-generated))
    return rows


def report(rows):
    now=datetime.datetime.now(datetime.timezone.utc)
    lines=['# Evaluation progress and recovery handoff','',
           f'Updated: {now.astimezone(ZoneInfo("Europe/Berlin")).isoformat()} (Europe/Berlin).','',
           'User approved resuming only the 1,614 missing IFEval prompts. Preserve all 550 original responses. No full ProbVAA and no adapter changes. Four persistent tmux workers use GPUs 4–7. Each independently resumes, validates, merges into a new directory, runs the pinned official scorer, and writes strict/loose per-example analysis. A CPU watcher creates the final comparison once all four finish.','',
           '| Model | Saved prompts | Missing | Worker status | Phase |','|---|---:|---:|---|---|']
    for row in rows:lines.append(f"| {row['model']} | {row['completed']}/541 | {row['missing']} | {row['status']} | {row['phase']} |")
    lines += ['', 'Errors:', *[f"- {r['model']}: {r.get('error','') }" for r in rows if r['status']=='failed'], '',
              'Actual OS process/GPU checks are required after a disconnection; JSON status alone cannot establish that a worker is alive. Per-model logs, raw suffixes, checkpoint.json and recovery_state.json are under outputs/evaluation_recovery_20261011/{Model}/. Original prefixes remain under outputs/evaluation_stage_20261011/{Model}/ifeval/. Exact tmux launch and watcher commands are preserved in outputs/evaluation_recovery_20261011/executed_commands.json and executed_commands.md. Prepared CPU completion commands are in commands.md. Historical preflight/evidence and prior handoff remain in the recovery directory.', '',
              'After completion: merged raw/scorer inputs and provenance in {Model}/complete/; official strict/loose scorer files in complete/scoring/; joined per-example results and four accuracy measures in {Model}/analysis/; thesis comparison in final_comparison/. Every output directory is exclusive and original hashes are checked again before merging. Interrupted original runs lack final runtime/CUDA peaks; retain resumed measurements separately and do not invent full-run measurements.', '',
              'Official IFEval commit: e6890f85757dd84e27ca6df2dd30651dafad28e0; 541 prompts and 834 constraints. Same training-format USER/ASSISTANT wrapper, greedy max_new_tokens=1280, seed42, same effective runtime/package versions, same adapter hashes as the smoke configuration. Pinned source/data hashes and model metadata are preserved. Official scorer unchanged, with the same fixed Python and language-detector seeds.', '',
              'All four political pilots are completed: 288 generations/model, 12 unique variants, paired prompts/seeds. Pilot parsing: Full91.67%, NoLeft88.19%, NoRight90.97%, LRBalanced92.01%. These policy scores are diagnostic only; recovered templates/wrapper and unique-variant/seed choices are reconstructed protocol choices. Reference parser and scoring formulas remain unchanged. Pilot report: outputs/evaluation_stage_20261011/pilot_report.md. CPU-only political preparation artifacts, when present, are under outputs/probvaa_preparation_20261011/.']
    tmp=REPO/'EVALUATION_PROGRESS.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(REPO/'EVALUATION_PROGRESS.md')
    tmp=ROOT/'live_progress.tmp';tmp.write_text(json.dumps(dict(updated_utc=now.isoformat(),models=rows),indent=2));tmp.replace(ROOT/'live_progress.json')


def main():
    last_errors=None
    while True:
        rows=snapshot();report(rows)
        if any(r['status']=='failed' for r in rows):
            errors=[(r['model'],r.get('error')) for r in rows if r['status']=='failed']
            if errors!=last_errors:print('Recovery error: inspect per-model logs.',errors,flush=True)
            last_errors=errors
        if all(r['status']=='complete' for r in rows):break
        time.sleep(30)
    out=ROOT/'final_comparison'
    with (ROOT/'comparison.log').open('x') as log:
        subprocess.run([sys.executable,str(REPO/'code/evaluation/compare_ifeval.py'),'--analysis-root',str(ROOT),'--output-dir',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    with (REPO/'EVALUATION_PROGRESS.md').open('a') as handle:
        handle.write('\nAll four official IFEval analyses and the thesis comparison are complete: outputs/evaluation_recovery_20261011/final_comparison/report.md.\n')
    print('All four models scored and comparison complete.',flush=True)


if __name__=='__main__':main()
