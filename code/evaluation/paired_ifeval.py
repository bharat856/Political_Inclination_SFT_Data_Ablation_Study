"""Paired prompt-cluster percentile bootstrap of saved official IFEval outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
from ifeval_artifacts import read_jsonl

def compare(full,other,draws=20000,seed=20261012):
    b={r['key']:r for r in other}
    if len(b)!=len(full) or set(b)!={r['key'] for r in full}:raise ValueError('Unpaired prompt IDs')
    for r in full:
        s=b[r['key']]
        if r['prompt']!=s['prompt'] or r['instruction_ids']!=s['instruction_ids']:raise ValueError('Unpaired prompt/constraints')
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(full),size=(draws,len(full)))
    n=np.array([len(r['instruction_ids']) for r in full]);result={}
    for mode in ('strict','loose'):
        for level in ('prompt','instruction'):
            f=np.array([all(r[mode]) if level=='prompt' else sum(r[mode]) for r in full],dtype=float)
            o=np.array([all(b[r['key']][mode]) if level=='prompt' else sum(b[r['key']][mode]) for r in full],dtype=float)
            den=np.ones(len(full)) if level=='prompt' else n
            boot=(o-f)[idx].sum(axis=1)/den[idx].sum(axis=1)
            result[mode+'_'+level]={'difference_percentage_points':float(100*(o-f).sum()/den.sum()),'ci95_percentage_points':(100*np.quantile(boot,[.025,.975])).tolist(),'denominator':int(den.sum()),'bootstrap_draws':draws,'seed':seed}
            if level=='prompt':result[mode+'_'+level].update(other_only_pass=int(((o==1)&(f==0)).sum()),full_only_pass=int(((o==0)&(f==1)).sum()))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    full=read_jsonl(a.input_root/'Full/analysis/per_example.jsonl')
    result={m:compare(full,read_jsonl(a.input_root/m/'analysis/per_example.jsonl')) for m in ('NoLeft','NoRight','LRBalanced')}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'results.json').write_text(json.dumps({'method':'Paired prompt-cluster percentile bootstrap; 20,000 resamples; seed 20261012; pointwise 95% intervals; official outcomes unchanged','scope':'Benchmark-prompt sampling uncertainty conditional on these trained adapters; not training-seed variability or causal uncertainty. No multiple-comparison adjustment. Instruction ratios preserve within-prompt dependence. Official punctuation fallback retained.','comparisons':result},indent=2))
    lines=['# Paired IFEval differences against Full','', '| Model | Strict prompt Δ [95% CI] | Loose prompt Δ [95% CI] | Strict instruction Δ [95% CI] | Loose instruction Δ [95% CI] |','|---|---:|---:|---:|---:|']
    for m,v in result.items():
        cells=[]
        for metric in ('strict_prompt','loose_prompt','strict_instruction','loose_instruction'):
            r=v[metric];lo,hi=r['ci95_percentage_points'];cells.append(f"{r['difference_percentage_points']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    lines.extend(['','Differences and intervals are percentage points. Paired prompt-cluster bootstrap, 20,000 resamples, fixed seed 20261012. All intervals are pointwise and reflect benchmark-prompt uncertainty conditional on the trained models, not variability over training runs. Instruction-level bootstrap retains each prompt’s constraints together and recomputes the micro-accuracy denominator. Official scorer punctuation fallback is retained unchanged.'])
    (a.output_dir/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
