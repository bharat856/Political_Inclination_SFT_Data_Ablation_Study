"""Explicit candidate protocol: CPU plan by default; --execute gates inference.
Saved segments are immutable. Any damaged/inconsistent segment stops recovery.
"""
import argparse, fcntl, json, os, time
from pathlib import Path
from ifeval_artifacts import digest, read_jsonl
from prepare_probvaa_candidate import candidate_jobs
from audit_probvaa_candidate import audit
from probvaa_config import model_spec, references
from probvaa_data import asset_report
from probvaa_parsing import map_response, label
from probvaa_supplementary import supplementary

ADAPTERS = {'Full':'olmo3_instruct_full_4gpu_v4','NoLeft':'olmo3_instruct_noleft_4gpu_v4','NoRight':'olmo3_instruct_noright_4gpu_v5','LRBalanced':'olmo3_instruct_lrbalanced_4gpu_v5'}

def validate_saved(records, prompts, samples, model, parameters):
    expected={(p['formulation_id'],p['template_variant_id']):p for p in prompts}
    seen=set()
    for r in records:
        key=(r['formulation_id'],r['template_variant_id'],r['sample_id'])
        p=expected.get(key[:2])
        if p is None or key in seen or not 0<=key[2]<samples:raise ValueError('Duplicate/unexpected completed ID')
        if r['model']!=model or r['seed']!=p['seed_base']+key[2] or r['training_format_prompt']!=p['training_format_prompt']:raise ValueError('Saved prompt/seed/model mismatch')
        for field in ('statement_id','statement_text','negation','opposite'):
            if r[field]!=p[field]:raise ValueError('Saved statement mismatch')
        if r['generation_parameters']!=parameters or r['parsed_answer']!=label(map_response(r['raw_output'])):raise ValueError('Saved generation/parser mismatch')
        if not isinstance(r['output_token_ids'],list) or len(r['output_token_ids'])>64:raise ValueError('Invalid saved token record')
        seen.add(key)
    return seen

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',choices=ADAPTERS,required=True);a.add_argument('--samples',type=int,required=True)
    a.add_argument('--smoke-prompts',type=int,choices=range(1,13),help='Explicit diagnostic-only subset, at most 12 prompts')
    a.add_argument('--stop-after',type=int,help='Graceful checkpoint after this many NEW generations (smoke only)')
    a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--execute',action='store_true')
    args=a.parse_args()
    if not 1<=args.samples<=30:raise ValueError('Samples must be 1..30')
    root=Path(__file__).resolve().parents[2];os.chdir(root)
    pilot=root/'outputs/evaluation_stage_20261011'/args.model/'probvaa_pilot'
    baseline=json.loads((pilot/'metadata.json').read_text());parameters=baseline['runtime']['generation_kwargs']
    prompts=list(candidate_jobs())
    if args.smoke_prompts:prompts=prompts[:args.smoke_prompts]
    if args.stop_after is not None and (not args.smoke_prompts or args.stop_after<1):raise ValueError('stop-after is positive and smoke-only')
    adapter=root/'outputs/sft_runs'/ADAPTERS[args.model]
    config={'diagnostic_smoke_prompts':args.smoke_prompts,'references':references(),'assets':asset_report(),'model':args.model,'samples':args.samples,'prompts_sha256':__import__('hashlib').sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest(),'adapter_sha256':{p.name:digest(p) for p in adapter.glob('adapter*') if p.is_file()},'generation_parameters':parameters,'baseline_sha256':digest(pilot/'metadata.json'),'pipeline_sha256':{f:digest(root/'code/evaluation'/f) for f in ('probvaa_data.py','probvaa_parsing.py','probvaa_scoring.py','probvaa_ci.py','probvaa_prompting.py','prepare_probvaa_candidate.py','audit_probvaa_candidate.py','resume_probvaa_candidate.py','probvaa_supplementary.py')}}
    out=args.output_dir
    records=[];segments=[]
    if out.exists():
        if json.loads((out/'configuration.json').read_text())!=config:raise ValueError('Recovery configuration changed')
        for segment in sorted(out.glob('segment_*')):
            saved=read_jsonl(segment/'generations.jsonl');records.extend(saved);segments.append({'path':str(segment),'sha256':digest(segment/'generations.jsonl'),'count':len(saved)})
    seen=validate_saved(records,prompts,args.samples,args.model,parameters)
    missing=[(p,s) for p in prompts for s in range(args.samples) if (p['formulation_id'],p['template_variant_id'],s) not in seen]
    print(json.dumps({'completed':len(seen),'missing':len(missing),'total':len(prompts)*args.samples,'execute':args.execute}),flush=True)
    if not args.execute:return
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Prevent a second process using a stale pre-lock plan.
    current=[str(p) for p in sorted(out.glob('segment_*'))]
    if current!=[s['path'] for s in segments] or any(digest(Path(s['path'])/'generations.jsonl')!=s['sha256'] for s in segments):raise ValueError('Segments changed during preflight')
    if not (out/'configuration.json').exists():
        with (out/'configuration.json').open('x') as f:json.dump(config,f,indent=2)
    def progress(status):
        tmp=out/'progress.tmp';tmp.write_text(json.dumps({'status':status,'pid':os.getpid(),'completed':len(records),'expected':len(prompts)*args.samples,'updated':time.time()}));tmp.replace(out/'progress.json')
    progress('loading' if missing else 'scoring')
    if missing:
        from probvaa_generation import Generator
        import torch
        if not torch.cuda.is_available():raise ValueError('No available CUDA device')
        g=Generator(model_spec('full',adapter=str(adapter)),dtype='bfloat16');g.settings.update(max_new_tokens=64,do_sample=True,temperature=1.,top_k=50,top_p=1.)
        runtime=g.metadata()
        if runtime!=baseline['runtime']:raise ValueError('Runtime differs from corrected pilot')
        segment=out/f'segment_{len(segments):04d}';segment.mkdir(exist_ok=False)
        (segment/'metadata.json').write_text(json.dumps({'configuration':config,'runtime':runtime,'previous_segments':segments},indent=2))
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with (segment/'generations.jsonl').open('x') as f:
            for new_index,(p,s) in enumerate(missing):
                g.seed=p['seed_base']+s;result=g.generate(p['training_format_prompt'])
                r=p|result|{'model':args.model,'sample_id':s,'seed':g.seed,'parsed_answer':label(map_response(result['raw_output']))}
                f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno());records.append(r);progress('generating')
                if args.stop_after and new_index+1>=args.stop_after:break
        torch.cuda.synchronize();(segment/'measurements.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()},indent=2))
    if len(records)<len(prompts)*args.samples:
        progress('checkpointed');return
    summary,inputs=audit(records,prompts,args.samples,diagnostic_subset=bool(args.smoke_prompts))
    if (out/'analysis').exists():
        if json.loads((out/'analysis/summary.json').read_text())!=summary or read_jsonl(out/'analysis/scoring_inputs.jsonl')!=inputs:raise ValueError('Existing analysis is incomplete or inconsistent')
    else:
        (out/'analysis').mkdir();(out/'analysis/summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
        with (out/'analysis/scoring_inputs.jsonl').open('x') as f:
            for r in inputs:f.write(json.dumps(r)+'\n')
    supp=supplementary(records)
    sp=out/'analysis/supplementary.json'
    if sp.exists():
        if json.loads(sp.read_text())!=supp:raise ValueError('Supplementary output changed')
    else:
        with sp.open('x') as f:json.dump(supp,f,indent=2,allow_nan=False)
    progress('complete')

if __name__=='__main__':main()
