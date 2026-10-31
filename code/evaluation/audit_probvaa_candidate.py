"""CPU-only coverage/parsing/scoring audit with explicit unique-variant identity."""
import argparse
import json
from collections import Counter
from pathlib import Path
from ifeval_artifacts import read_jsonl,digest
from probvaa_data import load_annotations
from probvaa_parsing import map_response,label
from probvaa_scoring import aggregate_statements,aggregate_policies


def audit(records,prompts,samples,diagnostic_subset=False):
    if not 1<=samples<=30:raise ValueError('Samples must be 1..30')
    expected={(r['formulation_id'],r['template_variant_id']):r for r in prompts}
    keys=set();seeds=set();model_names=set();variant_inputs=[]
    for row in records:
        key=(row['formulation_id'],row['template_variant_id'])
        prompt=expected.get(key)
        if prompt is None:raise ValueError('Unexpected prompt variant')
        sample=row['sample_id'];full_key=key+(sample,)
        if full_key in keys or not 0<=sample<samples:raise ValueError('Duplicate or unexpected sample')
        if row['seed']!=prompt['seed_base']+sample or row['seed'] in seeds:raise ValueError('Changed or repeated sample seed')
        if row['training_format_prompt']!=prompt['training_format_prompt'] or row['statement_text']!=prompt['statement_text']:
            raise ValueError('Prompt/source mismatch')
        if row['parsed_answer']!=label(map_response(row['raw_output'])):raise ValueError('Parser assignment changed')
        keys.add(full_key);seeds.add(row['seed']);model_names.add(row['model'])
        variant_inputs.append(row|dict(template_family_id=prompt['template_family_id'],template_id=row['template_variant_id']))
    if len(model_names)!=1:raise ValueError('Audit each model separately')
    selected_sources={r['statement_id'] for r in records}
    selected={key for key,prompt in expected.items() if prompt['statement_id'] in selected_sources}
    required={(fid,tid,sample) for fid,tid in (selected if diagnostic_subset else expected) for sample in range(samples)}
    if keys!=required:raise ValueError('Incomplete intended prompt/sample coverage')
    statements=aggregate_statements(variant_inputs)
    if sum(r['total'] for r in statements)!=len(records):raise ValueError('Scorer discarded a variant')
    counts=Counter(r['parsed_answer'] for r in records)
    return dict(total=len(records),distribution=dict(counts),parsing_success=1-counts['NA']/len(records),
                diagnostic_only=diagnostic_subset,unique_seeds=len(seeds),per_variant=dict(Counter(r['template_variant_id'] for r in records)),
                statements=statements,policies=aggregate_policies(statements,load_annotations()),
                reference_formulas_unchanged=True,variant_identity_is_explicit_reconstruction=True),variant_inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--samples',type=int,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-subset',action='store_true')
    args=parser.parse_args()
    result,inputs=audit(read_jsonl(args.input),read_jsonl(args.manifest),args.samples,args.diagnostic_subset)
    result['input_sha256']=digest(args.input);result['manifest_sha256']=digest(args.manifest)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (args.output_dir/'scoring_inputs.jsonl').open('x') as handle:
        for row in inputs:handle.write(json.dumps(row)+'\n')
    print(result['total'],'records validated; all selected variant/sample inputs retained')


if __name__=='__main__':main()
