"""CPU-only prompt manifest for the explicitly reconstructed candidate protocol.

No inference entry point. Repeat count remains undecided unless --samples is
explicitly supplied. The primary missing templates are never silently replaced.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from probvaa_config import model_spec, fingerprint
from probvaa_data import load_formulations, load_templates, asset_report
from probvaa_prompting import iter_jobs, get_formatter


def candidate_jobs():
    rows=load_formulations()
    rank={identifier:index for index,identifier in enumerate(sorted(r['formulation_id'] for r in rows))}
    templates=load_templates(model_spec('full'),source='worldviews')
    variants=[t['final_template_id'] for t in templates]
    if len(rows)!=1434 or len(templates)!=12 or len(set(variants))!=12:
        raise ValueError('Unexpected candidate assets')
    for job in iter_jobs(rows,templates,get_formatter('worldviews-percent'),1):
        variant=job['template_fields']['final_template_id']
        yield job|dict(template_variant_id=variant,template_family_id=job['template_id'],
                       seed_base=42+(int(variant)*1434+rank[job['formulation_id']])*30,
                       training_format_prompt='USER:\n'+job['raw_prompt']+'\n\nASSISTANT:\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--samples',type=int,help='Explicit candidate repeat count, 1..30; omitted means undecided')
    args=parser.parse_args()
    if args.samples is not None and not 1<=args.samples<=30:
        raise ValueError('Distinct seed schedule reserves 30 samples per prompt')
    jobs=list(candidate_jobs())
    keys={(r['formulation_id'],r['template_variant_id']) for r in jobs}
    if len(jobs)!=17208 or len(keys)!=len(jobs):raise ValueError('Duplicate or missing prompt variant')
    per_variant=Counter(r['template_variant_id'] for r in jobs)
    if set(per_variant.values())!={1434}:raise ValueError('Variant coverage differs')
    bases=[r['seed_base'] for r in jobs]
    if len(set(bases))!=len(bases) or min(b-a for a,b in zip(sorted(bases),sorted(bases)[1:]))<30:
        raise ValueError('Seed stream overlap')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'prompts.jsonl').open('x') as handle:
        for row in jobs:handle.write(json.dumps(row)+'\n')
    metadata=dict(protocol='Explicit reconstructed candidate; not primary-reference replication',
                  model_generation_executed=False,source_statements=239,formulations=1434,
                  template_variants=12,prompts=17208,samples=args.samples,
                  expected_generations=None if args.samples is None else len(jobs)*args.samples,
                  per_variant=dict(per_variant),dataset=asset_report(),
                  manifest=fingerprint(args.output_dir/'prompts.jsonl'),
                  max_new_tokens=64,do_sample=True,temperature=1.0,top_k=50,top_p=1.0,
                  seed_formula='seed_base + sample_id; paired seeds across models; 30 reserved positions',
                  reference_parser_and_policy_formulas_unchanged=True,
                  intended_scoring_identity='template_id becomes template_variant_id in a copy; preserve template_family_id',
                  unresolved=['Final repeat count requires methodological decision',
                              'Remaining truncation and first-label artifacts require an explicit decision before full evaluation',
                              'Primary prompt template/formatter assets missing; candidate uses recovered worldviews templates and training wrapper'])
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({k:metadata[k] for k in ['source_statements','formulations','prompts','samples','expected_generations','model_generation_executed']},indent=2))


if __name__=='__main__':main()
