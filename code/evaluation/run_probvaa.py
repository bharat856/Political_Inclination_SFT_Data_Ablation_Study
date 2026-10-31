"""Manifest-only by --dry-run; inference requires explicit --execute.

Primary missing templates/formatter fail before any ML imports or model load.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from probvaa_config import (model_spec, references, DECLARED_GENERATION,
                            EFFECTIVE_GENERATION, MissingReferenceDependency, fingerprint)
from probvaa_data import load_formulations, load_templates, asset_report
from probvaa_prompting import get_formatter, iter_jobs, local_tokenizer_info
from probvaa_parsing import map_response, normalize, label


def prepare(spec, samples=30, statement_limit=None, templates_path=None,
            template_source='model-behavior', format_mode='model-behavior', formatter_dependency=None):
    if samples < 1 or (statement_limit is not None and statement_limit < 1):
        raise ValueError('Samples and statement limit must be positive')
    refs = references()
    report = asset_report()
    rows = load_formulations()
    uids = list(dict.fromkeys(r['statement_id'] for r in rows if r['formulation_type'] == 'original'))
    if statement_limit is not None:
        uids = uids[:statement_limit]
    rows = [r for r in rows if r['statement_id'] in set(uids)]
    templates = load_templates(spec, templates_path, template_source)
    formatter = get_formatter(format_mode, formatter_dependency)
    expected = len(rows) * len(templates) * samples
    # Construct every job once for validation; stream again for manifest/inference.
    factory = lambda: iter_jobs(rows, templates, formatter, samples)
    count = sum(1 for _ in factory())
    if count != expected:
        raise ValueError('Manifest arithmetic mismatch')
    metadata = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'references': refs,
        'model': spec.metadata(), 'dataset': report, 'selected_statement_ids': uids,
        'statements': len(uids), 'formulations': len(rows), 'templates': len(templates),
        'samples': samples, 'expected_total_generations': expected, 'manifest_count': count,
        'declared_generation': DECLARED_GENERATION, 'effective_generation': EFFECTIVE_GENERATION,
        'template_asset': fingerprint(templates[0]['template_source']),
        'template_source': template_source, 'format_mode': format_mode,
        'formatter_dependency': formatter_dependency,
        'tokenizer_json_info': local_tokenizer_info(spec),
        'replication_status': ('primary prompt dependencies supplied; upstream variants recovered, exact missing-file identity unverified'
                               if template_source == 'model-behavior' and format_mode == 'model-behavior'
                               else 'explicit upstream prompt reconstruction; NOT exact primary model-behavior prompt replication'),
    }
    if formatter_dependency:
        metadata['formatter_asset'] = fingerprint(formatter_dependency.rsplit(':', 1)[0])
    return metadata, factory


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    p.add_argument('--preset', choices=['full', 'official', 'base'], default='full')
    p.add_argument('--model-label', help='Distinct output identity, e.g. NoRight; defaults to preset')
    p.add_argument('--base-model')
    p.add_argument('--adapter')
    p.add_argument('--tokenizer')
    p.add_argument('--prompt-model-name', help='Explicit reference name used for base/instruct switches; adapter folder name is not inspected')
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--statement-limit', type=int)
    p.add_argument('--templates')
    p.add_argument('--template-source', choices=['model-behavior', 'worldviews'], default='model-behavior')
    p.add_argument('--format-mode', choices=['model-behavior', 'worldviews-percent'], default='model-behavior')
    p.add_argument('--formatter', help='Recovered primary formatter file.py:function')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--manifest', help='Optional JSONL manifest path, written exclusively')
    p.add_argument('--output-dir')
    p.add_argument('--allow-download', action='store_true', help='Manual inference only: allow HF model/tokenizer download')
    p.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], help='Explicit loading deviation; default inherits reference/library behavior')
    return p


def main():
    args = parser().parse_args()
    spec = model_spec(args.preset, args.base_model, args.adapter, args.tokenizer, args.prompt_model_name)
    if args.model_label:
        spec.model = args.model_label
    try:
        metadata, jobs = prepare(spec, args.samples, args.statement_limit, args.templates,
                                 args.template_source, args.format_mode, args.formatter)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'Cannot construct faithful jobs: {exc}') from exc
    metadata.update(seed=args.seed, seed_behavior='reset same seed before each generation',
                    dtype_override=args.dtype, dry_run=args.dry_run)
    print(json.dumps(metadata, indent=2))
    for j in itertools.islice((j for j in jobs() if j['sample_id'] == 0), 3):
        print(f"EXAMPLE {j['formulation_id']} template row {j['template_row']}:\n{j['raw_prompt']}")
    if args.manifest:
        with Path(args.manifest).open('x', encoding='utf-8') as f:
            for job in jobs():
                f.write(json.dumps(job, ensure_ascii=False) + '\n')
    if args.dry_run:
        print('Manifest validated. No tokenizer/model loaded; chat tokens deliberately not rendered.')
        return
    if not args.output_dir:
        raise SystemExit('--execute requires a new --output-dir')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Everything above is CPU-only standard library. Import ML only now.
    from probvaa_generation import Generator
    generator = Generator(spec, args.seed, args.allow_download, args.dtype)
    metadata['runtime'] = generator.metadata()
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (out / 'generations.jsonl').open('x', encoding='utf-8') as f:
        for job in jobs():
            # On a generation error stop; do not turn errors into fabricated NA responses.
            generated = generator.generate(job['raw_prompt'])
            answer = map_response(generated['raw_output'])
            normalized = normalize(answer, job['negation'], job['opposite'])
            row = job | spec.metadata() | generated | {
                'parsed_answer': label(answer), 'response_mapped': answer,
                'normalized_answer_after_inversion': label(normalized),
                'response_mapped_flipped': normalized, 'seed': args.seed,
                'reference_commits': {r['upstream_url']: r['commit'] for r in metadata['references']},
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()


if __name__ == '__main__':
    main()
