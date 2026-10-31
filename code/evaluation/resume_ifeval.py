"""Prepare a CPU-only resume plan; GPU execution requires explicit --execute.

Do not invoke --execute without the user's restart approval and a fresh process
check. Writes suffix results exclusively to a new directory, never to originals.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

from ifeval_artifacts import digest, read_jsonl, validate_configuration, validate_segment

REPO = Path(__file__).resolve().parents[2]
STAGE = REPO / 'outputs/evaluation_stage_20261011'
DIAG = REPO / 'outputs/evaluation_diagnostics_20261011'
DATA = DIAG / 'official_source/instruction_following_eval/data/input_data.jsonl'


def prepare(model):
    partial = STAGE / model / 'ifeval'
    metadata = json.loads((partial / 'metadata.json').read_text())
    smoke = json.loads((DIAG / 'ifeval_full/metadata.json').read_text())
    validate_configuration(metadata, smoke, DATA)
    examples = read_jsonl(DATA)
    start = validate_segment(examples, read_jsonl(partial / 'generations.jsonl'),
                             read_jsonl(partial / 'responses.jsonl'))
    provenance = json.loads((STAGE / 'provenance.json').read_text())
    for item in provenance['ifeval_provenance']['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Pinned official source/data changed')
    adapter = Path(metadata['model']['adapter_path'])
    audit = next(item for item in provenance['adapter_audit'] if item['path'] == str(adapter))
    if digest(adapter / 'adapter_model.safetensors') != audit['final']['sha256']:
        raise ValueError('Trained adapter weights changed')
    if len(examples) != 541 or start >= 541:
        raise ValueError('Expected an incomplete 541-example benchmark')
    plan = dict(model=model, original_directory=str(partial), completed=start,
                start_index=start, remaining=541-start, next_official_key=examples[start]['key'],
                data_sha256=digest(DATA), adapter_sha256=audit['final']['sha256'],
                original_artifact_hashes={name: digest(partial / name) for name in
                ['metadata.json', 'generations.jsonl', 'responses.jsonl']},
                runtime=metadata['runtime'], seed=42, prompt_format=metadata['prompt_format'],
                created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                approval_required_for_execution=True,
                warning='Plan only. Obtain user restart approval and verify no live worker before --execute.')
    return plan, metadata, examples


def generate_suffix(plan, original, examples, output):
    # Imported only for explicitly requested execution, never CPU planning/tests.
    import torch
    from probvaa_config import model_spec
    from probvaa_generation import Generator
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for explicitly approved resume')
    spec = model_spec('full', adapter=original['model']['adapter_path'])
    spec.model = plan['model']
    loaded = time.perf_counter()
    generator = Generator(spec, dtype='bfloat16')
    generator.settings.update(do_sample=False, temperature=None, top_p=None, max_new_tokens=1280)
    generator.seed = 42
    if generator.metadata() != original['runtime']:
        raise ValueError('Resume runtime differs from original/smoke configuration')
    metadata = dict(original, start_index=plan['start_index'], rows=plan['remaining'],
                    limit=None, original_directory=plan['original_directory'],
                    original_artifact_hashes=plan['original_artifact_hashes'],
                    load_seconds=time.perf_counter()-loaded, resume_configuration_verified=True)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / 'generations.jsonl').open('x') as raw, (output / 'responses.jsonl').open('x') as responses:
        for index in range(plan['start_index'], len(examples)):
            row = examples[index]
            result = generator.generate('USER:\n' + row['prompt'] + '\n\nASSISTANT:\n')
            raw.write(json.dumps(row | result) + '\n')
            raw.flush()
            responses.write(json.dumps(dict(prompt=row['prompt'], response=result['raw_output'])) + '\n')
            responses.flush()
            checkpoint = dict(model=plan['model'], completed_overall=index+1, status='running',
                              elapsed_seconds=time.perf_counter()-started,
                              peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                              updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            temporary = output / 'checkpoint.tmp'
            temporary.write_text(json.dumps(checkpoint, indent=2))
            temporary.replace(output / 'checkpoint.json')
            print(plan['model'], index+1, '/541', flush=True)
    torch.cuda.synchronize()
    metadata.update(generation_seconds=time.perf_counter()-started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    checkpoint['status'] = 'complete'
    (output / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['Full','NoLeft','NoRight','LRBalanced'], required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--execute', action='store_true', help='GPU launch: obtain user approval first')
    args = parser.parse_args()
    plan, metadata, examples = prepare(args.model)
    plan['execution_requested'] = args.execute
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resume_plan.json').write_text(json.dumps(plan, indent=2))
    if args.execute:
        generate_suffix(plan, metadata, examples, output)
    else:
        print(json.dumps({k: plan[k] for k in ['model','completed','remaining','next_official_key','warning']}, indent=2))


if __name__ == '__main__':
    main()
