"""Validate and merge IFEval artifacts without generating or changing scores."""
import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f'{path}:{number}: blank record')
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f'{path}:{number}: invalid JSON') from error
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_segment(data, generations, responses, start=0):
    if len(generations) != len(responses):
        raise ValueError('Raw/scorer-input record counts differ')
    if start < 0 or start + len(generations) > len(data):
        raise ValueError('Segment lies outside benchmark')
    for index, (raw, response) in enumerate(zip(generations, responses), start):
        example = data[index]
        for key, value in example.items():
            if raw.get(key) != value:
                raise ValueError(f'Official example field differs at {index}: {key}')
        expected = 'USER:\n' + example['prompt'] + '\n\nASSISTANT:\n'
        if raw.get('rendered_model_input') != expected:
            raise ValueError(f'Prompt wrapper differs at {index}')
        if response != {'prompt': example['prompt'], 'response': raw['raw_output']}:
            raise ValueError(f'Scorer input differs from raw generation at {index}')
    if len({row['key'] for row in generations}) != len(generations):
        raise ValueError('Duplicate benchmark keys')
    return start + len(generations)


def validate_configuration(metadata, smoke, data_path):
    if metadata['input_sha256'] != digest(data_path):
        raise ValueError('Benchmark data hash differs')
    if metadata['prompt_format'] != smoke['prompt_format']:
        raise ValueError('Prompt format differs from smoke test')
    if metadata.get('seed') != 42:
        raise ValueError('Generation seed differs')
    # Includes inherited defaults, tokenizer identity and package versions.
    if metadata['runtime'] != smoke['runtime']:
        raise ValueError('Effective runtime differs from smoke test')


def merge_segments(data_path, segment_dirs, output_dir):
    """Require exact contiguous full coverage; preserve all source files."""
    data = read_jsonl(data_path)
    generations, responses, provenance = [], [], []
    previous = None
    for directory in map(Path, segment_dirs):
        metadata = json.loads((directory / 'metadata.json').read_text())
        if previous is not None:
            for key in ['model', 'runtime', 'seed', 'prompt_format', 'input_sha256']:
                if metadata[key] != previous[key]:
                    raise ValueError(f'Segment configuration differs: {key}')
        if metadata['input_sha256'] != digest(data_path):
            raise ValueError('Segment data hash differs')
        start = metadata.get('start_index', 0)
        if start != len(generations):
            raise ValueError('Segment gap, overlap or incorrect ordering')
        raw = read_jsonl(directory / 'generations.jsonl')
        scorer_inputs = read_jsonl(directory / 'responses.jsonl')
        validate_segment(data, raw, scorer_inputs, start)
        generations.extend(raw)
        responses.extend(scorer_inputs)
        provenance.append({'directory': str(directory.resolve()), 'start_index': start,
                           'count': len(raw), 'hashes': {name: digest(directory / name)
                           for name in ['metadata.json', 'generations.jsonl', 'responses.jsonl']}})
        previous = metadata
    if len(generations) != len(data):
        raise ValueError(f'Incomplete benchmark: {len(generations)}/{len(data)}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, content in [('generations.jsonl', generations), ('responses.jsonl', responses)]:
        with (output_dir / name).open('x') as handle:
            for row in content:
                handle.write(json.dumps(row) + '\n')
    metadata = dict(previous, rows=len(data), limit=None, start_index=0,
                    segments=provenance, complete_coverage_verified=True)
    # Interrupted segments have no trustworthy final CUDA peak or elapsed timer.
    metadata['generation_seconds'] = None
    metadata['peak_allocated_gib'] = None
    metadata['peak_reserved_gib'] = None
    metadata['measurement_note'] = 'Interrupted original segment lacks final runtime/peak measurements; use segment measurements separately.'
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def summarize_scores(data_path, generations_path, strict_path, loose_path):
    data = read_jsonl(data_path)
    generations = read_jsonl(generations_path)
    if len(generations) != len(data):
        raise ValueError('Refuse to publish full scores from partial generations')
    scores, per_example = {}, []
    modes = {'strict': read_jsonl(strict_path), 'loose': read_jsonl(loose_path)}
    for mode, rows in modes.items():
        if len(rows) != len(data):
            raise ValueError('Incomplete official scoring outputs')
        for example, row in zip(data, rows):
            if row['prompt'] != example['prompt'] or row['instruction_id_list'] != example['instruction_id_list']:
                raise ValueError('Official score/example alignment differs')
            if len(row['follow_instruction_list']) != len(example['instruction_id_list']):
                raise ValueError('Constraint count differs')
            if any(type(v) is not bool for v in row['follow_instruction_list']):
                raise ValueError('Non-boolean instruction result')
            if row['follow_all_instructions'] != all(row['follow_instruction_list']):
                raise ValueError('Prompt and instruction correctness disagree')
        correct = sum(sum(r['follow_instruction_list']) for r in rows)
        total = sum(len(r['follow_instruction_list']) for r in rows)
        prompt_correct = sum(r['follow_all_instructions'] for r in rows)
        scores[mode] = dict(prompt_correct=prompt_correct, prompt_total=len(data),
                            prompt_accuracy=prompt_correct / len(data),
                            instruction_correct=correct, instruction_total=total,
                            instruction_accuracy=correct / total)
    for example, raw, strict, loose in zip(data, generations, modes['strict'], modes['loose']):
        if raw['key'] != example['key'] or raw['prompt'] != example['prompt']:
            raise ValueError('Raw generation/example alignment differs')
        # The pinned official checker can randomize invalid letter-frequency
        # arguments (e.g. '#') differently in strict and loose passes. Preserve
        # official results and expose regressions rather than changing scores.
        regressions=[i for i,(a,b) in enumerate(zip(strict['follow_instruction_list'],loose['follow_instruction_list'])) if a and not b]
        if strict['response'] != raw['raw_output'] or loose['response'] != raw['raw_output']:
            raise ValueError('Official scorer used a different response')
        per_example.append(dict(key=example['key'], prompt=example['prompt'], response=raw['raw_output'],
                                instruction_ids=example['instruction_id_list'],
                                output_tokens=len(raw['output_token_ids']),
                                strict_pass_loose_fail_indices=regressions,
                                strict=strict['follow_instruction_list'], loose=loose['follow_instruction_list']))
    return scores, per_example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--segments', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    merge_segments(args.data, args.segments, args.output_dir)


if __name__ == '__main__':
    main()
