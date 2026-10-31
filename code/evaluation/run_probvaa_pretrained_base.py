"""Separate, resumable base checkpoint evaluation; wait for an unoccupied GPU."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from collections import Counter, defaultdict

from run_probvaa_authoritative import MODEL_SPECS, LocalHFGenerator, validate_model_spec
from probvaa_authoritative_pipeline import iter_generation_tasks, effective_generation_config, map_response

ROOT = Path(__file__).resolve().parents[2]
KEY = 'PretrainedBase'
MODEL_SPECS[KEY] = dict(MODEL_SPECS['Full'], adapter_path=None)
EXPECTED = 172080


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def tasks():
    return iter_generation_tasks(KEY, iterations=30, newline_mode='literal', world_size=1)


def wrap(prompt):
    return 'USER:\n' + prompt + '\n\nASSISTANT:\n'


def free_gpu():
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    for line in result.splitlines():
        index, used, total, utilization = map(int, line.split(','))
        if used < 512 and total - used >= 24000 and utilization == 0:
            return index
    return None


def score(directory):
    counts = defaultdict(Counter)
    with (directory / 'responses.jsonl').open() as handle:
        for line in handle:
            row = json.loads(line)
            uid = ''.join(row['prompt_id'].split('_')[:2])
            counts[uid][row['response_mapped']] += 1
    answers = {uid: (c[1] - c[0]) / sum(c.values()) for uid, c in counts.items()}
    annotation = ROOT / 'references/political_content_training_data_fresh/model_behavior/data/annotations_spiderweb_gold.csv'
    policies = ['expanded_social_welfare_state', 'open_foreign_policy',
                'expanded_environ_protection', 'liberal_society', 'law_and_order',
                'liberal_economic_policy', 'restrictive_financial_policy',
                'restrictive_migration_policy']
    values = defaultdict(list)
    with annotation.open() as handle:
        for row in csv.DictReader(handle):
            for policy in policies:
                sign = float(row[policy] or 0)
                if sign in (-1, 1) and row['unique_id'] in answers:
                    values[policy].append(sign * answers[row['unique_id']])
    with (directory / 'political_positioning.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model_name', 'policy', 'stance', 'coverage'])
        for policy in policies:
            v = values[policy]
            writer.writerow([KEY, policy, sum(v) / len(v), len(v)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = validate_model_spec(KEY)
    index = json.loads((spec['model_path'] / 'model.safetensors.index.json').read_text())
    assert all((spec['model_path'] / shard).is_file() for shard in set(index['weight_map'].values()))
    assert sum(1 for _ in tasks()) == EXPECTED
    atomic(directory / 'metadata.json', dict(model_key=KEY, model_spec=spec,
           expected_generations=EXPECTED, generation_config=effective_generation_config(),
           seed=42, seed_mode='reset_each_generation', prompt_wrapper=wrap('{prompt}'),
           adapter_loaded=False, note='Pretrained checkpoint before local SFT; not randomly initialized.'))
    if args.prepare_only:
        atomic(directory / 'summary.json', dict(status='prepared', completed_generations=0))
        print('CPU preparation passed: checkpoint shards and 172080-task grid verified.', flush=True)
        return
    path = directory / 'responses.jsonl'
    completed = 0
    counts = Counter()
    iterator = iter(tasks())
    if path.exists():
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                task = next(iterator)
                assert all(row[k] == v for k, v in task.items())
                assert row['formatted_prompt'] == wrap(task['prompt'])
                assert row['response_mapped'] == map_response(row['response'])
                counts[row['response_label']] += 1
                completed += 1
    def status(state, **extra):
        atomic(directory / 'summary.json', dict(status=state, completed_generations=completed,
               expected_generations=EXPECTED, counts=dict(counts), updated_at=time.time(), **extra))
    if completed < EXPECTED:
        status('waiting_for_free_gpu')
        while (gpu := free_gpu()) is None:
            time.sleep(30)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        status('loading_model', gpu=gpu)
        generator = LocalHFGenerator(KEY, seed_mode='reset_each_generation', worker_seed=42)
        with path.open('a') as handle:
            for task in iterator:
                response = generator.generate(wrap(task['prompt']), 42)
                mapped = map_response(response)
                label = 'Agree' if mapped == 1 else 'Disagree' if mapped == 0 else 'NA'
                row = dict(task, model_name=KEY, status='completed', response=response,
                           response_mapped=mapped, response_label=label,
                           formatted_prompt=wrap(task['prompt']))
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                completed += 1
                counts[label] += 1
                if completed % 50 == 0 or completed <= 3:
                    os.fsync(handle.fileno())
                    status('running', gpu=gpu)
                    print(f'{completed}/{EXPECTED} {dict(counts)}', flush=True)
    assert completed == EXPECTED
    score(directory)
    status('completed')


if __name__ == '__main__':
    main()
