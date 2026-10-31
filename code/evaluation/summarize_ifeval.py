"""Join full official IFEval scores with preserved raw responses (CPU only)."""
import argparse
import json
from pathlib import Path

from ifeval_artifacts import digest, read_jsonl, summarize_scores, validate_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    args = parser.parse_args()
    examples = read_jsonl(args.data)
    generations = read_jsonl(args.input_dir / 'generations.jsonl')
    responses = read_jsonl(args.input_dir / 'responses.jsonl')
    validate_segment(examples, generations, responses)
    scores, joined = summarize_scores(args.data, args.input_dir / 'generations.jsonl',
                                      args.input_dir / 'scoring/eval_results_strict.jsonl',
                                      args.input_dir / 'scoring/eval_results_loose.jsonl')
    if len(examples) != 541 or scores['strict']['instruction_total'] != 834:
        raise ValueError('This CLI requires the complete official 541-prompt/834-constraint benchmark')
    metadata = json.loads((args.input_dir / 'metadata.json').read_text())
    if metadata['input_sha256'] != digest(args.data):
        raise ValueError('Data hash differs from generation metadata')
    provenance = json.loads(args.provenance.read_text())
    for item in provenance['ifeval_provenance']['files']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Pinned official source/data changed')
    adapter = Path(metadata['model']['adapter_path'])
    audit = next(item for item in provenance['adapter_audit'] if item['path'] == str(adapter))
    if digest(adapter / 'adapter_model.safetensors') != audit['final']['sha256']:
        raise ValueError('Trained adapter changed')
    summary = dict(scores=scores, generations=541, metadata=metadata,
                   strict_pass_loose_fail_cases=[dict(key=r['key'],instruction_indices=r['strict_pass_loose_fail_indices']) for r in joined if r['strict_pass_loose_fail_indices']],
                   benchmark_provenance=provenance['ifeval_provenance'],
                   adapter_sha256=audit['final']['sha256'],
                   token_budget_hits=sum(len(r['output_token_ids']) == 1280 for r in generations),
                   input_hashes={name: digest(args.input_dir / name) for name in
                                 ['generations.jsonl','responses.jsonl','metadata.json',
                                  'scoring/eval_results_strict.jsonl','scoring/eval_results_loose.jsonl']})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (args.output_dir / 'per_example.jsonl').open('x') as handle:
        for row in joined:
            handle.write(json.dumps(row) + '\n')
    print(json.dumps(scores, indent=2))


if __name__ == '__main__':
    main()
