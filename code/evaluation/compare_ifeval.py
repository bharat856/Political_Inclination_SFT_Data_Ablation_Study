"""Compare four completed official IFEval analyses; no generation or scoring."""
import argparse
import csv
import json
from pathlib import Path
from ifeval_artifacts import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True,
                        help='Contains {Full,NoLeft,NoRight,LRBalanced}/analysis/')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    results, baseline, config = {}, None, None
    lines = ['# Completed official IFEval comparison', '',
             '| Model | Strict prompt | Loose prompt | Strict instruction | Loose instruction |',
             '|---|---:|---:|---:|---:|']
    for model in ['Full','NoLeft','NoRight','LRBalanced']:
        directory = args.analysis_root / model / 'analysis'
        result = json.loads((directory / 'summary.json').read_text())
        examples = read_jsonl(directory / 'per_example.jsonl')
        if result['generations'] != 541 or len(examples) != 541:
            raise ValueError('Incomplete model analysis')
        paired = [(r['key'],r['prompt'],r['instruction_ids']) for r in examples]
        current = {key:result['metadata'][key] for key in ['runtime','seed','prompt_format','input_sha256']}
        if baseline is None:
            baseline, config = paired, current
        elif baseline != paired or config != current:
            raise ValueError('Models evaluated on different prompts/configurations')
        results[model] = result
        strict, loose = result['scores']['strict'], result['scores']['loose']
        lines.append(f"| {model} | {100*strict['prompt_accuracy']:.2f}% | {100*loose['prompt_accuracy']:.2f}% | {100*strict['instruction_accuracy']:.2f}% | {100*loose['instruction_accuracy']:.2f}% |")
    lines += ['', 'All models use the same pinned official benchmark, prompt format, generation settings and seed. Exact counts, versions and file hashes are in comparison.json. These scores do not by themselves establish causal significance or broad capability.', '',
              'Interrupted original segments lack trustworthy final runtime and CUDA peak measurements. Full-run totals are unavailable; resumed segment measurements are preserved separately. Political pilot scores remain diagnostic and are not final thesis results.']
    lines += ['', 'Official scorer caveat: keys 1122 and 1129 request punctuation frequencies ("#" and "!"), but the pinned letter-frequency checker accepts only ASCII letters and substitutes seeded random letters. Strict and loose passes can choose different fallback letters, so an official strict pass can become a loose failure. These are 2 of 834 instructions. Scores here preserve the official outputs without correction. Per-model anomalies are retained in summary.json and per_example.jsonl.']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'comparison.json').write_text(json.dumps(results,indent=2))
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
    table=[]
    for model,result in results.items():
        row={'model':model}
        for mode in ['strict','loose']:
            for key,value in result['scores'][mode].items():row[mode+'_'+key]=value
        table.append(row)
    with (args.output_dir/'comparison.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)


if __name__ == '__main__':main()
