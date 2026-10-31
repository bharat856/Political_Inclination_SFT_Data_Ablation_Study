"""CPU-only raw JSONL -> statement/policy CSV and JSON summaries."""
import argparse
import csv
import json
from pathlib import Path
from probvaa_config import references
from probvaa_data import load_annotations
from probvaa_scoring import aggregate_statements, aggregate_policies


def summarize(paths, weighted=False):
    records = []
    for path in paths:
        with Path(path).open(encoding='utf-8') as f:
            records.extend(json.loads(line) for line in f if line.strip())
    statements = aggregate_statements(records)
    return {'statements': statements, 'policies': aggregate_policies(statements, load_annotations(), weighted),
            'input_paths': [str(Path(p).resolve()) for p in paths],
            'statement_formula': '(Agree-Disagree)/(Agree+Disagree+NA)',
            'weighted': weighted, 'references': references()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', help='Generation JSONL files, chronological order; last duplicate wins')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--weighted', action='store_true', help='Verified optional branch; final reference plots use unweighted')
    args = p.parse_args()
    result = summarize(args.inputs, args.weighted)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for kind in ('statements', 'policies'):
        rows = result[kind]
        if rows:
            with (out / f'{kind}.csv').open('x', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == '__main__':
    main()
