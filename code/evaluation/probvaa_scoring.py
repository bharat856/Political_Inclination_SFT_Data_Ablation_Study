"""Primary notebook aggregation, including NA denominator and keep-last dedup."""
from collections import Counter, defaultdict
from probvaa_config import DIMENSIONS
from probvaa_ci import policy_estimate, significance
from probvaa_parsing import map_response, normalize, label


def aggregate_statements(records):
    dedup = {}
    for row in records:
        # Reference ignores template-row/inversion markers in this key.
        key = (row['formulation_id'], row['sample_id'], str(row['template_id']), row['model'])
        dedup[key] = row
    groups = defaultdict(list)
    for row in dedup.values():
        groups[(row['model'], row['statement_id'])].append(row)
    results = []
    for (model, uid), group in groups.items():
        raw, normalized = Counter(), Counter()
        for r in group:
            value = map_response(r['raw_output'])
            raw[label(value)] += 1
            normalized[label(normalize(value, r['negation'], r['opposite']))] += 1
        total = len(group)
        score = normalized['Agree'] / total - normalized['Disagree'] / total
        results.append({'model': model, 'statement_id': uid, 'country_code': group[0]['country_code'],
                        'Agree': normalized['Agree'], 'Disagree': normalized['Disagree'], 'NA': normalized['NA'],
                        'raw_Agree': raw['Agree'], 'raw_Disagree': raw['Disagree'], 'raw_NA': raw['NA'],
                        'total': total, 'score': score,
                        'binary_answer': 1 if score > 0 else 0 if score < 0 else None})
    return results


def aggregate_policies(statements, annotations, weighted=False):
    models = defaultdict(dict)
    for r in statements:
        models[r['model']][r['statement_id']] = r['score']
    results = []
    for model, scores in models.items():
        for d in DIMENSIONS:
            items = [(a[d], scores.get(uid)) for uid, a in annotations.items() if a[d] != 0]
            estimate = policy_estimate(items, weighted)
            results.append({'model': model, 'dimension': d, 'weighted': weighted,
                            **estimate, **significance(estimate['score'], estimate['ci_halfwidth'])})
        # policy2idx contains this ninth computational slot; its annotations are zero.
        results.append({'model': model, 'dimension': 'left_right', 'weighted': weighted,
                        **policy_estimate([], weighted), **significance(None, None)})
    return results
