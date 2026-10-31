"""Recover formulations from actual upstream rows; never synthesize variants."""
import csv
from collections import Counter, defaultdict
from pathlib import Path
from probvaa_config import PRIMARY, WORLDVIEWS, DIMENSIONS, MissingReferenceDependency, fingerprint


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def decode_id(identifier):
    country, number, code, language = identifier.split('_')
    if len(code) != 7 or not code.isdigit():
        raise ValueError(f'Invalid code: {identifier}')
    if code[4] == '1':
        kind = 'negation'
    elif code[3] == '1':
        kind = 'opposite'
    elif code[0] == '1' and code[2] != '0':
        kind = 'paraphrase_' + code[2]
    elif code[0] == '1' and code[2] == '0':
        kind = 'original'
    else:
        raise ValueError(f'Unsupported formulation: {identifier}')
    return dict(statement_id=country + number, country_code=country,
                local_statement_id=number, formulation_id=identifier,
                formulation_type=kind, code=code, language_code=language,
                negation=code[4] == '1', opposite=code[3] == '1')


def load_formulations():
    primary_path = PRIMARY / 'model_behavior/data/prob_vaa_en.csv'
    upstream_path = WORLDVIEWS / 'data/vaa/probvaa.csv'
    primary = read_csv(primary_path)
    upstream = [r for r in read_csv(upstream_path) if r['ID'].endswith('_en')]
    upstream_map = {r['ID']: r for r in upstream}
    if len(upstream_map) != len(upstream):
        raise ValueError('Duplicate upstream formulation IDs')
    rows = []
    for r in primary:
        if upstream_map.get(r['ID'], {}).get('statement') != r['statement']:
            raise ValueError(f'Upstream text mismatch: {r["ID"]}')
        rows.append(decode_id(r['ID']) | {'statement_text': r['statement'],
                    'policy_domain': r['policy_domain'], 'source_asset': str(primary_path),
                    'source_record': r})
    domains = {r['statement_id']: r['policy_domain'] for r in rows}
    primary_ids = {r['formulation_id'] for r in rows}
    for r in upstream:
        if r['ID'] in primary_ids:
            continue
        item = decode_id(r['ID'])
        if item['formulation_type'] not in ('negation', 'opposite'):
            raise ValueError(f'Unexpected extra variant: {r["ID"]}')
        rows.append(item | {'statement_text': r['statement'],
                    'policy_domain': domains[item['statement_id']],
                    'source_asset': str(upstream_path), 'source_record': r})
    groups = defaultdict(list)
    for r in rows:
        groups[r['statement_id']].append(r['formulation_type'])
    required = {'original', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'negation', 'opposite'}
    if any(set(v) != required or len(v) != 6 for v in groups.values()):
        raise ValueError('Incomplete or repeated formulation family')
    return rows


def load_annotations():
    p = PRIMARY / 'model_behavior/data/annotations_spiderweb_gold.csv'
    rows = read_csv(p)
    result = {}
    for r in rows:
        uid = ''.join(r['ID'].split('_')[:2])
        if uid in result:
            raise ValueError(f'Duplicate annotation: {uid}')
        values = {d: float(r[d] or 0) for d in DIMENSIONS}
        if any(v not in (-1, 0, 1) for v in values.values()):
            raise ValueError(f'Invalid annotation: {uid}')
        result[uid] = values
    other = read_csv(WORLDVIEWS / 'data/human_annotations/annotations_spiderweb_gold.csv')
    if len(rows) != len(other) or any(any(a[k] != b[k] for k in b) for a, b in zip(rows, other)):
        raise ValueError('Shared annotation fields differ between references')
    return result


def asset_report():
    rows, annotations = load_formulations(), load_annotations()
    originals = [r for r in rows if r['formulation_type'] == 'original']
    if set(annotations) != {r['statement_id'] for r in originals}:
        raise ValueError('Statement/annotation ID mismatch')
    return {
        'statements': len(originals), 'countries': dict(Counter(r['country_code'] for r in originals)),
        'formulations': len(rows), 'distinct_texts': len({r['statement_text'] for r in rows}),
        'formulation_types': dict(Counter(r['formulation_type'] for r in rows)),
        'dimensions': {d: sum(a[d] != 0 for a in annotations.values()) for d in DIMENSIONS},
        'annotation_memberships': dict(Counter(sum(v != 0 for v in a.values()) for a in annotations.values())),
        'assets': [fingerprint(p) for p in (
            PRIMARY / 'model_behavior/data/prob_vaa_en.csv',
            PRIMARY / 'model_behavior/data/annotations_spiderweb_gold.csv',
            WORLDVIEWS / 'data/vaa/probvaa.csv',
            WORLDVIEWS / 'data/human_annotations/annotations_spiderweb_gold.csv')],
        'variant_provenance_limit': '478 upstream English rows recovered; byte identity with missing primary negation/opposite file is unverified',
    }


def load_templates(spec, path=None, source='model-behavior'):
    if path is None:
        if source == 'worldviews':
            path = WORLDVIEWS / 'data/prompt_instructions/en_clf_templates_final.csv'
        else:
            name = 'en_clf_templates_final.csv' if spec.uses_chat else 'en_base_templates_final.csv'
            path = PRIMARY / 'model_behavior/data' / name
    if not Path(path).is_file():
        raise MissingReferenceDependency(f'Missing model-behavior template asset: {path}. No automatic substitute is used.')
    rows = read_csv(path)
    if source == 'worldviews':
        rows = [r for r in rows if r['model_type'] == ('chat' if spec.uses_chat else 'generation')]
    if not rows or any('template' not in r or 'template_id' not in r for r in rows):
        raise ValueError('Templates must contain template and template_id')
    # Primary iteration is row-based, even if template IDs repeat. Preserve both.
    return [r | {'template_row': i, 'template_source': str(Path(path).resolve()),
                 'template_source_profile': source} for i, r in enumerate(rows)]
