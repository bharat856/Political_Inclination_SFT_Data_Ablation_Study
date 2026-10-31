"""Standard-library-only asset audit; does not run the evaluator or models."""
import json
from probvaa_config import references, PRIMARY, WORLDVIEWS
from probvaa_data import asset_report, read_csv


def validate():
    report = asset_report()
    report['references'] = references()
    report['template_inventory'] = {}
    for filename in ('en_clf_templates_final.csv', 'en_gpt_templates_final.csv'):
        rows = read_csv(WORLDVIEWS / 'data/prompt_instructions' / filename)
        report['template_inventory'][filename] = {
            'rows': len(rows), 'chat': sum(r['model_type'] == 'chat' for r in rows),
            'generation': sum(r['model_type'] == 'generation' for r in rows),
            'distinct_template_ids': len({r['template_id'] for r in rows}),
        }
    report['missing_primary_dependencies'] = [str(PRIMARY / 'model_behavior/data' / n)
        for n in ('prob_vaa_en_negation_opposite.csv', 'en_clf_templates_final.csv', 'en_base_templates_final.csv')]
    report['missing_primary_formatter'] = 'utils.format_prompt'
    return report


if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
