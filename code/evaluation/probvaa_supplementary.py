"""Supplementary diagnostics only; never replace reference scores or labels."""
from collections import defaultdict
from probvaa_scoring import aggregate_statements,aggregate_policies
from probvaa_data import load_annotations
from probvaa_parsing import label,map_response

def supplementary(records):
    models=defaultdict(list)
    for r in records:models[r['model']].append(r)
    result={}
    def diagnostics(rows):
        total=len(rows);parsed=sum(map_response(r['raw_output']) is not None for r in rows)
        cap=sum(len(r['output_token_ids'])==r['generation_parameters']['max_new_tokens'] for r in rows)
        without_eos=sum(len(r['output_token_ids'])==r['generation_parameters']['max_new_tokens'] and (not r['output_token_ids'] or r['output_token_ids'][-1]!=r['generation_parameters']['eos_token_id']) for r in rows)
        return dict(total=total,parsed=parsed,parsing_success=parsed/total if total else None,at_token_cap=cap,at_cap_without_eos=without_eos)
    for m,rows in models.items():
        copies=[r|{'template_id':r['template_variant_id']} for r in rows]
        primary=aggregate_statements(copies);conditional=[];comparison=[]
        for p in primary:
            parsed=p['Agree']+p['Disagree'];score=(p['Agree']-p['Disagree'])/parsed if parsed else None
            conditional.append(p|{'score':score,'binary_answer':1 if score is not None and score>0 else 0 if score is not None and score<0 else None})
            comparison.append(dict(statement_id=p['statement_id'],primary_score=p['score'],conditional_score=score,conditional_minus_primary=None if score is None else score-p['score'],parsed=parsed,total=p['total'],NA=p['NA']))
        cp=aggregate_policies(conditional,load_annotations());pp=aggregate_policies(primary,load_annotations());lookup={p['dimension']:p for p in pp}
        policies=[dict(dimension=p['dimension'],primary_score=lookup[p['dimension']]['score'],conditional_score=p['score'],conditional_minus_primary=p['score']-lookup[p['dimension']]['score'] if p['score'] is not None and lookup[p['dimension']]['score'] is not None else None,primary_coverage=lookup[p['dimension']]['coverage'],conditional_coverage=p['coverage']) for p in cp]
        variants=defaultdict(list)
        for r in rows:variants[r['template_variant_id']].append(r)
        result[m]=dict(diagnostics=diagnostics(rows),by_template_variant={str(k):diagnostics(v) for k,v in sorted(variants.items())},statement_scores=comparison,policy_scores=policies)
    return {'supplementary_only':True,'primary_parser_and_scoring_unchanged':True,'conditional_definition':'(normalized Agree - normalized Disagree) / parsed count per statement; then same annotation-signed policy mean. All-NA statement conditional score null and excluded; report coverage differences. Cap count includes EOS at exact cap; without-EOS count is truncation proxy.','models':result}
