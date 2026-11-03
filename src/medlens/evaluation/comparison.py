"""Compare frozen rankings on one judgment pool and expose missing-label uncertainty."""

import copy
import itertools
import statistics

from .metrics import evaluate,macro_average

METRICS=('ndcg@10','recall@10','mrr@10','precision@10')


def paired_deltas(per_query, pairs):
    result={}
    for candidate,baseline in pairs:
        metrics={}
        for metric in METRICS:
            values=[]
            for row in per_query:
                left=row['methods'][candidate][metric];right=row['methods'][baseline][metric]
                if left is not None and right is not None:values.append(left-right)
            metrics[metric]={'mean':statistics.mean(values) if values else None,
                'median':statistics.median(values) if values else None,'query_count':len(values),
                'wins':sum(v>1e-12 for v in values),'ties':sum(abs(v)<=1e-12 for v in values),
                'losses':sum(v< -1e-12 for v in values)}
        result[f'{candidate}-minus-{baseline}']=metrics
    return result


def compare(rows,qrels,primary_methods,pairs=()):
    if not rows:raise ValueError('Comparison requires queries')
    ids=[row['query_id'] for row in rows]
    methods=list(rows[0]['rankings'])
    if (len(set(ids))!=len(ids) or set(ids)!=set(qrels) or not methods
            or not set(primary_methods)<=set(methods) or not primary_methods):
        raise ValueError('Comparison query or method membership differs')
    if any(not {a,b}<=set(methods) for a,b in pairs):raise ValueError('Unknown paired method')
    evaluated=[]
    for row in rows:
        if set(row['rankings'])!=set(methods):raise ValueError('Methods differ across queries')
        metrics={m:evaluate(row['rankings'][m],row['candidates'],qrels[row['query_id']],10) for m in methods}
        evaluated.append({'query_id':row['query_id'],'methods':metrics})
    common=[row for row in evaluated if all(row['methods'][m]['unjudged_returned_count']==0 for m in primary_methods)]
    return {'comparison_status':'fully-judged-top-k' if len(common)==len(rows) else 'provisional-incomplete-judgments',
        'per_query':evaluated,
        'macro':{m:macro_average([row['methods'][m] for row in evaluated],10) for m in methods},
        'paired_deltas':paired_deltas(evaluated,pairs),
        'common_topk_complete':{'criterion_methods':list(primary_methods),'query_ids':[r['query_id'] for r in common],
            'query_count':len(common),'total_queries':len(rows),
            'macro':{m:macro_average([row['methods'][m] for row in common],10) for m in methods},
            'paired_deltas':paired_deltas(common,pairs)}}


def missing_label_sensitivity(rows,qrels,unresolved_pairs,primary_methods,pairs=(),*,max_scenarios=729):
    compare(rows,qrels,primary_methods,pairs)
    keys=[(p['query_id'],p['pmid']) for p in unresolved_pairs]
    if len(set(keys))!=len(keys) or any(q not in qrels or p not in qrels[q] or qrels[q][p] is not None for q,p in keys):
        raise ValueError('Sensitivity pairs must be distinct explicit null judgments')
    if 3**len(keys)>max_scenarios:raise ValueError('Too many exact missing-label scenarios')
    method_values={m:{metric:[] for metric in METRICS} for m in rows[0]['rankings']}
    delta_values={f'{a}-minus-{b}':{metric:[] for metric in METRICS} for a,b in pairs}
    for grades in itertools.product((0,1,2),repeat=len(keys)):
        alternative=copy.deepcopy(qrels)
        for (qid,pmid),grade in zip(keys,grades):alternative[qid][pmid]=grade
        result=compare(rows,alternative,primary_methods,pairs)
        for method in method_values:
            for metric in METRICS:
                value=result['macro'][method][metric]['mean']
                if value is not None:method_values[method][metric].append(value)
        for pair in delta_values:
            for metric in METRICS:
                value=result['paired_deltas'][pair][metric]['mean']
                if value is not None:delta_values[pair][metric].append(value)
    def bounds(values):
        return {key:{metric:{'min':min(v) if v else None,'max':max(v) if v else None,
                             'defined_scenarios':len(v)} for metric,v in metrics.items()}
                for key,metrics in values.items()}
    return {'scenario_count':3**len(keys),'unresolved_pairs':unresolved_pairs,
        'scope':'Only the explicitly listed null judgments vary; unpooled candidate labels remain unknown.',
        'macro_bounds':bounds(method_values),'paired_delta_bounds':bounds(delta_values)}
