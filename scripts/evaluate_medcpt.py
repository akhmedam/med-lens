"""Evaluate saved MedCPT and historical baselines on a versioned common pilot pool."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
from rescore_pilot import rescore
from medlens.evaluation.comparison import compare,missing_label_sensitivity
from medlens.evaluation.formats import parse_queries,parse_qrels
from medlens.generation.context import digest
from medlens.reranking.articles import MedCPTRanker,aggregate_windows
from medlens.reranking.medcpt import MODEL_ID,REVISION,sha_file

PAIRS=(('dense_medcpt','dense'),('medcpt','dense'),('medcpt','dense_medcpt'),('medcpt','dense_qwen'))


def read_entry(root,profile,name):
    entry=profile[name];path=root/entry['path'];body=path.read_bytes()
    if hashlib.sha256(body).hexdigest()!=entry['sha256']:raise ValueError(f'{name} checksum mismatch')
    return body


def load_comparison(profile_path,*,root=ROOT):
    profile_bytes=Path(profile_path).read_bytes();profile=json.loads(profile_bytes)
    if profile['schema_version']!=1 or profile['split']!='pilot':raise ValueError('Require a pilot comparison profile')
    if profile['primary_methods']!=['pubmed','dense','dense_medcpt','medcpt']:
        raise ValueError('Unexpected primary comparison methods')
    base_bytes=read_entry(root,profile,'base_profile');base_profile=json.loads(base_bytes)
    historical,_,_=rescore(root/profile['base_profile']['path'],root=root)
    query_path=root/base_profile['queries']['path'];query_bytes=query_path.read_bytes()
    queries=parse_queries(query_bytes,query_path.parent)
    if any(q.split!='pilot' for q in queries):raise ValueError('Only development pilot queries are permitted')
    base_qrels_body=(root/base_profile['qrels']['path']).read_bytes()
    base_qrels=parse_qrels(base_qrels_body,queries)
    qrel_bytes=read_entry(root,profile,'qrels');qrels=parse_qrels(qrel_bytes,queries)
    for qid,judgments in base_qrels.items():
        if any(p not in qrels[qid] or qrels[qid][p]!=grade for p,grade in judgments.items()):
            raise ValueError('Existing v2 judgments must remain unchanged')
    items=[json.loads(line) for line in read_entry(root,profile,'supplement_items').splitlines()]
    reviews=[json.loads(line) for line in read_entry(root,profile,'supplement_reviews').splitlines()]
    provenance=json.loads(read_entry(root,profile,'provenance'))
    for key,name in (('items_sha256','supplement_items'),('reviews_sha256','supplement_reviews'),('qrels_sha256','qrels')):
        if provenance[key]!=profile[name]['sha256']:raise ValueError('Supplement provenance checksum differs')
    if provenance['base_qrels_sha256']!=base_profile['qrels']['sha256']:raise ValueError('Supplement base profile differs')
    item_map={(r['query_id'],r['pmid']):r for r in items}
    review_map={(r['query_id'],r['pmid']):r for r in reviews}
    new_pairs={(q,p) for q,judgments in qrels.items() for p in judgments if p not in base_qrels[q]}
    if (len(item_map)!=len(items) or len(review_map)!=len(reviews)
            or set(item_map)!=new_pairs or set(review_map)!=new_pairs):
        raise ValueError('Supplement must cover exactly the additional judgment pairs')
    for key,review in review_map.items():
        item=item_map[key]
        grade=review['relevance']
        if ((grade is not None and (type(grade) is not int or grade not in {0,1,2}))
                or review['confidence'] not in {'low','medium','high'}
                or review['relevance']!=qrels[key[0]][key[1]] or not isinstance(review['rationale'],str)
                or not review['rationale'].strip() or review['item_sha256']!=digest(item)
                or review['dataset_id']!=item['dataset_id'] or review['text_version_id']!=item['text_version_id']):
            raise ValueError('Supplement label, source or rationale mismatch')
    counts={'pair_count':sum(len(v) for v in qrels.values()),
        'judged_count':sum(g is not None for v in qrels.values() for g in v.values()),
        'unjudged_count':sum(g is None for v in qrels.values() for g in v.values())}
    if any(profile['annotation'][k]!=v for k,v in counts.items()):
        raise ValueError('Annotation counts differ from actual qrels')
    if (provenance['total_pair_count']!=counts['pair_count']
            or provenance['judged_count']!=counts['judged_count']
            or provenance['unjudged_count']!=counts['unjudged_count']
            or provenance['added_pair_count']!=len(new_pairs)
            or provenance['preserved_base_pair_count']!=sum(len(v) for v in base_qrels.values())):
        raise ValueError('Supplement provenance counts differ')
    unresolved={(q,p) for q,judgments in qrels.items() for p,g in judgments.items() if g is None}
    configured=[(p['query_id'],p['pmid']) for p in profile['unresolved_pairs']]
    if len(set(configured))!=len(configured) or set(configured)!=unresolved:
        raise ValueError('Unresolved-pair configuration differs from actual qrels')
    cached=json.loads(read_entry(root,profile,'rankings'))
    if (cached['status']!='complete' or cached['heldout_used'] or cached['relevance_metrics_evaluated']
            or cached['top_k']!=10 or cached['dense_depth']!=10):
        raise ValueError('Require completed pre-evaluation pilot rankings')
    if cached['queries_sha256']!=hashlib.sha256(query_bytes).hexdigest():raise ValueError('Ranking query file differs')
    if ([q.query_id for q in queries]!=[r['query_id'] for r in cached['per_query']]
            or [q.query_id for q in queries]!=[r['query_id'] for r in historical['per_query']]):
        raise ValueError('Ranking query order/membership differs')
    input_ids={'dense':[r['dense']['run_id'] for r in cached['per_query']],
        'medcpt':[{m:r['run_id'] for m,r in row['methods'].items()} for row in cached['per_query']]}
    identity={k:cached[k] for k in ('queries_sha256','device','batch_size','dense_depth','top_k','code_sha256')}
    if cached['input_run_ids']!=input_ids or cached['run_id']!=digest({**identity,'input_run_ids':input_ids}):
        raise ValueError('Ranking artifact identity differs')
    rows=[];source_checks=[]
    for query,saved,old in zip(queries,cached['per_query'],historical['per_query']):
        ranker=MedCPTRanker(query.dataset,root/'data/windows/pilot-v1-002'/query.query_id)
        candidates=list(ranker.articles)
        if not set(qrels[query.query_id])<=set(candidates):raise ValueError('Judgment outside candidate snapshot')
        if saved['query']!=query.text or saved['dense']['query']!=query.text:raise ValueError('Ranking question differs')
        if (saved['dense']['dataset_id']!=ranker.dataset.manifest['dataset_id']
                or saved['dense']['window_artifact_id']!=ranker.window_manifest['artifact_id']
                or old['dataset_id']!=ranker.dataset.manifest['dataset_id']):
            raise ValueError('Ranking dataset/window identity differs')
        dense=[h['pmid'] for h in saved['dense']['results']]
        if dense!=old['methods']['dense']['ranking']:
            raise ValueError('Fresh Dense differs from the frozen historical Dense baseline')
        if candidates[:10]!=old['methods']['pubmed']['ranking']:raise ValueError('PubMed source order differs')
        methods={m:list(old['methods'][m]['ranking']) for m in ('pubmed','dense')}
        if set(saved['methods'])!={'dense_medcpt','medcpt'}:raise ValueError('Unexpected MedCPT modes')
        for method,result in saved['methods'].items():
            order=dense if method=='dense_medcpt' else candidates
            windows=ranker.eligible_windows(order);records=result['window_scores']
            if (result['eligible_pmids']!=order or result['dataset_id']!=ranker.dataset.manifest['dataset_id']
                    or result['window_artifact_id']!=ranker.window_manifest['artifact_id'] or result['query']!=query.text
                    or len(records)!=len(windows) or result['scoring']['pair_count']!=len(windows)
                    or result['runtime']['model_id']!=MODEL_ID or result['runtime']['revision']!=REVISION):
                raise ValueError('MedCPT eligibility, source or model differs')
            for window,record in zip(windows,records):
                if (any(record[k]!=window[k] for k in ('pmid','window_id','text_version_id'))
                        or type(record['input_tokens']) is not int or not 3<=record['input_tokens']<=512):
                    raise ValueError('MedCPT score/window mapping differs')
            expected=aggregate_windows(windows,[r['score'] for r in records],order)[:10]
            if len(expected)!=len(result['results']):raise ValueError('MedCPT result count differs')
            for n,((pmid,score,window),hit) in enumerate(zip(expected,result['results']),1):
                if (hit['pmid']!=pmid or hit['score']!=score or hit['rank']!=n
                        or hit['text_version_id']!=window['text_version_id']
                        or set(hit['best_window'])!={'window_id','kind','section_index','section_label','source_path',
                            'char_start','char_end','text','provenance'}
                        or any(window[k]!=value for k,value in hit['best_window'].items())):
                    raise ValueError('MedCPT maximum-window ranking does not replay')
            methods[method]=[h['pmid'] for h in result['results']]
        for m in ('bm25','rrf','dense_qwen','rrf_qwen'):methods[m]=list(old['methods'][m]['ranking'])
        for (qid,pmid),item in item_map.items():
            if qid!=query.query_id:continue
            article=ranker.articles[pmid]
            if (item['dataset_id']!=ranker.dataset.manifest['dataset_id'] or item['question']!=query.text
                    or item['title']!=article['title'] or item['abstract']!=article.get('abstract')
                    or item['text_version_id']!=article['text_version']['text_version_id']
                    or item['provenance']!=article['provenance']):
                raise ValueError('Annotation evidence differs from frozen source')
        rows.append({'query_id':query.query_id,'candidates':candidates,'rankings':methods})
        source_checks.append({'query_id':query.query_id,'dataset_id':ranker.dataset.manifest['dataset_id'],
            'window_artifact_id':ranker.window_manifest['artifact_id'],'candidate_count':len(candidates),
            'new_judgment_count':sum(q==query.query_id for q,p in new_pairs)})
    targets=[(p['query_id'],p['pmid']) for p in profile['conservative_direct_pairs']]
    expected_targets={key for key,r in review_map.items() if r['relevance']==2 and r['confidence']=='medium'}
    if len(set(targets))!=len(targets) or set(targets)!=expected_targets:
        raise ValueError('Conservative sensitivity must include all new medium-confidence direct judgments')
    conservative=copy.deepcopy(qrels)
    for item in profile['conservative_direct_pairs']:
        key=(item['query_id'],item['pmid']);review=review_map.get(key)
        if review is None or review['relevance']!=2 or review['confidence']!='medium':
            raise ValueError('Conservative sensitivity must target new medium-confidence direct judgments')
        conservative[key[0]][key[1]]=1
    return profile,rows,qrels,conservative,source_checks,provenance


def evaluate_profile(profile_path,*,root=ROOT):
    profile,rows,qrels,conservative,sources,provenance=load_comparison(profile_path,root=root)
    result=compare(rows,qrels,profile['primary_methods'],PAIRS)
    result['missing_label_sensitivity']=missing_label_sensitivity(rows,qrels,profile['unresolved_pairs'],profile['primary_methods'],PAIRS)
    alternative=compare(rows,conservative,profile['primary_methods'],PAIRS)
    result['conservative_supplement_sensitivity']={'changed_pairs':profile['conservative_direct_pairs'],
        'policy':'Only new medium-confidence grade-2 judgments become grade 1; all v2 judgments stay fixed.',
        'macro':alternative['macro'],'paired_deltas':alternative['paired_deltas'],
        'common_topk_complete':alternative['common_topk_complete']}
    identity={'profile_sha256':sha_file(profile_path),'input_hashes':{k:profile[k]['sha256'] for k in
        ('base_profile','rankings','qrels','supplement_items','supplement_reviews','provenance')},
        'code_sha256':{name:sha_file(ROOT/name) for name in ('scripts/evaluate_medcpt.py',
            'src/medlens/evaluation/comparison.py','src/medlens/evaluation/metrics.py',
            'scripts/rescore_pilot.py','src/medlens/evaluation/formats.py','src/medlens/reranking/articles.py',
            'src/medlens/reranking/medcpt.py','src/medlens/reranking/listwise.py',
            'src/medlens/retrieval/windows.py','src/medlens/parsing/dataset.py',
            'src/medlens/generation/context.py')}}
    report={'schema_version':1,'run_id':digest(identity),**identity,'evaluation_version':profile['evaluation_version'],
        'annotation':profile['annotation'],'supplement_provenance':provenance,'split':'pilot','k':10,
        'query_count':len(rows),'primary_methods':profile['primary_methods'],'source_checks':sources,
        'heldout_used':False,'inference_run':False,'recall_scope':'Known relevant judgments in the expanded pooled candidate set, not PubMed-wide recall.',
        **result}
    local={**report,'input_rankings':rows}
    return local,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=ROOT/'benchmarks/pilot-v1/medcpt-evaluation.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.summary.exists() or args.output.resolve()==args.summary.resolve():
        parser.error('Choose two distinct unused paths')
    local,summary=evaluate_profile(args.profile)
    for path,report in ((args.output,local),(args.summary,summary)):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'comparison_status':summary['comparison_status'],
        'primary_macro':{m:summary['macro'][m] for m in summary['primary_methods']}},indent=2))


if __name__=='__main__':main()
