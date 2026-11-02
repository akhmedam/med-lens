"""Save both MedCPT article rankings on pilot snapshots, without relevance metrics."""

import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
from medlens.evaluation.formats import parse_queries
from medlens.generation.context import digest
from medlens.reranking.articles import MedCPTRanker
from medlens.reranking.medcpt import MedCPTScorer,sha_file
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries',type=Path,default=ROOT/'benchmarks/pilot-v1/queries.jsonl')
    parser.add_argument('--windows-root',type=Path,default=ROOT/'data/windows/pilot-v1-002')
    parser.add_argument('--embeddings-root',type=Path,default=ROOT/'data/embeddings/pilot-v1')
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache/models/medcpt-cross-encoder')
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args()
    if (args.output.exists() or args.summary.exists() or args.output.resolve()==args.summary.resolve()
            or args.batch_size<=0):parser.error('Choose fresh paths and a positive batch size')
    queries=parse_queries(args.queries.read_bytes(),args.queries.parent)
    if not queries or any(q.split!='pilot' for q in queries):
        parser.error('Only the development pilot is permitted; do not run the held-out split')
    identity={'queries_sha256':sha_file(args.queries),'device':args.device,'batch_size':args.batch_size,
        'dense_depth':10,'top_k':10,'code_sha256':{name:sha_file(ROOT/name) for name in (
            'scripts/prepare_medcpt_rankings.py','src/medlens/reranking/articles.py',
            'src/medlens/reranking/medcpt.py','src/medlens/retrieval/dense.py',
            'src/medlens/retrieval/embeddings.py')}}
    report={'schema_version':1,'status':'running','run_id':None,**identity,
        'started_at':datetime.now(timezone.utc).isoformat(),'per_query':[],
        'relevance_metrics_evaluated':False,'heldout_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    def checkpoint():
        temporary=args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.output)
    try:
        tick=perf_counter()
        rankers={q.query_id:MedCPTRanker(q.dataset,args.windows_root/q.query_id) for q in queries}
        encoder=BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',ROOT/'.cache/tokenizers/bge-base-en-v1.5',device=args.device)
        report['bge_load_seconds']=encoder.load_seconds
        dense={}
        try:
            for query in queries:
                index=DenseIndex(query.dataset,args.windows_root/query.query_id,args.embeddings_root/query.query_id)
                dense[query.query_id]=index.run(query.text,encoder,top_k=10)
                ranker=rankers[query.query_id]
                if (dense[query.query_id]['dataset_id']!=ranker.dataset.manifest['dataset_id']
                        or dense[query.query_id]['window_artifact_id']!=ranker.window_manifest['artifact_id']):
                    raise ValueError('Dense and MedCPT source identities differ')
        finally:
            del encoder
            gc.collect()
            if args.device=='mps':
                import torch
                torch.mps.empty_cache()
        report['dense_phase_seconds']=perf_counter()-tick
        scorer=MedCPTScorer(args.assets,device=args.device,batch_size=args.batch_size)
        report['medcpt_runtime']=scorer.metadata
        for query in queries:
            ranker=rankers[query.query_id]
            selection=dense[query.query_id]
            eligible=[hit['pmid'] for hit in selection['results']]
            row={'query_id':query.query_id,'query':query.text,'dense':selection,'methods':{}}
            report['per_query'].append(row)
            for method,allowed in (('dense_medcpt',eligible),('medcpt',None)):
                result=ranker.run(query.text,scorer,eligible_pmids=allowed)
                row['methods'][method]=result
                checkpoint()
                print(query.query_id,method,result['eligible_count'],result['scoring']['pair_count'],
                    round(result['ranking_seconds'],4),flush=True)
            complete={w['window_id']:w['score'] for w in row['methods']['medcpt']['window_scores']}
            drift=max((abs(w['score']-complete[w['window_id']])
                for w in row['methods']['dense_medcpt']['window_scores']),default=0.0)
            row['shared_window_score_check']={'max_abs_difference':drift,'within_atol_1e-4':drift<=1e-4}
            if drift>1e-4:raise ValueError('Shared-window scores differ across modes beyond tolerance')
            checkpoint()
        # Bind the complete artifact to every dataset/window/vector/model configuration.
        report['input_run_ids']={'dense':[row['dense']['run_id'] for row in report['per_query']],
            'medcpt':[{method:result['run_id'] for method,result in row['methods'].items()}
                      for row in report['per_query']]}
        report['run_id']=digest({**identity,'input_run_ids':report['input_run_ids']})
        report['status']='complete'
    except (Exception,KeyboardInterrupt) as exc:
        report['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        report['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['process_peak_rss_bytes']=peak if sys.platform=='darwin' else peak*1024
        checkpoint()
    summary={key:value for key,value in report.items() if key!='per_query'}
    summary['local_report_sha256']=sha_file(args.output)
    summary['per_query']=[{'query_id':row['query_id'],
        'dataset_id':row['dense']['dataset_id'],'window_artifact_id':row['dense']['window_artifact_id'],
        'candidate_count':row['dense']['candidate_count'],
        'dense_pmids':[h['pmid'] for h in row['dense']['results']],
        'shared_window_score_check':row['shared_window_score_check'],
        'methods':{method:{'run_id':result['run_id'],'eligible_count':result['eligible_count'],
            'ranked_article_count':result['ranked_article_count'],'pair_count':result['scoring']['pair_count'],
            'pairs_sha256':result['scoring'].get('pairs_sha256'),
            'ranking_seconds':result['ranking_seconds'],'score_seconds':result['scoring']['score_seconds'],
            'ranked_pmids':[h['pmid'] for h in result['results']],
            'best_window_ids':[h['best_window']['window_id'] for h in result['results']]}
            for method,result in row['methods'].items()}} for row in report['per_query']]
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    with args.summary.open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
