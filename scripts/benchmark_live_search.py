"""Measure live PubMed request costs; no relevance or answer-quality judgments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
from medlens.evaluation.formats import parse_queries
from medlens.inference.backends import add_runtime_arguments
from medlens.generation.context import source_section,locate_quote
from medlens.ingestion.snapshots import load_snapshot
from medlens.retrieval.dense import DenseIndex


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def audit(directory):
    r=json.loads((directory/'run.json').read_bytes())
    if r['status']!='complete' or not r['live']['fresh_search_complete']:
        raise ValueError('Incomplete live request: '+str(directory))
    snap=load_snapshot(directory/'snapshot')
    if snap.manifest['snapshot_id']!=r['live']['snapshot_id']:raise ValueError('Snapshot binding differs')
    index=DenseIndex(directory/'dataset',directory/'windows',directory/'embeddings')
    if index.dataset.manifest['dataset_id']!=r['live']['dataset_id']:raise ValueError('Dataset binding differs')
    for w in index.windows:source_section(index.articles[w['pmid']],w)
    windows={w['window_id']:w for w in index.windows}
    for s in r['context']['sources']:
        w=windows[s['window_id']]
        if any(s[k]!=w[k] for k in ('pmid','text','provenance','char_start','char_end','text_version_id','source_path')):
            raise ValueError('Evidence mapping differs')
    visible=0
    for c in r['answer'].get('claims',[]):
        for citation in c['citations']:
            location=locate_quote(r['context'],citation['citation_id'],citation['quote'])
            if any(citation[k]!=location[k] for k in location):raise ValueError('Visible citation differs')
            visible+=1
    return r,{'window_spans_checked':len(index.windows),'selected_sources_checked':len(r['context']['sources']),
        'visible_citations_checked':visible,'vector_rows':len(index.vectors),'vector_bytes':int(index.vectors.nbytes)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    add_runtime_arguments(parser)
    args=parser.parse_args()
    if args.summary.exists():raise FileExistsError(args.summary)
    args.output.mkdir(parents=True,exist_ok=False)
    queries_path=ROOT/'benchmarks/pilot-v1/queries.jsonl'
    queries=parse_queries(queries_path.read_bytes(),queries_path.parent)
    if len(queries)!=10 or any(q.split!='pilot' for q in queries):raise ValueError('Require fixed development queries')
    chosen=[q for q in queries if q.query_id in {'p03','p06'}]
    code_files=[ROOT/'scripts/ask_live.py',Path(__file__),*[ROOT/'src/medlens'/n for n in
        ('live.py','pubmed_query.py','pipeline.py','reporting.py','ingestion/pubmed.py','ingestion/snapshots.py',
         'parsing/dataset.py','parsing/pubmed.py','retrieval/windows.py','retrieval/embeddings.py','retrieval/vector_cache.py',
         'generation/answers.py','generation/reviews.py','inference/qwen.py','inference/backends.py',
         'inference/mlx.py','inference/mlx_worker.py')]]
    code={str(p.relative_to(ROOT)):sha(p) for p in code_files}
    report={'schema_version':1,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
        'scope':'Two fixed depression/anxiety development questions; 30/100/300 fresh candidates, one cold and one warm embedding-cache request per combination. Runtime, integrity and pipeline status only; no relevance or answer-quality scoring.',
        'heldout_used':False,'qrels_used':False,'quality_evaluation_performed':False,'queries_sha256':sha(queries_path),
        'engine':args.engine,'query_ids':[q.query_id for q in chosen],'platform':platform.platform(),'machine':platform.machine(),
        'cache_scope':'Cold starts with an absent vector-cache directory; warm reuses it but searches and downloads PubMed anew. Model/OS caches are not flushed. Each request uses a new process and freshly generates/reviews drafts.',
        'timing_scope':'CLI request time excludes interpreter/import startup and final checkpoint write. Command wall time includes process startup and exit. Component times are nested, not additive.',
        'memory_scope':'Each child reports process RSS high-water, sampled MPS allocations and Qwen conversion samples separately. They overlap; do not sum them. No instantaneous device peak is claimed.',
        'code_sha256':code,'runs':[]}
    engine_args=['--engine',args.engine]
    if args.mlx_model_dir is not None:engine_args+=['--mlx-model-dir',str(args.mlx_model_dir)]
    if args.mlx_python is not None:engine_args+=['--mlx-python',str(args.mlx_python)]
    write(args.output/'summary.json',report)
    try:
        for q in chosen:
            for limit in (30,100,300):
                cache=ROOT/'.cache/embeddings'/('live-benchmark-'+args.output.name)/q.query_id/str(limit)
                if cache.exists():raise FileExistsError('Cold cache already exists: '+str(cache))
                for state in ('cold','warm'):
                    name=f'{q.query_id}-{limit}-{state}';directory=args.output/name
                    print('START',name,flush=True)
                    tick=perf_counter()
                    with (args.output/(name+'.log')).open('w') as log:
                        result=subprocess.run([sys.executable,str(ROOT/'scripts/ask_live.py'),'--query',q.text,
                            '--limit',str(limit),'--cache',str(cache),'--output-dir',str(directory),*engine_args],
                            stdout=log,stderr=subprocess.STDOUT)
                    wall=perf_counter()-tick
                    if result.returncode not in (0,2):raise RuntimeError('Live request failed: '+name)
                    r,checks=audit(directory)
                    calls=r['answer'].get('model_calls',[])
                    if any(c.get('cache_reused') for c in calls):raise ValueError('Generation must be fresh')
                    row={'query_id':q.query_id,'limit':limit,'cache_state':state,'command_wall_seconds':wall,
                        'cli_request_seconds':r['request_seconds'],'snapshot_id':r['live']['snapshot_id'],
                        'dataset_id':r['live']['dataset_id'],'window_artifact_id':r['live']['window_artifact_id'],
                        'embedding_artifact_id':r['live']['embedding_artifact_id'],
                        'submitted_query':r['live']['query_plan']['submitted_query'],
                        'query_translation':r['live']['query_translation'],
                        'total_pubmed_matches':r['live']['total_pubmed_matches'],
                        'fetched_articles':r['live']['fetched_article_count'],'request_attempts':r['live']['request_attempts'],
                        'preparation_timings':r['preparation_timings'],'timings':r['timings'],
                        'passage_cache':r['live']['passage_cache'],'runtime_load_count':r['runtime_load_count'],
                        'answer_status':r['answer']['status'],'validation_error':r['answer'].get('validation_error'),
                        'visible_claim_count':len(r['answer'].get('claims',[])),
                        'model_calls':[{'stage':c['stage'],'seconds':c['seconds'],'new_tokens':c['new_tokens'],
                            'hit_output_limit':c['hit_output_limit'],'input_tokens':c['prompt']['input_tokens']} for c in calls],
                        'runtime':r.get('runtime'),'runtime_closed':r.get('runtime_closed'),'memory_samples':r['memory_samples'],'integrity_checks':checks,
                        'local_run_sha256':sha(directory/'run.json')}
                    if state=='cold' and row['passage_cache']['cache_hit_inputs']!=0:
                        raise ValueError('Unexpected cold vector-cache hit')
                    report['runs'].append(row);write(args.output/'summary.json',report)
                    print('DONE',name,round(wall,2),row['answer_status'],flush=True)
        if code!={str(p.relative_to(ROOT)):sha(p) for p in code_files}:raise ValueError('Measured code changed during run')
        report.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        write(args.output/'summary.json',report);raise
    write(args.output/'summary.json',report)
    args.summary.parent.mkdir(parents=True,exist_ok=True);write(args.summary,report)


if __name__=='__main__':main()
