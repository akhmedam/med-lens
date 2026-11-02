"""Rank local candidates with direct MedCPT or Dense top-10 followed by MedCPT."""

import argparse
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

from medlens.reranking.articles import MedCPTRanker
from medlens.reranking.medcpt import MedCPTScorer

ROOT=Path(__file__).resolve().parents[1]


def dense_selection(dataset,windows,embeddings,question,model,tokenizer,device):
    from medlens.retrieval.dense import DenseIndex
    from medlens.retrieval.embeddings import BGEEncoder
    index=DenseIndex(dataset,windows,embeddings)
    encoder=BGEEncoder(model,tokenizer,device=device)
    try:
        return index.run(question,encoder,top_k=10)
    finally:
        del encoder
        gc.collect()
        if device=='mps':
            import torch
            torch.mps.empty_cache()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--windows',type=Path,required=True)
    parser.add_argument('--mode',choices=('dense-medcpt','medcpt'),required=True)
    parser.add_argument('--embeddings',type=Path)
    parser.add_argument('--query',required=True)
    parser.add_argument('--top-k',type=int,default=10)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache/models/medcpt-cross-encoder')
    parser.add_argument('--bge-model',type=Path,default=ROOT/'.cache/models/bge-base-en-v1.5')
    parser.add_argument('--bge-tokenizer',type=Path,default=ROOT/'.cache/tokenizers/bge-base-en-v1.5')
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if (not args.query.strip() or not 1<=args.top_k<=10 or args.batch_size<=0
            or (args.mode=='dense-medcpt' and args.embeddings is None)
            or (args.mode=='medcpt' and args.embeddings is not None)):
        parser.error('Supply a question, valid sizes, and embeddings only for dense-medcpt mode')
    if args.output.exists():parser.error('Output exists; choose a fresh path')
    started=perf_counter()
    report={'schema_version':1,'status':'running','stage':'setup','mode':args.mode,'query':args.query,
        'started_at':datetime.now(timezone.utc).isoformat(),'dense_selection':None,
        'input_paths':{key:str(getattr(args,key).resolve()) for key in ('dataset','windows')},
        'timings':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    def checkpoint(stage):
        report['stage']=stage
        temporary=args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.output)
        print(stage,report['status'],flush=True)
    try:
        ranker=MedCPTRanker(args.dataset,args.windows)
        eligible=None
        tick=perf_counter()
        if args.mode=='dense-medcpt':
            checkpoint('dense-selection')
            selection=dense_selection(args.dataset,args.windows,args.embeddings,args.query,
                args.bge_model,args.bge_tokenizer,args.device)
            if (selection['dataset_id']!=ranker.dataset.manifest['dataset_id']
                    or selection['window_artifact_id']!=ranker.window_manifest['artifact_id']):
                raise ValueError('Dense selection and MedCPT sources differ')
            report['dense_selection']=selection
            report['input_paths']['embeddings']=str(args.embeddings.resolve())
            eligible=[hit['pmid'] for hit in selection['results']]
            checkpoint('dense-complete')
        report['timings']['dense_selection_seconds']=perf_counter()-tick
        checkpoint('medcpt-scoring')
        order=list(ranker.articles) if eligible is None else eligible
        scorer=None
        if ranker.eligible_windows(order):
            scorer=MedCPTScorer(args.assets,device=args.device,batch_size=args.batch_size)
        report['ranking']=ranker.run(args.query,scorer,eligible_pmids=eligible,top_k=args.top_k)
        report['status']='complete'
    except (Exception,KeyboardInterrupt) as exc:
        report['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        report['error']={'type':type(exc).__name__,'message':str(exc),'stage':report['stage']}
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        report['timings']['request_seconds']=perf_counter()-started
        checkpoint('complete' if report['status']=='complete' else report['stage'])
    if report['status']!='complete':
        print(report['error']['message'],file=sys.stderr)
        return 130 if report['status']=='interrupted' else 1
    for hit in report['ranking']['results']:
        print(f"{hit['rank']:2}. {hit['pmid']}  {hit['score']:.4f}  {hit['title'] or 'Untitled article'}")
    print(args.output.resolve())
    return 0


if __name__=='__main__':raise SystemExit(main())
