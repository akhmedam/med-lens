"""Measure pinned MedCPT on development windows without evaluating relevance."""

import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
from medlens.generation.context import source_section
from medlens.parsing.dataset import load_dataset
from medlens.reranking.medcpt import (MedCPTScorer,PairBudgetError,download_assets,
    load_tokenizer,prepare_pair,sha_file)
from medlens.retrieval.windows import load_windows


def development_pairs(tokenizer):
    query_path=ROOT/'benchmarks/pilot-v1/queries.jsonl'
    queries=[json.loads(line) for line in query_path.read_text().splitlines()]
    pairs,entries,artifacts=[],[],[]
    for query in queries:
        if query['split']!='pilot':raise ValueError('Probe only accepts pilot development questions')
        dataset=load_dataset(query_path.parent/query['dataset'])
        manifest,windows=load_windows(ROOT/'data/windows/pilot-v1-002'/query['query_id'])
        if (manifest['dataset_id']!=dataset.manifest['dataset_id']
                or manifest['articles_sha256']!=dataset.manifest['articles_sha256']):
            raise ValueError('Development windows/dataset mismatch')
        articles={a['pmid']:a for a in dataset.articles}
        artifacts.append({'query_id':query['query_id'],'dataset_id':dataset.manifest['dataset_id'],
            'window_artifact_id':manifest['artifact_id']})
        for window in windows:
            source_section(articles[window['pmid']],window)
            triple=(query['text'],window['title'],window['text'])
            row={'query_id':query['query_id'],'window_id':window['window_id'],'pmid':window['pmid']}
            try:
                encoded=prepare_pair(*triple,tokenizer)
            except PairBudgetError as exc:
                # Recount the original pair for the audit; never feed it to the model.
                length=len(tokenizer(triple[0],triple[1]+'\n'+triple[2],
                    truncation=False,verbose=False)['input_ids'])
                entries.append({**row,'input_tokens':length,'status':'over-budget','error':str(exc)})
            else:
                length=len(encoded['input_ids'])
                entries.append({**row,'input_tokens':length,'status':'fits'})
                pairs.append({**row,'input_tokens':length,'pair':triple})
    return pairs,entries,{'queries_sha256':sha_file(query_path),'artifacts':artifacts}


def max_difference(left,right):
    if len(left)!=len(right):raise ValueError('Score counts differ')
    return max((abs(a-b) for a,b in zip(left,right)),default=0.0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,default=ROOT/'.cache/models/medcpt-cross-encoder')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--devices',nargs='+',choices=('cpu','mps'),default=['cpu','mps'])
    parser.add_argument('--batch-sizes',nargs='+',type=int,default=[1,4,8,16])
    parser.add_argument('--pairs',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,required=True)
    args=parser.parse_args()
    if (args.output.exists() or args.summary.exists() or args.output.resolve()==args.summary.resolve()
            or args.pairs<2 or args.repeats<1 or any(n<=0 for n in args.batch_sizes)
            or 1 not in args.batch_sizes or len(set(args.devices))!=len(args.devices)
            or len(set(args.batch_sizes))!=len(args.batch_sizes)):
        parser.error('Choose fresh paths, unique devices/batches, positive sizes and include batch size one')
    if args.download:download_assets(args.assets)
    tokenizer,token_info=load_tokenizer(args.assets)
    all_pairs,audit,identity=development_pairs(tokenizer)
    ordered=sorted(all_pairs,key=lambda row:(row['input_tokens'],row['query_id'],row['window_id']))
    count=min(args.pairs,len(ordered))
    if count<2:raise ValueError('Not enough compatible development pairs')
    selected=[ordered[round(i*(len(ordered)-1)/(count-1))] for i in range(count)]
    pairs=[row['pair'] for row in selected]
    base=len(prepare_pair('question','title','',tokenizer)['input_ids'])
    boundary=('question','title','word '*(512-base))
    overflow=('question','title','word '*(513-base))
    assert len(prepare_pair(*boundary,tokenizer)['input_ids'])==512
    try:prepare_pair(*overflow,tokenizer)
    except PairBudgetError:overflow_rejected=True
    else:raise ValueError('513-token pair was accepted')
    report={'schema_version':1,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
        **identity,'tokenizer':token_info,'code_sha256':{name:sha_file(ROOT/name) for name in
            ('scripts/probe_medcpt.py','src/medlens/reranking/medcpt.py')},
        'audit':audit,'selected_pairs':selected,'selection':'length-quantile-sample-ascending-length',
        'boundary_checks':{'accepted_tokens':512,'rejected_tokens':513,'overflow_rejected':overflow_rejected},
        'devices':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    def checkpoint():
        temporary=args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        temporary.replace(args.output)
    reference=None
    try:
        for device in args.devices:
            scorer=MedCPTScorer(args.assets,device=device)
            device_report={'metadata':scorer.metadata,'batches':[],
                'boundary_score':scorer.score([boundary])['scores'][0]}
            baseline=None
            for batch_size in sorted(args.batch_sizes):
                warmup=scorer.score(pairs,batch_size=batch_size)
                runs=[scorer.score(pairs,batch_size=batch_size) for _ in range(args.repeats)]
                if batch_size==1:baseline=runs[0]['scores']
                differences=[max_difference(baseline,r['scores']) for r in runs]
                row={'batch_size':batch_size,'warmup':warmup,'runs':runs,
                    'median_forward_seconds':statistics.median(r['forward_seconds'] for r in runs),
                    'median_score_seconds':statistics.median(r['score_seconds'] for r in runs),
                    'max_abs_difference_from_batch1':max(differences)}
                device_report['batches'].append(row)
                print(device,batch_size,round(row['median_score_seconds'],4),flush=True)
            device_report['batch_consistency_atol_1e-4']=all(
                b['max_abs_difference_from_batch1']<=1e-4 for b in device_report['batches'])
            if reference is None:
                reference=(device,baseline)
            else:
                drift=max_difference(reference[1],baseline)
                device_report['comparison_with_first_device']={'reference':reference[0],
                    'max_abs_difference':drift,'within_atol_1e-3':drift<=1e-3}
            report['devices'].append(device_report)
            checkpoint()
            del scorer
            gc.collect()
            if device=='mps':
                import torch
                torch.mps.empty_cache()
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
    summary={key:report[key] for key in ('schema_version','status','started_at','finished_at',
        'queries_sha256','artifacts','tokenizer','code_sha256','selection','boundary_checks','process_peak_rss_bytes')}
    summary.update(local_report_sha256=sha_file(args.output),audited_pair_count=len(audit),
        over_budget_count=sum(r['status']=='over-budget' for r in audit),
        original_pair_token_range=[min(r['input_tokens'] for r in audit),max(r['input_tokens'] for r in audit)],
        selected_pair_count=count,selected_pair_token_range=[selected[0]['input_tokens'],selected[-1]['input_tokens']],
        repeats=args.repeats,devices=[])
    for item in report['devices']:
        device_summary={key:value for key,value in item.items() if key not in {'batches','boundary_score'}}
        device_summary['batches']=[{key:value for key,value in batch.items() if key not in {'warmup','runs'}}
                                  | {'sampled_max_mps_tensor_bytes':max(r['sampled_max_mps_tensor_bytes'] or 0 for r in batch['runs']) or None,
                                     'sampled_max_mps_driver_bytes':max(r['sampled_max_mps_driver_bytes'] or 0 for r in batch['runs']) or None}
                                  for batch in item['batches']]
        summary['devices'].append(device_summary)
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    with args.summary.open('x') as stream:json.dump(summary,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
