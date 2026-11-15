"""Prepare fixed search-only confirmation sources; no model answers or human labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(path):
    from medlens.api_contract import SearchRequest
    from medlens.evaluation.biomedical_annotation import strict_json
    path=Path(path).expanduser()
    if path.is_symlink() or not path.is_file():raise ValueError('Require a regular frozen plan file')
    body=path.read_bytes()
    pin=path.with_suffix('.sha256')
    if pin.is_symlink() or not pin.is_file():raise ValueError('Missing regular frozen plan SHA256')
    if pin.read_text().strip()!=hashlib.sha256(body).hexdigest()+'  '+path.name:
        raise ValueError('Frozen plan SHA256 differs; do not alter a prepared confirmation set')
    plan = strict_json(body)
    if (not isinstance(plan,dict) or type(plan.get('schema_version')) is not int or plan.get('schema_version') != 1 or plan.get('kind') != 'medlens-human-confirmation-plan'
            or plan.get('split') != 'test' or plan.get('dataset_id') != 'human-confirmation-v1'
            or type(plan.get('candidate_limit')) is not int or plan.get('candidate_limit') != 100
            or type(plan.get('annotation_depth_per_method')) is not int or plan.get('annotation_depth_per_method') != 10
            or type(plan.get('question_count')) is not int or plan.get('question_count') != 20 or not isinstance(plan.get('cases'), list)
            or len(plan['cases']) != 20):
        raise ValueError('Require the explicit fixed 20-question human-confirmation-v1 plan')
    ids, groups, questions = set(), set(), set()
    for case in plan['cases']:
        if not isinstance(case,dict):raise ValueError('Each case must be an object')
        if any(not isinstance(case.get(key),str) or not case[key].strip() for key in
                ('query_id','group_id','question','relevance_intent','specialty','question_type')):
            raise ValueError('Incomplete confirmation question')
        qid = case['query_id']
        if (re.fullmatch('[A-Za-z0-9_-]{1,64}',qid) is None or qid in ids
                or case['group_id'] in groups or case['question'].casefold() in questions):
            raise ValueError('Duplicate or unsafe confirmation identity')
        SearchRequest(question=case['question'],generate=False)
        ids.add(qid);groups.add(case['group_id']);questions.add(case['question'].casefold())
    return body, plan


def runtime_hashes(root):
    paths = list((root/'src/medlens').glob('*.py'))
    for directory in ('ingestion','parsing','retrieval','generation','inference'):
        paths.extend((root/'src/medlens'/directory).rglob('*.py'))
    paths.append(Path(__file__))
    return {str(path.relative_to(root)):sha(path) for path in sorted(paths)}


def save(path, value):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'benchmarks/human-confirmation-v1/plan.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'.cache/embeddings/bge-input-v1')
    args=parser.parse_args(argv)
    try:
        body,plan=load_plan(args.plan)
        plan_path=args.plan.resolve();output=args.output.expanduser().resolve();cache=args.cache.expanduser().resolve()
        if args.output.is_symlink() or output.exists():raise ValueError('Choose a fresh source directory')
        if output.is_relative_to(cache) or cache.is_relative_to(output):raise ValueError('Source output and cache must be disjoint')
    except (OSError,ValueError) as exc:parser.error(str(exc))
    from medlens.api_backend import make_local_runner
    from medlens.api_contract import SearchRequest
    from medlens.mvp import policy_snapshot
    from medlens.pubmed_query import plan_query
    from medlens.evaluation.biomedical_annotation import canonical, strict_json
    selected=policy_snapshot()
    if (selected['source']['candidate_limit']['default'],selected['retrieval']['top_k']['default'],
        selected['retrieval']['device'],selected['answer']['style'],selected['answer']['model_profile'],
        selected['answer']['engine'])!=(100,10,'mps','source-findings','4b','mlx'):
        parser.error('Application policy differs from the frozen source-preparation protocol')
    hashes=runtime_hashes(ROOT)
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan-original.json').write_bytes(body)
    report={'schema_version':1,'kind':'human-confirmation-source-preparation','split':'test',
        'dataset_id':plan['dataset_id'],'created_at':datetime.now(timezone.utc).isoformat(),
        'status':'running','plan_sha256':hashlib.sha256(body).hexdigest(),
        'application_policy':selected,'acquisition_runtime_sha256':hashes,
        'candidate_limit':100,'top_k':10,'expected_questions':20,'runs':[],
        'answer_generation_performed':False,'human_labels_generated':False,
        'quality_scores_computed':False,'latency_acceptance_evaluated':False,
        'note':'This is reference preparation, not a full-generation latency or quality evaluation. Empty cases are retained.'}
    path=output/'preparation.json';save(path,report)
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').addFilter(lambda row:row.levelno>=logging.ERROR)
    runner=make_local_runner(ROOT,cache)
    try:
        for case in plan['cases']:
            if plan_path.read_bytes()!=body or runtime_hashes(ROOT)!=hashes:raise ValueError('Frozen acquisition plan/runtime changed')
            qid=case['query_id'];print(qid+' start',flush=True)
            started=datetime.now(timezone.utc).isoformat()
            row={'query_id':qid,'started_at':started,'status':'started'}
            report['runs'].append(row);save(path,report)
            request=SearchRequest(question=case['question'],limit=100,top_k=10,generate=False)
            run=runner(request,output/qid)
            run_path=output/qid/'run.json'
            if run_path.is_symlink() or not run_path.is_file():raise ValueError('Missing regular persisted source run')
            persisted=strict_json(run_path.read_bytes())
            if canonical(persisted)!=canonical(run):raise ValueError('Returned and persisted source runs differ')
            row.update({'status':run.get('status'),
                'run_sha256':sha(run_path) if run_path.is_file() else None,
                'fetched_articles':(run.get('live') or {}).get('fetched_article_count'),
                'ranked_articles':len((run.get('retrieval') or {}).get('results',[])),
                'snapshot_id':(run.get('live') or {}).get('snapshot_id'),
                'dataset_id':(run.get('live') or {}).get('dataset_id'),
                'source_only_seconds':run.get('request_seconds'),'runtime_load_count':run.get('runtime_load_count')})
            save(path,report)
            fetched=(run.get('live') or {}).get('fetched_article_count')
            empty=type(fetched) is int and fetched==0
            actual_top=run.get('top_k',10 if empty else None)
            if (run.get('status')!='complete' or run.get('question')!=case['question']
                    or (run.get('live') or {}).get('fresh_search_complete') is not True
                    or type((run.get('live') or {}).get('candidate_limit')) is not int
                    or (run.get('live') or {}).get('candidate_limit')!=100
                    or type(fetched) is not int or not 0<=fetched<=100
                    or run.get('generate',False if empty else None) is not False
                    or run.get('rerank',False if empty else None) is not False
                    or type(actual_top) is not int or actual_top!=10
                    or (run.get('live') or {}).get('query_plan')!=plan_query(case['question'])
                    or type(run.get('runtime_load_count')) is not int or run['runtime_load_count']!=0
                    or (run.get('answer') or {}).get('status')!='not_requested'
                    or (run.get('answer') or {}).get('model_calls',[])!=[]
                    or (run.get('answer') or {}).get('claims',[])!=[]):
                raise RuntimeError('Incomplete or non-search-only source case: '+qid+'; preserve the attempt')
            print(qid+' complete '+str(row['fetched_articles']),flush=True)
        if plan_path.read_bytes()!=body or runtime_hashes(ROOT)!=hashes:raise ValueError('Frozen acquisition changed')
        report['status']='complete-awaiting-manual-human-annotation'
        return 0
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            error={'type':type(exc).__name__,'message':str(exc)})
        return 130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat();save(path,report)


if __name__=='__main__':raise SystemExit(main())
