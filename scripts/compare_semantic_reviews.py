"""Compare review protocols on identical frozen drafts and verify source-bound replay."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'

from medlens.generation.answers import accepted_claim_ids, generate_answer, parse_draft, parse_review
from medlens.generation.contracts import LEGACY_REVIEW_VERSION, REVIEW_VERSION
from medlens.generation.quotes import quote_catalog, resolve_quote
from medlens.generation.request_contexts import load_request_contexts
from medlens.inference.qwen import MODEL_ID,REVISION
from generate_answers import control_context


def verify_frozen_drafts(before,after):
    for r in (before,after):
        if (r.get('status')!='complete' or r.get('heldout_used') is not False
                or r.get('resource_scope')!='rebuilt-pilot' or r.get('relevance_judgments_bound') is not False):
            raise ValueError('Require complete rebuilt development reports')
        if r.get('case_ids')!=[x['case_id'] for x in r['per_case']] or len(set(r['case_ids']))!=len(r['case_ids']):
            raise ValueError('Case identity is missing or duplicated')
    for key in ('context_report_sha256','profile_sha256','case_ids','prompt_version'):
        if not before.get(key) or before[key]!=after.get(key):raise ValueError('Frozen identity differs: '+key)
    if before.get('qrels_sha256') is not None or after.get('qrels_sha256') is not None:
        raise ValueError('Synthetic/rebuilt review comparison cannot bind historical judgments')
    for key in ('model_id','revision','device','quantization','packages'):
        if not before.get('runtime',{}).get(key) or before['runtime'][key]!=after.get('runtime',{}).get(key):
            raise ValueError('Runtime identity differs: '+key)
    for report in (before,after):
        for row in report['per_case']:
            stages=[c['stage'] for c in row['model_calls']]
            if not set(stages)<={'draft','support-review'} or len(stages)!=len(set(stages)):
                raise ValueError('Unexpected or duplicate model stage')
    for a,b in zip(before['per_case'],after['per_case']):
        for key in ('case_id','kind','question','context_id','expected'):
            if a.get(key)!=b.get(key):raise ValueError('Case differs: '+key)
        old=[c for c in a['model_calls'] if c['stage']=='draft']
        new=[c for c in b['model_calls'] if c['stage']=='draft']
        if len(old)!=len(new) or len(old)>1:raise ValueError('Draft call membership differs')
        for x,y in zip(old,new):
            if (x['text']!=y['text'] or x['max_new_tokens']!=y['max_new_tokens']
                    or x['prompt']!=y['prompt'] or x['hit_output_limit']!=y['hit_output_limit']
                    or y.get('cache_reused') is not True):
                raise ValueError('Draft changed or was regenerated')
        for x in a['model_calls']+b['model_calls']:
            if x['stage']=='support-review' and x['max_new_tokens']!=512:
                raise ValueError('Review budget changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    p.add_argument('--requests',type=Path,required=True);p.add_argument('--collections',type=Path,required=True)
    p.add_argument('--inspection',type=Path,default=ROOT/'docs/experiments/day21-source-inspection.json')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    if args.output.exists():p.error('Choose a fresh output path')
    old_bytes,new_bytes=args.baseline.read_bytes(),args.candidate.read_bytes()
    old,new=json.loads(old_bytes),json.loads(new_bytes);verify_frozen_drafts(old,new)
    if old['review_version']!=LEGACY_REVIEW_VERSION or new['review_version']!=REVIEW_VERSION:
        raise ValueError('Require the explicit legacy and semantic protocols')
    inspection_bytes=args.inspection.read_bytes();inspection=json.loads(inspection_bytes)
    if inspection['baseline_report_sha256']!=hashlib.sha256(old_bytes).hexdigest():
        raise ValueError('Inspection belongs to another baseline')
    if new['cache_source_sha256']!=hashlib.sha256(old_bytes).hexdigest():
        raise ValueError('Candidate draft-cache identity differs')
    for name,checksum in new['code_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=checksum:raise ValueError('Candidate code changed: '+name)
    artifact,checksum=load_request_contexts(ROOT,args.requests,args.collections)
    if checksum!=old['context_report_sha256']:raise ValueError('Saved requests differ')
    contexts={row['query_id']:row['context'] for row in artifact['per_query']}
    from transformers import AutoTokenizer
    t=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,token=False)
    contexts['empty-context']=control_context(artifact['per_query'][0]['context'],[],t)
    owner=next(r['context'] for r in artifact['per_query'] if any(s['pmid']=='17003660' for s in r['context']['sources']))
    source=next(s for s in owner['sources'] if s['pmid']=='17003660')
    contexts['wrong-population']=control_context(owner,[source],t)
    def refuse(*a,**k):raise AssertionError('Unexpected new inference')
    runtime=SimpleNamespace(tokenizer=t,generate=refuse)
    cases=[];citations=catalog_entries=replayed_calls=0
    for before,after in zip(old['per_case'],new['per_case']):
        ctx=contexts[before['case_id']];indexed=quote_catalog(ctx,t)
        for qid in indexed['catalog']:resolve_quote(ctx,indexed['catalog'],qid);catalog_entries+=1
        draft_calls=[c for c in before['model_calls'] if c['stage']=='draft']
        if draft_calls and not draft_calls[0]['hit_output_limit']:
            draft=parse_draft(draft_calls[0]['text'],ctx,catalog=indexed['catalog'],citation_format='strings')
            if draft['claims']:
                call=next(c for c in before['model_calls'] if c['stage']=='support-review')
                if call['hit_output_limit']:raise ValueError('Baseline review was truncated')
                legacy=parse_review(call['text'],{c['claim_id'] for c in draft['claims']},protocol=LEGACY_REVIEW_VERSION)
                accepted=accepted_claim_ids(legacy,protocol=LEGACY_REVIEW_VERSION)
                if before['review']!=legacy or before['claims']!=[c for c in draft['claims'] if c['claim_id'] in accepted]:
                    raise ValueError('Baseline legacy gates do not reproduce')
        replay=generate_answer(after['question'],ctx,runtime,cached_calls={c['stage']:c for c in after['model_calls']})
        for key in ('status','claims','review','limitations','reason','context_id'):
            if replay[key]!=after[key]:raise ValueError('Candidate replay differs: '+key)
        if not all(c['cache_reused'] for c in replay['model_calls']):raise ValueError('Candidate call did not replay')
        if len(replay['model_calls'])!=len(after['model_calls']) or any(
                a['prompt']!=b['prompt'] for a,b in zip(replay['model_calls'],after['model_calls'])):
            raise ValueError('Recorded candidate prompts do not reproduce')
        replayed_calls+=len(replay['model_calls'])
        for claim in after['claims']:
            for cite in claim['citations']:
                if resolve_quote(ctx,indexed['catalog'],cite['quote_id'])!=cite:raise ValueError('Citation changed')
                citations+=1
        a,b={c['claim_id'] for c in before['claims']},{c['claim_id'] for c in after['claims']}
        cases.append({'case_id':before['case_id'],'kind':before['kind'],'baseline_status':before['status'],
            'candidate_status':after['status'],'draft_claim_count':len(before.get('draft',{}).get('claims',[])),
            'baseline_visible_claims':sorted(a),'candidate_visible_claims':sorted(b),
            'newly_withheld':sorted(a-b),'newly_accepted':sorted(b-a),'retained':sorted(a&b),
            'candidate_validation_error':after.get('validation_error'),
            'candidate_review':after['review'],
            'fresh_review_seconds':sum(c['seconds'] for c in after['model_calls'] if c['stage']=='support-review' and not c.get('cache_reused'))})
    by_id={r['case_id']:r for r in cases};inspection_outcomes=[]
    for note in inspection['per_claim']:
        row=by_id[note['case_id']]
        inspection_outcomes.append({**note,'candidate_visible':note['claim_id'] in row['candidate_visible_claims'],
            'candidate_case_status':row['candidate_status']})
    result={'schema_version':1,'status':'complete','model_inference':False,'heldout_used':False,
        'baseline_sha256':hashlib.sha256(old_bytes).hexdigest(),'candidate_sha256':hashlib.sha256(new_bytes).hexdigest(),
        'inspection_sha256':hashlib.sha256(inspection_bytes).hexdigest(),'context_manifest_sha256':checksum,
        'paired_frozen_drafts':True,'clinical_accuracy_measured':False,'cases':cases,'inspection_outcomes':inspection_outcomes,
        'counts':{key:sum(len(r[key]) for r in cases if r['kind']=='pilot') for key in
            ('baseline_visible_claims','candidate_visible_claims','newly_withheld','newly_accepted','retained')},
        'candidate_pilot_statuses':dict(Counter(r['candidate_status'] for r in cases if r['kind']=='pilot')),
        'source_checks':{'cases_replayed':len(cases),'cached_calls_replayed':replayed_calls,
            'catalog_entries_including_controls':catalog_entries,'displayed_citations_resolved':citations,
            'source_windows_and_metadata_verified':True,'candidate_code_hashes_match':True},
        'interpretation':'Fixed drafts isolate the bundled review-protocol change (axes, instructions and evidence visibility), not individual mechanisms. AI source inspection is not independent clinical gold; withholding is not proof that a claim is false.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'counts':result['counts'],'candidate_pilot_statuses':result['candidate_pilot_statuses'],'source_checks':result['source_checks']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
