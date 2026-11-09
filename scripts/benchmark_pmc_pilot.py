"""Compare paired abstract/full-text PMC indexes on fixed development questions."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'

from medlens.dates import publication_interval,select_publication_dates
from medlens.evaluation.formats import parse_queries
from medlens.generation.context import build_context,source_section
from medlens.inference.qwen import REVISION as QWEN_REVISION
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder
from medlens.pipeline import tokenizer_identity
from medlens.resources import load_collection_profile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def timed(fn):
    t=perf_counter();v=fn();return v,perf_counter()-t
def dist(values):
    return {'min':min(values),'median':statistics.median(values),'max':max(values),'count':len(values)}
def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform=='darwin' else value*1024

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    import argparse
    import numpy as np
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<3:parser.error('Use at least three paired repetitions')
    if args.output.exists():raise FileExistsError(args.output)
    snapshot=ROOT/'data/pmc/20261109/snapshot'
    abstract_root=ROOT/'data/pmc/20261109/abstract'
    full_root=ROOT/'data/pmc/20261109/fulltext'
    base=ROOT/'data/collections/shared-20261108/shared'
    profile_path=ROOT/'configs/collections/recovery-20261106.json'
    profile,profile_sha,sources=load_collection_profile(ROOT,profile_path)
    query_path=ROOT/profile['queries']['path'];queries=parse_queries(query_path.read_bytes(),query_path.parent)
    if len(queries)!=10 or any(q.split!='pilot' for q in queries):raise ValueError('Expected the ten fixed development questions')
    snapshot_manifest=json.loads((snapshot/'manifest.json').read_bytes())
    if snapshot_manifest['status']!='complete' or len(snapshot_manifest['selected_pmids'])!=6:
        raise ValueError('Expected the six-record permitted PMC snapshot')
    if any(('file' in r)!=r['selected'] for r in snapshot_manifest['responses']):
        raise ValueError('Rejected PMC records must not retain XML')
    if not all(r.get('license_uri') for r in snapshot_manifest['responses'] if r['selected']):
        raise ValueError('Selected articles need explicit licenses')
    aidx,asetup=timed(lambda:DenseIndex(abstract_root/'dataset',abstract_root/'windows',abstract_root/'embeddings'))
    fidx,fsetup=timed(lambda:DenseIndex(full_root/'dataset',full_root/'windows',full_root/'embeddings'))
    if list(aidx.articles)!=list(fidx.articles) or aidx.dataset.manifest['dataset_id']!=fidx.dataset.manifest['dataset_id']:
        raise ValueError('Paired indexes must share article membership and records')
    if [w for w in aidx.windows if w['kind']=='abstract'] != [w for w in fidx.windows if w['kind']=='abstract']:
        raise ValueError('Full-text build must preserve abstract windows byte-for-byte')
    for w in fidx.windows:source_section(fidx.articles[w['pmid']],w)
    for row in snapshot_manifest['responses']:
        if row['selected']:
            assert row['license_uri'] in {'http://creativecommons.org/licenses/by/2.0',
                'http://creativecommons.org/licenses/by/3.0/','http://creativecommons.org/licenses/by/4.0/'}
    model,constructor=timed(lambda:BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',
        ROOT/'.cache/tokenizers/bge-base-en-v1.5',device='mps',batch_size=16))
    aidx.verify_encoder(model);fidx.verify_encoder(model)
    qtokenizer,tokenizer_load=timed(lambda:AutoTokenizer.from_pretrained(
        ROOT/'.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots'/QWEN_REVISION,
        local_files_only=True,trust_remote_code=False))
    context_tokenizer_info=tokenizer_identity(qtokenizer)
    _,warmup=timed(lambda:model.encode_queries([queries[0].text]))
    rows=[]
    for query in queries:
        qrow={'query_id':query.query_id,'runs':[]}
        for repeat in range(args.repeats):
            vectors,qtime=model.encode_queries([query.text]);vector=vectors[0]
            order=('abstract','fulltext') if repeat%2==0 else ('fulltext','abstract')
            results={}
            for mode in order:
                idx=aidx if mode=='abstract' else fidx
                hits,search_time=timed(lambda:idx.search(vector,top_k=10))
                scores=np.clip(idx.vectors@vector,-1.0,1.0)
                context,context_time=timed(lambda:build_context(idx.dataset,idx.window_manifest,idx.windows,
                    [h['pmid'] for h in hits],scores,qtokenizer,context_tokenizer_info))
                results[mode]={'ranked_pmids':[h['pmid'] for h in hits],
                    'best_window_kinds':[h['best_window']['kind'] for h in hits],
                    'search_seconds':search_time,'context_score_and_pack_seconds':context_time,
                    'context_sources':len(context['sources']),'evidence_tokens':context['evidence_tokens'],
                    'abstract_context_sources':sum(s.get('source_kind','abstract')=='abstract' for s in context['sources']),
                    'fulltext_context_sources':sum(s.get('source_kind')=='fulltext' for s in context['sources']),
                    'source_characters':sum(len(s['text']) for s in context['sources'])}
            aorder=results['abstract']['ranked_pmids'];forder=results['fulltext']['ranked_pmids']
            concordant=discordant=0
            for i in range(len(aorder)):
                for j in range(i+1,len(aorder)):
                    sign=(aorder.index(aorder[i])-aorder.index(aorder[j]))*(forder.index(aorder[i])-forder.index(aorder[j]))
                    concordant+=sign>0;discordant+=sign<0
            qrow['runs'].append({'query_encode_seconds':qtime['encode_seconds'],
                'same_rank_positions':sum(x==y for x,y in zip(aorder,forder)),
                'kendall_tau':(concordant-discordant)/max(1,concordant+discordant),
                'top1_same':aorder[0]==forder[0],'mps':{
                'sampled_tensor_bytes':qtime['sampled_max_mps_tensor_bytes'],
                'sampled_driver_bytes':qtime['sampled_max_mps_driver_bytes']},'modes':results,
                'top6_overlap':len(set(results['abstract']['ranked_pmids'])&set(results['fulltext']['ranked_pmids']))})
        for mode in ('abstract','fulltext'):
            rankings=[r['modes'][mode]['ranked_pmids'] for r in qrow['runs']]
            if any(x!=rankings[0] for x in rankings):raise ValueError('Repeated ranking changed within a mode')
        rows.append(qrow);print('paired',query.query_id,flush=True)
    allruns=[r for q in rows for r in q['runs']]
    aggregate={}
    for mode in ('abstract','fulltext'):
        aggregate[mode]={}
        for field in ('search_seconds','context_score_and_pack_seconds','evidence_tokens','abstract_context_sources','fulltext_context_sources','source_characters'):
            values=[r['modes'][mode][field] for r in allruns]
            if field.endswith('_seconds') or field in {'evidence_tokens','source_characters'}:
                aggregate[mode][field]=dist(values)
            else:aggregate[mode][field]={'median':statistics.median(values),'min':min(values),'max':max(values)}
        aggregate[mode]['index_validation_seconds']=asetup if mode=='abstract' else fsetup
        aggregate[mode]['vector_bytes']=int(aidx.vectors.nbytes if mode=='abstract' else fidx.vectors.nbytes)
    articles=aidx.dataset.articles
    intervals=[publication_interval(a) for a in articles]
    known=[v for v in intervals if v]
    if not known:raise ValueError('No verified journal issue dates')
    cutoff=sorted(v[0] for v in known)[len(known)//2].isoformat()
    eligible,selection=select_publication_dates(articles,published_from=cutoff)
    # Inclusive boundaries: every known interval must survive its own endpoint.
    boundary_checks=all(a['pmid'] in select_publication_dates([a],published_from=publication_interval(a)[1].isoformat())[0]
        and a['pmid'] in select_publication_dates([a],published_until=publication_interval(a)[0].isoformat())[0]
        for a in articles if publication_interval(a))
    rows_public=[]
    for r in snapshot_manifest['responses']:
        if r['selected']:
            rows_public.append({'pmid':r['pmid'],'pmcid':r['pmcid'],'license_uri':r['license_uri'],
                'retrieved_at':r['retrieved_at'],'raw_xml_sha256':r['sha256']})
    abstract_manifest=json.loads((abstract_root/'manifest.json').read_bytes())
    full_manifest=json.loads((full_root/'manifest.json').read_bytes())
    code_paths=[Path(__file__),ROOT/'src/medlens/pmc.py',ROOT/'src/medlens/dates.py',ROOT/'src/medlens/retrieval/dense.py',
        ROOT/'src/medlens/retrieval/windows.py',ROOT/'src/medlens/retrieval/vector_cache.py',ROOT/'src/medlens/generation/context.py',ROOT/'src/medlens/pipeline.py']
    cache_report=full_manifest['measurements']['cache']
    report={'schema_version':1,'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'machine':platform.machine(),'heldout_used':False,'qrels_used':False,
        'question_set_sha256':profile['queries']['sha256'],'question_ids':[q.query_id for q in queries],
        'scope':'Six deterministically selected, explicitly CC-BY PMC articles linked from the rebuilt development pool. Feasibility and cost only; not a representative sample or evidence-quality estimate.',
        'selection':{'method':'ascending PMID; first six eligible records within sequential candidate cap',
            'candidates_with_pmcid':snapshot_manifest['candidate_count'],'fetched':len(snapshot_manifest['responses']),
            'selected':len(snapshot_manifest['selected_pmids']),
            'rejected_by_reason':__import__('collections').Counter(r.get('reason','accepted') for r in snapshot_manifest['responses']),
            'retained_source_xml_files':sum('file'in r for r in snapshot_manifest['responses']),
            'rejected_source_xml_retained':False,'articles':rows_public},
        'snapshot_id':snapshot_manifest['snapshot_id'],'base_collection_id':snapshot_manifest['base_collection_id'],
        'paired_artifacts':{'abstract':{k:abstract_manifest[k] for k in ('manifest_id','dataset_id','window_artifact_id','embedding_artifact_id','counts','measurements')},
            'fulltext':{k:full_manifest[k] for k in ('manifest_id','dataset_id','window_artifact_id','embedding_artifact_id','counts','measurements')},
            'same_dataset':abstract_manifest['dataset_id']==full_manifest['dataset_id'],
            'abstract_windows_preserved':True,'all_fulltext_spans_verified':True},
        'timing_scope':'One validated resident model and both in-memory indexes. Same query vector used by both modes; order alternated. Warm-up excluded. Context score calculation and evidence packing reported separately from exact retrieval. Artifact preparation excludes Python startup and tokenizer loading.',
        'memory_scope':'Separate process RSS high-water and sampled post-forward MPS tensor/driver allocations; counters overlap and are not added. MPS measurements are not instantaneous peaks.',
        'runtime':model.info,'model_constructor_seconds':constructor,'query_tokenizer_load_seconds':tokenizer_load,
        'warmup_seconds_excluded':warmup,'index_validation_seconds':{'abstract':asetup,'fulltext':fsetup},
        'preparation':{'abstract':abstract_manifest['measurements'],'fulltext':full_manifest['measurements'],
            'shared_input_vectors_reused':cache_report['cache_hit_inputs'],
            'new_fulltext_vectors_encoded':cache_report['encoded_inputs'],'encoder_loads':cache_report['encoder_loads']},
        'date_filter':{'policy':'inclusive overlap using PubMed journal_issue date interval only',
            'unknown_dates_excluded_when_bounded':True,'inclusive_day_boundary_checks_passed':boundary_checks,
            'example_published_from':cutoff,'example_selected':len(eligible),**selection},
        'repeated_search':{'repeats_per_question':args.repeats,'aggregate':aggregate,
            'candidate_set_overlap_by_question':{q['query_id']:q['runs'][0]['top6_overlap'] for q in rows},
            'rank_agreement_by_question':{q['query_id']:{
                'same_rank_positions_median':statistics.median(r['same_rank_positions'] for r in q['runs']),
                'kendall_tau_median':statistics.median(r['kendall_tau'] for r in q['runs']),
                'top1_same_repeats':sum(r['top1_same'] for r in q['runs'])} for q in rows},
            'per_query':rows,'sampled_mps_tensor_bytes_max':max(r['mps']['sampled_tensor_bytes'] for r in allruns),
            'sampled_mps_driver_bytes_max':max(r['mps']['sampled_driver_bytes'] for r in allruns),
            'process_peak_rss_bytes':rss()},
        'decision':'Keep full text as an opt-in, license-gated experiment. The 6-record sample adds measurable body coverage and rankings can shift, but the current data have no independent relevance or answer-quality judgments; do not change the primary abstract workflow based on these results.',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code_paths}}
    report['selection']['rejected_by_reason']=dict(report['selection']['rejected_by_reason'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
