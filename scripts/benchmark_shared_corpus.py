"""Measure immutable corpus growth, cache reuse and development-query search costs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

from medlens.corpus import build_collection, inspect_collection
from medlens.evaluation.formats import parse_queries
from medlens.resources import load_collection_profile
from medlens.retrieval.bge_tokenizer import load_tokenizer
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def timed(call):
    tick = perf_counter()
    value = call()
    return value, perf_counter()-tick


def distribution(values):
    return {'min':min(values), 'median':statistics.median(values), 'max':max(values), 'count':len(values)}


def index(root):
    return DenseIndex(root/'dataset', root/'windows', root/'embeddings')


def encoder():
    return BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5', ROOT/'.cache/tokenizers/bge-base-en-v1.5',
        device='mps', batch_size=16)


def input_hashes(sources):
    return {str(p.relative_to(ROOT)):sha(p) for s in sources for k in ('dataset','windows','embeddings')
        for p in sorted(s[k].iterdir()) if p.is_file()}


def build(sources, tokenizer, info, cache, output, **kwargs):
    report, wall = timed(lambda:build_collection(sources, tokenizer, info, cache, output,
        encoder_factory=encoder, **kwargs))
    return {'collection_id':report['collection_id'], 'manifest_id':report['manifest_id'],
        'identity':report['identity'], 'counts':report['counts'], 'measurements':report['measurements'],
        'wall_seconds_including_final_validation':wall}


def search_worker(sources, profile, output, repeats):
    import numpy as np
    pooled, setup = timed(lambda:index(output/'shared'))
    cold, cold_setup = timed(lambda:index(output/'cold'))
    if pooled.windows != cold.windows:
        raise ValueError('Cold/cached window mappings differ')
    model, constructor = timed(encoder)
    pooled.verify_encoder(model)
    cold.verify_encoder(model)
    queries_path = ROOT/profile['queries']['path']
    queries = parse_queries(queries_path.read_bytes(), queries_path.parent)
    if len(queries) != 10 or any(q.split != 'pilot' for q in queries):
        raise ValueError('Only ten development queries are permitted')
    _, warmup = timed(lambda:model.encode_queries([queries[0].text]))
    rows = []
    for query, spec in zip(queries, sources):
        if query.query_id != spec['query_id']:
            raise ValueError('Query/source order differs')
        small, small_setup = timed(lambda:DenseIndex(spec['dataset'],spec['windows'],spec['embeddings']))
        small.verify_encoder(model)
        row = {'query_id':query.query_id,'small_articles':len(small.articles),'small_windows':len(small.windows),
            'small_vector_bytes':int(small.vectors.nbytes),'small_index_validation_seconds':small_setup,'runs':[]}
        for repeat in range(repeats):
            vectors, qstats = model.encode_queries([query.text])
            results = {};times = {}
            for name in (('small','shared') if repeat%2 == 0 else ('shared','small')):
                hits, times[name] = timed(lambda:(small if name=='small' else pooled).search(vectors[0]))
                results[name] = [h['pmid'] for h in hits]
            cold_hits = cold.search(vectors[0])
            if [h['pmid'] for h in cold_hits] != results['shared']:
                raise ValueError('Cold/cached top ten differ')
            row['runs'].append({'query_encoding':qstats,'search_seconds':times,
                'ranked_pmids':results,'top10_overlap':len(set(results['small'])&set(results['shared']))})
        if any(r['ranked_pmids'] != row['runs'][0]['ranked_pmids'] for r in row['runs']):
            raise ValueError('Ranking changed across repeats')
        rows.append(row)
        print('search',query.query_id,flush=True)
    return {'status':'complete','rows':rows,'shared_index_validation_seconds':setup,
        'cold_index_validation_seconds':cold_setup,'shared_vector_bytes':int(pooled.vectors.nbytes),
        'max_cold_cached_vector_abs_difference':float(np.max(np.abs(pooled.vectors-cold.vectors))),
        'all_cold_cached_top10_equal':True,'model_constructor_seconds':constructor,
        'model_internal_load_seconds':model.load_seconds,'warmup_seconds':warmup,'runtime':model.info,
        'process_peak_rss_bytes':rss()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collections',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh benchmark and artifact directory')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--summary',type=Path,help='Fresh public aggregate JSON, without source text')
    parser.add_argument('--worker',choices=('cold','search'),help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 2:parser.error('Use at least two repetitions')
    if args.summary and args.summary.exists():parser.error('Summary already exists')
    profile, profile_sha, sources = load_collection_profile(ROOT,args.collections)
    if args.worker == 'search':
        write(args.output/'search.json',search_worker(sources,profile,args.output,args.repeats))
        return
    tokenizer, info = load_tokenizer(ROOT/'.cache/tokenizers/bge-base-en-v1.5')
    if args.worker == 'cold':
        result = build(sources,tokenizer,info,args.output/'cold-cache',args.output/'cold',seed=False)
        result['process_peak_rss_bytes'] = rss()
        write(args.output/'cold.json',result)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    code_paths = [Path(__file__),ROOT/'src/medlens/corpus.py',ROOT/'src/medlens/retrieval/vector_cache.py',
        ROOT/'src/medlens/retrieval/dense.py',ROOT/'src/medlens/retrieval/windows.py',ROOT/'src/medlens/retrieval/embeddings.py']
    code = {str(p.relative_to(ROOT)):sha(p) for p in code_paths}
    before = input_hashes(sources)
    cache = args.output/'cache'
    stages = {}
    for name, selected, seed, reuse in [('small',[sources[0]],True,None),('shared',sources,True,None),
            ('repeat',sources,False,None),('reversed',list(reversed(sources)),False,None),
            ('removed',[sources[0]],False,None),('reseeded',sources,False,args.output/'shared')]:
        stages[name] = build(selected,tokenizer,info,args.output/'reseeded-cache' if reuse else cache,
            args.output/name,seed=seed,reuse=reuse)
        print('build',name,stages[name]['counts'],flush=True)
    if not (stages['small']['collection_id']==stages['removed']['collection_id'] and
            len({stages[n]['collection_id'] for n in ('shared','repeat','reversed','reseeded')})==1):
        raise ValueError('Cached rebuild identities differ')
    for worker in ('cold','search'):
        subprocess.run([sys.executable,str(Path(__file__)),'--collections',str(args.collections),
            '--output',str(args.output),'--repeats',str(args.repeats),'--worker',worker],check=True)
    cold = json.loads((args.output/'cold.json').read_bytes())
    search = json.loads((args.output/'search.json').read_bytes())
    if before != input_hashes(sources):raise ValueError('Source inputs changed')
    for name in stages:inspect_collection(args.output/name)
    runs = [r for row in search['rows'] for r in row['runs']]
    aggregate = {'query_encoding_seconds':distribution([r['query_encoding']['encode_seconds'] for r in runs]),
        'small_search_seconds':distribution([r['search_seconds']['small'] for r in runs]),
        'shared_search_seconds':distribution([r['search_seconds']['shared'] for r in runs]),
        'top10_overlap_by_query':{r['query_id']:r['runs'][0]['top10_overlap'] for r in search['rows']},
        'small_validation_seconds':distribution([r['small_index_validation_seconds'] for r in search['rows']]),
        'sampled_max_mps_tensor_bytes':max(r['query_encoding']['sampled_max_mps_tensor_bytes'] for r in runs),
        'sampled_max_mps_driver_bytes':max(r['query_encoding']['sampled_max_mps_driver_bytes'] for r in runs)}
    report = {'schema_version':1,'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),
        'platform':platform.platform(),'machine':platform.machine(),'heldout_used':False,'qrels_used':False,
        'scope':'Rebuilt abstract-only development collections; start of Small to Medium transition, not large-corpus or quality validation.',
        'timing_scope':'Single preparation run per scenario. OS filesystem caches not flushed. Cold means an empty vector cache and actual BGE inference, not cold OS/model storage. Search reuses one model and resident validated indexes; one query vector per repetition for both searches, alternating order. Warm-up excluded. Validation, model construction and preparation reported separately.',
        'memory_scope':'Process RSS high-water marks and sampled post-forward MPS tensor/driver allocations are separate overlapping views; do not add. Parent preparation excludes child RSS. No instantaneous device-peak claim.',
        'repeats':args.repeats,'profile_sha256':profile_sha,'queries_sha256':profile['queries']['sha256'],
        'code_sha256':code,'source_files_sha256':before,'source_files_unchanged':True,
        'cached_rebuild_identities_equal':True,'removal_restores_small_identity':True,
        'stages':stages,'cold':cold,'cached_preparation_process_peak_rss_bytes':rss(),
        'search':{k:v for k,v in search.items() if k!='rows'},'aggregate':aggregate,
        'cache_entries':sum(1 for _ in cache.glob('*/*.bin'))}
    write(args.output/'summary.json',report)
    if args.summary:
        args.summary.parent.mkdir(parents=True,exist_ok=True)
        write(args.summary,report)


if __name__ == '__main__':
    main()
