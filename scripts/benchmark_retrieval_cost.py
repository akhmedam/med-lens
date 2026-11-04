"""Measure local pilot pool/depth costs in separate BGE and MedCPT processes."""

import argparse
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
from medlens.evaluation.formats import parse_queries, parse_qrels
from medlens.evaluation.metrics import evaluate, macro_average
from medlens.reranking.medcpt import sha_file

POOLS = (10, 20, 30)
DEPTHS = (10, 20, 30)


def scoped_index(index, order, vectors=None):
    """In-memory prefix view; never rewrite or relabel the frozen source artifact."""
    if not order or len(set(order)) != len(order) or not set(order) <= set(index.articles):
        raise ValueError('Expected a nonempty unique subset of candidate PMIDs')
    view = copy.copy(index)
    allowed = set(order)
    positions = [i for i, w in enumerate(index.windows) if w['pmid'] in allowed]
    view.windows = tuple(index.windows[i] for i in positions)
    view.vectors = index.vectors[positions] if vectors is None else vectors
    if len(view.vectors) != len(positions):
        raise ValueError('Subset vectors do not match source windows')
    view.articles = {p: index.articles[p] for p in order}
    view.positions = {p: [] for p in order}
    for i, w in enumerate(view.windows):
        view.positions[w['pmid']].append(i)
    return view


def timed(call):
    start = perf_counter()
    value = call()
    return value, perf_counter() - start


def distribution(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values), 'count': len(values)}


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value * 1024


def write(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bge_worker(queries, repeats, report, checkpoint):
    import numpy as np
    from medlens.retrieval.dense import DenseIndex
    from medlens.retrieval.embeddings import BGEEncoder
    encoder, elapsed = timed(lambda: BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',
        ROOT/'.cache/tokenizers/bge-base-en-v1.5', device='mps', batch_size=16))
    report['model_constructor_seconds'] = elapsed
    report['runtime'] = encoder.info
    # Separate shape warm-up, excluded from measured rows.
    report['warmup_seconds'] = timed(lambda: encoder.encode_queries([queries[0].text]))[1]
    for query in queries:
        index, setup = timed(lambda: DenseIndex(query.dataset,
            ROOT/'data/windows/pilot-v1-002'/query.query_id, ROOT/'data/embeddings/pilot-v1'/query.query_id))
        index.verify_encoder(encoder)
        if len(index.articles) != 30:
            raise ValueError('Expected frozen 30-article pilot snapshots')
        report.setdefault('sources', {})[query.query_id] = {
            'dataset_id': index.dataset.manifest['dataset_id'],
            'window_artifact_id': index.window_manifest['artifact_id'],
            'embedding_artifact_id': index.embedding_manifest['artifact_id'],
            'full_index_setup_seconds': setup}
        for pool in POOLS:
            order = list(index.articles)[:pool]
            view = scoped_index(index, order)
            row = {'query_id': query.query_id, 'pool': pool, 'pmids': order,
                'windows': len(view.windows), 'vector_bytes': int(view.vectors.nbytes), 'runs': []}
            report['rows'].append(row)
            for _ in range(repeats):
                start = perf_counter()
                qvectors, qstats = encoder.encode_queries([query.text])
                hits, search_time = timed(lambda: view.search(qvectors[0], pool))
                cached = perf_counter() - start
                start = perf_counter()
                vectors, pstats = encoder.encode([w['input_ids'] for w in view.windows])
                fresh_view = scoped_index(index, order, vectors)
                fresh_hits, fresh_search = timed(lambda: fresh_view.search(qvectors[0], pool))
                passage_build = perf_counter() - start
                if [h['pmid'] for h in fresh_hits] != [h['pmid'] for h in hits]:
                    raise ValueError('Fresh and cached Dense article order differs')
                row['runs'].append({'cached_seconds': cached, 'query': qstats,
                    'cosine_search_seconds': search_time, 'passage_encode': pstats,
                    'passage_build_search_seconds': passage_build,
                    'fresh_total_seconds': cached - search_time + passage_build,
                    'fresh_search_seconds': fresh_search,
                    'max_vector_drift': float(np.max(np.abs(vectors - view.vectors))),
                    'ranked_pmids': [h['pmid'] for h in hits]})
                checkpoint()
            if any(r['ranked_pmids'] != row['runs'][0]['ranked_pmids'] for r in row['runs']):
                raise ValueError('Dense ordering changed across repeats')
            print('bge', query.query_id, pool, flush=True)


def medcpt_worker(queries, repeats, dense, report, checkpoint):
    from medlens.reranking.articles import MedCPTRanker
    from medlens.reranking.medcpt import MedCPTScorer
    scorer, elapsed = timed(lambda: MedCPTScorer(ROOT/'.cache/models/medcpt-cross-encoder', device='mps', batch_size=4))
    report['model_constructor_seconds'] = elapsed
    report['runtime'] = scorer.metadata
    first = MedCPTRanker(queries[0].dataset, ROOT/'data/windows/pilot-v1-002'/queries[0].query_id)
    w = first.windows[0]
    report['warmup_seconds'] = timed(lambda: scorer.score([(queries[0].text, w['title'], w['text'])]))[1]
    selections = {(r['query_id'], r['pool']): r for r in dense['rows']}
    for query in queries:
        ranker, setup = timed(lambda: MedCPTRanker(query.dataset, ROOT/'data/windows/pilot-v1-002'/query.query_id))
        source = dense['sources'][query.query_id]
        if (source['dataset_id'] != ranker.dataset.manifest['dataset_id'] or
                source['window_artifact_id'] != ranker.window_manifest['artifact_id']):
            raise ValueError('Dense/MedCPT source mismatch')
        for pool in POOLS:
            selected = selections[query.query_id, pool]
            if selected['pmids'] != list(ranker.articles)[:pool]:
                raise ValueError('Candidate prefix order differs')
            scenarios = [('medcpt', None, selected['pmids'])] + [
                ('dense_medcpt', depth, selected['runs'][0]['ranked_pmids'][:depth])
                for depth in DEPTHS if depth <= pool]
            # Alternate traversal to reduce a systematic direct-first order effect.
            for repeat in range(repeats):
                traversal = scenarios if repeat % 2 == 0 else list(reversed(scenarios))
                for method, depth, eligible in traversal:
                    result, elapsed = timed(lambda: ranker.run(query.text, scorer, eligible_pmids=eligible))
                    report['rows'].append({'query_id': query.query_id, 'pool': pool, 'method': method,
                        'depth': depth, 'repeat': repeat, 'source_setup_seconds': setup,
                        'request_seconds': elapsed, 'eligible_pmids': eligible,
                        'pair_count': result['scoring']['pair_count'],
                        'scoring': result['scoring'],
                        'window_scores': result['window_scores'],
                        'ranked_pmids': [h['pmid'] for h in result['results']]})
                    checkpoint()
            print('medcpt', query.query_id, pool, flush=True)


def summarize(dense, med, qrels):
    groups = {}
    for row in dense['rows']:
        key = f"dense/pool-{row['pool']}"
        entry = groups.setdefault(key, {'metrics': [], 'query_rows': []})
        runs = row['runs']
        ranking = runs[0]['ranked_pmids'][:10]
        scores = evaluate(ranking, row['pmids'], qrels[row['query_id']])
        entry['metrics'].append(scores)
        entry['query_rows'].append({'query_id': row['query_id'], 'ranked_pmids': ranking, 'metrics': scores,
            'pair_count': 0, 'passage_count': row['windows'], 'vector_bytes': row['vector_bytes'],
            'sampled_mps_tensor_bytes': max(max(r[k]['sampled_max_mps_tensor_bytes'] or 0 for k in ('query', 'passage_encode')) for r in runs) if 'query' in runs[0] else None,
            'sampled_mps_driver_bytes': max(max(r[k]['sampled_max_mps_driver_bytes'] or 0 for k in ('query', 'passage_encode')) for r in runs) if 'query' in runs[0] else None,
            'cached_seconds': distribution([r['cached_seconds'] for r in runs]),
            'fresh_seconds': distribution([r['fresh_total_seconds'] for r in runs])})
    for row in med['rows']:
        if row['repeat'] != 0:
            continue
        key = f"{row['method']}/pool-{row['pool']}/depth-{row['depth']}"
        repeats = [r for r in med['rows'] if all(r[k] == row[k] for k in ('query_id', 'pool', 'method', 'depth'))]
        if any(r['ranked_pmids'] != row['ranked_pmids'] for r in repeats):
            raise ValueError('MedCPT top ten changed across repeats')
        base = next(r for r in dense['rows'] if r['query_id'] == row['query_id'] and r['pool'] == row['pool'])
        scores = evaluate(row['ranked_pmids'], base['pmids'], qrels[row['query_id']])
        entry = groups.setdefault(key, {'metrics': [], 'query_rows': []})
        entry['metrics'].append(scores)
        med_times = [r['request_seconds'] for r in repeats]
        cached = [r['cached_seconds'] for r in base['runs']]
        fresh = [r['fresh_total_seconds'] for r in base['runs']]
        entry['query_rows'].append({'query_id': row['query_id'], 'ranked_pmids': row['ranked_pmids'],
            'metrics': scores, 'pair_count': row['pair_count'],
            'sampled_mps_tensor_bytes': max(r['scoring']['sampled_max_mps_tensor_bytes'] or 0 for r in repeats) if 'scoring' in row else None,
            'sampled_mps_driver_bytes': max(r['scoring']['sampled_max_mps_driver_bytes'] or 0 for r in repeats) if 'scoring' in row else None,
            'medcpt_seconds': distribution(med_times),
            'cached_seconds': distribution([t + (cached[i] if row['method'] == 'dense_medcpt' else 0)
                for i, t in enumerate(med_times)]),
            'fresh_seconds': distribution([t + (fresh[i] if row['method'] == 'dense_medcpt' else 0)
                for i, t in enumerate(med_times)])})
    for entry in groups.values():
        entry['macro'] = macro_average(entry.pop('metrics'), 10)
        for field in ('sampled_mps_tensor_bytes', 'sampled_mps_driver_bytes'):
            values = [r[field] for r in entry['query_rows'] if r[field] is not None]
            entry[field] = max(values) if values else None
        entry['total_pairs_per_repeat'] = sum(r['pair_count'] for r in entry['query_rows'])
        for kind in ('cached_seconds', 'fresh_seconds'):
            entry[kind] = distribution([r[kind]['median'] for r in entry['query_rows']])
        entry['fully_judged_queries'] = [r['query_id'] for r in entry['query_rows'] if r['metrics']['unjudged_returned_count'] == 0]
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh run directory')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--summary', type=Path, help='Optional fresh compact public report')
    parser.add_argument('--summarize-only', action='store_true', help='Replay complete worker reports without model inference')
    parser.add_argument('--worker', choices=('bge', 'medcpt'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.summary and args.summary.exists():
        parser.error('Choose a fresh summary path')
    if args.repeats < 2:
        parser.error('At least two repeats required')
    queries_path = ROOT/'benchmarks/pilot-v1/queries.jsonl'
    profile_path = ROOT/'benchmarks/pilot-v1/medcpt-evaluation.json'
    profile = json.loads(profile_path.read_bytes())
    qrels_path = ROOT/profile['qrels']['path']
    if sha_file(qrels_path) != profile['qrels']['sha256']:
        raise ValueError('Frozen qrels checksum differs')
    base_path = ROOT/profile['base_profile']['path']
    if sha_file(base_path) != profile['base_profile']['sha256']:
        raise ValueError('Base profile checksum differs')
    base = json.loads(base_path.read_bytes())
    if sha_file(queries_path) != base['queries']['sha256']:
        raise ValueError('Frozen query checksum differs')
    queries = parse_queries(queries_path.read_bytes(), queries_path.parent)
    if len(queries) != 10 or any(q.split != 'pilot' for q in queries):
        raise ValueError('Only the frozen ten-query development pilot is permitted')
    if args.worker:
        output = args.output/f'{args.worker}.json'
        if output.exists():
            raise FileExistsError(output)
        report = {'status': 'running', 'rows': [], 'started_at': datetime.now(timezone.utc).isoformat()}
        write(output, report)
        start = perf_counter()
        try:
            checkpoint = lambda: write(output, report)
            if args.worker == 'bge':
                bge_worker(queries, args.repeats, report, checkpoint)
            else:
                dense = json.loads((args.output/'bge.json').read_bytes())
                if dense['status'] != 'complete':
                    raise ValueError('Incomplete BGE stage')
                medcpt_worker(queries, args.repeats, dense, report, checkpoint)
            report['status'] = 'complete'
        except (Exception, KeyboardInterrupt) as exc:
            report['status'] = 'failed'
            report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            report['elapsed_seconds'] = perf_counter() - start
            report['process_peak_rss_bytes'] = peak_rss()
            write(output, report)
        return
    if not args.summarize_only:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'benchmark-source.py').write_bytes(Path(__file__).read_bytes())
    code = {str(p.relative_to(ROOT)): sha_file(p) for p in [Path(__file__),
        ROOT/'src/medlens/retrieval/dense.py', ROOT/'src/medlens/retrieval/embeddings.py',
        ROOT/'src/medlens/reranking/articles.py', ROOT/'src/medlens/reranking/medcpt.py']}
    if not args.summarize_only:
        for worker in ('bge', 'medcpt'):
            subprocess.run([sys.executable, str(Path(__file__)), '--output', str(args.output),
                '--repeats', str(args.repeats), '--worker', worker], check=True)
    dense, med = [json.loads((args.output/f'{w}.json').read_bytes()) for w in ('bge', 'medcpt')]
    if any(r['status'] != 'complete' for r in (dense, med)):
        raise ValueError('Both workers must complete before summarizing')
    if (len(dense['rows']) != 30 or any(len(r['runs']) != args.repeats for r in dense['rows'])
            or len(med['rows']) != 90 * args.repeats):
        raise ValueError('Incomplete or mismatched repeat counts')
    qrels = parse_qrels(qrels_path.read_bytes(), queries)
    measured_source = ast.parse((args.output/'benchmark-source.py').read_text())
    current_source = ast.parse(Path(__file__).read_text())
    measured_functions = ('bge_worker', 'medcpt_worker', 'scoped_index', 'timed', 'peak_rss', 'write')
    def bodies(tree):
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in measured_functions}
    if bodies(measured_source) != bodies(current_source):
        raise ValueError('Measurement functions differ from the saved run source')
    groups = summarize(dense, med, qrels)
    common = sorted(set.intersection(*(set(g['fully_judged_queries']) for g in groups.values())))
    for group in groups.values():
        group['common_fully_judged_macro'] = macro_average(
            [r['metrics'] for r in group['query_rows'] if r['query_id'] in common], 10)
    report = {'schema_version': 1, 'status': 'complete', 'heldout_used': False,
        'common_fully_judged_queries': common,
        'quality_status': 'provisional-incomplete-judgments',
        'annotation': profile['annotation'],
        'pubmed_quality': {str(pool): macro_average([evaluate(r['pmids'][:10], r['pmids'], qrels[r['query_id']])
            for r in dense['rows'] if r['pool'] == pool], 10) for pool in POOLS},
        'timing_scope': 'Reused model and validated in-memory data. Fresh Dense adds passage encoding from existing windows; excludes fetch, parsing, windowing and disk artifact writes. Cascade totals are sums of separately measured stages, not co-resident end-to-end runs.',
        'quality_scope': 'Fixed full 30-candidate qrels-v3 pool denominators at every prefix/depth. Unjudged results contribute zero to pooled arithmetic and reduce coverage; they are not relabeled irrelevant. AI-assisted development judgments, not expert or held-out truth.',
        'memory_scope': 'Separate worker process RSS high-water marks, and post-forward sampled MPS allocations. Neither is an instantaneous device peak; do not add RSS to MPS or sum stage memory.',
        'platform': platform.platform(), 'machine': platform.machine(),
        'repeats': args.repeats, 'pools': POOLS, 'depths': DEPTHS,
        'queries_sha256': sha_file(queries_path), 'qrels_sha256': sha_file(qrels_path),
        'profile_sha256': sha_file(profile_path), 'code_sha256': code,
        'measurement_script_sha256': sha_file(args.output/'benchmark-source.py'),
        'local_reports_sha256': {w: sha_file(args.output/f'{w}.json') for w in ('bge', 'medcpt')},
        'sources': dense['sources'],
        'runtime': {w: {k: r[k] for k in ('runtime', 'model_constructor_seconds', 'warmup_seconds',
            'process_peak_rss_bytes', 'elapsed_seconds')} for w, r in (('bge', dense), ('medcpt', med))},
        'groups': groups}
    write(args.output/'summary.json', report)
    if args.summary:
        compact = {**report, 'local_detailed_summary_sha256': sha_file(args.output/'summary.json'),
            'groups': {key: {k: v for k, v in group.items() if k != 'query_rows'}
                for key, group in report['groups'].items()}}
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open('x') as stream:
            stream.write(json.dumps(compact, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
