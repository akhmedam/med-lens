"""Freeze search-only biomedical development cases for independent human annotation.

No answer generation, labels, relevance metrics or held-out files are used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def sha(body):
    return hashlib.sha256(body).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_plan(path):
    body = path.read_bytes()
    plan = json.loads(body)
    cases = plan.get('cases')
    if (plan.get('schema_version') != 1 or plan.get('split') != 'dev'
            or plan.get('heldout_used') is not False or not isinstance(cases, list) or not cases):
        raise ValueError('Require an explicit development-only annotation plan')
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or any(not isinstance(case.get(k), str) or not case[k].strip()
                for k in ('query_id', 'question', 'relevance_intent', 'specialty', 'question_type')):
            raise ValueError('Incomplete biomedical development case')
        qid = case['query_id']
        if (not qid.isascii() or not qid.replace('-', '').replace('_', '').isalnum()
                or qid in seen):
            raise ValueError('Require unique safe query IDs')
        seen.add(qid)
    limit = plan.get('candidate_limit')
    if type(limit) is not int or not 1 <= limit <= 300:
        raise ValueError('Plan must declare a bounded candidate_limit')
    return body, plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=ROOT/'benchmarks/biomedical-dev-v1/plan.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh local directory, outside the embedding cache')
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/embeddings/biomedical-dev-v1')
    parser.add_argument('--device', choices=('mps', 'cpu'), default='mps')
    args = parser.parse_args(argv)
    try:
        body, plan = load_plan(args.plan)
        if args.output.exists() or args.output.is_symlink():
            raise ValueError('Choose a fresh output directory')
        output, cache = args.output.resolve(), args.cache.resolve()
        if output.is_relative_to(cache) or cache.is_relative_to(output):
            raise ValueError('Case output and embedding cache must be disjoint')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output, args.cache = output, cache
    from medlens.pubmed_query import plan_query
    plans = {case['query_id']: plan_query(case['question']) for case in plan['cases']}
    implementation = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
        (Path(__file__), ROOT/'scripts/ask_live.py', ROOT/'src/medlens/pubmed_query.py')}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan-original.json').write_bytes(body)
    report = {'schema_version': 1, 'kind': 'biomedical-development-source-preparation',
        'status': 'running', 'created_at': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(body), 'candidate_limit': plan['candidate_limit'], 'query_plans': plans,
        'implementation_sha256': implementation, 'split': 'dev', 'heldout_used': False,
        'human_labels_available': False, 'quality_evaluation_performed': False,
        'answer_generation_performed': False, 'runs': []}
    path = args.output/'preparation.json'
    save(path, report)
    try:
        for case in plan['cases']:
            if args.plan.read_bytes() != body:
                raise ValueError('Frozen annotation plan changed during acquisition')
            qid = case['query_id']
            print('START', qid, flush=True)
            start = perf_counter()
            with (args.output/(qid+'.log')).open('x') as log:
                result = subprocess.run([sys.executable, str(ROOT/'scripts/ask_live.py'),
                    '--query', case['question'], '--limit', str(plan['candidate_limit']),
                    '--search-only', '--answer-model', '4b', '--cache', str(args.cache),
                    '--device', args.device, '--output-dir', str(args.output/qid)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            wall = perf_counter()-start
            run_path = args.output/qid/'run.json'
            run = json.loads(run_path.read_bytes()) if run_path.is_file() else {}
            row = {'query_id': qid, 'exit_code': result.returncode,
                'status': run.get('status', 'no-checkpoint'), 'command_wall_seconds': wall,
                'run_sha256': sha(run_path.read_bytes()) if run_path.is_file() else None,
                'snapshot_id': run.get('live', {}).get('snapshot_id'),
                'dataset_id': run.get('live', {}).get('dataset_id'),
                'total_pubmed_matches': run.get('live', {}).get('total_pubmed_matches'),
                'fetched_article_count': run.get('live', {}).get('fetched_article_count'),
                'query_translation': run.get('live', {}).get('query_translation'),
                'window_count': run.get('scope', {}).get('window_count'),
                'passage_cache': run.get('live', {}).get('passage_cache'),
                'preparation_timings': run.get('preparation_timings'),
                'timings': run.get('timings'), 'query_plan': plans[qid]}
            report['runs'].append(row)
            save(path, report)
            if (result.returncode != 0 or run.get('status') != 'complete'
                    or run.get('question') != case['question']
                    or run.get('live', {}).get('query_plan') != plans[qid]
                    or run.get('live', {}).get('fresh_search_complete') is not True
                    or run.get('runtime_load_count') != 0
                    or (run.get('retrieval', {}).get('results') and run.get('generate') is not False)
                    or run.get('answer', {}).get('status') != 'not_requested'):
                raise RuntimeError('Incomplete or unexpected search-only case: '+qid+'; inspect its checkpoint/log')
            if not run.get('retrieval', {}).get('results'):
                raise RuntimeError('No searchable candidates for '+qid+'; preserve this run and revise a new plan explicitly')
            print('COMPLETE', qid, row['fetched_article_count'], round(wall, 2), flush=True)
        if args.plan.read_bytes() != body:
            raise ValueError('Frozen annotation plan changed')
        if any(sha((ROOT/name).read_bytes()) != value for name, value in implementation.items()):
            raise ValueError('Acquisition implementation changed during preparation')
        report['status'] = 'complete-awaiting-manual-human-annotation'
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error={'type': type(exc).__name__, 'message': str(exc)})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(path, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
