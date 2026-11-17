"""Freeze then execute one operational held-out confirmation, without tuning.

Freeze hashes question bytes but never parses the question rows. Run consumes
the one-shot execution marker before parsing them. Raw outputs remain local.
"""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from threading import Event, Thread
import time


QUERIES = 'benchmarks/heldout-v1/queries.jsonl'
QUERY_SHA256 = '7d1fa94cbf999bca44a139f60a3646e505565244712a6b60291ccae311a7035f'
PLAN = 'benchmarks/heldout-v1/candidate-plan.json'
PLAN_SHA256 = '45d3680c16ffc9c595817243b65099d5fcd5a5918881d5b86593dce45dfa138e'
POLICY = {'answer_style': 'source-findings', 'limit': 100, 'top_k': 10,
    'generate': True, 'pubmed_query': None, 'engine': 'mlx', 'answer_model': '4b',
    'encoder_device': 'mps', 'question_order': 'original-file-order',
    'one_attempt_per_question': True, 'prompt_or_model_tuning': False,
    'candidate_expression_policy': 'actual unchanged question through the current recorded lexical planner',
    'cache_policy': 'existing immutable passage cache; disclose actual hits/misses, not a cold benchmark',
    'network_retry_policy': 'only the existing bounded PubMed transport retries; no harness case retries'}
SELECTOR_PINS = {
    'src/medlens/generation/source_findings.py': '91b9aaaf9b6f2a13ca1476a23524183521d6cd001c7fd0207826773602edaf88',
    'src/medlens/generation/context.py': '75613d2e16f8f6a71d1ece07142b98c9915c4df7ee90af2537438ea7d1835c05',
    'src/medlens/generation/source_scope.py': '3abb04d7ff53272431a93cf70c6e96664d2d8aeec4fe146a317de2d48d3f4925'}
ASSET_MANIFESTS = (
    '.cache/models/bge-base-en-v1.5/model-source.json',
    '.cache/tokenizers/bge-base-en-v1.5/tokenizer-source.json',
    '.cache/models/qwen3.5-4b-mlx-4bit-g64/medlens-source.json',
    '.cache/tokenizers/qwen3.5-4b/tokenizer-source.json')
PACKAGES = ('numpy', 'torch', 'transformers', 'tokenizers', 'psutil', 'fastapi', 'uvicorn')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_new(path, value):
    body = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def constants(path, names):
    found = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise ValueError('Required pinned constants were not found: ' + str(path))
    return found


def packages():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def source_hashes(root):
    paths = sorted(p for p in (root / 'src/medlens').rglob('*.py') if '__pycache__' not in p.parts)
    paths += [root / name for name in ('pyproject.toml', 'scripts/validate_mvp.py',
        'docs/final-validation-protocol.md', 'scripts/prepare_models.py',
        'requirements-macos-arm64.lock',
        'requirements-mlx-macos-arm64.lock', 'configs/model-assets.json', 'configs/mlx-assets.json')]
    return {str(p.relative_to(root)): file_sha(p) for p in paths}


def application_policy(root):
    """Execute only the stdlib central policy module, before reading test text."""
    spec = importlib.util.spec_from_file_location('_validation_mvp_policy', root / 'src/medlens/mvp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.policy_snapshot()
    actual = (value['source']['candidate_limit']['default'], value['retrieval']['top_k']['default'],
        value['retrieval']['device'], value['answer']['style'],
        value['answer']['model_profile'], value['answer']['engine'], value['answer']['independent_validation'])
    if actual != (100, 10, 'mps', 'source-findings', '4b', 'mlx', False):
        raise ValueError('Central application policy differs from the prospective run')
    return value


def asset_identity(root):
    manifests, files = {}, {}
    for name in ASSET_MANIFESTS:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Require a regular asset manifest: ' + name)
        value = load(path)
        manifests[name] = {'sha256': file_sha(path), 'metadata': value}
        for filename in value['files']:
            target = path.parent / filename
            if Path(filename).name != filename or target.is_symlink() or not target.is_file():
                raise ValueError('Invalid regular pinned asset file: ' + str(target))
            actual = {'sha256': file_sha(target), 'bytes': target.stat().st_size}
            expected = value['files'][filename]
            expected_sha = expected if isinstance(expected, str) else expected['sha256']
            if actual['sha256'] != expected_sha or (
                    isinstance(expected, dict) and actual['bytes'] != expected['bytes']):
                raise ValueError('Asset file differs from its pinned checksum/size: ' + str(target))
            files[str(target.relative_to(root))] = actual
    return {'manifests': manifests, 'actual_asset_files': files}


def worker_environment(root):
    python = root / '.cache/venvs/mlx/bin/python'
    command = ('import importlib.metadata as m,json,sys; '
        'print(json.dumps({"python":sys.version,"packages":{n:m.version(n) '
        'for n in ("mlx","mlx-lm","mlx-metal","transformers","tokenizers")}}))')
    result = subprocess.run([str(python), '-I', '-c', command],
        capture_output=True, text=True, timeout=30, check=True)
    return {'executable': str(python), **json.loads(result.stdout)}


def freeze(root, output):
    """Hash opaque question bytes before any question deserialization."""
    if file_sha(root / QUERIES) != QUERY_SHA256 or file_sha(root / PLAN) != PLAN_SHA256:
        raise ValueError('Original held-out bytes differ from the predeclared identities')
    for name, expected in SELECTOR_PINS.items():
        if file_sha(root / name) != expected:
            raise ValueError('The selected V2 algorithm changed: ' + name)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Choose a fresh validation directory')
    cache = (root / '.cache/embeddings/bge-input-v1').resolve()
    if output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError('Validation output and immutable cache must be disjoint')
    selector = constants(root / 'src/medlens/generation/source_findings.py',
        {'SELECTOR_SYSTEM', 'PROMPT_VERSION', 'ANSWER_POLICY', 'SELECTION_CAP', 'MAX_FINDINGS',
         'MAX_QUOTE_CHARS', 'MAX_QUOTES_PER_FINDING'})
    prompt = selector.pop('SELECTOR_SYSTEM')
    profile = constants(root / 'src/medlens/inference/profiles.py', {'PROFILES'})['PROFILES']['4b']
    manifest = {'schema_version': 1, 'kind': 'prospective-mvp-operational-validation',
        'frozen_at': now(), 'root': str(root), 'policy': POLICY,
        'application_policy': application_policy(root),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'question_file': {'path': QUERIES, 'sha256': QUERY_SHA256, 'expected_count': 12},
        'original_candidate_plan': {'path': PLAN, 'sha256': PLAN_SHA256,
            'unchanged': True, 'applied_to_this_run': False,
            'difference': 'Original cap30/four-method comparison retained; this separately declared operational MVP pass uses cap100, actual question planner, BGE and source-findings only.'},
        'prefreeze_exposure_declaration': {
            'candidate_pubmed_expressions_read_on': '2026-11-15',
            'question_texts_read': False, 'model_outputs_read': False,
            'note': 'Operator declaration, not a programmatically provable blindness guarantee. Candidate expressions were inspected before this freeze; question texts and outputs were not.'},
        'source_sha256': source_hashes(root), 'model_profile': profile,
        'selector': {**selector, 'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()},
        'assets': asset_identity(root), 'parent_packages': packages(),
        'parent_python': sys.version, 'worker_environment': worker_environment(root),
        'reference_policy': {'retrieval_labels': 'not-collected-for-new-heldout-sources',
            'answer_labels': 'not-collected', 'independent_validation': False,
            'quality_estimates_permitted': False},
        'acceptance': {'every_case_terminal_success': True, 'every_visible_citation_exact_source_bound': True,
            'public_projection_matches_private_report': True, 'every_loaded_runtime_closed': True,
            'observed_native_workers_reaped': True,
            'no_performance_sla_predeclared': True,
            'independent_quality_gate': 'not-established-without-new-finalized-human-reference'}}
    manifest['manifest_id'] = digest(manifest)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / 'manifest.json', manifest)
    print(json.dumps({'status': 'frozen', 'manifest_id': manifest['manifest_id'],
        'question_texts_parsed': False, 'output': str(output)}, ensure_ascii=False), flush=True)
    return manifest


def verify_frozen(root, manifest):
    if manifest.get('manifest_id') != digest({k: v for k, v in manifest.items() if k != 'manifest_id'}):
        raise ValueError('Frozen manifest identity differs')
    if manifest['policy'] != POLICY or manifest['root'] != str(root):
        raise ValueError('Frozen run policy/root differs')
    if application_policy(root) != manifest['application_policy']:
        raise ValueError('Central application policy changed after freeze')
    if file_sha(root / QUERIES) != QUERY_SHA256 or file_sha(root / PLAN) != PLAN_SHA256:
        raise ValueError('Frozen held-out inputs changed')
    if source_hashes(root) != manifest['source_sha256']:
        raise ValueError('Source changed after freeze; do not inspect test outputs to tune it')
    if asset_identity(root) != manifest['assets'] or packages() != manifest['parent_packages']:
        raise ValueError('Model/tokenizer assets or packages changed after freeze')
    if worker_environment(root) != manifest['worker_environment'] or sys.version != manifest['parent_python']:
        raise ValueError('Python/native worker environment changed after freeze')


class MemorySampler:
    """Separate current RSS observations; never sum overlapping processes/devices."""
    def __init__(self):
        import psutil
        self.psutil, self.parent = psutil, psutil.Process()
        self.stop = Event()
        self.samples, self.native = [], {}
        self.thread = Thread(target=self._loop, daemon=True)

    def sample(self):
        row = {'at': now(), 'parent_current_rss_bytes': self.parent.memory_info().rss, 'descendants': []}
        for child in self.parent.children(recursive=True):
            try:
                argv = child.cmdline()
                native = any(Path(arg).name == 'mlx_worker.py' for arg in argv)
                info = {'pid': child.pid, 'created_at': child.create_time(),
                    'current_rss_bytes': child.memory_info().rss, 'native_mlx_worker': native}
                row['descendants'].append(info)
                if native:
                    self.native[(child.pid, info['created_at'])] = info
            except self.psutil.Error:
                continue
        self.samples.append(row)
        return row

    def _loop(self):
        while not self.stop.wait(0.25):
            self.sample()

    def __enter__(self):
        self.before = self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.after = self.sample()

    def result(self):
        workers = []
        for (pid, created), item in self.native.items():
            try:
                process = self.psutil.Process(pid)
                still_running = process.create_time() == created and process.is_running()
            except self.psutil.NoSuchProcess:
                still_running = False
            workers.append({'pid': pid, 'created_at': created, 'reaped': not still_running})
        return {'scope': 'Observed parent and child current RSS separately. No sum, high-water leak claim, swap peak, device/process non-overlap or whole-machine peak is inferred.',
            'before': self.before, 'after': self.after,
            'parent_max_sampled_current_rss_bytes': max(row['parent_current_rss_bytes'] for row in self.samples),
            'observed_native_workers': workers,
            'observed_native_workers_all_reaped': all(w['reaped'] for w in workers) if workers else None,
            'samples': self.samples}


def audit_sources(run, public):
    from medlens.api_contract import public_report
    from medlens.generation.context import locate_quote
    from medlens.generation.request_contexts import verify_context_sources
    from medlens.retrieval.dense import DenseIndex
    if public != public_report(run):
        raise ValueError('Public job result differs from persisted private projection')
    if run.get('answer_style') != 'source-findings':
        raise ValueError('Private run did not use the selected source-findings mode')
    context = run.get('context') or {}
    sources, claims = context.get('sources', []), (run.get('answer') or {}).get('claims', [])
    if not sources:
        if claims:
            raise ValueError('Claims exist without a source context')
        return {'status': 'passed', 'source_count': 0, 'visible_claims': 0, 'citations': 0}
    paths = run['input_paths']
    index = DenseIndex(paths['dataset'], paths['windows'], paths['embeddings'])
    verify_context_sources(context, index)
    citations = 0
    for claim in claims:
        if claim['text'] != ' '.join(c['quote'] for c in claim['citations']):
            raise ValueError('Source-findings prose differs from exact joined quotations')
        for citation in claim['citations']:
            location = locate_quote(context, citation['citation_id'], citation['quote'])
            for key in ('pmid', 'window_id', 'text_version_id', 'source_path', 'section_index'):
                if citation.get(key) != location.get(key):
                    raise ValueError('Citation source coordinate differs: ' + key)
            matches = citation.get('matches')
            if not isinstance(matches, list) or not matches or any(span not in location['matches'] for span in matches):
                raise ValueError('Citation span differs from original source')
            if citation['pmid'] != claim['source_article_id']:
                raise ValueError('Finding mixes source articles')
            citations += 1
    return {'status': 'passed', 'source_count': len(sources), 'visible_claims': len(claims), 'citations': citations}


@contextmanager
def ignore_sigint():
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def execute(root, output):
    manifest = load(output / 'manifest.json')
    verify_frozen(root, manifest)
    # The marker is exclusive and precedes the first semantic read of test rows.
    save_new(output / 'execution-started.json', {'started_at': now(),
        'manifest_id': manifest['manifest_id'], 'restart_resume_or_force_permitted': False})
    rows = [json.loads(line) for line in (root / QUERIES).read_bytes().splitlines() if line.strip()]
    if (len(rows) != 12 or len({r['query_id'] for r in rows}) != 12
            or len({r['group_id'] for r in rows}) != 12
            or any(set(r) != {'query_id', 'text', 'split', 'group_id', 'dataset'} or r['split'] != 'test'
                or re.fullmatch(r'[A-Za-z0-9_-]+', r['query_id']) is None for r in rows)):
        raise ValueError('Unexpected frozen question schema or membership')
    sys.path.insert(0, str(root / 'src'))
    from medlens.api_backend import make_local_runner
    from medlens.api_contract import SearchRequest
    from medlens.api_service import JobService
    jobs = output / 'jobs'
    service = JobService(jobs, make_local_runner(root, root / '.cache/embeddings/bge-input-v1'))
    records, attempted, accepted, failure = [], [], [], None
    try:
        for query in rows:
            if source_hashes(root) != manifest['source_sha256']:
                raise ValueError('Source changed during the frozen run; remaining cases are not attempted')
            if not service.health()['accepting_requests']:
                failure = {'type': 'ServiceUnavailable', 'message': 'Remaining cases were not attempted because cleanup/storage health was uncertain.'}
                break
            print(query['query_id'] + ' start', flush=True)
            attempted.append(query['query_id'])
            save_new(output / ('attempt-' + query['query_id'] + '.json'),
                {'query_id': query['query_id'], 'started_at': now(), 'manifest_id': manifest['manifest_id']})
            request = SearchRequest(question=query['text'], limit=100, top_k=10,
                generate=True, answer_style='source-findings', pubmed_query=None)
            started = time.perf_counter()
            with MemorySampler() as memory:
                job = service.submit(request)
                accepted.append({'query_id': query['query_id'], 'request_id': job['request_id']})
                save_new(output / ('accepted-' + query['query_id'] + '.json'), accepted[-1])
                while True:
                    job = service.get(job['request_id'])
                    if job['status'] in {'succeeded', 'failed', 'interrupted'}:
                        break
                    time.sleep(0.25)
            request_wall_seconds = time.perf_counter() - started
            folder = jobs / job['request_id']
            if job.get('mvp_policy') != manifest['application_policy']:
                raise ValueError('Persisted job used a different central MVP policy')
            raw_path = folder / 'run/run.json'
            run = load(raw_path) if raw_path.is_file() else None
            audit_started = time.perf_counter()
            try:
                source_audit = audit_sources(run, job['result']) if run is not None else {'status': 'failed', 'error_type': 'MissingRunArtifact'}
            except Exception as exc:
                source_audit = {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}
            count = (run or {}).get('runtime_load_count')
            cleanup = type(count) is int and count >= 0 and (count == 0 or run.get('runtime_closed') is True)
            observation = memory.result()
            answer = (job.get('result') or {}).get('answer') or {}
            record = {'query_id': query['query_id'], 'group_id': query['group_id'],
                'request_id': job['request_id'], 'status': job['status'], 'answer_status': answer.get('status'),
                'error': job['error'], 'request_wall_seconds': request_wall_seconds,
                'post_request_source_audit_seconds': time.perf_counter() - audit_started,
                'pipeline_request_seconds': (run or {}).get('request_seconds'),
                'fresh_search_complete': ((run or {}).get('live') or {}).get('fresh_search_complete'),
                'fetched_articles': ((run or {}).get('live') or {}).get('fetched_article_count'),
                'passage_cache': ((run or {}).get('live') or {}).get('passage_cache'),
                'runtime_load_count': count, 'runtime_closed': (run or {}).get('runtime_closed'),
                'runtime_cleanup_confirmed': cleanup, 'source_audit': source_audit,
                'memory': observation,
                'pipeline_memory_samples': (run or {}).get('memory_samples'),
                'pipeline_memory_scope': (run or {}).get('memory_scope'),
                'independent_validation': False,
                'job_sha256': file_sha(folder / 'job.json'), 'run_sha256': file_sha(raw_path) if run is not None else None}
            record['operational_pass'] = (job['status'] == 'succeeded'
                and record['fresh_search_complete'] is True and source_audit['status'] == 'passed'
                and cleanup and observation['observed_native_workers_all_reaped'] is not False)
            save_new(output / ('case-' + query['query_id'] + '.json'), record)
            records.append(record)
            print(query['query_id'] + ' ' + job['status'] + ' ' + str(answer.get('status')), flush=True)
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        with ignore_sigint():
            service.close()
        try:
            frozen_sources_unchanged = source_hashes(root) == manifest['source_sha256']
        except OSError:
            frozen_sources_unchanged = False
        if not frozen_sources_unchanged and failure is None:
            failure = {'type': 'SourceDrift', 'message': 'Frozen source identity changed before final acceptance.'}
        summary = {'schema_version': 1, 'kind': 'mvp-operational-validation-summary',
            'finished_at': now(), 'manifest_id': manifest['manifest_id'],
            'expected_queries': 12, 'attempted_queries': len(attempted), 'recorded_queries': len(records),
            'accepted_jobs': accepted,
            'accepted_job_count': len(accepted),
            'attempted_query_ids': attempted,
            'unattempted_query_ids': [r['query_id'] for r in rows if r['query_id'] not in attempted],
            'complete': len(records) == 12 and failure is None,
            'operational_pass': len(records) == 12 and all(r['operational_pass'] for r in records) and failure is None,
            'failure': failure, 'case_records': ['case-' + r['query_id'] + '.json' for r in records],
            'frozen_sources_unchanged': frozen_sources_unchanged,
            'answer_statuses': {s: sum(str(r['answer_status']) == s for r in records) for s in sorted({str(r['answer_status']) for r in records})},
            'reference_coverage': {'retrieval_judgments': None, 'answer_judgments': None,
                'retrieval_quality_metrics': None, 'independent_answer_quality': None},
            'independent_validation': False, 'clinical_validation': False,
            'quality_gate': 'not-established-without-new-finalized-human-reference',
            'release_quality_conditions_satisfied': False,
            'interpretation': 'One frozen operational confirmation, not model selection, a quality estimate, an SLA or independent clinical validation.'}
        save_new(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    try:
        result = freeze(root, output) if args.action == 'freeze' else execute(root, output)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print('Validation stopped: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    return 0 if args.action == 'freeze' or result['operational_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
