"""Prospective fixed-20 AI diagnostic: freeze, one CLI pass, reconcile, review.

No semantic confirmation inputs are parsed by freeze. Source/answer labels are
scoring-only inputs after the application is frozen and its live pass finishes.
This tool never turns assistant judgments into independent human validation.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, RLock, Thread
import time

import validate_mvp as legacy

PLAN = 'benchmarks/human-confirmation-v1/plan.json'
PLAN_SHA256 = 'b830b3e9afd4d376f78f1eb9c7c90820782fa25cb6beed32115529bd17af6dbd'
COUNT = 20
ORIGIN = 'ai'
YES_NO = {'yes', 'no', 'unknown'}
CORE_FACETS = ('population_system', 'intervention_comparison', 'outcome_target')
SUPPORT_FACETS = ('factual_support', 'qualifiers_preserved')
CARD_FACETS = (*CORE_FACETS, *SUPPORT_FACETS)
POLICY = {'selected_path': 'actual-cli-subprocess', 'limit': 100, 'top_k': 10,
    'generate': True, 'answer_style': 'source-findings', 'pubmed_query': None,
    'order': 'unchanged-plan-order', 'attempts_per_case': 1,
    'harness_retries': False, 'outer_watchdog_seconds': 90,
    'outer_cleanup_seconds': 5,
    'wall_limit_seconds': 60, 'useful_fraction_min': 0.5,
    'false_abstention_fraction_max': 0.5,
    'reference_origin': 'assistant; never independent human annotation',
    'cache_policy': 'existing immutable passage cache; record actual cache observations',
    'network_retries': 'only frozen application transport retries within the same request'}
STRESS = {'inputs': 'previously observed development material only',
    'candidate_limits': [1, 300], 'top_k': [1, 10],
    'passage_cache': ['cold-isolated-directory', 'warm-repeat-same-development-case'],
    'matrix': 'full boundary cross product; independent of the 20 confirmation cases',
    'source_sizes': 'record actual fetched article and source-text bytes; do not infer a full size bound',
    'long_session': {'requests_min': 20, 'inputs': 'old development questions only',
        'measure': 'parent and child current RSS separately, native cleanup after each request',
        'claim_limit': 'finite observed series; no proof of leak freedom'}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Require a regular JSON file: ' + str(path))
    def pairs(rows):
        value = {}
        for key, item in rows:
            require(key not in value, 'Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def source_hashes(root):
    paths = sorted((root / 'src/medlens').rglob('*.py'))
    paths += [root / 'pyproject.toml', root / 'scripts/validate_mvp.py',
        root / 'scripts/validate_confirmation.py', root / 'scripts/prepare_models.py',
        root / 'scripts/prepare_mlx.py', root / 'scripts/prepare_mlx_4b.py']
    paths += sorted(root.glob('requirements*.lock'))
    paths += sorted((root / 'configs').rglob('*.json'))
    return {str(path.relative_to(root)): legacy.file_sha(path) for path in paths}


def application_policy(root):
    spec = importlib.util.spec_from_file_location('_confirmation_policy', root / 'src/medlens/mvp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = module.policy_snapshot()
    require((value['source']['candidate_limit']['default'], value['retrieval']['top_k']['default'],
        value['answer']['style'], value['answer']['model_profile'], value['answer']['engine']) ==
        (100, 10, 'source-findings', '4b', 'mlx'), 'Actual selected application policy differs')
    return value


def freeze(root, output):
    """Hash the original plan as opaque bytes; never deserialize cases here."""
    require(legacy.file_sha(root / PLAN) == PLAN_SHA256, 'Original fixed-20 plan bytes differ')
    require(not output.exists() and not output.is_symlink(), 'Choose a fresh confirmation directory')
    selector = legacy.constants(root / 'src/medlens/generation/source_findings.py',
        {'SELECTOR_SYSTEM', 'PROMPT_VERSION', 'ANSWER_POLICY', 'SELECTION_CAP', 'MAX_FINDINGS',
         'MAX_QUOTE_CHARS', 'MAX_QUOTES_PER_FINDING'})
    prompt = selector.pop('SELECTOR_SYSTEM')
    value = {'schema_version': 1, 'kind': 'prospective-fixed20-ai-confirmation',
        'created_at': legacy.now(), 'root': str(root), 'policy': POLICY,
        'plan': {'path': PLAN, 'sha256': PLAN_SHA256, 'expected_count': COUNT,
            'semantic_cases_parsed': False},
        'source_sha256': source_hashes(root), 'application_policy': application_policy(root),
        'selector': {**selector, 'system_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()},
        'model_profile': legacy.constants(root / 'src/medlens/inference/profiles.py', {'PROFILES'})['PROFILES']['4b'],
        'assets': legacy.asset_identity(root), 'parent_packages': legacy.packages(),
        'parent_python': sys.version, 'parent_executable': str(Path(sys.executable).resolve()),
        'worker_environment': legacy.worker_environment(root),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'planned_development_stress': STRESS,
        'prefreeze_exposure': 'Operator must declare prior exposure separately; a hash is not proof of blindness.',
        'acceptance': {'useful': 'explicit actual-output review yes / all 20 >= .5',
            'false_abstention': 'unnecessary semantic abstentions / reconfirmed answerable yes <= .5; A=0 not established',
            'wall': 'every full CLI subprocess start-to-final-stdout-and-process-exit <=60 seconds',
            'critical_scope_errors': 'zero confirmed; unknown can block establishment',
            'source_integrity': 'all source/public projection audits pass',
            'native_cleanup': 'confirmed process reaping and owned process group absence',
            'independent_human_validation': False, 'clinical_validation': False}}
    value['manifest_id'] = legacy.digest(value)
    output.mkdir(parents=True, exist_ok=False)
    legacy.save_new(output / 'manifest.json', value)
    return {'status': 'frozen', 'manifest_id': value['manifest_id'], 'semantic_cases_parsed': False}


def verify_manifest(root, output, *, assets=False):
    value = load(output / 'manifest.json')
    require(value.get('manifest_id') == legacy.digest({k: v for k, v in value.items() if k != 'manifest_id'}), 'Manifest identity differs')
    require(value.get('root') == str(root) and value.get('policy') == POLICY, 'Frozen root/policy differs')
    require(value['plan']['sha256'] == PLAN_SHA256 and legacy.file_sha(root / PLAN) == PLAN_SHA256, 'Frozen plan differs')
    require(source_hashes(root) == value['source_sha256'], 'Application/harness/config source changed after freeze')
    require(application_policy(root) == value['application_policy'], 'Application policy changed')
    require(sys.version == value['parent_python'] and str(Path(sys.executable).resolve()) == value['parent_executable']
        and legacy.packages() == value['parent_packages'], 'Parent runtime changed')
    if assets:
        require(legacy.asset_identity(root) == value['assets'] and
            legacy.worker_environment(root) == value['worker_environment'], 'Model/tokenizer/native runtime identity changed')
    return value


def journal(output, event):
    """Append with exclusive creation on first write and durable flush each event."""
    path = output / 'journal.jsonl'
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'ab') as stream:
        stream.write(legacy.canonical({'at': legacy.now(), **event}) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def parse_plan(root):
    plan = load(root / PLAN)
    rows = plan.get('cases')
    require(plan.get('question_count') == COUNT and isinstance(rows, list) and len(rows) == COUNT,
        'Require the original twenty-question plan')
    ids = [row.get('query_id') for row in rows]
    import re
    require(len(set(ids)) == COUNT and all(isinstance(qid, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', qid) for qid in ids), 'Unsafe/duplicate case identity')
    require(all(isinstance(row.get('question'), str) and row['question'].strip() for row in rows), 'Missing planned question')
    return plan, rows


def cleanup_confirmed(run, observation):
    proof = (run or {}).get('native_worker_cleanup') or {}
    # A forcibly terminated request deliberately has unknown Python model-load
    # count. Physical parent reaping/group absence still proves native cleanup;
    # runtime_closed alone never does. Timeout remains an operational failure.
    return (proof.get('confirmed') is True and proof.get('worker_reaped') is True
        and proof.get('group_absent') is True and observation['observed_native_workers_all_reaped'] is not False)


def read_run_artifacts(job_path):
    """Keep partial bytes/diagnostics separate from authoritative supervisor proof."""
    values, hashes, errors = {}, {}, {}
    for name, path in {'run': job_path.parent / 'run/run.json',
            'worker_outcome': job_path.parent / 'worker-outcome.json'}.items():
        hashes[name] = legacy.file_sha(path) if path.is_file() and not path.is_symlink() else None
        if path.exists() or path.is_symlink():
            try:
                value = load(path)
                require(isinstance(value, dict), 'Run artifact must be an object')
                values[name] = value
            except (OSError, ValueError, TypeError) as exc:
                errors[name] = type(exc).__name__
    outcome_exists = (job_path.parent / 'worker-outcome.json').exists()
    outcome = values.get('worker_outcome') if outcome_exists else values.get('run')
    return values.get('run'), outcome, hashes, errors


class OwnedProcessTree:
    """Observe only descendants of a pinned CLI child and clean their owned sessions.

    PID+birth identity and current session membership are checked again before
    each signal. An old numeric PGID never grants authority to kill a new group.
    Sampling proves only the observed tree; it cannot prove no unseen fork ever
    escaped before its parent exited. Production worker cleanup remains separate.
    """
    def __init__(self, process, *, interval=.025):
        import psutil
        self.psutil, self.process, self.interval = psutil, process, interval
        self.lock, self.stop = RLock(), Event()
        self.members, self.groups, self.errors, self.signals = {}, {}, [], []
        self.root_seen = False
        try:
            root = psutil.Process(process.pid)
            self._capture(root, parent=None)
            self.root_seen = True
        except (psutil.Error, OSError, ValueError) as exc:
            self.errors.append(type(exc).__name__)
        self.observe()
        self.thread = Thread(target=self._loop, daemon=True)

    def _capture(self, process, *, parent):
        created, pid = process.create_time(), process.pid
        pgid, sid = os.getpgid(pid), os.getsid(pid)
        previous = self.members.get(pid)
        if previous and previous['created_at'] != created:
            raise ValueError('Observed PID was reused')
        self.members[pid] = {'pid': pid, 'created_at': created,
            'parent_observed_pid': parent, 'process_group': pgid, 'session': sid}
        # Only a verified new session can be group-signalled. A descendant in an
        # existing/foreign group can still be signalled individually by birth ID.
        if pid == pgid == sid and pgid > 1 and pgid != os.getpgrp():
            self.groups.setdefault(pgid, {'leader_pid': pid, 'leader_created_at': created, 'session': sid})

    def _current(self, member):
        try:
            process = self.psutil.Process(member['pid'])
            return process if process.create_time() == member['created_at'] else None
        except self.psutil.NoSuchProcess:
            return None
        except self.psutil.Error as exc:
            self.errors.append(type(exc).__name__)
            return None

    def observe(self):
        with self.lock:
            # A previously observed owned descendant remains an ancestry anchor
            # after the CLI exits; continue discovering its children during drain.
            root = self.members.get(self.process.pid)
            anchors = [root] if root and self._current(root) is not None else list(self.members.values())
            for member in anchors:
                parent = self._current(member)
                if parent is None:
                    continue
                try:
                    self._capture(parent, parent=member['parent_observed_pid'])
                    for child in parent.children(recursive=True):
                        self._capture(child, parent=parent.pid)
                except (self.psutil.NoSuchProcess, ProcessLookupError):
                    continue
                except (self.psutil.Error, OSError, ValueError) as exc:
                    self.errors.append(type(exc).__name__)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.observe()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        try:
            if args and args[0] is not None:
                # A harness interruption/pipe exception must not abandon the
                # already accepted CLI and its separately owned request group.
                with legacy.ignore_sigint():
                    self.cleanup(seconds=POLICY['outer_cleanup_seconds'])
        finally:
            self.stop.set()
            self.thread.join()
            self.observe()

    def _group_members(self, pgid):
        if pgid not in self.groups or pgid <= 1 or pgid == os.getpgrp():
            return []
        matching = []
        for member in list(self.members.values()):
            process = self._current(member)
            if process is None:
                continue
            try:
                if os.getpgid(process.pid) == pgid and os.getsid(process.pid) == self.groups[pgid]['session']:
                    matching.append(member)
            except ProcessLookupError:
                continue
            except OSError as exc:
                self.errors.append(type(exc).__name__)
        return matching

    def signal_owned(self, signum, *, include_cli=True):
        self.observe()
        with self.lock:
            signalled = set()
            # Signal child sessions before their CLI parent, giving the parent a
            # chance to reap its direct children before its own termination.
            for pgid in sorted(self.groups, key=lambda group: group == self.process.pid):
                if not include_cli and pgid == self.process.pid:
                    continue
                matching = self._group_members(pgid)  # fresh PID+birth+PGID+SID
                if not matching:
                    continue
                try:
                    os.killpg(pgid, signum)
                    self.signals.append({'signal': signum, 'process_group': pgid,
                        'verified_member_pids': [member['pid'] for member in matching]})
                    signalled.update(member['pid'] for member in matching)
                except ProcessLookupError:
                    continue
                except OSError as exc:
                    self.errors.append(type(exc).__name__)
            for member in list(self.members.values()):
                if member['pid'] in signalled or member['pid'] == os.getpid():
                    continue
                if not include_cli and member['pid'] == self.process.pid:
                    continue
                process = self._current(member)
                if process is None:
                    continue
                try:
                    # psutil.send_signal additionally protects against PID reuse.
                    process.send_signal(signum)
                    self.signals.append({'signal': signum, 'pid': member['pid'],
                        'created_at': member['created_at'], 'individual_signal': True})
                except self.psutil.NoSuchProcess:
                    continue
                except self.psutil.Error as exc:
                    self.errors.append(type(exc).__name__)

    def result(self):
        self.process.poll()  # waitpid(WNOHANG) reaps our direct child when exited.
        self.observe()
        with self.lock:
            survivors = []
            for member in self.members.values():
                process = self._current(member)
                if process is not None:
                    try:
                        survivors.append({**member, 'status': process.status()})
                    except self.psutil.NoSuchProcess:
                        pass
                    except self.psutil.Error as exc:
                        self.errors.append(type(exc).__name__)
            remaining_groups = []
            group_probe_errors = {}
            for pgid in self.groups:
                try:
                    os.killpg(pgid, 0)  # absence probe only, never an authority to signal.
                    remaining_groups.append(pgid)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    remaining_groups.append(pgid)
                    # A transient EPERM while an owned group is exiting is a
                    # current unknown. A later ESRCH resolves this absence
                    # probe; ancestry/identity observation errors stay sticky.
                    group_probe_errors[pgid] = type(exc).__name__
            reaped = self.process.returncode is not None
            return {'confirmed_observed_tree_absent': self.root_seen and reaped and not survivors and not remaining_groups and not self.errors,
                'cli_reaped': reaped, 'owned_members': list(self.members.values()),
                'owned_sessions': self.groups, 'survivors': survivors,
                'remaining_process_groups': remaining_groups, 'signals': list(self.signals),
                'group_probe_errors': group_probe_errors,
                'observation_errors': sorted(set(self.errors)),
                'observation_interval_seconds': self.interval,
                'unobserved_descendant_absence_established': False,
                'scope': 'periodically observed descendants with PID+birth identity; orphan zombies are survivors, never claimed reaped'}

    def cleanup(self, *, seconds=5):
        started = time.monotonic()
        deadline = started + seconds
        # Keep the CLI alive first: its supervisor can reap the native request
        # leader, avoiding orphan zombies when the CLI itself is responsive.
        self.signal_owned(signal.SIGTERM, include_cli=False)
        term_deadline = min(started + .3, deadline)
        while time.monotonic() < term_deadline and not self.result()['confirmed_observed_tree_absent']:
            time.sleep(.01)
        if not self.result()['confirmed_observed_tree_absent']:
            self.signal_owned(signal.SIGKILL, include_cli=False)
            reap_grace = min(time.monotonic() + .15, deadline)
            while time.monotonic() < reap_grace:
                result = self.result()
                if not any(p['pid'] != self.process.pid for p in result['survivors']):
                    break
                time.sleep(.01)
            self.signal_owned(signal.SIGTERM)
            cli_grace = min(time.monotonic() + .3, deadline)
            while time.monotonic() < cli_grace and not self.result()['confirmed_observed_tree_absent']:
                time.sleep(.01)
        if not self.result()['confirmed_observed_tree_absent']:
            self.signal_owned(signal.SIGKILL)
        while time.monotonic() < deadline and not self.result()['confirmed_observed_tree_absent']:
            time.sleep(.01)
        result = self.result()
        result['cleanup_seconds'] = time.monotonic() - started
        result['forced'] = True
        return result


def execute_cli(root, directory, case, *, runner=None, sampler=None):
    """Actual public CLI. Injectable fixture runner exists solely for fictional tests."""
    sampler = sampler or legacy.MemorySampler
    directory.mkdir(parents=True, exist_ok=False)
    jobs = directory / 'jobs'
    command = [sys.executable, '-m', 'medlens.cli', '--question', case['question'],
        '--root', str(root), '--storage-dir', str(jobs), '--limit', '100', '--top-k', '10']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(root / 'src')
    started = time.perf_counter()
    forced = False
    delivered = True
    with sampler() as memory:
        process = (runner or subprocess.Popen)(command, cwd=root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        with OwnedProcessTree(process) as owned:
            try:
                stdout, stderr = process.communicate(timeout=POLICY['outer_watchdog_seconds'])
                outer_cleanup = owned.result()
                if not outer_cleanup['confirmed_observed_tree_absent']:
                    outer_cleanup = owned.cleanup(seconds=POLICY['outer_cleanup_seconds'])
            except subprocess.TimeoutExpired:
                forced = True
                outer_cleanup = owned.cleanup(seconds=POLICY['outer_cleanup_seconds'])
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except subprocess.TimeoutExpired as exc:
                    delivered = False
                    stdout, stderr = exc.output or b'', exc.stderr or b''
    elapsed = time.perf_counter() - started
    (directory / 'stdout.bin').write_bytes(stdout)
    (directory / 'stderr.bin').write_bytes(stderr)
    observation = memory.result()
    paths = sorted(jobs.glob('*/job.json')) if jobs.is_dir() else []
    job, run, outcome, error = None, None, None, None
    source_audit = {'status': 'unknown', 'error_type': 'MissingTerminalArtifact'}
    artifact_hashes, artifact_errors = {}, {}
    delivery = False
    if len(paths) == 1:
        job = load(paths[0])
        run_path = paths[0].parent / 'run/run.json'
        outcome_path = paths[0].parent / 'worker-outcome.json'
        run, outcome, artifact_hashes, artifact_errors = read_run_artifacts(paths[0])
        if outcome is not None:
            try:
                source_audit = legacy.audit_sources(outcome, job.get('result'))
            except Exception as exc:
                source_audit = {'status': 'failed', 'error_type': type(exc).__name__, 'message': str(exc)}
        from medlens.cli_output import render_result
        delivery = render_result(job).encode('utf-8') in stdout
        error = job.get('error')
    answer = ((job or {}).get('result') or {}).get('answer') or {}
    outer_forced = forced or bool(outer_cleanup.get('forced'))
    clean = cleanup_confirmed(outcome, observation) and outer_cleanup['confirmed_observed_tree_absent']
    result = {'query_id': case['query_id'], 'question_sha256': legacy.digest(case['question']),
        'request_id': (job or {}).get('request_id'), 'job_status': (job or {}).get('status'),
        'answer_status': answer.get('status'), 'error': error, 'returncode': process.returncode,
        'full_wall_seconds': elapsed, 'wall_pass': elapsed <= 60 and not outer_forced,
        'wall_scope': 'before CLI process creation including startup, validation and admission through full stdout/stderr receipt and process exit after service/native cleanup',
        'harness_post_request_audit_included': False, 'stdout_delivery_matches_terminal_public_result': delivery,
        'harness_process_observation_included': True,
        'stdout_sha256': legacy.file_sha(directory / 'stdout.bin'),
        'stderr_sha256': legacy.file_sha(directory / 'stderr.bin'),
        'watchdog_forced_kill': forced, 'native_cleanup_confirmed': clean,
        'outer_forced_cleanup': outer_forced, 'outer_cleanup': outer_cleanup,
        'stdout_stderr_receipt_complete': delivered,
        'native_worker_cleanup': (outcome or {}).get('native_worker_cleanup'), 'memory': observation,
        'service_timing': (job or {}).get('timing'),
        'source_audit': source_audit,
        'fresh_search_complete': ((run or {}).get('live') or {}).get('fresh_search_complete'),
        'passage_cache': ((run or {}).get('live') or {}).get('passage_cache'),
        'job_path': str(paths[0].relative_to(directory)) if len(paths) == 1 else None,
        'job_sha256': legacy.file_sha(paths[0]) if len(paths) == 1 else None,
        'run_sha256': artifact_hashes.get('run'),
        'worker_outcome_sha256': artifact_hashes.get('worker_outcome'),
        'artifact_read_errors': artifact_errors,
        'application_policy_matches_freeze': (job or {}).get('mvp_policy') == application_policy(root)}
    result['operational_pass'] = ((job or {}).get('status') == 'succeeded' and process.returncode == 0
        and delivery and delivered and clean and not outer_forced and source_audit['status'] == 'passed'
        and result['fresh_search_complete'] is True and result['application_policy_matches_freeze']
        and not artifact_errors and run is not None)
    return result


def execute(root, output):
    manifest = verify_manifest(root, output, assets=True)
    # Exclusive durable marker is consumed BEFORE parsing the semantic question plan.
    legacy.save_new(output / 'execution-started.json', {'manifest_id': manifest['manifest_id'],
        'started_at': legacy.now(), 'resume_force_retry_permitted': False})
    journal(output, {'event': 'execution-started', 'manifest_id': manifest['manifest_id'], 'semantic_plan_parsed': False})
    records, rows, failure = [], [], None
    try:
        _, rows = parse_plan(root)
        for case in rows:
            verify_manifest(root, output)
            qid = case['query_id']
            journal(output, {'event': 'case-started', 'query_id': qid, 'attempt': 1,
                'question_sha256': legacy.digest(case['question'])})
            print(qid + ' start', flush=True)
            record = execute_cli(root, output / 'cases' / qid, case)
            try:
                verify_manifest(root, output)
                record['frozen_source_unchanged_after'] = True
            except (OSError, ValueError):
                record['frozen_source_unchanged_after'] = False
                raise
            finally:
                legacy.save_new(output / ('case-' + qid + '.json'), record)
                records.append(record)
                journal(output, {'event': 'case-finished', 'query_id': qid,
                    'record_sha256': legacy.file_sha(output / ('case-' + qid + '.json'))})
            print(qid + ' ' + str(record['job_status']) + ' ' + str(round(record['full_wall_seconds'], 3)), flush=True)
            require(record['native_cleanup_confirmed'], 'Unsafe/unknown native cleanup; preserve remaining cases as unattempted failures')
    except BaseException as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        journal(output, {'event': 'execution-stopped', 'error_type': type(exc).__name__})
        raise
    finally:
        summary = {'schema_version': 1, 'kind': 'fixed20-live-pass', 'manifest_id': manifest['manifest_id'],
            'finished_at': legacy.now(), 'planned_count': COUNT, 'recorded_count': len(records),
            'planned_query_ids': [r['query_id'] for r in rows], 'recorded_query_ids': [r['query_id'] for r in records],
            'unattempted_query_ids': [r['query_id'] for r in rows if r['query_id'] not in {v['query_id'] for v in records}],
            'complete': len(records) == COUNT and failure is None, 'failure': failure,
            'operational_pass': len(records) == COUNT and failure is None and all(r['operational_pass'] for r in records),
            'timing_pass': len(records) == COUNT and all(r['wall_pass'] for r in records),
            'records': {r['query_id']: legacy.file_sha(output / ('case-' + r['query_id'] + '.json')) for r in records},
            'quality_gate': 'not-established-pending-explicit-actual-output-review',
            'independent_human_validation': False, 'clinical_validation': False}
        legacy.save_new(output / 'live-summary.json', summary)
    return summary


def source_key(item):
    """Bind exact question/text/version without XML position or acquisition metadata.

    Moving an unchanged article within EFetch changes source_path but not its
    reviewed content. Preserve section order, labels, raw/normalized text and
    abstract attributes (including OtherAbstract language/type), not copyright.
    """
    def abstract_text(value):
        if value is None:
            return None
        return {'attributes': value.get('attributes', {}), 'sections': [
            {key: section.get(key) for key in ('label', 'nlm_category', 'text', 'raw_text')}
            for section in value.get('sections', [])]}
    value = {key: item.get(key) for key in ('query_id', 'question', 'pmid',
        'title', 'raw_title', 'abstract_status', 'text_version_id')}
    value['abstract'] = abstract_text(item.get('abstract'))
    value['other_abstracts'] = [abstract_text(a) for a in item.get('other_abstracts', [])]
    return legacy.digest(value)


def actual_sources(root, output):
    """Reconstruct the real PubMed/Dense union and verify original XML text."""
    from medlens.evaluation.biomedical_annotation import _pool
    from medlens.ingestion.snapshots import load_snapshot
    from medlens.parsing.dataset import load_dataset
    from medlens.parsing.pubmed import parse_pubmed_xml
    _, cases = parse_plan(root)
    items, bindings = [], []
    summary = load(output / 'live-summary.json')
    for case in cases:
        qid = case['query_id']
        def unavailable(state, error):
            bindings.append({'query_id': qid, 'question_sha256': legacy.digest(case['question']),
                'source_keys': [], 'pool_state': state, 'pool_error': error})
        record_path = output / ('case-' + qid + '.json')
        if not record_path.is_file():
            unavailable('unavailable', 'UnattemptedOrMissingCaseRecord')
            continue
        record = load(record_path)
        require(legacy.file_sha(record_path) == summary['records'][qid], 'Live case changed')
        directory = output / 'cases' / qid
        job_path = directory / (record['job_path'] or '__missing__')
        if not job_path.is_file() or legacy.file_sha(job_path) != record['job_sha256']:
            unavailable('integrity_failed', 'LiveJobChangedOrMissing')
            continue
        run_path = job_path.parent / 'run/run.json'
        if record['source_audit']['status'] == 'failed':
            unavailable('integrity_failed', record['source_audit'].get('error_type', 'LiveSourceAuditFailed'))
            continue
        if record.get('run_sha256') is not None and (not run_path.is_file() or legacy.file_sha(run_path) != record['run_sha256']):
            unavailable('integrity_failed', 'LivePrivateRunChangedOrMissing')
            continue
        if (record.get('job_status') != 'succeeded' or not record.get('operational_pass')
                or record['source_audit']['status'] != 'passed'):
            unavailable('unavailable', 'OperationalRequestIncompleteOrFailed')
            continue
        if record.get('run_sha256') is None:
            unavailable('integrity_failed', 'SuccessfulRequestMissingPrivateRun')
            continue
        try:
            run = load(run_path)
        except (OSError, ValueError, TypeError):
            unavailable('integrity_failed', 'SuccessfulRequestInvalidPrivateRun')
            continue
        if ((run.get('live') or {}).get('fresh_search_complete') is not True
                or not isinstance(run.get('retrieval'), dict) or not isinstance(run['retrieval'].get('results'), list)):
            unavailable('integrity_failed', 'SuccessfulRequestMissingSourcePool')
            continue
        try:
            require(run.get('question') == case['question'], 'Live question binding differs')
            snapshot = load_snapshot(run_path.parent / 'snapshot')
            dataset = load_dataset(run['input_paths']['dataset'])
            require(dataset.manifest['source_snapshot_id'] == snapshot.manifest['snapshot_id'], 'Actual source snapshot binding differs')
            parsed = {}
            for response in snapshot.manifest['responses'][1:]:
                for article in parse_pubmed_xml(snapshot.responses[response['file']], source_file=response['file'],
                        snapshot_id=snapshot.manifest['snapshot_id'], retrieved_at=response['retrieved_at']):
                    require(article.pmid not in parsed, 'Duplicate actual raw source PMID')
                    parsed[article.pmid] = asdict(article)
            require(list(dataset.articles) == [json.loads(legacy.canonical(parsed[p])) for p in snapshot.manifest['pmids']], 'Actual source text differs from original XML')
            selected, _ = _pool(case, run, snapshot, dataset, depth=10, plan_sha=PLAN_SHA256) if snapshot.manifest['pmids'] else ([], {})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            unavailable('integrity_failed', 'VerifiedSourceArtifact:' + type(exc).__name__)
            continue
        for item in selected:
            item['source_key'] = source_key(item)
        items += selected
        bindings.append({'query_id': qid, 'question_sha256': legacy.digest(case['question']),
            'source_keys': sorted(i['source_key'] for i in selected), 'pool_state': 'available'})
    return items, bindings


def reconcile_values(old_items, old_queries, labels, actual_items, actual_bindings):
    """Pure mapping: surviving article labels, blank deltas and pool reconfirmation."""
    require(labels.get('judgments_origin') == ORIGIN and labels.get('human_annotation') is False,
        'Only explicitly assistant-origin preparation labels are accepted; never relabel human/AI origin')
    require(labels.get('status') == 'complete-ai-assessment', 'Preparation reference must be complete AI assessment')
    require(labels.get('queries') == old_queries, 'Preparation AI question text/order differs')
    old = {source_key(item): item for item in old_items}
    require(len(old) == len(old_items), 'Duplicate preparation source identity')
    by_item = {row['item_id']: row for row in labels['article_judgments']}
    require(len(by_item) == len(labels['article_judgments']) and set(by_item) == {i['item_id'] for i in old_items}, 'Preparation labels do not cover exactly the old pool')
    for item in old_items:
        label = by_item[item['item_id']]
        require(all(label.get(k) == item[k] for k in ('query_id', 'pmid', 'text_version_id', 'item_sha256')), 'Preparation article label/source binding differs')
        require(label.get('state') in {'rated', 'unjudgeable'} and
            (type(label.get('relevance')) is int and label['relevance'] in (0, 1, 2) if label['state'] == 'rated' else label.get('relevance') is None), 'Incomplete/invalid preparation article labels')
    old_pools = {row['query_id']: sorted(source_key(i) for i in old_items if i['query_id'] == row['query_id']) for row in old_queries}
    old_q = {row['query_id']: row for row in old_queries}
    qlabels = {row['query_id']: row for row in labels['query_judgments']}
    require(set(qlabels) == set(old_q) and len(qlabels) == len(labels['query_judgments']), 'Preparation answerability coverage differs')
    for qid, query in old_q.items():
        require(qlabels[qid].get('state') in YES_NO, 'Preparation answerability finalization differs')
    mapped, delta, queries = [], [], []
    for item in actual_items:
        key = source_key(item)
        if key in old:
            before = old[key]
            label = by_item[before['item_id']]
            mapped.append({'source_key': key, 'old_item_id': before['item_id'], 'new_item_id': item['item_id'],
                'new_item_sha256': item['item_sha256'], 'query_id': item['query_id'],
                'state': label['state'], 'relevance': label['relevance'], 'rationale': label.get('rationale', ''), 'judgments_origin': ORIGIN})
        else:
            delta.append({**item, 'state': 'pending', 'relevance': None, 'rationale': '', 'judgments_origin': None})
    for binding in actual_bindings:
        qid = binding['query_id']
        require(qid in old_q and binding['question_sha256'] == legacy.digest(old_q[qid]['question']), 'Actual/preparation question identity differs')
        changed = binding['pool_state'] != 'available' or old_pools[qid] != sorted(binding['source_keys'])
        queries.append({**binding, 'pool_changed': changed,
            'state': 'pending' if changed else qlabels[qid]['state'],
            'supporting_pmids': [] if changed else [next(i['pmid'] for i in old_items if i['item_id'] == iid) for iid in qlabels[qid].get('supporting_item_ids', [])],
            'rationale': '' if changed else qlabels[qid].get('rationale', ''),
            'judgments_origin': None if changed else ORIGIN})
    return {'mapped': mapped, 'delta_items': delta, 'queries': queries,
        'rule': 'question+PMID+exact title/abstract+text-version; timestamp is not pool membership',
        'answerability_scope': 'actual-frozen-PubMed-BestMatch10-union-Dense10',
        'source_labels_establish_output_usefulness': False}


def reconcile(root, output, reference, labels_path, delta):
    verify_manifest(root, output)
    require(not delta.exists() and not delta.is_symlink(), 'Choose a fresh reconciliation directory')
    from medlens.evaluation.confirmation_annotation import load_packet
    packet = load_packet(reference)
    require(packet.package['plan_sha256'] == PLAN_SHA256, 'Preparation package plan differs')
    labels = load(labels_path)
    require(labels.get('source_package_id') == packet.package['package_id'] and
        labels.get('source_package_sha256') == legacy.file_sha(Path(reference) / 'package.json') and
        all(labels.get(k) == packet.package[k] for k in ('plan_sha256', 'items_sha256', 'source_bindings_sha256', 'query_bindings_sha256')), 'Preparation AI labels/package differs')
    require(labels.get('artifact_id') == legacy.digest({k: v for k, v in labels.items() if k != 'artifact_id'}), 'Preparation AI artifact identity differs')
    items, bindings = actual_sources(root, output)
    value = reconcile_values(packet.package['items'], packet.package['queries'], labels, items, bindings)
    value.update({'schema_version': 1, 'kind': 'fixed20-source-reconciliation',
        'manifest_id': load(output / 'manifest.json')['manifest_id'],
        'live_summary_sha256': legacy.file_sha(output / 'live-summary.json'),
        'preparation_package_id': packet.package['package_id'], 'preparation_labels_sha256': legacy.file_sha(labels_path),
        'actual_items': items, 'judgments_origin': ORIGIN, 'human_authored': False})
    value['reconciliation_id'] = legacy.digest(value)
    delta.mkdir(parents=True, exist_ok=False)
    legacy.save_new(delta / 'reconciliation.json', value)
    (delta / 'original-preparation-ai-labels.json').write_bytes(Path(labels_path).read_bytes())
    (delta / 'original-preparation-ai-labels.json').chmod(0o400)
    # Source review deliberately contains no generated answer or model abstention.
    legacy.save_new(delta / 'delta-source-review.json', {'kind': 'assistant-source-delta-review',
        'reconciliation_id': value['reconciliation_id'], 'judgments_origin': ORIGIN,
        'human_authored': False, 'finalized': False, 'items': value['delta_items'],
        'queries': value['queries']})
    return {'status': 'reconciled', 'reconciliation_id': value['reconciliation_id'],
        'mapped_articles': len(value['mapped']), 'blank_new_or_changed_articles': len(value['delta_items']),
        'answerability_reconfirmation_required': sum(q['pool_changed'] for q in value['queries'])}


def card_binding(claim):
    return legacy.digest(claim)


def validate_card_review(cards, critical_scope_mismatch):
    """Check only consistency of explicit AI decisions; infer no scientific label.

    A known core mismatch must remain a critical mismatch even if the query is
    judged not useful. Unknown core applicability cannot certify no mismatch.
    Support/qualifier failures have their own release gate because they need
    not be a population/intervention/outcome mismatch.
    """
    require(critical_scope_mismatch in YES_NO, 'Incomplete critical scope review')
    require(isinstance(cards, list), 'Incomplete card review')
    for card in cards:
        require(all(card.get(field) in YES_NO for field in CARD_FACETS), 'Incomplete card review')
    core_no = [index for index, card in enumerate(cards) if any(card[field] == 'no' for field in CORE_FACETS)]
    core_unknown = [index for index, card in enumerate(cards) if any(card[field] == 'unknown' for field in CORE_FACETS)]
    if core_no:
        require(critical_scope_mismatch == 'yes', 'A displayed card with a core scope no requires explicit critical scope mismatch yes')
    elif core_unknown:
        require(critical_scope_mismatch != 'no', 'Unknown displayed core scope cannot certify critical scope mismatch no')
    support_no = [index for index, card in enumerate(cards) if any(card[field] == 'no' for field in SUPPORT_FACETS)]
    support_unknown = [index for index, card in enumerate(cards) if any(card[field] == 'unknown' for field in SUPPORT_FACETS)]
    return {'gate': 'fail' if support_no else 'not_established' if support_unknown else 'pass',
        'core_no_card_indices': core_no, 'core_unknown_card_indices': core_unknown,
        'support_no_card_indices': support_no, 'support_unknown_card_indices': support_unknown,
        'card_index_base': 0}


def finalize_sources(reconciliation, review_path):
    """Freeze explicit AI delta grades/answerability before exposing actual outputs."""
    rec = load(reconciliation / 'reconciliation.json')
    require(rec.get('reconciliation_id') == legacy.digest({k: v for k, v in rec.items() if k != 'reconciliation_id'}),
        'Reconciliation receipt changed')
    review = load(review_path)
    require(review.get('kind') == 'assistant-source-delta-review' and
        review.get('reconciliation_id') == rec['reconciliation_id'] and
        review.get('judgments_origin') == ORIGIN and review.get('human_authored') is False
        and review.get('finalized') is True, 'Require finalized explicit AI source delta review')
    expected = rec['delta_items']
    actual = review.get('items')
    require(isinstance(actual, list) and len(actual) == len(expected), 'Delta article coverage differs')
    for row, item in zip(actual, expected):
        require(all(row.get(k) == v for k, v in item.items() if k not in ('state', 'relevance', 'rationale', 'judgments_origin')),
            'Delta article text/source/order changed')
        require(row.get('state') in {'rated', 'unjudgeable'} and row.get('judgments_origin') == ORIGIN,
            'Delta article decision/origin pending')
        require((type(row.get('relevance')) is int and row['relevance'] in (0, 1, 2)) if row['state'] == 'rated'
            else row.get('relevance') is None, 'Invalid delta article grade')
    queries = review.get('queries')
    require(isinstance(queries, list) and len(queries) == len(rec['queries']), 'Delta answerability coverage differs')
    for row, query in zip(queries, rec['queries']):
        require(all(row.get(k) == query[k] for k in ('query_id', 'question_sha256', 'source_keys', 'pool_state', 'pool_changed')),
            'Delta answerability pool/question binding differs')
        require(row.get('state') in YES_NO and row.get('judgments_origin') == ORIGIN, 'Delta answerability pending/origin differs')
        if not query['pool_changed']:
            require(row['state'] == query['state'] and row.get('supporting_pmids') == query['supporting_pmids'],
                'Unchanged source reference cannot be silently revised')
        allowed = {i['pmid'] for i in rec['actual_items'] if i['query_id'] == row['query_id']}
        support = row.get('supporting_pmids')
        require(isinstance(support, list) and len(set(support)) == len(support) and set(support) <= allowed,
            'Answerability supporting sources differ from actual pool')
        require(row['state'] != 'yes' or bool(support), 'Answerable yes requires explicitly reviewed supporting PMID')
        if query['pool_state'] != 'available':
            require(row['state'] == 'unknown', 'Unavailable source pool requires explicit unknown')
    result = {'kind': 'fixed20-finalized-ai-source-reference', 'reconciliation_id': rec['reconciliation_id'],
        'judgments_origin': ORIGIN, 'human_authored': False, 'finalized': True,
        'original_delta_review_sha256': legacy.file_sha(review_path), 'queries': queries,
        'new_article_judgments': actual, 'preserved_article_judgments': rec['mapped']}
    result['source_reference_id'] = legacy.digest(result)
    legacy.save_new(reconciliation / 'source-reference.json', result)
    original = reconciliation / 'original-ai-delta-source-review.json'
    with original.open('xb') as stream:
        stream.write(Path(review_path).read_bytes())
    original.chmod(0o400)
    return {'status': 'ai-source-reference-frozen', 'source_reference_id': result['source_reference_id']}


def review_template(output, reconciliation, destination):
    """Blank explicit review of actual output cards; never infer semantic labels."""
    value = load(reconciliation / 'reconciliation.json')
    source_reference = load(reconciliation / 'source-reference.json')
    require(source_reference['reconciliation_id'] == value['reconciliation_id'] and
        source_reference.get('finalized') is True, 'Finalize source deltas before output review')
    summary = load(output / 'live-summary.json')
    require(value['live_summary_sha256'] == legacy.file_sha(output / 'live-summary.json'), 'Reconciliation/live pass differs')
    rows = []
    for query in value['queries']:
        qid = query['query_id']
        record = load(output / ('case-' + qid + '.json')) if qid in summary['records'] else None
        job = load(output / 'cases' / qid / record['job_path']) if record and record['job_path'] else None
        claims = (((job or {}).get('result') or {}).get('answer') or {}).get('claims', [])
        rows.append({'query_id': qid, 'question_sha256': query['question_sha256'],
            'case_record_sha256': summary['records'].get(qid),
            'source_pool_sha256': legacy.digest(query['source_keys']),
            'useful_selection': 'unknown', 'critical_scope_mismatch': 'unknown',
            'answerable': 'unknown', 'unnecessary_abstention': 'unknown', 'rationale': '',
            'cards': [{'claim_sha256': card_binding(claim), 'factual_support': 'unknown',
                'population_system': 'unknown', 'intervention_comparison': 'unknown',
                'outcome_target': 'unknown', 'qualifiers_preserved': 'unknown', 'rationale': ''} for claim in claims]})
    template = {'schema_version': 1, 'kind': 'fixed20-assistant-actual-output-review',
        'manifest_id': value['manifest_id'], 'reconciliation_id': value['reconciliation_id'],
        'source_reference_id': source_reference['source_reference_id'],
        'judgments_origin': ORIGIN, 'human_authored': False,
        'independent_human_validation': False, 'clinical_validation': False,
        'finalized': False, 'finalized_at': None, 'queries': rows}
    legacy.save_new(destination, template)
    return template


def ratio_gate(low, high, threshold, *, minimum):
    if minimum:
        return 'pass' if low >= threshold else 'fail' if high < threshold else 'not_established'
    return 'pass' if high <= threshold else 'fail' if low > threshold else 'not_established'


def score_rows(rows, *, expected_count=COUNT):
    """Fixed denominator and exact possible-world abstention bounds for unknowns."""
    require(len(rows) == expected_count and len({r['query_id'] for r in rows}) == expected_count, 'Every planned case must remain in the quality denominator')
    useful_yes = sum(r['useful_selection'] == 'yes' for r in rows)
    useful_unknown = sum(r['useful_selection'] == 'unknown' for r in rows)
    a_yes = sum(r['answerable'] == 'yes' for r in rows)
    a_unknown = sum(r['answerable'] == 'unknown' for r in rows)
    false_yes = sum(r['unnecessary_abstention'] == 'yes' for r in rows)
    # For each question enumerate only logically possible A/F assignments.
    # F can be true only for a successful semantic insufficient_evidence result.
    states = {(0, 0)}
    for row in rows:
        choices = []
        for a in ([True, False] if row['answerable'] == 'unknown' else [row['answerable'] == 'yes']):
            # Definition of unnecessary semantic abstention is fixed by A and
            # actual terminal status. A reviewer cannot mark a known answerable
            # insufficient_evidence output as an appropriate abstention.
            f = a and row['semantic_abstention']
            if row['unnecessary_abstention'] == 'unknown' or f == (row['unnecessary_abstention'] == 'yes'):
                choices.append((int(a), int(f)))
        require(bool(choices), 'False-abstention yes contradicts answerability/actual semantic abstention')
        states = {(a+x, f+y) for a, f in states for x, y in choices}
    rates = [f/a for a, f in states if a > 0]
    zero_possible = any(a == 0 for a, _ in states)
    bounds = [min(rates), max(rates)] if rates else [None, None]
    abstention_gate = 'not_established' if not rates else ratio_gate(*bounds, .5, minimum=False)
    if zero_possible and abstention_gate == 'pass':
        abstention_gate = 'not_established'
    useful_bounds = [useful_yes/expected_count, (useful_yes+useful_unknown)/expected_count]
    critical_yes = sum(r['critical_scope_mismatch'] == 'yes' for r in rows)
    critical_unknown = sum(r['critical_scope_mismatch'] == 'unknown' for r in rows)
    return {'useful': {'yes': useful_yes, 'unknown': useful_unknown, 'denominator': expected_count,
            'rate': useful_yes/expected_count, 'bounds': useful_bounds,
            'gate': ratio_gate(*useful_bounds, .5, minimum=True)},
        'false_abstention': {'confirmed_count': false_yes, 'answerable_yes': a_yes,
            'answerable_unknown': a_unknown, 'rate': false_yes/a_yes if a_yes else None,
            'bounds': bounds, 'zero_denominator_possible': zero_possible, 'gate': abstention_gate},
        'critical_scope': {'yes': critical_yes, 'unknown': critical_unknown,
            'gate': 'fail' if critical_yes else 'not_established' if critical_unknown else 'pass'}}


def finalize(root, output, reconciliation, review_path):
    verify_manifest(root, output)
    rec = load(reconciliation / 'reconciliation.json')
    require(rec['reconciliation_id'] == legacy.digest({k: v for k, v in rec.items() if k != 'reconciliation_id'}), 'Reconciliation receipt changed')
    summary = load(output / 'live-summary.json')
    source_reference = load(reconciliation / 'source-reference.json')
    require(source_reference['source_reference_id'] == legacy.digest({k: v for k, v in source_reference.items() if k != 'source_reference_id'})
        and source_reference['reconciliation_id'] == rec['reconciliation_id'], 'Frozen source reference differs')
    reference_queries = {q['query_id']: q for q in source_reference['queries']}
    require(rec['live_summary_sha256'] == legacy.file_sha(output / 'live-summary.json'), 'Live summary changed since reconciliation')
    current_items, current_bindings = actual_sources(root, output)
    require(current_items == rec['actual_items'] and
        all(all(current.get(k) == previous.get(k) for k in ('query_id', 'question_sha256', 'source_keys', 'pool_state'))
            for current, previous in zip(current_bindings, rec['queries'])) and len(current_bindings) == len(rec['queries']),
        'Actual raw sources or reviewed source pools changed after reconciliation')
    review = load(review_path)
    require(review.get('kind') == 'fixed20-assistant-actual-output-review' and review.get('judgments_origin') == ORIGIN
        and review.get('human_authored') is False and review.get('finalized') is True
        and isinstance(review.get('finalized_at'), str) and bool(review['finalized_at'])
        and review.get('independent_human_validation') is False and review.get('clinical_validation') is False,
        'Require an explicitly finalized AI actual-output review; cannot manufacture independent validation')
    require(review.get('manifest_id') == rec['manifest_id'] and review.get('reconciliation_id') == rec['reconciliation_id'], 'Actual-output review/freeze/source binding differs')
    require(review.get('source_reference_id') == source_reference['source_reference_id'], 'Output review source-reference identity differs')
    by_query = {r['query_id']: r for r in review['queries']}
    require(len(by_query) == COUNT and len(review['queries']) == COUNT and set(by_query) == {q['query_id'] for q in rec['queries']}, 'Output review must cover exactly all twenty cases')
    rows, source_unknown, source_failed, native_unknown, timing_failure = [], [], [], [], []
    card_audits = {}
    for query in rec['queries']:
        qid, reviewed = query['query_id'], by_query[query['query_id']]
        require(reviewed.get('question_sha256') == query['question_sha256'] and reviewed.get('source_pool_sha256') == legacy.digest(query['source_keys'])
            and reviewed.get('case_record_sha256') == summary['records'].get(qid), 'Output review question/case/pool differs')
        for field in ('useful_selection', 'critical_scope_mismatch', 'answerable', 'unnecessary_abstention'):
            require(reviewed.get(field) in YES_NO, 'Unknown review decision: ' + field)
        require(reviewed['answerable'] == reference_queries[qid]['state'], 'Output review answerability differs from frozen source-only reference')
        record = load(output / ('case-' + qid + '.json')) if qid in summary['records'] else None
        if record:
            require(legacy.file_sha(output / ('case-' + qid + '.json')) == summary['records'][qid], 'Case record changed after live pass')
        job = load(output / 'cases' / qid / record['job_path']) if record and record['job_path'] else None
        if job:
            require(legacy.file_sha(output / 'cases' / qid / record['job_path']) == record['job_sha256'], 'Actual job changed after live pass')
        answer = (((job or {}).get('result') or {}).get('answer') or {})
        claims = answer.get('claims', [])
        cards = reviewed.get('cards')
        require(isinstance(cards, list) and [c['claim_sha256'] for c in cards] == [card_binding(c) for c in claims], 'Actual card review bindings/order differ')
        card_audits[qid] = validate_card_review(cards, reviewed['critical_scope_mismatch'])
        successful = bool(record and record['operational_pass'])
        semantic = successful and answer.get('status') == 'insufficient_evidence'
        if not successful:
            require(reviewed['useful_selection'] == 'no', 'An operationally failed request must be explicitly reviewed as not useful')
        if reviewed['useful_selection'] == 'yes':
            require(successful and claims and reviewed['critical_scope_mismatch'] == 'no'
                and all(all(c[field] == 'yes' for field in ('factual_support', 'population_system', 'intervention_comparison', 'outcome_target', 'qualifiers_preserved')) for c in cards), 'Useful yes requires a successful applicable supported actual output and no misleading card')
        if not semantic:
            require(reviewed['unnecessary_abstention'] == 'no', 'Operational failures/non-abstentions are not semantic abstentions')
        if reviewed['answerable'] == 'no':
            require(reviewed['unnecessary_abstention'] != 'yes', 'Unanswerable query cannot have false abstention yes')
        if reviewed['unnecessary_abstention'] == 'yes':
            require(reviewed['answerable'] == 'yes', 'Confirmed false abstention requires confirmed answerability')
        if semantic and reviewed['answerable'] == 'yes':
            require(reviewed['unnecessary_abstention'] == 'yes', 'Confirmed answerable semantic abstention must be explicitly marked unnecessary')
        if semantic and reviewed['answerable'] == 'unknown':
            require(reviewed['unnecessary_abstention'] == 'unknown', 'Unknown answerability must remain unknown for a semantic abstention')
        if query['pool_state'] != 'available' or not record or record['source_audit']['status'] != 'passed':
            source_unknown.append(qid)
            require(reviewed['useful_selection'] != 'yes' and reviewed['answerable'] == 'unknown', 'Missing source coverage cannot create positive usefulness/answerability')
        if query['pool_state'] == 'integrity_failed' or (record and record['source_audit']['status'] == 'failed'):
            source_failed.append(qid)
        if not record or not record['native_cleanup_confirmed']:
            native_unknown.append(qid)
        if not record or not record['wall_pass']:
            timing_failure.append(qid)
        rows.append({**reviewed, 'semantic_abstention': semantic})
    quality = score_rows(rows)
    operational = 'pass' if summary['operational_pass'] else 'fail'
    support_gates = [value['gate'] for value in card_audits.values()]
    gates = {'useful': quality['useful']['gate'], 'false_abstention': quality['false_abstention']['gate'],
        'critical_scope': quality['critical_scope']['gate'], 'timing': 'fail' if timing_failure else 'pass',
        'displayed_card_support': 'fail' if 'fail' in support_gates else 'not_established' if 'not_established' in support_gates else 'pass',
        'source_integrity': 'fail' if source_failed else 'not_established' if source_unknown else 'pass',
        'native_cleanup': 'not_established' if native_unknown else 'pass', 'operational': operational}
    result = {'schema_version': 1, 'kind': 'fixed20-ai-confirmation-report', 'manifest_id': rec['manifest_id'],
        'reconciliation_id': rec['reconciliation_id'], 'actual_output_review_sha256': legacy.file_sha(review_path),
        'judgments_origin': ORIGIN, 'human_authored': False, 'quality': quality, 'gates': gates,
        'overall_ai_gate': 'fail' if 'fail' in gates.values() else 'not_established' if 'not_established' in gates.values() else 'pass',
        'source_integrity_unknown_query_ids': source_unknown, 'native_cleanup_unknown_query_ids': native_unknown,
        'source_integrity_failed_query_ids': source_failed,
        'displayed_card_review_consistency': card_audits,
        'timing_failed_or_unattempted_query_ids': timing_failure,
        'full_wall_seconds': {qid: load(output / ('case-' + qid + '.json'))['full_wall_seconds'] for qid in summary['records']},
        'independent_human_validation': False, 'clinical_validation': False,
        'interpretation': 'AI diagnostic on one fixed sample; no independent human or clinical validation; operational success alone never establishes usefulness.'}
    legacy.save_new(output / 'ai-confirmation-report.json', result)
    original = output / 'original-ai-output-review.json'
    with original.open('xb') as stream:
        stream.write(Path(review_path).read_bytes())
    original.chmod(0o400)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'reconcile', 'finalize-sources', 'review-template', 'finalize'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--reconciliation', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args(argv)
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    sys.path.insert(0, str(root / 'src'))
    try:
        if args.action == 'freeze':
            result = freeze(root, output)
        elif args.action == 'run':
            result = execute(root, output)
        elif args.action == 'reconcile':
            require(all((args.reference, args.labels, args.reconciliation)), 'reconcile requires --reference --labels --reconciliation')
            result = reconcile(root, output, args.reference, args.labels, args.reconciliation)
        elif args.action == 'review-template':
            require(args.reconciliation and args.destination, 'review-template requires --reconciliation --destination')
            result = review_template(output, args.reconciliation, args.destination)
            result = {'status': 'blank-output-review-created', 'questions': len(result['queries']), 'decisions_prefilled': False}
        elif args.action == 'finalize-sources':
            require(args.reconciliation and args.review, 'finalize-sources requires --reconciliation --review')
            result = finalize_sources(args.reconciliation, args.review)
        else:
            require(args.reconciliation and args.review, 'finalize requires --reconciliation --review')
            result = finalize(root, output, args.reconciliation, args.review)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print('Confirmation stopped: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
