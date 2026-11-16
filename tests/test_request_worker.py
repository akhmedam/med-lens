"""Actual process-group/deadline checks with tiny Python workers and no ML/network."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from medlens.api_backend import make_local_runner
from medlens.api_contract import SearchRequest
from medlens.api_service import JobService, ServiceUnavailable
from medlens.request_deadline import RequestBudget
from medlens.request_worker import IsolatedRunner

QUESTION = "Does fictional intervention reduce symptoms?"
FAKE_WORKER = r'''import json, os, pathlib, signal, subprocess, sys, time
path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text())
mode = json.loads((pathlib.Path(value['root']) / 'mode.json').read_text())
(path.parent / 'worker-identity.json').write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))
(path.parent / 'worker-progress.json').write_text(json.dumps({'stage':'answer','status':'running'}))
output = pathlib.Path(value['output'])
output.mkdir()
(output / 'run.json').write_text('partial checkpoint')
if mode.get('hang'):
    # The normal TERM path reaps its descendant before exiting, just as MLX
    # close does. A second fixture verifies KILL when the worker ignores TERM.
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    (path.parent / 'descendant.pid').write_text(str(child.pid))
    def stop(*args):
        try:child.wait(timeout=.2)
        except subprocess.TimeoutExpired:child.kill(); child.wait()
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, stop)
    time.sleep(30)
if mode.get('ignore_term'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(30)
if mode.get('exit'):
    raise SystemExit(2)
if mode.get('wait'):
    time.sleep(30)
report = {'schema_version':1,'status':'complete','stage':'complete',
    'runtime_load_count':0,'question':value['request']['question'],
    'answer_style':value['request']['answer_style'],
    'answer':{'status':'insufficient_evidence' if value['request']['generate'] else 'not_requested','claims':[]}}
if 'report' in mode:report = mode['report']
report.update(mode.get('report_update', {}))
for key in mode.get('drop_fields', []):report.pop(key, None)
envelope = {'protocol':value['protocol'], 'nonce':value['nonce'],
    'request_sha256':value['request_sha256'], 'report':report}
envelope.update(mode.get('envelope_update', {}))
if not mode.get('missing_result'):
    (path.parent / 'worker-result.json').write_text(json.dumps(envelope if not mode.get('bare_report') else report))
'''


class RequestWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / 'fake-worker.py'
        self.script.write_text(FAKE_WORKER)
        self.mode({})
        self.request = SearchRequest(question=QUESTION)

    def mode(self, value):
        (self.root / 'mode.json').write_text(json.dumps(value))

    def runner(self):
        return IsolatedRunner(self.root, self.root / 'cache', worker_script=self.script)

    def budget(self):
        return RequestBudget(time.monotonic(), seconds=1.3, cleanup_reserve=.65, delivery_reserve=.15)

    def test_normal_worker_is_fresh_reaped_and_cleanup_evidence_is_durable(self):
        runner, pids = self.runner(), []
        for index in range(2):
            output = self.root / str(index) / 'run'
            report = runner.run_with_budget(self.request, output, None, self.budget())
            self.assertEqual(report['status'], 'complete')
            cleanup = report['native_worker_cleanup']
            self.assertTrue(cleanup['confirmed'])
            self.assertTrue(cleanup['worker_reaped'])
            self.assertTrue(cleanup['group_absent'])
            self.assertFalse(cleanup['forced'])
            pids.append(cleanup['worker_pid'])
            self.assertEqual(json.loads((output / 'run.json').read_text()), report)
            self.assertEqual(json.loads((output.parent / 'worker-outcome.json').read_text()), report)
            with self.assertRaises(ProcessLookupError):os.kill(cleanup['worker_pid'], 0)
        self.assertNotEqual(pids[0], pids[1])

    def test_candidate_and_topk_boundaries_share_budget_and_reach_fresh_child_verbatim(self):
        runner = self.runner()
        for limit in (1, 300):
            for top_k in (1, 10):
                with self.subTest(limit=limit, top_k=top_k):
                    request = SearchRequest(question=QUESTION, limit=limit, top_k=top_k)
                    output = self.root / f'boundary-{limit}-{top_k}' / 'run'
                    report = runner.run_with_budget(request, output, None, self.budget())
                    actual = json.loads((output.parent / 'worker-request.json').read_text())['request']
                    self.assertEqual((actual['limit'], actual['top_k']), (limit, top_k))
                    self.assertTrue(report['native_worker_cleanup']['confirmed'])
                    self.assertEqual(report['request_timing']['timeout_seconds'], 1.3)

    def test_timeout_kills_descendant_reaps_worker_and_preserves_partial_checkpoint(self):
        self.mode({'hang':True})
        stages = []
        budget = self.budget()
        output = self.root / 'timeout' / 'run'
        report = self.runner().run_with_budget(self.request, output,
            lambda stage, status:stages.append(stage), budget)
        self.assertLess(budget.elapsed(), budget.seconds)
        self.assertTrue(report['deadline_exceeded'])
        self.assertEqual(report['answer']['status'], 'not_completed')
        self.assertTrue(report['native_worker_cleanup']['confirmed'])
        self.assertTrue(report['native_worker_cleanup']['forced'])
        self.assertEqual((output / 'run.json').read_text(), 'partial checkpoint')
        self.assertIn('answer', stages)
        descendant = int((output.parent / 'descendant.pid').read_text())
        with self.assertRaises(ProcessLookupError):os.kill(descendant, 0)
        self.assertTrue(json.loads((output / 'native-worker-cleanup.json').read_text())['confirmed'])

    def test_uncooperative_worker_is_killed_before_cleanup_budget_expires(self):
        self.mode({'ignore_term':True})
        budget = self.budget()
        report = self.runner().run_with_budget(self.request, self.root / 'kill' / 'run', None, budget)
        self.assertLess(budget.elapsed(), budget.seconds)
        self.assertTrue(report['native_worker_cleanup']['confirmed'])
        self.assertTrue(report['deadline_exceeded'])

    def test_expired_transport_budget_does_not_launch_a_process(self):
        budget = RequestBudget(time.monotonic() - 61)
        runner = self.runner()
        with patch('medlens.request_worker.subprocess.Popen') as launch:
            report = runner.run_with_budget(self.request, self.root / 'expired' / 'run', None, budget)
        launch.assert_not_called()
        self.assertTrue(report['deadline_exceeded'])
        self.assertTrue(report['native_worker_cleanup']['confirmed'])

    def test_malformed_or_foreign_worker_report_never_becomes_success(self):
        modes = [
            {'bare_report':True, 'report':{'status':'complete'}},
            {'report':{'status':'complete'}},
            {'drop_fields':['schema_version']},
            {'drop_fields':['question']},
            {'drop_fields':['answer_style']},
            {'drop_fields':['stage']},
            {'drop_fields':['answer']},
            {'report_update':{'schema_version':True}},
            {'report_update':{'question':'A different biomedical request?'}},
            {'report_update':{'answer':{'status':'unknown','claims':[]}}},
            {'report_update':{'answer':{'status':'not_completed','claims':[]}}},
            {'report_update':{'answer':{'status':'not_requested','claims':[]}}},
            {'report_update':{'answer':{'status':'partial'}}},
            {'report_update':{'stage':'answer'}},
            {'report_update':{'generate':False}},
            {'report_update':{'top_k':1}},
            {'envelope_update':{'nonce':'foreign-nonce'}},
            {'envelope_update':{'request_sha256':'f'*64}},
            {'envelope_update':{'protocol':True}},
            {'missing_result':True},
        ]
        for index, mode in enumerate(modes):
            with self.subTest(mode=mode):
                self.mode(mode)
                report = self.runner().run_with_budget(self.request,
                    self.root / f'malformed-{index}' / 'run', None, self.budget())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error']['type'], 'RequestWorkerError')
                self.assertEqual(report['answer']['status'], 'not_completed')
                self.assertTrue(report['native_worker_cleanup']['confirmed'])

    def test_generate_mode_is_validated_independently_of_cleanup(self):
        request = SearchRequest(question=QUESTION, generate=False)
        valid = self.runner().run_with_budget(request, self.root / 'search-only' / 'run', None, self.budget())
        self.assertEqual(valid['status'], 'complete')
        self.assertEqual(valid['answer']['status'], 'not_requested')
        self.mode({'report_update':{'answer':{'status':'insufficient_evidence','claims':[]}}})
        invalid = self.runner().run_with_budget(request, self.root / 'foreign-generate' / 'run', None, self.budget())
        self.assertEqual(invalid['status'], 'failed')
        self.assertEqual(invalid['error']['type'], 'RequestWorkerError')
        self.assertTrue(invalid['native_worker_cleanup']['confirmed'])

    def test_malformed_worker_protocol_is_a_failed_job_with_ready_clean_lane(self):
        self.mode({'report':{'status':'complete'}})
        service = JobService(self.root / 'malformed-service', self.runner())
        self.addCleanup(service.close)
        job = service.submit(self.request)
        service._future.result(timeout=3)
        job = service.get(job['request_id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error']['code'], 'search_worker_failed')
        self.assertEqual(job['result']['answer']['status'], 'not_completed')
        self.assertTrue(job['result']['native_worker_cleanup']['confirmed'])
        self.assertEqual(service.health()['status'], 'ready')

    def test_progress_callback_failure_terminates_worker_and_preserves_cleanup_proof(self):
        self.mode({'wait':True})
        def progress(stage, status):
            raise OSError('/private/fixture-progress-storage-failed')
        budget = self.budget()
        output = self.root / 'progress-error' / 'run'
        report = self.runner().run_with_budget(self.request, output, progress, budget)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error']['type'], 'RequestProgressError')
        self.assertTrue(report['progress_persistence_failed'])
        self.assertTrue(report['native_worker_cleanup']['confirmed'])
        self.assertLess(budget.elapsed(), budget.seconds)
        durable = json.loads((output.parent / 'worker-outcome.json').read_text())
        self.assertTrue(durable['native_worker_cleanup']['confirmed'])
        self.assertTrue(durable['progress_persistence_failed'])

    def test_failed_progress_persistence_closes_admission_even_if_terminal_save_recovers(self):
        self.mode({'wait':True})
        service = JobService(self.root / 'progress-service', self.runner())
        self.addCleanup(service.close)
        persist = service._persist
        def fail_progress(job):
            if job['status'] == 'running' and job['stage'] == 'answer':
                raise OSError('/private/fixture-progress-storage-failed')
            return persist(job)
        with patch.object(service, '_persist', side_effect=fail_progress):
            job = service.submit(self.request)
            service._future.result(timeout=3)
        job = service.get(job['request_id'])
        self.assertEqual(job['status'], 'failed')
        self.assertEqual(job['error']['code'], 'storage_error')
        self.assertNotIn('/private/fixture', json.dumps(job))
        self.assertTrue(job['result']['native_worker_cleanup']['confirmed'])
        self.assertEqual(service.health()['status'], 'unavailable')
        with self.assertRaises(ServiceUnavailable):service.submit(self.request)

    def test_worker_crash_is_operational_failure_with_confirmed_outer_cleanup(self):
        self.mode({'exit':True})
        report = self.runner().run_with_budget(self.request, self.root / 'crash' / 'run', None, self.budget())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['error']['type'], 'RequestWorkerError')
        self.assertEqual(report['answer']['status'], 'not_completed')
        self.assertTrue(report['native_worker_cleanup']['confirmed'])

    def test_default_adapter_never_imports_tensor_modules_in_parent_after_real_job(self):
        # The -S parent cannot import ML; its child reports a sanitized setup
        # failure. This checks execution, not just lazy runner construction.
        source = Path(__file__).resolve().parents[1] / 'src'
        code = ("import sys; sys.path.insert(0, " + repr(str(source)) + "); "
            "from medlens.api_backend import make_local_runner; "
            "from medlens.api_contract import SearchRequest; "
            "r=make_local_runner(" + repr(str(self.root)) + "," + repr(str(self.root/'cache')) + ")"
            "(SearchRequest(question=" + repr(QUESTION) + ")," + repr(str(self.root/'lazy'/'run')) + "); "
            "assert r['native_worker_cleanup']['confirmed']; "
            "assert not any(n==p or n.startswith(p+'.') for n in sys.modules "
            "for p in ('torch','transformers','numpy','mlx','medlens.live')); print('isolated')")
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'isolated')

    def test_service_timeout_projection_and_origin_include_pre_admission_time(self):
        class DeadlineRunner:
            def __call__(self, *args):
                raise AssertionError('service bypassed bounded runner')
            def run_with_budget(inner, request, output, progress, budget):
                self.assertGreaterEqual(budget.elapsed(), 2)
                return {'status':'failed','stage':'answer','deadline_exceeded':True,
                    'answer':{'status':'not_completed'},
                    'native_worker_cleanup':{'confirmed':True,'worker_reaped':True,'group_absent':True}}
        service = JobService(self.root / 'service', DeadlineRunner())
        self.addCleanup(service.close)
        job = service.submit_started(self.request, time.monotonic() - 2)
        service._future.result(timeout=3)
        job = service.get(job['request_id'])
        self.assertEqual(job['error']['code'], 'search_timeout')
        self.assertEqual(job['result']['answer']['status'], 'not_completed')
        self.assertGreaterEqual(job['timing']['terminal_after_cleanup_seconds'], 2)
        self.assertTrue(job['timing']['cleanup_confirmed'])
        self.assertEqual(service.health()['status'], 'ready')
        service.observe_delivery(job['request_id'], scope='http-asgi-body-sent')
        delivered = service.get(job['request_id'])['timing']['delivery_observation']
        self.assertEqual(delivered['scope'], 'http-asgi-body-sent')
        self.assertGreaterEqual(delivered['seconds'], 2)

    def test_unknown_native_group_cleanup_blocks_admission(self):
        def runner(request, output, progress):
            return {'status':'complete','stage':'complete','runtime_load_count':0,
                'native_worker_cleanup':{'confirmed':False,'worker_reaped':True,'group_absent':False}}
        service = JobService(self.root / 'unknown', runner)
        self.addCleanup(service.close)
        job = service.submit(self.request)
        service._future.result(timeout=3)
        self.assertEqual(service.get(job['request_id'])['error']['code'], 'runtime_cleanup_failed')
        with self.assertRaises(ServiceUnavailable):service.submit(self.request)


if __name__ == '__main__':
    unittest.main()
