"""Confirmation acquisition keeps every planned case and never generates answers."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medlens.pubmed_query import plan_query


def load_tool():
    path = Path(__file__).resolve().parents[1] / 'scripts/prepare_confirmation_sources.py'
    spec = importlib.util.spec_from_file_location('prepare_confirmation_source_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fictional_plan():
    return {'schema_version': 1, 'kind': 'medlens-human-confirmation-plan',
        'dataset_id': 'human-confirmation-v1', 'split': 'test',
        'question_count': 20, 'candidate_limit': 100, 'annotation_depth_per_method': 10,
        'pool_methods': ['pubmed-best-match', 'dense'],
        'cases': [{'query_id': f'hc{i:02d}', 'group_id': f'fictional-group-{i:02d}',
            'question': f'Does fictional intervention A{i:02d} reduce fictional outcome B{i:02d}?',
            'relevance_intent': f'Fictional scope {i:02d}; this is not a biomedical label.',
            'specialty': 'fictional', 'question_type': 'treatment'} for i in range(1, 21)]}


class ConfirmationSourcesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.tool = load_tool()
        self.plan_path = self.root / 'plan.json'
        self.write_plan(fictional_plan())
        self.output = self.root / 'cache-sources'
        self.cache = self.root / 'cache'
        self.requests = []

    def write_plan(self, value):
        body = (json.dumps(value, indent=2) + '\n').encode()
        self.plan_path.write_bytes(body)
        self.plan_path.with_suffix('.sha256').write_text(hashlib.sha256(body).hexdigest() + '  ' + self.plan_path.name + '\n')
        return body

    def report(self, request, directory, *, empty=False):
        report = {'status': 'complete', 'stage': 'complete', 'question': request.question,
            'runtime_load_count': 0, 'request_seconds': .001,
            'answer_style': request.answer_style,
            'live': {'fresh_search_complete': True, 'candidate_limit': request.limit,
                'query_plan': plan_query(request.question), 'fetched_article_count': 0 if empty else 2,
                'snapshot_id': 'snapshot-' + directory.name,
                'dataset_id': 'dataset-' + directory.name},
            'retrieval': {'results': [] if empty else [{'pmid': '1001'}, {'pmid': '1002'}]},
            'answer': {'status': 'not_requested', 'claims': [], 'model_calls': []}}
        # The actual empty live branch legitimately omits these fields.
        if not empty:
            report.update(generate=False, rerank=False, top_k=request.top_k)
        return report

    def write_run(self, directory, report):
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'run.json').write_text(json.dumps(report, indent=2) + '\n')

    def run_main(self, runner, *, output=None, runtime_hashes=None):
        output = output or self.output
        hashes = runtime_hashes or (lambda root: {'fixture_runtime.py': 'f' * 64})
        with patch.object(self.tool, 'ROOT', self.root), \
             patch.object(self.tool, 'runtime_hashes', side_effect=hashes), \
             patch('medlens.api_backend.make_local_runner', return_value=runner) as factory, \
             redirect_stdout(io.StringIO()):
            code = self.tool.main(['--plan', str(self.plan_path), '--output', str(output),
                                   '--cache', str(self.cache)])
        return code, factory

    def test_twenty_search_only_cases_keep_empty_results_and_source_hashes(self):
        empty_ids = {'hc02', 'hc11', 'hc20'}
        original = self.plan_path.read_bytes()
        def runner(request, output):
            self.requests.append((request, output))
            run = self.report(request, output, empty=output.name in empty_ids)
            self.write_run(output, run)
            return run
        code, factory = self.run_main(runner)
        self.assertEqual(code, 0)
        factory.assert_called_once_with(self.root, self.cache)
        self.assertEqual(len(self.requests), 20)
        report = json.loads((self.output / 'preparation.json').read_text())
        self.assertEqual(report['status'], 'complete-awaiting-manual-human-annotation')
        self.assertEqual(report['split'], 'test')
        self.assertEqual(report['expected_questions'], 20)
        self.assertEqual([r['query_id'] for r in report['runs']], [f'hc{i:02d}' for i in range(1, 21)])
        for field in ('answer_generation_performed', 'human_labels_generated',
                      'quality_scores_computed', 'latency_acceptance_evaluated'):
            self.assertIs(report[field], False)
        self.assertEqual((self.output / 'plan-original.json').read_bytes(), original)
        self.assertEqual(self.plan_path.read_bytes(), original)
        for row, (request, output) in zip(report['runs'], self.requests):
            self.assertEqual((request.generate, request.limit, request.top_k, request.answer_style),
                             (False, 100, 10, 'source-findings'))
            self.assertEqual(row['run_sha256'], hashlib.sha256((output / 'run.json').read_bytes()).hexdigest())
            self.assertEqual(row['snapshot_id'], 'snapshot-' + output.name)
            self.assertEqual(row['dataset_id'], 'dataset-' + output.name)
            self.assertEqual(row['runtime_load_count'], 0)
        self.assertEqual({r['query_id'] for r in report['runs'] if r['fetched_articles'] == 0}, empty_ids)
        self.assertEqual(len(list(self.output.glob('hc*/run.json'))), 20)

    def test_companion_sha_is_required_and_mismatch_is_rejected(self):
        companion = self.plan_path.with_suffix('.sha256')
        companion.unlink()
        with self.assertRaises((OSError, ValueError)):
            self.tool.load_plan(self.plan_path)
        companion.write_text('0' * 64 + '  ' + self.plan_path.name + '\n')
        with self.assertRaises(ValueError):
            self.tool.load_plan(self.plan_path)

    def test_wrong_kind_split_count_and_duplicate_or_unsafe_cases_are_rejected(self):
        original = fictional_plan()
        changes = [
            lambda p: p.update(kind='biomedical-development-annotation-plan'),
            lambda p: p.update(split='dev'),
            lambda p: p.update(question_count=19),
            lambda p: p['cases'].pop(),
            lambda p: p['cases'][1].update(query_id=p['cases'][0]['query_id']),
            lambda p: p['cases'][1].update(group_id=p['cases'][0]['group_id']),
            lambda p: p['cases'][1].update(question=p['cases'][0]['question'].upper()),
            lambda p: p['cases'][0].update(query_id='../outside'),
            lambda p: p['cases'][0].update(query_id='/tmp/outside'),
            lambda p: p['cases'][0].update(query_id='sources/hc01'),
        ]
        for change in changes:
            plan = deepcopy(original); change(plan); self.write_plan(plan)
            with self.subTest(change=repr(change)), self.assertRaises(ValueError):
                self.tool.load_plan(self.plan_path)

    def test_rejected_plan_and_output_paths_never_construct_runner_or_overwrite(self):
        marker = self.root / 'already-present'
        marker.mkdir(); (marker / 'keep').write_bytes(b'unchanged')
        link = self.root / 'output-link'; link.symlink_to(marker, target_is_directory=True)
        outputs = [marker, link, self.cache / 'nested', self.root / 'outer']
        caches = [self.cache, self.cache, self.cache, self.root / 'outer' / 'nested']
        for output, cache in zip(outputs, caches):
            with self.subTest(output=output), patch('medlens.api_backend.make_local_runner') as factory, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                self.tool.main(['--plan', str(self.plan_path), '--output', str(output), '--cache', str(cache)])
            self.assertEqual(caught.exception.code, 2)
            factory.assert_not_called()
        self.assertEqual((marker / 'keep').read_bytes(), b'unchanged')
        plan = fictional_plan(); plan['cases'][0]['query_id'] = '../outside'; self.write_plan(plan)
        with patch('medlens.api_backend.make_local_runner') as factory, \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.tool.main(['--plan', str(self.plan_path), '--output', str(self.output)])
        factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_plan_and_runtime_drift_preserve_first_completed_checkpoint(self):
        for mode in ('plan', 'runtime'):
            with self.subTest(mode=mode):
                self.write_plan(fictional_plan())
                changed = False
                calls = []
                output = self.root / ('drift-' + mode)
                def hashes(root):
                    return {'fixture_runtime.py': ('0' if changed and mode == 'runtime' else 'f') * 64}
                def runner(request, directory):
                    nonlocal changed
                    calls.append(request)
                    run = self.report(request, directory)
                    self.write_run(directory, run)
                    if mode == 'plan':
                        self.plan_path.write_bytes(self.plan_path.read_bytes() + b' ')
                    changed = True
                    return run
                code, _ = self.run_main(runner, output=output, runtime_hashes=hashes)
                self.assertEqual(code, 1)
                self.assertEqual(len(calls), 1)
                report = json.loads((output / 'preparation.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['expected_questions'], 20)
                self.assertEqual(len(report['runs']), 1)
                self.assertEqual(report['runs'][0]['query_id'], 'hc01')
                self.assertIsNotNone(report['finished_at'])
                self.assertTrue((output / 'hc01/run.json').is_file())

    def test_generation_or_wrong_source_configuration_fails_without_starting_second_case(self):
        mutations = [
            lambda r: r.update(runtime_load_count=1),
            lambda r: r['answer'].update(model_calls=[{'stage': 'generation'}]),
            lambda r: r.update(generate=True),
            lambda r: r.update(top_k=9),
            lambda r: r['live'].update(candidate_limit=300),
        ]
        for i, mutate in enumerate(mutations):
            calls = []
            def runner(request, directory):
                calls.append(request)
                run = self.report(request, directory); mutate(run)
                self.write_run(directory, run)
                return run
            with self.subTest(mutation=i):
                output = self.root / f'invalid-run-{i}'
                code, _ = self.run_main(runner, output=output)
                self.assertEqual(code, 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads((output / 'preparation.json').read_text())['status'], 'failed')

    def test_missing_different_typed_or_symlinked_run_artifact_is_not_success(self):
        for i, mode in enumerate(('missing', 'different', 'typed', 'symlink')):
            calls = []
            def runner(request, directory):
                calls.append(request)
                run = self.report(request, directory)
                directory.mkdir(parents=True, exist_ok=False)
                saved = deepcopy(run)
                if mode == 'different': saved['question'] = 'Different fictional scope?'
                if mode == 'typed': saved['runtime_load_count'] = False
                if mode != 'missing':
                    target = directory / ('target.json' if mode == 'symlink' else 'run.json')
                    target.write_text(json.dumps(saved))
                    if mode == 'symlink': (directory / 'run.json').symlink_to(target)
                return run
            with self.subTest(mode=mode):
                output = self.root / f'artifact-{i}'
                code, _ = self.run_main(runner, output=output)
                self.assertEqual(code, 1)
                self.assertEqual(len(calls), 1)
                report = json.loads((output / 'preparation.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(len(report['runs']), 1)

    def test_raised_runner_retains_accepted_attempt_in_failed_report(self):
        calls = []
        def runner(request, output):
            calls.append(request)
            raise RuntimeError('fictional preparation failure')
        code, _ = self.run_main(runner)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        report = json.loads((self.output / 'preparation.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['expected_questions'], 20)
        self.assertEqual(len(report['runs']), 1)
        self.assertEqual(report['runs'][0]['query_id'], 'hc01')


if __name__ == '__main__':
    unittest.main()
