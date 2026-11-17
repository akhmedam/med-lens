"""The held-out harness must fail closed before test text is exposed."""

from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/validate_mvp.py'
spec = importlib.util.spec_from_file_location('mvp_validation', SCRIPT)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class FinalValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'out'

    def test_real_checkout_source_fingerprint_covers_installed_runtime_locks(self):
        root = SCRIPT.parents[1]
        hashes = validation.source_hashes(root)
        for name in ('requirements-macos-arm64.lock', 'requirements-mlx-macos-arm64.lock'):
            with self.subTest(lock=name):
                self.assertEqual(hashes[name], hashlib.sha256((root / name).read_bytes()).hexdigest())
        self.assertNotIn('requirements-mvp-macos-arm64.lock', hashes)
        self.assertNotIn('requirements-api.lock', hashes)

    def test_freeze_never_deserializes_question_text(self):
        selector = {'SELECTOR_SYSTEM': 'fictional fixed prompt', 'PROMPT_VERSION': 'v2',
            'ANSWER_POLICY': 'v2', 'SELECTION_CAP': 512, 'MAX_FINDINGS': 3,
            'MAX_QUOTE_CHARS': 532, 'MAX_QUOTES_PER_FINDING': 3}
        def opaque_hash(path):
            name = str(Path(path).relative_to(self.root))
            return {validation.QUERIES: validation.QUERY_SHA256,
                validation.PLAN: validation.PLAN_SHA256, **validation.SELECTOR_PINS}[name]
        def pinned_constants(path, names):
            return {'PROFILES': {'4b': {'model_id': 'fixture', 'revision': 'fixed'}}} if names == {'PROFILES'} else selector.copy()
        with ExitStack() as stack:
            stack.enter_context(patch.object(validation, 'file_sha', side_effect=opaque_hash))
            stack.enter_context(patch.object(validation, 'constants', side_effect=pinned_constants))
            stack.enter_context(patch.object(validation, 'application_policy', return_value={'id': 'fixture-policy'}))
            stack.enter_context(patch.object(validation, 'source_hashes', return_value={'fixture.py': 'fixed'}))
            stack.enter_context(patch.object(validation, 'asset_identity', return_value={}))
            stack.enter_context(patch.object(validation, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(validation, 'packages', return_value={}))
            stack.enter_context(patch.object(validation.subprocess, 'check_output', return_value='fixture-head'))
            # Neither a question JSON parse nor a semantic read is allowed by freeze.
            stack.enter_context(patch.object(validation.json, 'loads', side_effect=AssertionError('question parse before freeze')))
            stack.enter_context(patch.object(Path, 'read_bytes', side_effect=AssertionError('unexpected content read')))
            with redirect_stdout(io.StringIO()):
                manifest = validation.freeze(self.root, self.output)
        self.assertTrue((self.output / 'manifest.json').is_file())
        self.assertEqual(manifest['question_file']['sha256'], validation.QUERY_SHA256)
        self.assertFalse(manifest['prefreeze_exposure_declaration']['question_texts_read'])
        self.assertEqual(manifest['application_policy']['id'], 'fixture-policy')

    def test_existing_execution_marker_blocks_second_attempt_before_question_parse(self):
        self.output.mkdir()
        (self.output / 'manifest.json').write_text('{"manifest_id":"fixture"}')
        marker = self.output / 'execution-started.json'
        marker.write_text('original attempt')
        with patch.object(validation, 'verify_frozen'), patch.object(Path, 'read_bytes',
                side_effect=AssertionError('questions must not be read on a second attempt')):
            with self.assertRaises(FileExistsError):
                validation.execute(self.root, self.output)
        self.assertEqual(marker.read_text(), 'original attempt')

    def test_source_drift_rejected_before_asset_checks_or_question_deserialization(self):
        manifest = {'policy': validation.POLICY, 'root': str(self.root),
            'application_policy': {'id': 'fixed'}, 'source_sha256': {'file.py': 'old'}}
        manifest['manifest_id'] = validation.digest(manifest)
        with patch.object(validation, 'application_policy', return_value={'id': 'fixed'}), \
             patch.object(validation, 'file_sha', side_effect=[validation.QUERY_SHA256, validation.PLAN_SHA256]), \
             patch.object(validation, 'source_hashes', return_value={'file.py': 'changed'}), \
             patch.object(validation, 'asset_identity') as assets, \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('question text is still unopened')):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                validation.verify_frozen(self.root, manifest)
        assets.assert_not_called()

    def test_same_size_corrupt_asset_cannot_be_blessed_by_freeze(self):
        assets = self.root / 'assets'
        assets.mkdir()
        (assets / 'weights.bin').write_bytes(b'tampered')
        declared = {'files': {'weights.bin': {'sha256': '0' * 64, 'bytes': 8}}}
        (assets / 'model-source.json').write_text(json.dumps(declared))
        with patch.object(validation, 'ASSET_MANIFESTS', ('assets/model-source.json',)):
            with self.assertRaisesRegex(ValueError, 'pinned checksum/size'):
                validation.asset_identity(self.root)
        declared['files']['weights.bin'] = hashlib.sha256(b'tampered').hexdigest()
        (assets / 'model-source.json').write_text(json.dumps(declared))
        with patch.object(validation, 'ASSET_MANIFESTS', ('assets/model-source.json',)):
            self.assertEqual(validation.asset_identity(self.root)['actual_asset_files']['assets/weights.bin']['bytes'], 8)

    def test_forged_citation_span_is_rejected_without_a_model(self):
        from medlens.api_contract import public_report
        from medlens.generation.context import digest, render_evidence
        source = {'citation_id': 'E1', 'pmid': '1', 'window_id': 'w', 'text_version_id': 'v',
            'source_path': './PubmedArticle/Abstract/AbstractText', 'section_index': 0,
            'char_start': 10, 'char_end': 24, 'text': 'fictional text',
            'title': 'Fictional source', 'journal': None, 'publication_date': None, 'section': None}
        context = {'sources': [source], 'sources_sha256': digest([source]),
            'evidence_payload': render_evidence([source])}
        citation = {k: source[k] for k in ('citation_id', 'pmid', 'window_id', 'text_version_id', 'source_path', 'section_index')}
        citation.update(quote='fictional', matches=[{'char_start': 11, 'char_end': 20}])
        run = {'status': 'complete', 'answer_style': 'source-findings', 'context': context,
            'input_paths': {'dataset': 'd', 'windows': 'w', 'embeddings': 'e'},
            'answer': {'status': 'partial', 'claims': [{'text': 'fictional',
                'source_article_id': '1', 'citations': [citation]}]}}
        with patch('medlens.retrieval.dense.DenseIndex'), \
             patch('medlens.generation.request_contexts.verify_context_sources'):
            with self.assertRaisesRegex(ValueError, 'Citation span'):
                validation.audit_sources(run, public_report(run))
            fixed = copy.deepcopy(run)
            fixed['answer']['claims'][0]['citations'][0]['matches'] = [{'char_start': 10, 'char_end': 19}]
            self.assertEqual(validation.audit_sources(fixed, public_report(fixed))['citations'], 1)

    def execute_fixture(self, *, interrupt=False, drift_after_job=False):
        """Fictional rows exercise the journal, never the real held-out file."""
        from medlens.mvp import policy_snapshot
        self.output.mkdir()
        manifest = {'manifest_id': 'fictional', 'source_sha256': {'fixture.py': 'fixed'},
                    'application_policy': policy_snapshot()}
        validation.save_new(self.output / 'manifest.json', manifest)
        queries = self.root / 'fictional.jsonl'
        queries.write_text('\n'.join(json.dumps({'query_id': 'fake' + str(i), 'group_id': 'g' + str(i),
            'text': 'What does the fictional biomedical study report?', 'split': 'test', 'dataset': 'unused'})
            for i in range(12)))
        output = self.output
        state = {'submitted': 0, 'closed': False, 'drift': False}
        class Service:
            def health(self):
                return {'accepting_requests': True}
            def submit(self, request):
                state['submitted'] += 1
                rid = str(state['submitted']).zfill(32)
                folder = output / 'jobs' / rid
                (folder / 'run').mkdir(parents=True)
                raw = {'status': 'complete', 'answer_style': 'source-findings',
                    'runtime_load_count': 0, 'live': {'fresh_search_complete': True}}
                self.job = {'request_id': rid, 'mvp_policy': policy_snapshot(),
                    'status': 'succeeded', 'error': None,
                    'result': {'answer': {'status': None if interrupt else 'insufficient_evidence'}}}
                validation.save_new(folder / 'job.json', self.job)
                validation.save_new(folder / 'run/run.json', raw)
                return self.job
            def get(self, request_id):
                if interrupt:
                    raise KeyboardInterrupt()
                state['drift'] = drift_after_job
                return self.job
            def close(self):
                state['closed'] = True
        class Memory:
            def __enter__(self):return self
            def __exit__(self, *args):return None
            def result(self):return {'observed_native_workers_all_reaped': None}
        with ExitStack() as stack:
            stack.enter_context(patch.object(validation, 'verify_frozen'))
            stack.enter_context(patch.object(validation, 'QUERIES', 'fictional.jsonl'))
            stack.enter_context(patch.object(validation, 'source_hashes', side_effect=lambda root:
                {'fixture.py': 'changed' if state['drift'] else 'fixed'}))
            stack.enter_context(patch.object(validation, 'MemorySampler', Memory))
            stack.enter_context(patch.object(validation, 'audit_sources', return_value={'status': 'passed'}))
            stack.enter_context(patch('medlens.api_service.JobService', return_value=Service()))
            stack.enter_context(patch('medlens.api_backend.make_local_runner', return_value=lambda *args: None))
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt if interrupt else ValueError):
                    validation.execute(self.root, self.output)
        self.assertTrue(state['closed'])
        return json.loads((self.output / 'summary.json').read_text()), state

    def test_interrupted_accepted_attempt_does_not_disappear_from_summary(self):
        summary, state = self.execute_fixture(interrupt=True)
        self.assertEqual(state['submitted'], 1)
        self.assertEqual(summary['attempted_queries'], 1)
        self.assertEqual(summary['accepted_job_count'], 1)
        self.assertEqual(summary['recorded_queries'], 0)
        self.assertEqual(summary['accepted_jobs'][0]['query_id'], 'fake0')
        self.assertFalse(summary['operational_pass'])
        self.assertEqual(len(summary['unattempted_query_ids']), 11)

    def test_midrun_source_drift_stops_before_second_question_and_cannot_pass(self):
        summary, state = self.execute_fixture(drift_after_job=True)
        self.assertEqual(state['submitted'], 1)
        self.assertEqual(summary['attempted_queries'], 1)
        self.assertEqual(summary['recorded_queries'], 1)
        self.assertFalse(summary['frozen_sources_unchanged'])
        self.assertFalse(summary['operational_pass'])
        self.assertEqual(len(summary['unattempted_query_ids']), 11)


if __name__ == '__main__':
    unittest.main()
