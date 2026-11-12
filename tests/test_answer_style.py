"""An explicit experimental answer style must never replace the default path."""
from collections import Counter
from contextlib import redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import numpy as np

from medlens.generation.context import COVERAGE_POLICY, build_context
from medlens.live import run_live_question
from medlens.pipeline import run_question
from medlens.retrieval.dense import DenseIndex
from pubmed_helpers import make_client, search_json
from test_answers import FakeRuntime, draft
from test_evidence_coverage import fixture as coverage_fixture
from test_live import Encoder as LiveEncoder, xml
from test_pipeline import Encoder, Tokenizer, index_fixture
from test_shared_corpus import INFO
from test_windows import Tokenizer as BgeTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODULE = 'medlens.generation.source_findings'


def guarded_module():
    module = ModuleType(MODULE)
    def generate(question, context, instance, *, source_articles):
        return {'status': 'insufficient_evidence', 'claims': [], 'model_calls': [],
                'answer_style': 'source-findings', 'reason': 'fictional-stub'}
    module.generate_source_findings_answer = Mock(side_effect=generate)
    return module


def structured_index():
    dataset, manifest, windows, _ = coverage_fixture(4)
    index = DenseIndex.__new__(DenseIndex)
    index.dataset, index.window_manifest, index.windows = dataset, manifest, windows
    index.articles = {a['pmid']: a for a in dataset.articles}
    index.positions = {p: [n for n, w in enumerate(windows) if w['pmid'] == p] for p in index.articles}
    index.vectors = np.zeros((len(windows), 768), np.float32)
    for n in range(len(windows)):
        score = .95 - n*.01
        index.vectors[n, 0], index.vectors[n, 1] = score, np.sqrt(1-score*score)
    index.embedding_manifest, index.missing_pmids = {'artifact_id': 'e'}, []
    index.verify_encoder = Mock()
    return index


def runtime(outputs=()):
    value = FakeRuntime(outputs)
    value.tokenizer, value.metadata = Tokenizer(), {'fixture': True}
    value.close = Mock()
    return value


class AnswerStyleTests(unittest.TestCase):
    def test_default_draft_keeps_768_512_calls_without_loading_experimental_module(self):
        review = [{'claim_id': 'C1', 'verdict': 'supported', 'population_match': 'match',
                   'comparison_match': 'not_requested', 'outcome_match': 'match', 'reason': 'supported_by_excerpt'}]
        instance = runtime([draft(), review])
        with patch.dict(sys.modules, {MODULE: None}):
            report = run_question(index_fixture(), 'Question?', encoder_factory=Encoder,
                tokenizer=Tokenizer(), runtime_factory=lambda: instance, generate=True)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['answer_style'], 'draft')
        self.assertEqual([limit for _, limit in instance.calls], [768, 512])
        instance.close.assert_called_once_with()

    def test_experimental_style_packs_three_articles_and_passes_actual_bound_articles(self):
        index = structured_index(); instance = runtime(); module = guarded_module()
        with patch.dict(sys.modules, {MODULE: module}):
            report = run_question(index, 'Question?', encoder_factory=Encoder,
                tokenizer=Tokenizer(), runtime_factory=lambda: instance, generate=True,
                answer_style='source-findings')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['answer_style'], 'source-findings')
        self.assertEqual(report['context']['selection_policy'], COVERAGE_POLICY)
        self.assertEqual(report['context']['article_ranking'], ['1', '2', '3', '4'])
        self.assertLessEqual(len(report['context']['sources']), 6)
        self.assertEqual(set(s['pmid'] for s in report['context']['sources']), {'1', '2', '3'})
        self.assertTrue(all(n <= 2 for n in Counter(s['pmid'] for s in report['context']['sources']).values()))
        call = module.generate_source_findings_answer.call_args
        self.assertIs(call.kwargs['source_articles'], index.articles)
        self.assertIs(call.args[2], instance)
        self.assertEqual(instance.calls, [])
        self.assertIn('generation/source_scope.py', report['implementation_sha256'])
        self.assertIn('generation/source_findings.py', report['implementation_sha256'])

    def test_rerank_repack_uses_the_same_coverage_config(self):
        index = structured_index(); instance = runtime([['D4', 'D3', 'D2', 'D1']]); module = guarded_module()
        with patch.dict(sys.modules, {MODULE: module}), patch('medlens.pipeline.build_context', wraps=build_context) as pack:
            report = run_question(index, 'Question?', encoder_factory=Encoder, tokenizer=Tokenizer(),
                runtime_factory=lambda: instance, generate=True, rerank=True, answer_style='source-findings')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(pack.call_count, 2)
        self.assertIs(pack.call_args_list[0].kwargs['config'], pack.call_args_list[1].kwargs['config'])
        self.assertEqual(pack.call_args.kwargs['config'].selection_mode, 'results-methods')
        self.assertEqual(report['context']['article_ranking'], ['4', '3', '2', '1'])
        self.assertEqual(set(s['pmid'] for s in report['context']['sources']), {'4', '3', '2'})
        self.assertEqual([limit for _, limit in instance.calls], [128])

    def test_empty_context_abstains_without_qwen_factory(self):
        module = guarded_module(); factory = Mock(side_effect=AssertionError('No model load'))
        factory.identity = {'engine': 'fictional'}
        with patch.dict(sys.modules, {MODULE: module}):
            report = run_question(index_fixture(True), 'Question?', encoder_factory=Encoder,
                tokenizer=Tokenizer(), runtime_factory=factory, generate=True, answer_style='source-findings')
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['answer']['status'], 'insufficient_evidence')
        self.assertEqual(report['runtime_load_count'], 0)
        factory.assert_not_called()
        self.assertEqual(module.generate_source_findings_answer.call_args.args[1]['sources'], [])

    def test_bad_style_fails_before_any_encoder_model_or_output(self):
        forbidden = Mock(side_effect=AssertionError('No work should start'))
        for style in ('typo', None, True, []):
            with self.subTest(style=style), self.assertRaises(ValueError):
                run_question(None, 'Question?', encoder_factory=forbidden, tokenizer=None,
                    runtime_factory=forbidden, answer_style=style)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'request'
            with self.assertRaises(ValueError):
                run_live_question('Question?', output, Path(directory)/'cache',
                    tokenizer_loader=forbidden, encoder_info={}, encoder_factory=forbidden,
                    answer_tokenizer_loader=forbidden, runtime_factory=forbidden, answer_style='bad')
            self.assertFalse(output.exists())
        forbidden.assert_not_called()

    def test_style_is_explicit_in_request_identity_even_without_generation(self):
        index = structured_index()
        default = run_question(index, 'Question?', encoder_factory=Encoder,
            tokenizer=Tokenizer(), runtime_factory=lambda: None, generate=False)
        with patch.dict(sys.modules, {MODULE: None}):
            explicit = run_question(index, 'Question?', encoder_factory=Encoder,
                tokenizer=Tokenizer(), runtime_factory=lambda: None, generate=False, answer_style='source-findings')
        self.assertNotEqual(default['request_id'], explicit['request_id'])
        self.assertEqual(default['answer_style'], 'draft')
        self.assertEqual(explicit['answer_style'], 'source-findings')

    def test_live_request_forwards_style_and_empty_search_records_it_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); module = guarded_module(); instance = runtime()
            client, _, _ = make_client([search_json(['1'], 1), xml(body='A fictional complete finding.')])
            with patch.dict(sys.modules, {MODULE: module}):
                report = run_live_question('Question?', root/'one', root/'cache', client=client,
                    tokenizer_loader=lambda: (BgeTokenizer(), INFO), encoder_info=LiveEncoder().info,
                    encoder_factory=LiveEncoder, answer_tokenizer_loader=Tokenizer,
                    runtime_factory=lambda: instance, answer_style='source-findings')
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(report['answer_style'], 'source-findings')
            self.assertEqual(module.generate_source_findings_answer.call_count, 1)
            self.assertEqual(set(module.generate_source_findings_answer.call_args.kwargs['source_articles']), {'1'})
            client, _, _ = make_client([search_json([], 0)])
            forbidden = Mock(side_effect=AssertionError('Empty request must not load weights'))
            empty = run_live_question('Question?', root/'empty', root/'cache', client=client,
                tokenizer_loader=lambda: (BgeTokenizer(), INFO), encoder_info={}, encoder_factory=forbidden,
                answer_tokenizer_loader=forbidden, runtime_factory=forbidden, answer_style='source-findings')
            self.assertEqual(empty['status'], 'complete')
            self.assertEqual(empty['answer_style'], 'source-findings')
            self.assertTrue(empty['request_id'])
            forbidden.assert_not_called()

    def test_both_cli_help_pages_expose_opt_in_without_model_loading(self):
        for filename in ('ask.py', 'ask_live.py'):
            path = ROOT/'scripts'/filename
            spec = spec_from_file_location('style_help_'+filename.replace('.', '_'), path)
            module = module_from_spec(spec); spec.loader.exec_module(module)
            stdout = StringIO()
            with self.subTest(filename=filename), redirect_stdout(stdout), self.assertRaises(SystemExit) as caught:
                module.main(['--help'])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn('--answer-style {draft,source-findings}', stdout.getvalue())
            self.assertIn('experimental', stdout.getvalue())
            self.assertIn('source quotations', stdout.getvalue())


if __name__ == '__main__':unittest.main()
