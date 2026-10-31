"""Search survives generation failures; one Qwen instance handles every LLM stage."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from medlens.pipeline import run_question
from medlens.reporting import render_report
from medlens.retrieval.dense import DenseIndex
import test_answers as answers
import test_evidence_context as evidence


class Tokenizer(answers.Tokenizer):
    backend_tokenizer = SimpleNamespace(to_str=lambda: 'test-tokenizer')


def index_fixture(empty=False):
    index = DenseIndex.__new__(DenseIndex)
    index.dataset, index.window_manifest, index.windows = evidence.fixtures()
    index.articles = {a['pmid']: a for a in index.dataset.articles}
    index.positions = {p: list(range(i*3, i*3+3)) for i,p in enumerate(index.articles)}
    index.vectors = np.zeros((9,768), np.float32)
    for i in range(9):
        cosine = .9 - i * .04
        index.vectors[i,0], index.vectors[i,1] = cosine, np.sqrt(1-cosine*cosine)
    index.embedding_manifest = {'artifact_id':'e'}
    index.missing_pmids = []
    if empty:
        index.windows = []
        index.vectors = np.empty((0,768), np.float32)
        index.positions = {p:[] for p in index.articles}
        index.missing_pmids = list(index.articles)
    def verify(encoder):
        if encoder.info != {'fixture':True}:raise ValueError('Wrong encoder')
    index.verify_encoder = verify
    return index


class Encoder:
    info = {'fixture':True}
    def encode_queries(self, questions):
        self.questions = questions
        vector = np.zeros((1,768),np.float32);vector[0,0] = 1
        return vector, {'encode_seconds':0}


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, outputs=(), **kwargs):
        self.loads = 0
        self.runtime = answers.FakeRuntime(outputs)
        self.runtime.tokenizer = Tokenizer()
        self.runtime.metadata = {'fixture':True}
        def factory():
            self.loads += 1
            return self.runtime
        return run_question(kwargs.pop('index',index_fixture()), 'Research question?',
            encoder_factory=Encoder, tokenizer=Tokenizer(), runtime_factory=factory, **kwargs)

    def test_retrieval_only_never_loads_qwen_and_has_source_report(self):
        result = self.run_pipeline()
        self.assertEqual(result['status'],'complete')
        self.assertEqual(self.loads,0)
        self.assertEqual(result['answer']['status'],'not_requested')
        self.assertEqual([h['pmid'] for h in result['retrieval']['results']],['1','2','3'])
        self.assertTrue(result['context']['sources'])
        self.assertIn('Generation was not requested',render_report(result))

    def test_same_runtime_reranks_then_generates_and_reviews(self):
        review = [{'claim_id':'C1','verdict':'supported','population_match':'match',
            'comparison_match':'not_requested','outcome_match':'match','reason':'supported_by_excerpt'}]
        result = self.run_pipeline([['D3','D1','D2'],answers.draft(),review],generate=True,rerank=True)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(self.loads,1)
        self.assertEqual(result['runtime_load_count'],1)
        self.assertEqual([limit for _,limit in self.runtime.calls],[128,768,512])
        self.assertEqual(result['context']['article_ranking'],['3','1','2'])
        self.assertEqual(result['answer']['claims'][0]['citations'][0]['pmid'],'3')
        self.assertIn('requires independent review',render_report(result))

    def test_invalid_rerank_preserves_dense_order_and_still_packs_evidence(self):
        result = self.run_pipeline(['["D99"]'],rerank=True)
        self.assertEqual(result['reranking']['status'],'fallback')
        self.assertEqual(result['context']['article_ranking'],['1','2','3'])
        self.assertIn('dense search order was retained',render_report(result))

    def test_invalid_draft_is_withheld_but_search_and_evidence_remain(self):
        result = self.run_pipeline(['not json'],generate=True)
        self.assertEqual(result['answer']['status'],'validation_failed')
        self.assertEqual(len(result['retrieval']['results']),3)
        page = render_report(result)
        self.assertIn('no answer is shown',page)
        self.assertIn('Selected source excerpts',page)
        self.assertIn('alpha beta gamma',page)

    def test_empty_search_abstains_without_runtime_even_if_both_flags_enabled(self):
        result = self.run_pipeline(index=index_fixture(True),generate=True,rerank=True)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['answer']['status'],'insufficient_evidence')
        self.assertEqual(self.loads,0)
        self.assertEqual(result['answer']['model_calls'],[])

    def test_model_load_failure_keeps_completed_search_and_context(self):
        checkpoints = []
        def unavailable():raise RuntimeError('Model unavailable')
        result = run_question(index_fixture(),'Question',encoder_factory=Encoder,tokenizer=Tokenizer(),
            runtime_factory=unavailable,generate=True,checkpoint=lambda r:checkpoints.append(copy.deepcopy(r)))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error']['stage'],'answer')
        self.assertEqual(len(result['retrieval']['results']),3)
        self.assertTrue(result['context']['sources'])
        self.assertEqual(checkpoints[-1]['status'],'failed')
        self.assertIn('Completed search results remain',render_report(result))

    def test_reranker_load_failure_also_keeps_dense_evidence(self):
        def unavailable():raise RuntimeError('Model unavailable')
        result = run_question(index_fixture(),'Question',encoder_factory=Encoder,tokenizer=Tokenizer(),
            runtime_factory=unavailable,generate=True,rerank=True)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error']['stage'],'reranking')
        self.assertEqual(result['context']['article_ranking'],['1','2','3'])
        self.assertTrue(result['context']['sources'])

    def test_wrong_tokenizer_fails_before_inference(self):
        runtime = answers.FakeRuntime([])
        runtime.tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:'different'))
        result = run_question(index_fixture(),'Question',encoder_factory=Encoder,tokenizer=Tokenizer(),
            runtime_factory=lambda:runtime,generate=True)
        self.assertEqual(result['status'],'failed')
        self.assertIn('tokenizers differ',result['error']['message'])
        self.assertEqual(runtime.calls,[])

    def test_invalid_question_or_limit_rejected_before_factory(self):
        def forbidden():raise AssertionError('Factory must not run')
        for question,k in (('',10),('  ',10),('Question',0),('Question',11),('Question',True)):
            with self.subTest(question=question,k=k),self.assertRaises(ValueError):
                run_question(index_fixture(),question,encoder_factory=forbidden,tokenizer=Tokenizer(),
                    runtime_factory=forbidden,top_k=k)

    def test_report_escapes_untrusted_titles_question_and_source_text(self):
        result = self.run_pipeline()
        result['question'] = '<script>alert(1)</script>'
        result['retrieval']['results'][0]['title'] = '<img src=x onerror=alert(1)>'
        result['context']['sources'][0]['text'] = '<iframe src="https://evil.example">'
        page = render_report(result)
        self.assertNotIn('<script>',page)
        self.assertNotIn('<img ',page)
        self.assertNotIn('<iframe ',page)
        self.assertIn('&lt;script&gt;',page)


class AskCLITests(unittest.TestCase):
    def test_setup_failure_is_saved_and_existing_output_is_never_overwritten(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('ask_cli',Path(__file__).resolve().parents[1]/'scripts/ask.py')
        cli = importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            argv=['missing-dataset','--windows','missing-windows','--embeddings','missing-embeddings',
                '--query','Question','--output-dir',str(output)]
            with patch.object(cli,'DenseIndex',side_effect=ValueError('Checksum mismatch')):
                self.assertEqual(cli.main(argv),1)
            before=(output/'run.json').read_bytes()
            self.assertEqual(json.loads(before)['status'],'failed')
            self.assertTrue((output/'report.html').is_file())
            with self.assertRaises(SystemExit) as error:cli.main(argv)
            self.assertEqual(error.exception.code,2)
            self.assertEqual((output/'run.json').read_bytes(),before)
