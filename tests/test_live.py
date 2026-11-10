"""Fresh source membership, cache reuse and failure isolation across complete requests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from medlens.live import run_live_question
from medlens.pubmed_query import plan_query
from medlens.ingestion.pubmed import PubMedError
from pubmed_helpers import make_client, search_json
from test_shared_corpus import Encoder as PassageEncoder, INFO
from test_windows import Tokenizer
from test_pipeline import Tokenizer as AnswerTokenizer
from test_answers import FakeRuntime, draft

QUESTION='Does cognitive behavioral therapy help adults with depression?'


def xml(pmid='1', body='alpha beta gamma'):
    return ('<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>'+pmid+
        '</PMID><Article><ArticleTitle>Depression evidence</ArticleTitle><Abstract>'+
        '<AbstractText Label="RESULTS">'+body+'</AbstractText></Abstract></Article>'+
        '</MedlineCitation></PubmedArticle></PubmedArticleSet>').encode()


class Encoder(PassageEncoder):
    def encode_queries(self, texts):return self.encode([[101,1000,102] for _ in texts])


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.tokenizer=Tokenizer();self.loads=[]
        self.runtime=FakeRuntime([draft(),[{'claim_id':'C1','verdict':'supported',
            'population_match':'match','comparison_match':'not_requested','outcome_match':'match',
            'reason':'supported_by_excerpt'}]])
        self.runtime.tokenizer=AnswerTokenizer();self.runtime.metadata={'fixture':True}
        self.runtime.close=Mock()
        def factory():self.loads.append('bge');return Encoder()
        self.factory=factory
        def runtime():self.loads.append('qwen');return self.runtime
        self.runtime_factory=runtime
    def run_request(self,name,events,**kwargs):
        client,_,transport=make_client(events)
        self.transport=transport
        return run_live_question(QUESTION,self.root/name,self.root/'cache',client=client,
            tokenizer_loader=lambda:(self.tokenizer,INFO),encoder_info=Encoder().info,
            encoder_factory=self.factory,answer_tokenizer_loader=AnswerTokenizer,
            runtime_factory=self.runtime_factory,release_accelerator=lambda:self.loads.append('release'),**kwargs)

    def test_fresh_query_generates_and_reviews_exact_citation_without_reranker(self):
        r=self.run_request('one',[search_json(['1'],1),xml()])
        self.assertEqual(r['status'],'complete');self.assertTrue(r['live']['fresh_search_complete'])
        self.assertEqual(r['answer']['claims'][0]['citations'][0]['pmid'],'1')
        self.assertEqual(r['answer']['claims'][0]['citations'][0]['quote'],'alpha beta gamma')
        self.assertEqual(r['runtime_load_count'],1);self.assertEqual(r['reranking']['status'],'skipped')
        self.assertEqual([limit for _,limit in self.runtime.calls],[768,512])
        self.assertLess(self.loads.index('release'),self.loads.index('qwen'))
        self.assertTrue(r['runtime_closed']);self.runtime.close.assert_called_once_with()
        self.assertEqual(len(self.transport.calls),2)
        page=(self.root/'one/report.html').read_text();self.assertIn('Fresh PubMed',page)
        self.assertNotIn('provided local collection',page)

    def test_every_request_refetches_but_unchanged_vectors_reuse_cache(self):
        a=self.run_request('a',[search_json(['1'],1),xml()],generate=False)
        before=self.loads.count('bge')
        b=self.run_request('b',[search_json(['1'],1),xml()],generate=False)
        self.assertNotEqual(a['live']['snapshot_id'],b['live']['snapshot_id'])
        self.assertNotEqual(a['live']['dataset_id'],b['live']['dataset_id'])
        self.assertEqual(b['live']['passage_cache']['encoded_inputs'],0)
        self.assertEqual(self.loads.count('bge')-before,1) # fresh query embedding only
        self.assertNotEqual(a['context']['sources'][0]['provenance'],b['context']['sources'][0]['provenance'])
        self.assertNotIn('qwen',self.loads);self.assertEqual(len(self.transport.calls),2)

    def test_membership_changes_do_not_leave_previous_articles_in_results(self):
        self.run_request('a',[search_json(['1'],1),xml()],generate=False)
        b=self.run_request('b',[search_json(['2'],1),xml('2')],generate=False)
        self.assertEqual([h['pmid'] for h in b['retrieval']['results']],['2'])
        self.assertEqual(b['live']['passage_cache']['encoded_inputs'],0)

    def test_changed_text_reencodes_and_keeps_exact_new_evidence(self):
        self.run_request('a',[search_json(['1'],1),xml()],generate=False)
        b=self.run_request('b',[search_json(['1'],1),xml(body='new source finding')],generate=False)
        self.assertEqual(b['live']['passage_cache']['encoded_inputs'],1)
        self.assertEqual(b['context']['sources'][0]['text'],'new source finding')

    def test_empty_live_result_abstains_before_encoder_or_qwen_weights(self):
        r=self.run_request('empty',[search_json([],0)])
        self.assertEqual(r['status'],'complete');self.assertEqual(r['answer']['status'],'insufficient_evidence')
        self.assertNotIn('bge',self.loads);self.assertNotIn('qwen',self.loads)
        self.assertEqual(len(self.transport.calls),1)
        self.assertTrue((self.root/'empty/dataset/manifest.json').exists())

    def test_upstream_error_never_falls_back_to_earlier_success(self):
        self.run_request('prior',[search_json(['1'],1),xml()],generate=False)
        r=self.run_request('failure',[PubMedError('fixture upstream failure')])
        self.assertEqual(r['status'],'failed');self.assertFalse(r['live']['fresh_search_complete'])
        self.assertNotIn('retrieval',r);self.assertNotIn('qwen',self.loads)
        self.assertEqual(json.loads((self.root/'failure/snapshot/manifest.json').read_text())['status'],'failed')

    def test_corrupt_cache_stops_fresh_request_without_inference_or_overwrite(self):
        self.run_request('a',[search_json(['1'],1),xml()],generate=False)
        p=next((self.root/'cache').glob('*/*.bin'));data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]))
        r=self.run_request('corrupt',[search_json(['1'],1),xml()])
        self.assertEqual(r['status'],'failed');self.assertEqual(r['error']['type'],'CacheError')
        self.assertTrue(r['live']['fresh_search_complete']);self.assertNotIn('qwen',self.loads)

    def test_existing_output_and_overlapping_cache_rejected_without_network(self):
        self.run_request('a',[search_json([],0)])
        with self.assertRaises(FileExistsError):self.run_request('a',[])
        with self.assertRaises(ValueError):
            run_live_question(QUESTION,self.root/'other',self.root/'other/cache',
                tokenizer_loader=Mock(),encoder_info={},encoder_factory=Mock(),
                answer_tokenizer_loader=Mock(),runtime_factory=Mock())
        self.assertFalse((self.root/'other').exists())

    def test_generation_failure_preserves_fresh_search_and_evidence(self):
        self.runtime.generate=Mock(side_effect=RuntimeError('fixture model failed'))
        r=self.run_request('model',[search_json(['1'],1),xml()])
        self.assertEqual(r['status'],'failed');self.assertEqual(r['error']['stage'],'answer')
        self.assertEqual(r['context']['sources'][0]['pmid'],'1')
        self.assertTrue(r['live']['fresh_search_complete'])
        self.runtime.close.assert_called_once_with()


class QueryTests(unittest.TestCase):
    def test_conditions_comparison_or_and_transparent_qualifier_policy(self):
        r=plan_query('Compare CBT and SSRIs for adolescent depression')
        self.assertEqual(r['conditions'],['depression'])
        self.assertEqual(r['interventions'],['cbt','antidepressants'])
        self.assertIn(' OR ',r['submitted_query']);self.assertNotIn('adolescent',r['submitted_query'])
        self.assertIn('qualifiers',r['limitations'])
        self.assertEqual(plan_query('CBT for GAD')['conditions'],['generalized-anxiety'])
        self.assertEqual(plan_query('Exercise for anxiety')['conditions'],['anxiety'])
    def test_explicit_override_remains_recorded_and_scoped(self):
        r=plan_query(QUESTION,'"trial"[Title]')
        self.assertEqual(r['override'],'"trial"[Title]');self.assertIn('AND',r['submitted_query'])
        self.assertIn('Depressive Disorder',r['submitted_query'])
    def test_unscoped_empty_and_invalid_override_rejected(self):
        for q in ('','How to treat diabetes?','Депрессия и лечение'):
            with self.assertRaises(ValueError):plan_query(q)
        for override in ('','\nfoo'):
            with self.assertRaises(ValueError):plan_query(QUESTION,override)
