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
    def test_original_question_and_every_lexical_transformation_are_recorded(self):
        question='Compare CBT and SSRIs for adolescent depression at 12 weeks?'
        result=plan_query(question)
        self.assertEqual(result['question'],question)
        self.assertEqual(result['submitted_query'],'CBT and SSRIs for adolescent depression at 12 weeks')
        self.assertEqual(result['method'],'lexical-question-scaffold-pubmed-atm-v1')
        self.assertEqual(result['conditions'],[])
        self.assertEqual(result['interventions'],[])
        removals=[change for change in result['query_transformations'] if 'char_start' in change]
        self.assertEqual([change['removed_text'] for change in removals],['Compare','?'])
        cursor=0;pieces=[]
        for change in removals:
            start,end=change['char_start'],change['char_end']
            self.assertEqual(question[start:end],change['removed_text'])
            pieces.extend((question[cursor:start],' '));cursor=end
        pieces.append(question[cursor:])
        self.assertEqual(' '.join(''.join(pieces).split()),result['submitted_query'])
        self.assertIn('query_translation',result['limitations'])

    def test_varied_biomedical_questions_do_not_need_a_disease_allowlist(self):
        questions=(
            'Does aspirin reduce myocardial infarction risk in older adults?',
            'Does metformin reduce HbA1c in adults with type 2 diabetes?',
            'Does remdesivir reduce hospitalization in adults with COVID-19?',
            'Does exercise reduce migraine frequency in adolescents?',
            'Does oral rehydration reduce diarrhea duration in children under 5?',
            'How does TNF-α regulate NF-κB signaling in macrophages?',
            'Is β-amyloid associated with cognitive decline in APOE ε4 carriers?',
        )
        for question in questions:
            with self.subTest(question=question):
                result=plan_query(question)
                self.assertEqual(result['question'],question)
                self.assertTrue(result['submitted_query'])
                self.assertNotIn('Depressive Disorder',result['submitted_query'])
        self.assertEqual(plan_query('How does TNF-α regulate NF-κB signaling?')['submitted_query'],
            'TNF-α regulate NF-κB signaling')

    def test_question_operators_are_not_required_pubmed_content_terms(self):
        question='In adults, how does treatment X compare with treatment Y for outcome Z?'
        result=plan_query(question)
        self.assertEqual(result['submitted_query'],'In adults, treatment X with treatment Y for outcome Z')
        self.assertEqual(result['question'],question)
        self.assertIn('compare',[change.get('removed_text') for change in result['query_transformations']])

    def test_reference_role_scaffolding_does_not_remove_the_named_reference(self):
        result=plan_query('What is the accuracy of sensor X using assay Y as a gold standard?')
        self.assertEqual(result['submitted_query'],'the accuracy of sensor X assay Y')
        self.assertIn('assay Y',result['submitted_query'])
        self.assertIn('as a gold standard',[change.get('removed_text') for change in result['query_transformations']])

    def test_quotes_biomarkers_and_directly_negated_predicates_are_protected(self):
        question='Does REDUCE-IT improve HER2+ and BRCA1/2 outcomes involving TNF‑associated signaling?'
        term=plan_query(question)['submitted_query']
        for token in ('REDUCE-IT','HER2+','BRCA1/2','TNF‑associated'):
            self.assertIn(token,term)
        self.assertEqual(plan_query('Does "compare" affect β-amyloid?')['submitted_query'],'"compare" β-amyloid')
        self.assertEqual(plan_query('Does X not reduce Y?')['submitted_query'],'X not reduce Y')
        self.assertEqual(plan_query('IS expression and DO concentration')['submitted_query'],'IS expression and DO concentration')

    def test_explicit_override_is_exact_without_scaffold_or_boolean_injection(self):
        expression='  (metformin[Title/Abstract] OR GLP-1[Title/Abstract]) AND HbA1c  '
        result=plan_query(QUESTION,expression)
        self.assertEqual(result['override'],expression)
        self.assertEqual(result['submitted_query'],expression)
        self.assertEqual(result['method'],'explicit-pubmed-query-v2')
        self.assertEqual(result['query_transformations'],[])
        self.assertEqual(plan_query(QUESTION,'"compare"[Journal]')['submitted_query'],'"compare"[Journal]')
        self.assertEqual(plan_query(QUESTION,QUESTION)['submitted_query'],QUESTION)

    def test_transport_validation_does_not_claim_language_or_topic_detection(self):
        for question in ('What is the weather?', 'Лечение диабета', '  HER2+ breast cancer  '):
            result=plan_query(question)
            self.assertEqual(result['question'],question)
            self.assertEqual(result['language_detection'],'not-performed')
            self.assertEqual(result['topic_detection'],'not-performed')
            self.assertFalse(result['automatic_translation'])
            self.assertEqual(result['supported_scope'],'English biomedical research questions')

    def test_question_and_override_boundaries_and_no_searchable_terms(self):
        self.assertEqual(plan_query('a'*8000)['submitted_query'],'a'*8000)
        self.assertEqual(plan_query(QUESTION,'β'*4000)['submitted_query'],'β'*4000)
        for question in (None,True,42,'','   ','a'*8001,'Does compare reduce?'):
            with self.subTest(question_type=type(question).__name__),self.assertRaises(ValueError):
                plan_query(question)
        for expression in (True,42,'','   ','a'*4001):
            with self.subTest(override_type=type(expression).__name__),self.assertRaises(ValueError):
                plan_query(QUESTION,expression)

    def test_control_characters_are_rejected_without_changing_valid_unicode(self):
        for character in ('\x00','\t','\n','\r','\x1f','\x7f','\x85','\u200b','\u202e','\u2028','\u2029','\ud800'):
            with self.subTest(control=ascii(character)):
                with self.assertRaises(ValueError):plan_query('IL-6'+character+' diabetes')
                with self.assertRaises(ValueError):plan_query(QUESTION,'HER2'+character+' cancer')
        expression='TNF-α[Title/Abstract] AND β-blockers AND HER2+'
        self.assertEqual(plan_query('BRCA1/2 and HER2+ breast cancer',expression)['submitted_query'],expression)
