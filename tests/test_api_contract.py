"""Public inputs and privacy boundaries without any model or network execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from medlens.api_contract import RequestValidationError, SearchRequest, public_report


SECRET = 'SECRET_SERVER_DIAGNOSTIC'
PRIVATE = '/Users/server/private/model-cache/' + SECRET
XML_PATH = './PubmedArticle[1]/MedlineCitation/Article/Abstract/AbstractText[1]'
QUOTE = 'Adults receiving agent A had lower symptom scores (β = 0.2).'


def fixture(style='source-findings'):
    citation = {'citation_id': 'E1', 'quote_id': 'E1.Q1', 'pmid': '123',
        'window_id': 'window-1', 'text_version_id': 'pubmed:123:version',
        'source_path': XML_PATH, 'section_index': 0, 'quote': QUOTE,
        'matches': [{'char_start': 0, 'char_end': len(QUOTE), 'private': PRIVATE}],
        'ambiguous': False, 'provenance': {'source_file': PRIVATE},
        'tokens': [SECRET], 'model_calls': [SECRET]}
    source = {**citation, 'title': 'A source title', 'text': QUOTE,
        'char_start': 0, 'char_end': len(QUOTE), 'section': 'RESULTS',
        'journal': 'A journal', 'publication_date': '2026-01',
        'article_rank': 1, 'window_cosine': .8,
        'quote_sentence_spans': [{'char_start': 0, 'char_end': len(QUOTE), 'path': PRIVATE}]}
    return {'schema_version': 1, 'request_id': 'request-1', 'question': 'Does agent A reduce symptoms?',
        'status': 'complete', 'stage': 'complete', 'answer_style': style,
        'input_paths': {'model': PRIVATE}, 'cache': PRIVATE,
        'runtime': {'model_dir': PRIVATE, 'stderr_path': PRIVATE, 'tokens': [SECRET]},
        'error': {'message': PRIVATE}, 'cleanup_error': {'message': PRIVATE},
        'model_calls': [SECRET], 'memory_samples': [{'path': PRIVATE}],
        'independent_validation': True,
        'scope': {'source': 'live-pubmed-candidate-subset', 'candidate_count': 2,
                  'searchable_count': 2, 'window_count': 3, 'directory': PRIVATE},
        'live': {'source': 'PubMed ESearch/EFetch', 'fresh_search_complete': True,
                 'candidate_limit': 100, 'total_pubmed_matches': 5, 'fetched_article_count': 2,
                 'snapshot_directory': PRIVATE, 'passage_cache': {'path': PRIVATE},
                 'query_translation': 'agent A symptoms',
                 'query_plan': {'method': 'explicit-pubmed-query-v2',
                     'question': 'Does agent A reduce symptoms?', 'submitted_query': 'agent A symptoms',
                     'override': 'agent A symptoms', 'automatic_translation': False,
                     'query_transformations': [{'rule': 'test', 'char_start': 0, 'char_end': 4,
                                                'removed_text': 'Does', 'private_path': PRIVATE}],
                     'model_calls': [SECRET]}},
        'retrieval': {'method': 'dense-cosine-max-window', 'results': [
            {'rank': 1, 'pmid': '123', 'title': 'A source title', 'score': .8,
             'url': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'best_window': {'provenance': PRIVATE}},
            {'rank': 2, 'pmid': '456', 'title': 'Another source', 'score': .7,
             'url': 'file://' + PRIVATE}]},
        'context': {'sources': [source], 'evidence_payload': SECRET,
                    'tokenizer': {'path': PRIVATE}, 'evidence_tokens': 200},
        'answer': {'status': 'partial', 'answer_style': style,
            'claims': [{'claim_id': 'C1', 'text': QUOTE, 'citations': [citation],
                'source_title': 'A source title', 'source_article_id': '123',
                'scope_cautions': [{'code': 'population_unclear', 'message': 'Eligibility is uncertain.',
                    'source_proofs': [{'field': 'abstract:0', 'char_start': 0,
                        'char_end': 6, 'text': 'Adults', 'source_file': PRIVATE,
                        'provenance': {'path': PRIVATE}}], 'private': PRIVATE}],
                'prompt': SECRET}],
            'limitations': ['limited_excerpts'], 'quote_identity_verified': True,
            'semantic_support_verified': False, 'generated_factual_prose': False,
            'answer_completeness_established': False,
            'verification_status': 'scientific-review-required',
            'support_check': 'not-performed-source-quotation-only',
            'draft': {'text': SECRET}, 'model_calls': [{'prompt': SECRET}],
            'generator_evidence_payload': SECRET, 'validation_error': {'message': PRIVATE},
            'reason': PRIVATE},
        'timings': {'pipeline_seconds': 1.5, 'answer_seconds': 1.0, 'stderr_path': PRIVATE,
                    'UNKNOWN_' + SECRET: 99},
        'preparation_timings': {'pubmed_download_seconds': .5, 'cache_path': PRIVATE},
        'request_seconds': 2.0, 'runtime_closed': True}


class SearchRequestTests(unittest.TestCase):
    def test_defaults_are_frozen_and_keep_original_question(self):
        question = '  Does agent A reduce symptoms after two days?  '
        request = SearchRequest.from_payload({'question': question})
        self.assertEqual(asdict(request), {'question': question, 'pubmed_query': None,
            'limit': 100, 'top_k': 10, 'generate': True, 'answer_style': 'draft'})
        with self.assertRaises(FrozenInstanceError):request.question = 'changed'

    def test_exact_pubmed_override_and_parameter_boundaries(self):
        override = '"agent A"[Title] AND symptoms'
        for limit, top_k in ((1, 1), (300, 10)):
            request = SearchRequest.from_payload({'question': 'Why?', 'pubmed_query': override,
                'limit': limit, 'top_k': top_k, 'generate': False, 'answer_style': 'source-findings'})
            self.assertEqual(request.pubmed_query, override)
            self.assertEqual(request.question, 'Why?')
        self.assertIsNone(SearchRequest.from_payload({'question': 'Symptoms?', 'pubmed_query': None}).pubmed_query)

    def test_missing_unknown_and_nonobject_inputs_rejected_without_echoing_values(self):
        for payload in (None, [], 'question', {}, {'question': None}, {'question': 3},
                        {'question': ['symptoms']}, {'question': 'Symptoms?', 'model': PRIVATE},
                        {'question': 'Symptoms?', 'output_dir': PRIVATE},
                        {'question': 'Symptoms?', 'engine': 'mlx'}):
            with self.subTest(payload=payload), self.assertRaises(RequestValidationError) as caught:
                SearchRequest.from_payload(payload)
            self.assertNotIn(PRIVATE, str(caught.exception))

    def test_ints_do_not_accept_bools_numeric_strings_floats_or_null(self):
        for field, values in {'limit': [True, False, '100', 100.0, None, 0, 301],
                              'top_k': [True, '10', 10.0, None, 0, 11]}.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(RequestValidationError):
                    SearchRequest.from_payload({'question': 'Symptoms?', field: value})

    def test_boolean_and_style_are_strict(self):
        for value in (0, 1, 'true', None):
            with self.subTest(value=value), self.assertRaises(RequestValidationError):
                SearchRequest.from_payload({'question': 'Symptoms?', 'generate': value})
        for value in (None, True, 'synthesis', '', 4):
            with self.subTest(value=value), self.assertRaises(RequestValidationError):
                SearchRequest.from_payload({'question': 'Symptoms?', 'answer_style': value})

    def test_real_planner_limits_controls_and_empty_scaffold_are_reused(self):
        SearchRequest('a' * 8000, pubmed_query='b' * 4000)
        for payload in ({'question': 'a' * 8001}, {'question': 'Symptoms?', 'pubmed_query': 'b' * 4001},
                        {'question': 'Does?'}, {'question': '  '},
                        {'question': 'Symptoms?', 'pubmed_query': ''}):
            with self.subTest(payload=str(payload)[:100]), self.assertRaises(RequestValidationError):
                SearchRequest.from_payload(payload)
        for char in ('\n', '\r', '\t', '\x00', '\u200b', '\u2028', '\ud800'):
            for field in ('question', 'pubmed_query'):
                with self.subTest(field=field, char=repr(char)), self.assertRaises(RequestValidationError):
                    SearchRequest.from_payload({'question': 'Symptoms?', field: 'word' + char + 'word'})
        with self.assertRaises(RequestValidationError):SearchRequest('Symptoms?', pubmed_query=False)

    def test_fresh_import_and_validation_need_no_ml_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        code = '''import builtins,sys
sys.path.insert(0, SRC)
original = builtins.__import__
def blocked(name,*args,**kwargs):
    if name.split('.')[0] in {'numpy','torch','transformers','mlx','mlx_lm'}:
        raise AssertionError('Unexpected ML import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=blocked
from medlens.api_contract import SearchRequest,public_report
assert SearchRequest.from_payload({'question':'Symptoms?'}).limit == 100
assert public_report({})['independent_validation'] is False
print('stdlib-only')
'''.replace('SRC', repr(str(root/'src')))
        child = subprocess.run([sys.executable, '-c', code], cwd=root, text=True, capture_output=True)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stdout.strip(), 'stdlib-only')


class PublicReportTests(unittest.TestCase):
    def test_source_findings_keep_quotes_spans_cautions_without_private_artifacts(self):
        raw = fixture(); before = deepcopy(raw)
        result = public_report(raw)
        self.assertEqual(raw, before)
        claim = result['answer']['claims'][0]; citation = claim['citations'][0]
        self.assertEqual(claim['text'], QUOTE)
        self.assertEqual(claim['source_title'], 'A source title')
        self.assertEqual(citation['quote'], QUOTE)
        self.assertEqual(citation['source_path'], XML_PATH)
        self.assertEqual(citation['text_version_id'], 'pubmed:123:version')
        self.assertEqual(citation['matches'], [{'char_start': 0, 'char_end': len(QUOTE)}])
        self.assertEqual(claim['scope_cautions'][0]['source_proofs'][0],
                         {'field': 'abstract:0', 'char_start': 0, 'char_end': 6, 'text': 'Adults'})
        self.assertEqual(result['evidence'][0]['text'], QUOTE)
        self.assertEqual(result['articles'][1]['rank'], 2)
        self.assertEqual(result['articles'][1]['url'], 'https://pubmed.ncbi.nlm.nih.gov/456/')
        self.assertTrue(result['answer']['quote_identity_verified'])
        self.assertFalse(result['answer']['semantic_support_verified'])
        self.assertFalse(result['independent_validation'])
        text = json.dumps(result, ensure_ascii=False, allow_nan=False)
        self.assertNotIn(SECRET, text)
        self.assertNotIn(PRIVATE, text)
        for name in ('model_calls', 'input_paths', 'stderr_path', 'generator_evidence_payload', 'draft', 'tokens'):
            self.assertNotIn('"' + name + '"', text)
        result['answer']['claims'][0]['citations'][0]['matches'][0]['char_end'] = 1
        self.assertEqual(raw, before)

    def test_draft_review_is_not_promoted_to_independent_verification(self):
        raw = fixture('draft'); raw['answer']['status'] = 'answered'
        raw['answer']['support_check'] = 'same-model-review-not-independent'
        for name in ('quote_identity_verified', 'semantic_support_verified', 'generated_factual_prose'):
            raw['answer'].pop(name)
        raw['answer']['review'] = [{'verdict': 'supported', 'private': PRIVATE}]
        raw['answer']['independent_validation'] = True
        result = public_report(raw)
        self.assertEqual(result['answer']['status'], 'answered')
        self.assertEqual(result['answer']['support_check'], 'same-model-review-not-independent')
        self.assertNotIn('semantic_support_verified', result['answer'])
        self.assertNotIn('quote_identity_verified', result['answer'])
        self.assertNotIn('review', result['answer'])
        self.assertFalse(result['independent_validation'])

    def test_pipeline_failure_retains_completed_sources_and_separate_answer_status(self):
        raw = fixture(); raw.update(status='failed', stage='answer')
        raw['answer'].update(status='validation_failed')
        raw['answer']['claims'] = []
        result = public_report(raw)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['stage'], 'answer')
        self.assertEqual(result['answer']['status'], 'validation_failed')
        self.assertTrue(result['articles'])
        self.assertTrue(result['evidence'])
        self.assertNotIn('error', result)
        self.assertNotIn('reason', result['answer'])
        self.assertNotIn(SECRET, json.dumps(result))

    def test_known_program_reasons_preserved_but_exception_reasons_never_escape(self):
        for reason in ('empty-pubmed-result', 'model-abstained',
                       'no-claims-passed-support-review', 'generation-disabled',
                       'no-direct-scoped-source-findings'):
            raw = {'status': 'complete', 'answer': {
                'status': 'insufficient_evidence', 'claims': [], 'reason': reason}}
            self.assertEqual(public_report(raw)['answer']['reason'], reason)
        for status, reason in (('validation_failed', PRIVATE),
                               ('validation_failed', 'model-abstained'),
                               ('insufficient_evidence', 'unknown-program-code'),
                               ('insufficient_evidence', PRIVATE)):
            raw = {'status': 'complete', 'answer': {
                'status': status, 'claims': [], 'reason': reason}}
            self.assertNotIn('reason', public_report(raw)['answer'])

    def test_empty_pubmed_and_absent_context_are_valid_public_results(self):
        raw = {'status': 'complete', 'answer': {'status': 'insufficient_evidence', 'claims': []},
               'live': {'fresh_search_complete': True, 'total_pubmed_matches': 0,
                        'fetched_article_count': 0, 'query_plan': {'submitted_query': 'Symptoms'}},
               'context': {'sources': [], 'status': 'no-source-excerpts'}}
        result = public_report(raw)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['answer']['status'], 'insufficient_evidence')
        self.assertEqual(result['articles'], [])
        self.assertEqual(result['evidence'], [])
        self.assertEqual(result['search']['fetched_article_count'], 0)
        self.assertTrue(result['search']['fresh_search_complete'])
        self.assertIsNone(public_report({'status': 'failed'})['answer']['status'])

    def test_fulltext_citation_preserves_license_and_xml_location(self):
        raw = fixture()
        citation = raw['answer']['claims'][0]['citations'][0]
        citation.update(source_kind='fulltext', pmcid='PMC123',
                        license_uri='https://creativecommons.org/licenses/by/4.0/',
                        source_path='./body[1]/sec[2]/p[3]')
        projected = public_report(raw)['answer']['claims'][0]['citations'][0]
        for name in ('source_kind', 'pmcid', 'license_uri', 'source_path', 'text_version_id', 'quote'):
            self.assertEqual(projected[name], citation[name])
        self.assertEqual(projected['url'], 'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/')

    def test_nonfinite_timings_and_filesystem_locations_do_not_escape(self):
        raw = fixture()
        raw['timings'].update(answer_seconds=math.nan, context_seconds=math.inf,
                              pipeline_seconds=2.0, retrieval_seconds=True)
        raw['answer']['claims'][0]['citations'][0]['source_path'] = PRIVATE
        raw['answer']['claims'][0]['citations'][0]['license_uri'] = 'file://' + PRIVATE
        result = public_report(raw)
        self.assertEqual(result['timings'], {'pipeline_seconds': 2.0})
        self.assertNotIn('source_path', result['answer']['claims'][0]['citations'][0])
        self.assertNotIn('license_uri', result['answer']['claims'][0]['citations'][0])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':unittest.main()
