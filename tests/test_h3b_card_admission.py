"""Fictional adult-card admission checks; no native runtime, model or network."""
from copy import deepcopy
import json
import unittest

from medlens.generation import source_findings as c
from test_source_findings import make, plan_data, select
from test_answers import FakeRuntime

QUESTION = 'Does agent Z reduce pain in adults?'
ADULT_SETUP = 'Adult participants were randomized to agent Z or placebo.'
UNKNOWN_SETUP = 'Participants were randomized to agent Z or placebo.'
RESULT = 'Pain was reduced by agent Z compared with placebo.'


def article(setup, result=RESULT):
    return [('METHODS', setup), ('RESULTS', result)]


def selected(plans):
    return {'articles': [select(article_id=plan['article_id'],
        scope=plan['scope_quote_ids'][0] if plan['scope_required'] else None,
        findings=plan['finding_quote_ids'][:2])['articles'][0] for plan in plans],
        'limitations': []}


class RequiredAdultAdmission(unittest.TestCase):
    def run_fixture(self, abstracts, question=QUESTION):
        context, articles = make(abstracts)
        before_context, before_articles = deepcopy(context), deepcopy(articles)
        index, plans, checks = plan_data(context, articles, question)
        self.assertTrue(all(plan['allow_direct'] for plan in plans))
        raw = selected(plans)
        runtime = FakeRuntime([raw])
        answer = c.generate_source_findings_answer(question, context, runtime,
            source_articles=articles)
        self.assertEqual(context, before_context)
        self.assertEqual(articles, before_articles)
        self.assertEqual(len(runtime.calls), 1)
        self.assertEqual([cap for _, cap in runtime.calls], [768])
        self.assertEqual(answer['selection'], raw)
        self.assertEqual(json.loads(answer['model_calls'][0]['text']), raw)
        self.assertFalse(answer['semantic_support_verified'])
        self.assertLessEqual(len(answer['claims']), 3)
        self.assertTrue(all(len(claim['text']) <= 1600 for claim in answer['claims']))
        return answer, context, articles

    def test_mixed_cards_keep_explicit_adult_result(self):
        answer, context, articles = self.run_fixture([
            article(ADULT_SETUP), article(UNKNOWN_SETUP)])
        self.assertEqual(answer['status'], 'partial')
        self.assertEqual([claim['source_article_id'] for claim in answer['claims']], ['1'])
        self.assertNotIn('population_unclear', answer['limitations'])
        admission = answer['card_admission']
        self.assertEqual(admission['included_claim_ids'], ['C1'])
        self.assertEqual(admission['excluded_cards'][0]['old_claim_id'], 'C2')
        self.assertEqual([source['pmid'] for source in context['sources']], ['1', '1', '2', '2'])
        self.assertEqual(set(answer['scope_checks']), set(articles))

    def test_unknown_first_does_not_renumber_valid_second_claim(self):
        answer, _, _ = self.run_fixture([article(UNKNOWN_SETUP), article(ADULT_SETUP)])
        self.assertEqual([claim['claim_id'] for claim in answer['claims']], ['C2'])
        self.assertEqual([claim['claim_id'] for claim in answer['draft']['claims']], ['C1', 'C2'])
        admission = answer['card_admission']
        self.assertEqual(admission['included_claim_ids'], ['C2'])
        excluded = admission['excluded_cards'][0]
        self.assertEqual((excluded['old_claim_id'], excluded['source_article_id']), ('C1', '1'))
        self.assertEqual(excluded['classification'], 'unknown-not-mismatch')
        self.assertEqual(excluded['source_proofs'], [])
        check = answer['scope_checks']['1']
        self.assertEqual(excluded['source_input_sha256'], check['input_sha256'])
        self.assertEqual(excluded['text_version_id'], check['text_version_id'])

    def test_all_uncertain_is_scientific_refusal_with_raw_decisions_preserved(self):
        answer, _, _ = self.run_fixture([article(UNKNOWN_SETUP), article(UNKNOWN_SETUP)])
        self.assertEqual(answer['status'], 'insufficient_evidence')
        self.assertEqual(answer['reason'], 'required-adult-eligibility-not-established')
        self.assertEqual(answer['claims'], [])
        self.assertIn('population_unclear', answer['limitations'])
        self.assertNotIn('validation_error', answer)
        self.assertEqual(len(answer['draft']['claims']), 2)
        self.assertEqual([row['effective_relation'] for row in answer['source_relations']], ['direct', 'direct'])
        self.assertEqual([row['raw_relation'] for row in answer['source_relations']], ['direct', 'direct'])
        self.assertEqual(len(answer['card_admission']['excluded_cards']), 2)
        self.assertFalse(answer['card_admission']['positive_validation_performed'])

    def test_question_without_required_adults_keeps_unknown_age(self):
        answer, _, _ = self.run_fixture([article(UNKNOWN_SETUP)], 'Does agent Z reduce pain?')
        self.assertEqual(answer['status'], 'partial')
        self.assertEqual(answer['card_admission']['excluded_cards'], [])
        self.assertFalse(answer['card_admission']['requested_adults'])

    def test_negated_or_compound_adult_word_does_not_add_hard_admission(self):
        for question in ('Does agent Z reduce pain without adults?',
                         'Does agent Z reduce pain in adult-onset syndrome?',
                         'Does agent Z reduce pain in non-adults?',
                         'Does agent Z reduce pain in adult/pediatric cohorts?'):
            with self.subTest(question=question):
                answer, _, _ = self.run_fixture([article(UNKNOWN_SETUP)], question)
                self.assertEqual(answer['status'], 'partial')
                self.assertFalse(answer['card_admission']['requested_adults'])
                self.assertEqual(answer['card_admission']['excluded_cards'], [])

    def test_recognized_lower_age_bound_is_kept(self):
        answer, _, _ = self.run_fixture([
            article('Participants aged 18 to 75 years were randomized to agent Z or placebo.')])
        self.assertEqual(answer['status'], 'partial')
        self.assertEqual(answer['card_admission']['included_claim_ids'], ['C1'])
        self.assertEqual(answer['scope_checks']['1']['cautions'], [])

    def test_mean_age_alone_is_excluded_with_exact_source_proof(self):
        setup = 'Participants were randomized to agent Z or placebo; mean age was 42 years.'
        answer, _, articles = self.run_fixture([article(setup)])
        self.assertEqual(answer['status'], 'insufficient_evidence')
        excluded = answer['card_admission']['excluded_cards'][0]
        self.assertEqual(excluded['reason_code'], 'adult_eligibility_not_explicit')
        self.assertEqual(excluded['classification'], 'unknown-not-mismatch')
        self.assertTrue(excluded['source_proofs'])
        for proof in excluded['source_proofs']:
            field, index = proof['field'].split(':')
            self.assertEqual(field, 'abstract')
            text = articles['1']['abstract']['sections'][int(index)]['text']
            self.assertEqual(text[proof['char_start']:proof['char_end']], proof['text'])
            self.assertIn('mean age', proof['text'])
        self.assertNotIn('adult', answer['draft']['claims'][0]['text'].lower())

    def test_explicit_adult_null_finding_remains_useful(self):
        null = 'Pain was not reduced by agent Z compared with placebo.'
        answer, _, _ = self.run_fixture([article(ADULT_SETUP, null)])
        self.assertEqual(answer['status'], 'partial')
        self.assertIn(null, answer['claims'][0]['text'])
        self.assertEqual(answer['card_admission']['excluded_cards'], [])

    def test_existing_explicit_regimen_mismatch_still_blocks_before_model(self):
        context, articles = make([article('Adults were assigned to A plus dose-adjusted B or B.')],
            title='Agent A as add-on to dose-adjusted B versus B in a trial')
        before = deepcopy(context)
        runtime = FakeRuntime([])
        answer = c.generate_source_findings_answer('How does agent A compare with agent B in adults?',
            context, runtime, source_articles=articles)
        self.assertEqual(answer['status'], 'insufficient_evidence')
        self.assertEqual(runtime.calls, [])
        self.assertTrue(answer['scope_checks']['1']['blocks'])
        self.assertEqual(context, before)

    def test_three_article_pool_keeps_valid_cards_without_expanding_caps(self):
        answer, context, _ = self.run_fixture([
            article(ADULT_SETUP), article(UNKNOWN_SETUP), article(ADULT_SETUP)])
        self.assertEqual([claim['claim_id'] for claim in answer['claims']], ['C1', 'C3'])
        self.assertEqual(len(answer['draft']['claims']), 3)
        self.assertEqual(len(answer['selector_menu']['article_plans']), 3)
        self.assertEqual(answer['selector_menu']['max_sentence_characters'], 532)
        self.assertEqual(answer['selector_menu']['max_materialized_characters'], 1598)
        question_proof = answer['card_admission']['question_proofs'][0]
        self.assertEqual(QUESTION[question_proof['char_start']:question_proof['char_end']], 'adults')
        self.assertEqual(context['config']['evidence_tokens'], 4000)


if __name__ == '__main__':
    unittest.main()
