"""Adversarial candidate-broadening checks; timing remains in the research question."""
import unittest

from medlens.pubmed_query import plan_query


class TemporalQueryTests(unittest.TestCase):
    def temporal_changes(self, result):
        return [r for r in result['query_transformations']
                if r['rule'] == 'terminal-outcome-followup']

    def test_terminal_followup_broadens_candidates_preserves_question_and_audit(self):
        question = ('In children with acute otitis media, do antibiotics compared with placebo '
                    'reduce ear pain after two to three days?')
        result = plan_query(question)
        self.assertEqual(result['question'], question)
        self.assertEqual(result['method'], 'lexical-question-scaffold-pubmed-atm-v2')
        self.assertEqual(result['submitted_query'],
                         'In children with acute otitis media, antibiotics with placebo ear pain')
        changes = self.temporal_changes(result)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]['removed_text'], 'after two to three days')
        self.assertEqual(question[changes[0]['char_start']:changes[0]['char_end']],
                         'after two to three days')
        removals = [r for r in result['query_transformations'] if 'char_start' in r]
        cursor = 0
        pieces = []
        for row in removals:
            pieces.extend((question[cursor:row['char_start']], ' '))
            cursor = row['char_end']
        pieces.append(question[cursor:])
        self.assertEqual(' '.join(''.join(pieces).split()), result['submitted_query'])

    def test_generic_numeric_and_written_followups_need_no_disease_dictionary(self):
        for timing in ('at 48 hours', 'after two days', 'after 2 to 3 days',
                       'after 2–3 days', 'at 2-3 days', 'after 3.5 weeks',
                       'after twelve months', 'at 5 years'):
            with self.subTest(timing=timing):
                question = f'Does experimental treatment X improve outcome Y {timing}?'
                result = plan_query(question)
                self.assertEqual(result['submitted_query'], 'experimental treatment X outcome Y')
                self.assertEqual(self.temporal_changes(result)[0]['removed_text'], timing)
                self.assertEqual(result['question'], question)

    def test_intervention_duration_and_treatment_start_are_preserved(self):
        questions = (
            'Does two-day therapy reduce pain?',
            'Does starting treatment after 48 hours reduce pain?',
            'Does treatment X reduce pain after two days of treatment?',
            'Does X reduce pain when administered after 48 hours?',
            'Does X reduce pain following therapy after 48 hours?',
            'Does X reduce therapy duration after two days?',
            'Does X reduce daily administration after two days?',
            'Does X reduce pain with treatment at 48 hours?',
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(self.temporal_changes(plan_query(question)), [])

    def test_population_age_and_onset_clauses_are_preserved(self):
        questions = (
            'In infants at 48 hours, does X reduce pain?',
            'Does X reduce pain in infants at two days?',
            'Does X reduce pain in children aged two years?',
            'Does X reduce mortality among children at two days?',
            'Does X improve survival after symptom onset at 48 hours?',
            'Does X improve survival in patients exposed for two days?',
            'Does X reduce pain since injury after two days?',
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertEqual(self.temporal_changes(plan_query(question)), [])

    def test_explicit_and_quoted_timing_is_untouched(self):
        expressions = (
            '  X AND "pain after two to three days"[Title/Abstract]  ',
            'Does X reduce pain at 48 hours?',
        )
        for expression in expressions:
            result = plan_query('Does X reduce pain after two days?', expression)
            self.assertEqual(result['submitted_query'], expression)
            self.assertEqual(result['query_transformations'], [])
            self.assertEqual(result['method'], 'explicit-pubmed-query-v2')
        for question in ('Does X reduce "pain after 48 hours"?',
                         'Does X reduce pain after "two to three days"?',
                         'Does "reduce pain" after 48 hours?'):
            self.assertEqual(self.temporal_changes(plan_query(question)), [])

    def test_explicit_search_operators_and_field_syntax_do_not_trigger_broadening(self):
        for question in ('Does (X OR Y) reduce pain at 48 hours?',
                         'Does X[Title/Abstract] reduce pain at 48 hours?',
                         'Does X AND Y reduce pain after 48 hours?',
                         'Does X NOT Y reduce pain after 48 hours?'):
            with self.subTest(question=question):
                self.assertEqual(self.temporal_changes(plan_query(question)), [])

    def test_negated_predicate_or_timing_is_not_removed(self):
        for question in ('Does X not reduce pain at 48 hours?',
                         'Does X never improve pain after two days?',
                         'Does X reduce pain not after two days?',
                         'Does X reduce pain without improvement after two days?',
                         'Does not taking X reduce pain after two days?'):
            with self.subTest(question=question):
                self.assertEqual(self.temporal_changes(plan_query(question)), [])

    def test_identifiers_and_ambiguous_time_relations_are_retained(self):
        for question in ('Does X reduce time-to-event after two days?',
                         'Does X reduce AT-48 hours?',
                         'Does REDUCE-IT reduce pain after 48 hours?',
                         'Does X improve pain at 48 hours after surgery?',
                         'Does X reduce pain after two days or fever after three days?'):
            with self.subTest(question=question):
                result = plan_query(question)
                if question == 'Does REDUCE-IT reduce pain after 48 hours?':
                    self.assertIn('REDUCE-IT', result['submitted_query'])
                else:
                    self.assertEqual(self.temporal_changes(result), [])

    def test_no_effect_predicate_keeps_existing_followup_search_behavior(self):
        for question in ('Compare CBT and SSRIs for adolescent depression at 12 weeks?',
                         'What is test accuracy at 48 hours?',
                         'What happens after two days?'):
            with self.subTest(question=question):
                self.assertEqual(self.temporal_changes(plan_query(question)), [])


if __name__ == '__main__':
    unittest.main()
