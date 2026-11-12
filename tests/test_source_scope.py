import unittest
from medlens.generation.source_scope import check_source_scope


def check(question, title, *text, quotes=()):
    return check_source_scope(question, title,
                              [{'index': i, 'text': t} for i, t in enumerate(text)], quotes)


def codes(result, field='blocks'):
    return {r['code'] for r in result[field]}


class ScopeGuardTests(unittest.TestCase):
    def test_negated_all_both_and_background_care_do_not_veto(self):
        q = 'Does AX17 compared with placebo reduce pain?'
        for source in ('Not all participants received AX17; only the active arm did.',
                       'It is not true that both groups received AX17.',
                       'All participants received no AX17.',
                       'Participants did not receive concomitant AX17.'):
            with self.subTest(source=source):
                r = check(q, 'BQ28 trial', source)
                self.assertFalse(codes(r) & {'intervention_given_to_all_groups', 'intervention_is_cointervention'})

    def test_added_to_half_full_dose_question_preserves_requested_regimen(self):
        q = 'How does AX17 added to half-dose BQ28 compare with full-dose BQ28?'
        r = check(q, 'AX17 as add-on to dose-reduced BQ28 versus dose-titrated BQ28.')
        self.assertNotIn('different_addon_dose_regimen', codes(r))

    def test_requested_analysis_prevents_blanket_veto_from_another_reference(self):
        q = 'What is accuracy using RX42 as the gold standard?'
        r = check(q, 'Sensor evaluation', 'RX42 served as gold standard for the primary analysis.',
                  'SC53 was used as reference standard for a separate analysis.')
        self.assertNotIn('different_reference_standard', codes(r))
        self.assertIn('multiple_reference_analyses', codes(r, 'cautions'))
        r = check(q, 'Sensor evaluation', 'We were not using RX42 as gold standard.',
                  'SC53 served as reference standard.')
        self.assertIn('different_reference_standard', codes(r))

    def test_hyphenated_48_hour_assessment_equals_two_days(self):
        r = check('Does AX17 reduce pain after two days?', 'Assessment trial',
                  'At the 48-hour assessment, pain scores were recorded.')
        self.assertNotIn('outcome_time_not_established', codes(r))

    def test_negated_adult_background_does_not_suppress_caution(self):
        for source in ('This was not adults; children were studied.',
                       'No adult patients were enrolled.', 'Adults were not enrolled.',
                       'We excluded adults from the trial.'):
            with self.subTest(source=source):
                r = check('Does AX17 reduce pain in adults?', 'Trial', source)
                self.assertIn('adult_eligibility_not_explicit', codes(r, 'cautions'))

    def test_api_validation_and_policy_are_explicit(self):
        result = check('Does AX17 reduce pain?', 'Trial')
        self.assertEqual(result['policy_version'], 'source-scope-veto-v1')
        for args in [(None, 'Trial', []), (' ', 'Trial', []), ('Question', None, []),
                     ('Question', 'Trial', 'text'), ('Question', 'Trial', [{'text': 3}]),
                     ('Question', 'Trial', [{'text': 'a', 'index': True}]),
                     ('Question', 'Trial', [{'text': 'a', 'index': 0}, {'text': 'b', 'index': 0}])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                check_source_scope(*args)
        with self.assertRaises(ValueError):
            check_source_scope('Question', 'Trial', [], selected_quotes='quote')

    def test_explicit_addon_dose_title_vetoes_different_comparison(self):
        r = check('How does Vela compare with Mira?',
                  'Vela as add-on to dose-reduced Mira versus dose-titrated Mira.',
                  'A superior result was observed.')
        self.assertIn('different_addon_dose_regimen', codes(r))

    def test_requested_addon_and_ordinary_background_care_are_preserved(self):
        q = 'How does Vela plus dose-reduced Mira compare with Mira?'
        title = 'Vela as add-on to dose-reduced Mira versus dose-titrated Mira.'
        self.assertNotIn('different_addon_dose_regimen', codes(check(q, title)))
        self.assertNotIn('different_addon_dose_regimen', codes(check(
            'How does Vela compare with Mira?', 'Vela versus Mira with stable background care.')))

    def test_consensus_and_distinct_named_reference_veto(self):
        q = 'What is accuracy using violet assay as the reference standard?'
        for text in ('Episodes used the consensus-based reference standard.',
                     'Sensitivity was calculated against green assay as gold standard.'):
            self.assertIn('different_reference_standard', codes(check(q, 'Accuracy trial', text)))
        self.assertNotIn('different_reference_standard', codes(check(
            q, 'Accuracy trial', 'Sensitivity was calculated against violet assay as gold standard.')))

    def test_combined_clinical_reference_requires_combined_question(self):
        text = 'The endpoint was defined as one symptom combined with a positive violet assay.'
        q = 'What is accuracy using violet assay as the reference standard?'
        self.assertIn('combined_clinical_reference', codes(check(q, 'Accuracy study', text)))
        q = 'What is accuracy using symptoms plus violet assay as the reference standard?'
        self.assertNotIn('combined_clinical_reference', codes(check(q, 'Accuracy study', text)))

    def test_full_source_definition_is_checked_even_when_quote_omits_it(self):
        r = check('What is accuracy using violet assay as the reference standard?',
                  'Accuracy study', 'Sensitivity was 90%.',
                  'The endpoint was defined as symptoms combined with violet assay.',
                  quotes=['Sensitivity was 90%.'])
        self.assertIn('combined_clinical_reference', codes(r))

    def test_named_progression_target_and_possessive_normalization(self):
        q = "Is exposure associated with progression from initial state to Opal’s disorder?"
        self.assertIn('progression_target_not_established', codes(check(
            q, 'Progression study', 'Progression to Quartz disorder was observed.')))
        self.assertNotIn('progression_target_not_established', codes(check(
            q, 'Opal disorder study', 'Progression was observed.')))

    def test_explicit_acronym_for_requested_target_is_retained(self):
        q = 'Is exposure associated with progression from initial state to Opal progressive disorder?'
        self.assertNotIn('progression_target_not_established', codes(check(
            q, 'Progression study', 'Progression to OPD was observed.')))

    def test_time_conversion_accepts_exact_equivalent_points_and_ranges(self):
        for request, source in [('after two to three days', 'at 48–72h'),
                                ('at two hours', 'at 120 minutes'),
                                ('after two days', 'at 48 hours'),
                                ('at three years', 'after 36 months')]:
            with self.subTest(request=request, source=source):
                r = check('Does Vela reduce pain ' + request + '?', 'Vela study',
                          'Pain was measured ' + source + '.')
                self.assertNotIn('outcome_time_not_established', codes(r))

    def test_broader_interval_or_single_boundary_is_not_exact_range(self):
        q = 'Does Vela reduce pain after two to three days?'
        for source in ('Pain improved after two to seven days.',
                       'Pain improved at 48 hours.', 'No time was specified.'):
            with self.subTest(source=source):
                self.assertIn('outcome_time_not_established', codes(check(q, 'Vela trial', source)))

    def test_all_groups_and_concomitant_care_veto_wrong_intervention(self):
        q = 'Does Vela compared with placebo reduce pain?'
        for source in ('All participants also received Vela.',
                       'Participants in both groups received Vela.',
                       'Participants received concomitant Vela.',
                       'Both arms received Vela concomitantly.'):
            with self.subTest(source=source):
                self.assertTrue(codes(check(q, 'Mira versus placebo for pain', source)) & {
                    'intervention_given_to_all_groups', 'intervention_is_cointervention'})

    def test_background_care_for_another_intervention_does_not_block(self):
        q = 'Does Vela compared with placebo reduce pain?'
        r = check(q, 'Vela versus placebo for pain', 'Both groups received water.',
                  'Participants received concomitant hydration.')
        self.assertFalse(codes(r) & {'intervention_given_to_all_groups', 'intervention_is_cointervention'})

    def test_adult_missing_is_caution_only_and_central_age_not_eligibility(self):
        for source in ('Mean age 69 years.', 'Median age 72 years.',
                       'Average age 56 years.', 'Children aged 6 years participated.',
                       'Eligibility age >=18 months.'):
            with self.subTest(source=source):
                r = check('Does Vela reduce pain in adults?', 'Vela trial', source)
                self.assertFalse(r['blocks'])
                self.assertIn('adult_eligibility_not_explicit', codes(r, 'cautions'))
        for source in ('Adults were enrolled.', 'Participants aged ≥75 years were enrolled.',
                       'Participants aged 18–42 years were enrolled.', 'Eligibility age >=18.'):
            with self.subTest(source=source):
                self.assertNotIn('adult_eligibility_not_explicit', codes(check(
                    'Does Vela reduce pain in adults?', 'Vela trial', source), 'cautions'))

    def test_every_existing_proof_span_matches_original_source(self):
        q = 'Does Vela compared with placebo reduce pain after two to three days?'
        title = 'Mira versus placebo for pain'
        sections = [{'index': 0, 'text': 'All participants received Vela.'},
                    {'index': 1, 'text': 'Pain was measured at 24 hours.'}]
        r = check_source_scope(q, title, sections)
        fields = {'title': title, 'question': q, **{f'abstract:{s["index"]}': s['text'] for s in sections}}
        for row in [*r['blocks'], *r['cautions']]:
            for proof in [*row['source_proofs'], *([row['question_proof']] if 'question_proof' in row else [])]:
                self.assertEqual(fields[proof['field']][proof['char_start']:proof['char_end']], proof['text'])

    def test_unrecognized_language_never_claims_positive_validation(self):
        r = check('Explain an unfamiliar relation', 'Unfamiliar title', 'A complex relation exists.')
        self.assertEqual(r['recognized_checks'], [])
        self.assertEqual(r['blocks'], [])
        self.assertFalse(r['positive_validation_performed'])


if __name__ == '__main__':
    unittest.main()
