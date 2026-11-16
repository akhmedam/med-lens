"""Fictional primary-text endpoint contracts; no model, source acquisition or entities."""

import unittest

from medlens.generation import source_scope as guard


DIAGNOSTIC = 'What is the diagnostic accuracy of sensor K for clinically significant velum condition?'
PROGRESSION = 'Does agent M reduce progression to fulvous state?'
DIAGNOSTIC_CODE = 'selected_requested_diagnostic_qualifier_unestablished'
REGRESSION_CODE = 'selected_clinical_regression_instead_of_requested_progression'
COHORT = 'Adult participants were followed in a clinical cohort.'
ACCURACY = 'The sensitivity for velum condition was 0.81.'
REGRESSION = 'The probability of velum regression was 0.60.'


def section(index, label, text):
    return {'index': index, 'label': label, 'nlm_category': label.casefold(), 'text': text}


class SelectedEndpointContracts(unittest.TestCase):
    def check(self, question, sections, selected, *, title='Fictional endpoint study'):
        result = guard.check_selected_findings(question, title, sections, selected)
        self.assertFalse(result['positive_validation_performed'])
        fields = {f'abstract:{s["index"]}': s['text'] for s in sections}
        for block in result['blocks']:
            if block['code'] not in {DIAGNOSTIC_CODE, REGRESSION_CODE}:
                continue
            proof = block['question_proof']
            self.assertEqual(proof['field'], 'question')
            self.assertEqual(question[proof['char_start']:proof['char_end']], proof['text'])
            self.assertTrue(block['source_proofs'])
            for proof in block['source_proofs']:
                self.assertIn(proof['field'], fields)
                self.assertEqual(fields[proof['field']][proof['char_start']:proof['char_end']], proof['text'])
                self.assertIn(proof['text'], selected)
        return {block['code'] for block in result['blocks']}

    def diagnostic(self, *, extra=(), selected=None, title='Fictional endpoint study'):
        sections = [section(11, 'METHODS', COHORT), section(23, 'RESULTS', ACCURACY), *extra]
        return self.check(DIAGNOSTIC, sections, [ACCURACY] if selected is None else selected, title=title)

    def progression(self, *, extra=(), selected=None, cohort=COHORT):
        sections = [section(11, 'METHODS', cohort), section(23, 'RESULTS', REGRESSION), *extra]
        return self.check(PROGRESSION, sections, [REGRESSION] if selected is None else selected)

    def test_diagnostic_title_background_and_future_conclusion_do_not_establish_selected_endpoint(self):
        cases = [
            ('Accuracy for clinically significant velum condition', []),
            ('Fictional study', [section(31, 'BACKGROUND', 'Clinically significant velum condition is the intended target.')]),
            ('Fictional study', [section(31, 'CONCLUSIONS', 'Sensor K could detect clinically significant velum condition in future work.')]),
            ('Fictional study', [section(31, 'METHODS', 'Clinically significant lumen condition was defined as a separate endpoint.')]),
            ('Fictional study', [section(31, 'RESULTS', 'The sensitivity for clinically significant lumen condition was 0.76.')]),
        ]
        for title, extra in cases:
            with self.subTest(title=title, extra=extra):
                self.assertIn(DIAGNOSTIC_CODE, self.diagnostic(title=title, extra=extra))

    def test_actual_methods_qualified_definition_preserves_possible_result_shorthand(self):
        definition = 'Clinically significant velum condition was defined as the prespecified primary endpoint.'
        self.assertNotIn(DIAGNOSTIC_CODE, self.diagnostic(extra=[section(31, 'METHODS', definition)]))

    def test_actual_qualified_results_analysis_preserves_unqualified_selection(self):
        qualified = 'The sensitivity for clinically significant velum condition was 0.76.'
        self.assertNotIn(DIAGNOSTIC_CODE, self.diagnostic(extra=[section(31, 'RESULTS', qualified)]))

    def test_selected_qualified_observed_result_preserves_requested_contract(self):
        qualified = 'Accuracy for clinically significant velum condition was 0.77.'
        for role in ('RESULTS', 'CONCLUSIONS'):
            with self.subTest(role=role):
                self.assertNotIn(DIAGNOSTIC_CODE, self.diagnostic(
                    extra=[section(31, role, qualified)], selected=[ACCURACY, qualified]))
        unrelated = 'Accuracy for clinically significant lumen condition was 0.77.'
        self.assertIn(DIAGNOSTIC_CODE, self.diagnostic(
            extra=[section(31, 'RESULTS', unrelated)], selected=[ACCURACY, unrelated]))

    def test_uncertain_acronym_and_unrecognized_question_target_remain_inconclusive(self):
        alias = 'The sensitivity for velum condition was 0.81 using csVC as the endpoint.'
        sections = [section(11, 'METHODS', COHORT), section(23, 'RESULTS', alias)]
        self.assertNotIn(DIAGNOSTIC_CODE, self.check(DIAGNOSTIC, sections, [alias]))
        definition = 'The target VC was defined as the prespecified endpoint.'
        self.assertNotIn(DIAGNOSTIC_CODE, self.diagnostic(extra=[section(31, 'METHODS', definition)]))
        for alias_name in ('VC', 'csVC'):
            acronym_question = f'What is the diagnostic accuracy of sensor K for clinically significant {alias_name}?'
            self.assertNotIn(DIAGNOSTIC_CODE, self.check(acronym_question,
                [section(11, 'METHODS', COHORT), section(23, 'RESULTS', ACCURACY)], [ACCURACY]))

    def test_diagnostic_requires_selected_observed_results_with_same_literal_target(self):
        cases = [
            ('RESULTS', 'The sensitivity for lumen condition was 0.81.'),
            ('RESULTS', 'The sensitivity for velum condition could be 0.81 in future work.'),
            ('METHODS', ACCURACY),
            ('', ACCURACY),
        ]
        for role, quote in cases:
            with self.subTest(role=role, quote=quote):
                self.assertNotIn(DIAGNOSTIC_CODE, self.check(DIAGNOSTIC,
                    [section(11, 'METHODS', COHORT), section(23, role, quote)], [quote]))

    def test_observed_regression_endpoint_risk_hazard_probability_and_inverse_are_demoted(self):
        quotes = [
            'The probability of velum regression was 0.60.',
            'The risk of velum regression was lower in the exposed group.',
            'The hazard of velum regression was 0.70.',
            'The regression probability was 0.60.',
            'The regression risk was lower in the exposed group.',
            'The regression hazard was 0.70.',
        ]
        for quote in quotes:
            with self.subTest(quote=quote):
                self.assertIn(REGRESSION_CODE, self.check(PROGRESSION,
                    [section(11, 'METHODS', COHORT), section(23, 'RESULTS', quote)], [quote]))

    def test_local_statistical_regression_models_and_regression_to_mean_are_excluded(self):
        quotes = [
            'The probability of Cox regression model failure was 0.60.',
            'The risk of logistic regression model failure was 0.60.',
            'The probability of linear regression analysis failure was 0.60.',
            'The risk of regression model failure was 0.60.',
            'The probability of regression analysis failure was 0.60.',
            'The probability of regression to the mean was 0.60.',
        ]
        for quote in quotes:
            with self.subTest(quote=quote):
                self.assertNotIn(REGRESSION_CODE, self.check(PROGRESSION,
                    [section(11, 'METHODS', COHORT), section(23, 'RESULTS', quote)], [quote]))

    def test_regression_demotion_requires_actual_primary_clinical_context(self):
        for extra in ([], [section(31, 'BACKGROUND', COHORT)]):
            with self.subTest(extra=extra):
                self.assertNotIn(REGRESSION_CODE, self.progression(
                    cohort='Records were analyzed in a simulation.', extra=extra))

    def test_separate_selected_observed_progression_result_including_null_is_preserved(self):
        progression_results = [
            'Progression to fulvous state was lower in the exposed group.',
            'Progression to the fulvous state was lower in the exposed group.',
            'No progression to fulvous state was observed.',
            'Progression to fulvous state did not occur.',
            'No progression to fulvous state occurred.',
        ]
        for quote in progression_results:
            with self.subTest(quote=quote):
                self.assertNotIn(REGRESSION_CODE, self.progression(
                    extra=[section(31, 'RESULTS', quote)], selected=[REGRESSION, quote]))

    def test_nonselected_methods_and_future_progression_mentions_do_not_rescue_selection(self):
        cases = [
            ('METHODS', 'The primary endpoint was progression to fulvous state.', False),
            ('RESULTS', 'No progression to fulvous state was observed.', False),
            ('RESULTS', 'Agent M could reduce progression to fulvous state in future work.', True),
        ]
        for role, quote, selected in cases:
            with self.subTest(role=role, quote=quote, selected=selected):
                self.assertIn(REGRESSION_CODE, self.progression(extra=[section(31, role, quote)],
                    selected=[REGRESSION, quote] if selected else [REGRESSION]))

    def test_selected_observed_progression_from_literal_start_to_requested_target_is_preserved(self):
        for quote in (
            'Progression from velum condition to fulvous state was lower in the exposed group.',
            'No progression from velum condition to fulvous state occurred.',
        ):
            with self.subTest(quote=quote):
                self.assertNotIn(REGRESSION_CODE, self.progression(
                    extra=[section(31, 'RESULTS', quote)], selected=[REGRESSION, quote]))

    def test_literal_methods_definition_in_inverted_order_is_inconclusive(self):
        definitions = (
            'For this study, velum condition was defined as clinically significant disease; non-significant cases were excluded.',
            'For this study, we defined velum condition as clinically significant disease.',
        )
        for definition in definitions:
            with self.subTest(definition=definition):
                self.assertNotIn(DIAGNOSTIC_CODE, self.diagnostic(
                    extra=[section(31, 'METHODS', definition)]))
                unrelated = definition.replace('velum condition', 'lumen condition')
                self.assertIn(DIAGNOSTIC_CODE, self.diagnostic(
                    extra=[section(31, 'METHODS', unrelated)]))

    def test_alias_uncertainty_and_nonselected_or_nonresult_regression_are_inconclusive(self):
        alias = 'The hazard of progression to FS was 0.90.'
        self.assertNotIn(REGRESSION_CODE, self.progression(
            extra=[section(31, 'RESULTS', alias)], selected=[REGRESSION, alias]))
        for role in ('METHODS', 'BACKGROUND', 'CONCLUSIONS', ''):
            with self.subTest(role=role):
                self.assertNotIn(REGRESSION_CODE, self.check(PROGRESSION,
                    [section(11, 'METHODS', COHORT), section(23, role, REGRESSION)], [REGRESSION]))
        self.assertNotIn(REGRESSION_CODE, self.progression(selected=[]))


if __name__ == '__main__':
    unittest.main()
