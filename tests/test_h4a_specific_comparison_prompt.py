"""Fictional prompt-delivery and retained-guard controls; no model inference."""

import json
import unittest

from medlens.generation import source_findings as selector
from medlens.generation import source_scope as guard
from test_answers import FakeRuntime
from test_mixed_control_postonly import MIX, Q, CODE
from test_source_findings import make, plan_data, select


class H4aSpecificComparisonInstruction(unittest.TestCase):
    def answer(self, question, sections, *, title='Fictional study', applicability='answers_question'):
        context, articles = make([sections], title=title)
        _, plans, _ = plan_data(context, articles, question)
        self.assertTrue(plans[0]['allow_direct'])
        raw = select(scope=plans[0]['scope_quote_ids'][0] if plans[0]['scope_required'] else None,
            findings=plans[0]['finding_quote_ids'][:1], applicability=applicability)
        runtime = FakeRuntime([raw])
        answer = selector.generate_source_findings_answer(question, context, runtime, source_articles=articles)
        self.assertEqual(len(runtime.calls), 1)
        prompt, cap = runtime.calls[0]
        self.assertEqual(cap, 768)
        self.assertEqual(prompt['decoding_schema'], selector.selection_schema(plans))
        self.assertEqual(answer['selection'], raw)
        self.assertEqual(answer['question_contract'], guard.question_contract(question))
        self.assertFalse(answer['question_task_kind']['positive_validation_performed'])
        self.assertIn('Exact question anchors (recognized spans only):', prompt['messages'][1]['content'])
        self.assertIn(question, prompt['messages'][1]['content'])
        self.assertFalse(answer['semantic_support_verified'])
        self.assertEqual(selector.MAX_QUOTE_CHARS, 532)
        return answer, prompt

    def test_corrected_instruction_is_in_actual_prompt_and_pooled_guard_remains(self):
        answer, prompt = self.answer(Q, [('METHODS', MIX),
            ('RESULTS', 'Agent R reduced events versus pooled control (RR 0.74).')])
        system = prompt['messages'][0]['content']
        self.assertIn('selected scope and result together must isolate A versus B', system)
        self.assertIn('Pooled results across mixed comparators do not answer', system)
        self.assertIn('choose different_comparison or indirect_result', system)
        self.assertNotIn('reporting A versus comparators including B can answer A versus B', system)
        self.assertEqual(answer['status'], 'insufficient_evidence')
        self.assertIn(CODE, {b['code'] for b in answer['source_relations'][0]['selected_scope_checks']['blocks']})

    def test_review_with_isolated_b_result_remains_possible(self):
        result = 'In the placebo-controlled subgroup, agent R reduced events (RR 0.74).'
        answer, _ = self.answer(Q, [('METHODS', MIX), ('RESULTS', result)],
            title='Review of trials of agent R versus placebo or usual care')
        self.assertEqual(answer['status'], 'partial')
        self.assertTrue(answer['claims'][0]['text'].endswith(result))

    def test_scope_can_identify_b_without_repeating_b_name_in_result(self):
        result = 'Events were lower in the agent R group than in the control group.'
        answer, _ = self.answer(Q, [('METHODS', 'Adults were randomized to agent R or placebo.'),
            ('RESULTS', result)])
        self.assertEqual(answer['status'], 'partial')
        self.assertTrue(answer['claims'][0]['text'].endswith(result))

    def test_observational_association_does_not_gain_rct_requirement(self):
        question = 'Is exposure A associated with outcome B?'
        answer, prompt = self.answer(question, [
            ('METHODS', 'Adult participants were enrolled in a prospective observational cohort.'),
            ('RESULTS', 'Exposure A was associated with outcome B.')])
        self.assertEqual(answer['question_task_kind']['kind'], 'association')
        self.assertEqual(answer['status'], 'partial')
        self.assertIn('An association can answer an association', prompt['messages'][0]['content'])
        self.assertIn('a causal effect is not required', prompt['messages'][1]['content'])

    def test_animal_mechanism_without_human_restriction_remains_possible(self):
        question = 'How does factor A promote process B?'
        answer, prompt = self.answer(question, [
            ('METHODS', 'Mice were exposed to factor A.'),
            ('RESULTS', 'Factor A promoted process B in the mice.')])
        self.assertEqual(answer['question_task_kind']['kind'], 'mechanism')
        self.assertEqual(answer['status'], 'partial')
        self.assertIn('An animal/cell mechanism can answer a mechanism question', prompt['messages'][0]['content'])

    def test_within_arm_result_model_nonanswer_is_not_promoted(self):
        answer, _ = self.answer(Q, [('METHODS', 'Adults were randomized to agent R or placebo.'),
            ('RESULTS', 'Within the agent R group, events decreased from baseline.')],
            applicability='different_comparison')
        self.assertEqual(answer['status'], 'insufficient_evidence')
        self.assertEqual(answer['claims'], [])
        self.assertEqual(tuple(selector.APPLICABILITY), ('answers_question', 'different_population',
            'different_intervention', 'different_comparison', 'different_outcome', 'indirect_result',
            'missing_details', 'unrelated'))


if __name__ == '__main__':
    unittest.main()
