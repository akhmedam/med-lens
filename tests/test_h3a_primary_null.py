"""Fictional primary-null coverage checks; no native runtime, model or network."""
from copy import deepcopy
import json
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from medlens.generation import source_findings as c
from medlens.inference.structured import prepare_schema, required_order_regex
from test_source_findings import make, plan_data, select
from test_answers import FakeRuntime

QUESTION='Does agent R compared with placebo reduce pain in adults?'
SETUP='Adults were randomized to agent R or placebo.'
PRIMARY='The primary end-point was change in pain after treatment.'
NULL='The primary efficacy analysis did not reach statistical significance.'
EXPLORATORY='Exploratory analysis found lower pain with agent R than placebo.'


def fixture(primary=PRIMARY):
    return make([[('METHODS',SETUP+' '+primary),('RESULTS',NULL+' '+EXPLORATORY)]])


def regex_for(schema):
    # Exercise the real ordered-schema compiler with enum/const/tuple-only
    # schemas; native token-DFA compilation remains a separate root check.
    parent=ModuleType('outlines_core');child=ModuleType('outlines_core.json_schema')
    def unexpected(*args,**kwargs):raise AssertionError('No primitive fallback expected')
    child.build_regex_from_schema=unexpected
    with patch.dict(sys.modules,{'outlines_core':parent,'outlines_core.json_schema':child}):
        return re.compile(required_order_regex(schema,whitespace_pattern=''))


class PrimaryNullCoverage(unittest.TestCase):
    def setup(self,primary=PRIMARY):
        ctx,articles=fixture(primary);idx,plans,checks=plan_data(ctx,articles,QUESTION)
        return ctx,articles,idx,plans,checks

    def test_primary_null_is_required_and_exact_with_no_numeric_threshold(self):
        ctx,articles,idx,plans,checks=self.setup();plan=plans[0]
        required=plan['required_finding_quote_ids'];self.assertEqual(len(required),1)
        self.assertEqual(idx['catalog'][required[0]]['quote'],NULL)
        self.assertFalse(checks['1']['primary_null_requirement']['positive_validation_performed'])
        for group in ('primary_endpoint_proofs','primary_null_proofs'):
            for proof in checks['1']['primary_null_requirement'][group]:
                section=articles['1']['abstract']['sections'][int(proof['field'].split(':')[1])]
                self.assertEqual(section['text'][proof['char_start']:proof['char_end']],proof['text'])

    def test_positive_exploratory_cannot_hide_primary_null(self):
        ctx,articles,idx,plans,checks=self.setup();plan=plans[0];scope=plan['scope_quote_ids'][0]
        positive=next(q for q in plan['finding_quote_ids']if idx['catalog'][q]['quote']==EXPLORATORY)
        with self.assertRaises(c.AnswerError)as raised:
            c.materialize_selection(json.dumps(select(scope=scope,findings=[positive])),ctx,idx,plans,checks)
        self.assertEqual(raised.exception.code,'selection_primary_null')
        required=plan['required_finding_quote_ids'][0]
        runtime=FakeRuntime([select(scope=scope,findings=[required,positive])])
        answer=c.generate_source_findings_answer(QUESTION,ctx,runtime,source_articles=articles)
        self.assertEqual(answer['status'],'partial');self.assertEqual(len(runtime.calls),1)
        self.assertIn(NULL,answer['claims'][0]['text']);self.assertIn(EXPLORATORY,answer['claims'][0]['text'])
        self.assertLessEqual(len(answer['claims'][0]['text']),1600)

    def test_primary_null_alone_remains_eligible_and_source_is_not_mutated(self):
        ctx,articles,idx,plans,checks=self.setup();before=deepcopy(ctx);plan=plans[0]
        value=select(scope=plan['scope_quote_ids'][0],findings=plan['required_finding_quote_ids'])
        answer=c.generate_source_findings_answer(QUESTION,ctx,FakeRuntime([value]),source_articles=articles)
        self.assertEqual(answer['status'],'partial');self.assertEqual(ctx,before)
        self.assertIn(NULL,answer['claims'][0]['text']);self.assertFalse(answer['semantic_support_verified'])

    def test_question_about_secondary_endpoint_does_not_require_unrelated_primary_null(self):
        ctx,articles,idx,plans,checks=self.setup('The primary endpoint was breathlessness; secondary pain was measured.')
        self.assertEqual(plans[0]['required_finding_quote_ids'],[])
        self.assertEqual(checks['1']['primary_null_requirement']['primary_endpoint_proofs'],[])
        self.assertTrue(plans[0]['allow_direct'])

    def test_unrecognized_primary_definition_is_not_guessed(self):
        ctx,articles,idx,plans,checks=self.setup('Secondary pain was measured during follow-up.')
        self.assertEqual(plans[0]['required_finding_quote_ids'],[])
        self.assertTrue(plans[0]['allow_direct'])

    def test_previous_background_null_does_not_create_current_requirement(self):
        ctx,articles=make([[('BACKGROUND','Previous trials reported that the primary outcome showed no significant difference.'),
            ('METHODS',SETUP+' '+PRIMARY),('RESULTS',EXPLORATORY)]])
        idx,plans,checks=plan_data(ctx,articles,QUESTION)
        self.assertEqual(checks['1']['primary_null_requirement']['primary_null_proofs'],[])
        self.assertTrue(plans[0]['allow_direct'])

    def test_unlabeled_nlm_background_is_not_a_reported_primary_null(self):
        ctx,articles=make([[('METHODS',SETUP+' '+PRIMARY),
            (None,'An external trial reported that the primary outcome showed no significant difference.'),
            ('RESULTS',EXPLORATORY)]])
        articles['1']['abstract']['sections'][1]['nlm_category']='BACKGROUND'
        idx,plans,checks=plan_data(ctx,articles,QUESTION)
        self.assertEqual(checks['1']['primary_null_requirement']['primary_null_proofs'],[])
        self.assertTrue(plans[0]['allow_direct'])

    def test_unpacked_required_result_blocks_direct_without_adding_text(self):
        ctx,articles=make([[('METHODS',SETUP+' '+PRIMARY),('RESULTS',EXPLORATORY),('CONCLUSIONS',NULL)]])
        idx,plans,checks=plan_data(ctx,articles,QUESTION)
        self.assertFalse(plans[0]['allow_direct'])
        self.assertIn('recognized_primary_null_outside_bounded_findings',[x['code']for x in checks['1']['policy_blocks']])
        runtime=FakeRuntime([]);answer=c.generate_source_findings_answer(QUESTION,ctx,runtime,source_articles=articles)
        self.assertEqual(answer['status'],'insufficient_evidence');self.assertEqual(runtime.calls,[])

    def test_direct_grammar_requires_anchor_skip_remains_possible(self):
        ctx,articles,idx,plans,checks=self.setup();plan=plans[0];scope=plan['scope_quote_ids'][0]
        primary=plan['required_finding_quote_ids'][0];other=next(q for q in plan['finding_quote_ids']if q!=primary)
        schema,_=prepare_schema(c.selection_schema(plans),'literal',validate_regex=False)
        grammar=regex_for(schema)
        dumps=lambda value:json.dumps(value,separators=(',',':'))
        self.assertIsNotNone(grammar.fullmatch(dumps(select(scope=scope,findings=[primary]))))
        self.assertIsNotNone(grammar.fullmatch(dumps(select(scope=scope,findings=[primary,other]))))
        self.assertIsNone(grammar.fullmatch(dumps(select(scope=scope,findings=[other]))))
        self.assertIsNotNone(grammar.fullmatch(dumps(select(scope=None,findings=[],relation='none'))))
        self.assertNotIn('contains',json.dumps(schema));self.assertLess(len(json.dumps(schema).encode()),32768)

if __name__=='__main__':unittest.main()
