"""Semantic axes remain mandatory and synthetic controls cannot reward all-refusal."""

import copy
import json
from pathlib import Path
import unittest

import test_answers as helpers
import test_evidence_context as sources
from medlens.evaluation.semantics import load_challenges, summarize_challenges
from medlens.generation.answers import AnswerError, accepted_claim_ids, parse_draft, parse_review
from medlens.generation.context import ContextConfig, build_context
from medlens.generation.contracts import LEGACY_REVIEW_VERSION, REVIEW_VERSION, review_values
from medlens.generation.quotes import quote_catalog
from medlens.generation.reviews import review_messages


def supported(protocol=REVIEW_VERSION):
    return {'claim_id': 'C1', **{key: 'supported' if key == 'verdict' else 'supported_by_excerpt'
        if key == 'reason' else 'match' for key in review_values(protocol)}}


class SemanticReviewTests(unittest.TestCase):
    def test_all_seven_axes_gate_a_supported_verdict(self):
        axes = set(review_values()) - {'verdict','reason'}
        self.assertEqual(len(axes),7)
        self.assertEqual(accepted_claim_ids([supported()]), {'C1'})
        for axis in axes:
            for value in ('mismatch','unclear'):
                row = supported();row[axis] = value
                parsed = parse_review(json.dumps([row]), {'C1'},protocol=REVIEW_VERSION)
                with self.subTest(axis=axis,value=value): self.assertEqual(accepted_claim_ids(parsed),set())
        row=supported();row['comparison_match']='not_requested'
        self.assertEqual(accepted_claim_ids([row]),{'C1'})

    def test_missing_axes_and_wrong_not_requested_are_not_repaired(self):
        for axis in set(review_values()) - {'verdict','reason'}:
            row=supported();row.pop(axis)
            with self.assertRaises(AnswerError): parse_review(json.dumps([row]),{'C1'},protocol=REVIEW_VERSION)
            if axis != 'comparison_match':
                row=supported();row[axis]='not_requested'
                with self.assertRaises(AnswerError):parse_review(json.dumps([row]),{'C1'},protocol=REVIEW_VERSION)

    def test_legacy_protocol_is_explicit_and_cannot_satisfy_new_contract(self):
        row=supported(LEGACY_REVIEW_VERSION)
        self.assertEqual(parse_review(json.dumps([row]),{'C1'},protocol=LEGACY_REVIEW_VERSION),[row])
        with self.assertRaises(AnswerError):parse_review(json.dumps([row]),{'C1'},protocol=REVIEW_VERSION)
        with self.assertRaises(ValueError):review_values('invented')

    def test_all_selected_evidence_is_reviewed_without_laundering_citations(self):
        dataset,manifest,windows=sources.fixtures()
        ctx=build_context(dataset,manifest,windows,['1','2'],[.5]*len(windows),helpers.Tokenizer(),{},
            config=ContextConfig(max_fragments=2,max_per_article=1))
        indexed=quote_catalog(ctx,helpers.Tokenizer())
        draft=parse_draft(json.dumps(helpers.draft()),ctx,catalog=indexed['catalog'],citation_format='strings')
        _,new=review_messages('Question',ctx,draft['claims'],indexed)
        _,old=review_messages('Question',ctx,draft['claims'],indexed,protocol=LEGACY_REVIEW_VERSION)
        self.assertIn('E2.Q1',new)
        self.assertNotIn('"citation_id":"E2"',old)
        self.assertIn('"citations":["E1.Q1"]',new)
        broken=copy.deepcopy(draft['claims']);broken[0]['citations'][0]['quote']='fabricated'
        with self.assertRaises(ValueError):review_messages('Question',ctx,broken,indexed)

    def test_frozen_challenges_have_both_classes_and_valid_exact_quotes(self):
        path=Path(__file__).resolve().parents[1]/'benchmarks/semantic-dev-v1/cases.json'
        groups,checksum=load_challenges(path,helpers.Tokenizer())
        self.assertEqual(len(checksum),64)
        self.assertEqual(len(groups),7)
        for group in groups:
            indexed=quote_catalog(group['context'],helpers.Tokenizer())
            parsed=parse_draft(json.dumps(group['draft']),group['context'],catalog=indexed['catalog'],citation_format='strings')
            self.assertEqual(len(parsed['claims']),3)
            self.assertEqual(sum(r['expected_supported'] for r in group['labels']),1)
            for protocol in (LEGACY_REVIEW_VERSION,REVIEW_VERSION):
                messages=review_messages(group['question'],group['context'],parsed['claims'],indexed,protocol=protocol)
                self.assertNotIn('expected_supported',str(messages))
                self.assertNotIn('"focus"',str(messages))

    def test_review_input_overflow_withholds_without_truncating_sources(self):
        from medlens.generation.answers import generate_answer
        class Sized(helpers.Tokenizer):
            calls=0
            def apply_chat_template(self,messages,**kwargs):
                self.calls+=1
                return range(6500 if self.calls==1 else 6501)
        runtime=helpers.FakeRuntime([helpers.draft()]);runtime.tokenizer=Sized()
        context=helpers.context();original=copy.deepcopy(context)
        result=generate_answer('Question',context,runtime,review_protocol=REVIEW_VERSION)
        self.assertEqual(result['validation_error'],{'stage':'support-review','code':'input_limit'})
        self.assertEqual(result['claims'],[])
        self.assertEqual(len(runtime.calls),1)
        self.assertEqual(context,original)

    def test_all_refusal_preserves_false_rejection_denominator(self):
        rows=[{'group_id':'g','claim_id':str(i),'expected_supported':expected,'accepted':False,
               'review_status':'valid'} for i,expected in enumerate((True,False,False))]
        summary=summarize_challenges(rows)
        self.assertEqual(summary['supported_withheld'],1)
        self.assertEqual(summary['supported_retained'],0)
        self.assertEqual(summary['unsupported_withheld'],2)
        with self.assertRaises(ValueError):summarize_challenges(rows+rows)
        rows[0].update(review_status='validation_failed',accepted=True)
        with self.assertRaises(ValueError):summarize_challenges(rows)


class ReviewPromotionTests(unittest.TestCase):
    def test_default_preserves_legacy_and_candidate_is_explicit(self):
        from medlens.generation.answers import generate_answer
        from medlens.generation.contracts import DEFAULT_REVIEW_VERSION
        self.assertEqual(DEFAULT_REVIEW_VERSION,LEGACY_REVIEW_VERSION)
        for protocol in (LEGACY_REVIEW_VERSION,REVIEW_VERSION):
            runtime=helpers.FakeRuntime([helpers.draft(),[supported(protocol)]])
            kwargs={} if protocol==LEGACY_REVIEW_VERSION else {'review_protocol':protocol}
            result=generate_answer('Question',helpers.context(),runtime,**kwargs)
            self.assertEqual(result['review_version'],protocol)
            self.assertEqual(len(result['claims']),1)
            self.assertEqual([limit for _,limit in runtime.calls],[768,512])

    def test_unknown_candidate_fails_before_model(self):
        from medlens.generation.answers import generate_answer
        runtime=helpers.FakeRuntime([])
        with self.assertRaises(ValueError):generate_answer('Question',helpers.context(),runtime,review_protocol='invented')
        self.assertEqual(runtime.calls,[])


    def test_optin_reuses_draft_but_never_a_legacy_review(self):
        from medlens.generation.answers import generate_answer
        old=generate_answer('Question',helpers.context(),helpers.FakeRuntime([helpers.draft(),[supported(LEGACY_REVIEW_VERSION)]]))
        cache={c['stage']:c for c in old['model_calls']}
        runtime=helpers.FakeRuntime([[supported(REVIEW_VERSION)]])
        new=generate_answer('Question',helpers.context(),runtime,cached_calls=cache,review_protocol=REVIEW_VERSION)
        self.assertEqual(len(runtime.calls),1)
        self.assertEqual(runtime.calls[0][1],512)
        self.assertTrue(new['model_calls'][0]['cache_reused'])
        self.assertFalse(new['model_calls'][1]['cache_reused'])
        self.assertEqual(new['review_version'],REVIEW_VERSION)
