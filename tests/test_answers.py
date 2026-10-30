"""Malformed, uncited and unsupported output must not become a visible answer."""

import copy
import json
from types import SimpleNamespace
import unittest

from medlens.generation.context import build_context, ContextConfig
from medlens.generation.answers import (AnswerError, check_context, generate_answer,
    parse_draft, parse_review, render_answer, strict_json)
import test_evidence_context as helpers


def context(empty=False):
    dataset, manifest, windows = helpers.fixtures()
    return build_context(dataset, manifest, windows, [] if empty else ["1"], [.9,.8,.7]*3,
        helpers.CharacterTokenizer(), {}, config=ContextConfig(max_fragments=1))


def legacy_draft():
    return {"status": "partial", "claims": [{"text": "A scoped observation.",
        "citations": [{"citation_id": "E1", "quote": "alpha beta gamma"}]}], "limitations": ["limited_excerpts"]}


def draft():
    d=legacy_draft()
    d['claims'][0]['citations']=[{'quote_id':'E1.Q1'}]
    return d


class Tokenizer(helpers.CharacterTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return list(json.dumps(messages))


class FakeRuntime:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.tokenizer = Tokenizer()
        self.calls = []

    def generate(self, prompt, *, max_new_tokens):
        self.calls.append((prompt,max_new_tokens))
        out = next(self.outputs)
        return {"text": out if isinstance(out,str) else json.dumps(out), "hit_output_limit": False,
                "new_tokens": 20, "seconds": .1, "max_new_tokens": max_new_tokens}


class AnswerTests(unittest.TestCase):
    def test_valid_citation_maps_to_source_not_model_supplied_pmid(self):
        r = parse_draft(json.dumps(legacy_draft()), context())
        self.assertEqual(r['claims'][0]['citations'][0]['pmid'], '1')
        self.assertEqual(r['claims'][0]['citations'][0]['matches'], [{'char_start':0,'char_end':16}])
        d = legacy_draft();d['claims'][0]['citations'][0]['pmid']='999'
        with self.assertRaises(AnswerError):parse_draft(json.dumps(d),context())

    def test_invalid_drafts_rejected(self):
        bad=[]
        for modify in (
            lambda x:x.update(status=[]),
            lambda x:x.update(status='insufficient_evidence'),
            lambda x:x.update(limitations=['invented']),
            lambda x:x['claims'][0].update(citations=[]),
            lambda x:x['claims'][0].update(text='See PMID 999 or https://example.org'),
            lambda x:x['claims'][0]['citations'][0].update(citation_id='E99'),
            lambda x:x['claims'][0]['citations'][0].update(quote='Alpha beta gamma'),
        ):
            d=legacy_draft();modify(d);bad.append(json.dumps(d))
        bad+=['```json\n'+json.dumps(draft())+'\n```','null','{"status":"partial","status":"answered"}']
        for text in bad:
            with self.subTest(text=text),self.assertRaises(AnswerError):parse_draft(text,context())
        with self.assertRaises(AnswerError):strict_json('[NaN]')

    def test_review_must_cover_every_claim_once(self):
        good={'claim_id':'C1','verdict':'supported','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'supported_by_excerpt'}
        self.assertEqual(parse_review(json.dumps([good]),{'C1'}),[good])
        for rows in ([],[good,good],[{**good,'claim_id':'C99'}],[{**good,'verdict':[]}],[{**good,'reason':''}],[{**good,'reason':'Free-form explanation'}]):
            with self.subTest(rows=rows),self.assertRaises(AnswerError):parse_review(json.dumps(rows),{'C1'})

    def test_supported_claim_displayed_with_system_assigned_link(self):
        rt=FakeRuntime([draft(),[{'claim_id':'C1','verdict':'supported','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'supported_by_excerpt'}]])
        r=generate_answer('Research question?',context(),rt)
        self.assertEqual(len(r['claims']),1)
        self.assertIn('https://pubmed.ncbi.nlm.nih.gov/1/',render_answer(r))
        self.assertEqual([x[1] for x in rt.calls],[768,512])

    def test_rejected_claim_withheld_even_with_an_exact_quote(self):
        rt=FakeRuntime([draft(),[{'claim_id':'C1','verdict':'not_supported','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'population_mismatch'}]])
        r=generate_answer('Research question?',context(),rt)
        self.assertEqual(r['status'],'insufficient_evidence')
        self.assertEqual(r['claims'],[])
        self.assertNotIn('A scoped observation',render_answer(r))

    def test_empty_context_skips_model_and_corruption_fails_before_model(self):
        rt=FakeRuntime([])
        self.assertEqual(generate_answer('Question',context(True),rt)['reason'],'empty-context')
        self.assertEqual(rt.calls,[])
        c=context();c['sources'][0]['text']='changed'
        with self.assertRaises(AnswerError):generate_answer('Question',c,rt)
        self.assertEqual(rt.calls,[])

    def test_invalid_review_suppresses_entire_draft(self):
        rt=FakeRuntime([draft(),'not json'])
        r=generate_answer('Question',context(),rt)
        self.assertEqual(r['status'],'validation_failed')
        self.assertEqual(r['claims'],[])
        self.assertNotIn('A scoped observation',render_answer(r))

    def test_partial_review_preserves_only_supported_claims(self):
        d=draft()
        d['status']='answered'
        d['claims'].append({'text':'An unsupported extension.',
            'citations':[{'quote_id':'E1.Q1'}]})
        rt=FakeRuntime([d,[{'claim_id':'C1','verdict':'supported','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'supported_by_excerpt'},
                         {'claim_id':'C2','verdict':'unclear','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'missing_details'}]])
        r=generate_answer('Question',context(),rt)
        self.assertEqual(r['status'],'partial')
        self.assertEqual([c['claim_id'] for c in r['claims']],['C1'])
        self.assertNotIn('unsupported extension',render_answer(r))

    def test_truncated_draft_or_review_cannot_be_displayed(self):
        class TruncatedRuntime(FakeRuntime):
            def generate(self,prompt,*,max_new_tokens):
                r=super().generate(prompt,max_new_tokens=max_new_tokens)
                r['hit_output_limit']=len(self.calls)==self.truncate_call
                return r
        for stage in (1,2):
            rt=TruncatedRuntime([draft(),[{'claim_id':'C1','verdict':'supported','population_match':'match','comparison_match':'not_requested','outcome_match':'match','reason':'supported_by_excerpt'}]])
            rt.truncate_call=stage
            r=generate_answer('Question',context(),rt)
            self.assertEqual(r['status'],'validation_failed')
            self.assertEqual(r['claims'],[])


    def test_axis_mismatch_overrides_supported_verdict(self):
        for axis in ('population_match','comparison_match','outcome_match'):
            row={'claim_id':'C1','verdict':'supported','reason':'supported_by_excerpt',
                 'population_match':'match','comparison_match':'match','outcome_match':'match'}
            row[axis]='mismatch'
            result=generate_answer('Question',context(),FakeRuntime([draft(),[row]]))
            self.assertEqual(result['claims'],[])
            self.assertEqual(result['status'],'insufficient_evidence')

    def test_per_claim_limitations_are_explicitly_supported_without_text_repair(self):
        d=legacy_draft();d['claims'][0]['limitations']=['population_unclear'];d.pop('limitations')
        parsed=parse_draft(json.dumps(d),context())
        self.assertEqual(parsed['format_variant'],'per-claim-limitations')
        self.assertEqual(parsed['claims'][0]['text'],d['claims'][0]['text'])
        self.assertEqual(parsed['limitations'],['population_unclear'])
        d['claims'][0]['unexpected']='not permitted'
        with self.assertRaises(AnswerError):parse_draft(json.dumps(d),context())

    def test_cache_reuse_requires_exact_prompt_and_limit(self):
        row={'claim_id':'C1','verdict':'supported','reason':'supported_by_excerpt',
             'population_match':'match','comparison_match':'match','outcome_match':'match'}
        original=generate_answer('Question',context(),FakeRuntime([draft(),[row]]))
        cache={c['stage']:c for c in original['model_calls']}
        rt=FakeRuntime([])
        reused=generate_answer('Question',context(),rt,cached_calls=cache)
        self.assertEqual(rt.calls,[])
        self.assertEqual(reused['claims'],original['claims'])
        self.assertTrue(all(c['cache_reused'] for c in reused['model_calls']))
        changed=FakeRuntime([draft(),[row]])
        fresh=generate_answer('Changed question',context(),changed,cached_calls=cache)
        self.assertEqual(len(changed.calls),2)
        self.assertFalse(any(c['cache_reused'] for c in fresh['model_calls']))


    def test_quote_ids_resolve_exactly_and_cannot_invent_text(self):
        from medlens.generation.quotes import quote_catalog,resolve_quote
        c=context();catalog=quote_catalog(c,Tokenizer())
        value=parse_draft(json.dumps(draft()),c,catalog=catalog['catalog'])
        self.assertEqual(value['claims'][0]['citations'][0]['quote'],'alpha beta gamma')
        d=draft();d['claims'][0]['citations'][0]['quote_id']='E99.Q9'
        with self.assertRaises(AnswerError):parse_draft(json.dumps(d),c,catalog=catalog['catalog'])
        changed=copy.deepcopy(catalog['catalog']);changed['E1.Q1']['char_start']+=1
        with self.assertRaises(ValueError):resolve_quote(c,changed,'E1.Q1')
