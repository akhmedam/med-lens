"""Bounded JSON grammar and fail-closed decoding without models or GPU imports."""
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from medlens.generation.contracts import draft_contract, review_contract
from medlens.inference.structured import (Constraint, ConstraintCache, byte_decoder,
    check_schema_budget, prepare_schema, required_order_regex, require_finite)

HAS_CORE = importlib.util.find_spec('outlines_core') is not None


def tokenizer_fixture():
    decoder = byte_decoder()
    vocab = {character: value for character, value in decoder.items()}
    backend = {'decoder': {'type': 'ByteLevel'}, 'model': {'type': 'BPE', 'vocab': vocab}}
    return SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(backend)),
        get_vocab=lambda: dict(vocab), all_special_ids=[256, 257])


class SchemaBudgetTests(unittest.TestCase):
    def test_schema_bytes_depth_nodes_and_enum_budget(self):
        self.assertEqual(check_schema_budget({'const':'a'}),{'const':'a'})
        deep = {'const': 'a'}
        for _ in range(25):deep={'items':deep}
        for schema in ({'const':'я'*17000}, deep, {'enum':['x']*513},
                       {'values':['x']*4096}, {'const':float('nan')}):
            with self.subTest(schema_keys=list(schema)),self.assertRaises(ValueError):
                check_schema_budget(schema)

    def test_draft_lowering_is_explicit_and_only_for_exact_contract(self):
        original=draft_contract(['E1.Q1','E2.Q3'])
        lowered, info=prepare_schema(original,'draft',validate_regex=False)
        self.assertEqual(info['decode_relaxations'],['maxLength','uniqueItems'])
        self.assertIn('oneOf',lowered)
        self.assertEqual([row['properties']['claims']['maxItems'] for row in lowered['oneOf']],[3,0])
        changed=json.loads(json.dumps(original))
        changed['properties']['claims']['maxItems']=9
        with self.assertRaises(ValueError):prepare_schema(changed,'draft',validate_regex=False)
        self.assertIn('if',original)
        self.assertEqual(original['properties']['claims']['items']['properties']['text']['maxLength'],1600)

    def test_review_lowering_covers_each_claim_once(self):
        lowered, info=prepare_schema(review_contract(['C1','C2']),'review',validate_regex=False)
        self.assertEqual([row['properties']['claim_id']['const'] for row in lowered['prefixItems']],['C1','C2'])
        self.assertEqual(info['decode_relaxations'],[])

    def test_raw_guard_preserves_scores_and_rejects_bad_rows(self):
        usable=np.array([[-np.inf,2.,1.]])
        self.assertIs(require_finite(np,usable),usable)
        for row in ([np.nan,0.],[np.inf,0.],[-np.inf,-np.inf]):
            with self.assertRaises(FloatingPointError):require_finite(np,np.array([[1.,2.],row]))


@unittest.skipUnless(HAS_CORE,'outlines-core is installed in the isolated MLX environment')
class StructuredCoreTests(unittest.TestCase):
    def constraint(self,schema):
        return Constraint(schema,tokenizer_fixture(),vocab_size=288,eos=(256,257),whitespace_pattern='')

    def accepts(self,constraint,text):
        from outlines_core import Guide
        guide=Guide(constraint.index)
        try:
            for token in text.encode():guide.advance(token,return_tokens=False)
        except ValueError:return False
        return guide.is_finished()

    def test_required_order_survives_canonical_schema_transport(self):
        schema={'type':'object','additionalProperties':False,'required':['z','a'],
            'properties':{'a':{'enum':['ok']},'z':{'type':'string','minLength':1,'maxLength':8}}}
        schema=json.loads(json.dumps(schema,sort_keys=True))
        constraint=self.constraint(schema)
        self.assertTrue(self.accepts(constraint,'{"z":"café\\u03b1\\uD83D\\uDE00","a":"ok"}'))
        self.assertFalse(self.accepts(constraint,'{"a":"ok","z":"café"}'))
        self.assertFalse(self.accepts(constraint,'{"z":"café","a":"bad"}'))

    def test_unicode_scalar_length_bounds_and_byte_splits(self):
        constraint=self.constraint({'type':'string','minLength':1,'maxLength':1})
        for text in ('"a"','"é"','"\\u0061"','"\\uD83D\\uDE00"'):
            with self.subTest(text=text):self.assertTrue(self.accepts(constraint,text))
        for text in ('""','"aa"','"\\u0061\\u0062"','"\\uD83D"','"\\uDE00"'):
            with self.subTest(text=text):self.assertFalse(self.accepts(constraint,text))

    def test_mask_eos_lifecycle_fresh_state_and_exact_terminal_ids(self):
        constraint=self.constraint({'type':'string','minLength':1,'maxLength':2})
        processor=constraint.processor(np)
        prefix=[3,4]
        raw=np.arange(288,dtype=np.float32)[None,:]
        masked=processor(np.array(prefix),raw)
        self.assertTrue(np.isneginf(masked[0,256]))
        for token in '"é"'.encode():
            self.assertTrue(np.isfinite(masked[0,token]))
            self.assertEqual(masked[0,token],raw[0,token])
            prefix.append(token)
            masked=processor(np.array(prefix),raw)
        self.assertTrue(processor.guide.is_finished())
        self.assertTrue(all(np.isfinite(masked[0,token]) for token in (256,257)))
        self.assertTrue(np.isneginf(masked[0,258:]).all())
        prefix.append(256)
        terminal=processor(np.array(prefix),raw)
        self.assertEqual(np.isfinite(terminal).sum(),2)
        output=[*'"é"'.encode(),256]
        self.assertTrue(processor.finish(output)['eos_accepted'])
        with self.assertRaises(ValueError):processor.finish([1,*output[1:]])
        with self.assertRaises(ValueError):processor.finish(output,hit_output_limit=True)
        fresh=constraint.processor(np)
        self.assertFalse(fresh.guide.is_finished())

    def test_invalid_raw_scores_cannot_be_hidden_and_postmask_empty_fails(self):
        constraint=self.constraint({'const':'a'})
        for bad in (np.nan,np.inf):
            scores=np.zeros((1,288),dtype=np.float32);scores[0,287]=bad
            with self.assertRaises(FloatingPointError):constraint.processor(np)(np.array([1]),scores)
        scores=np.full((1,288),-np.inf,dtype=np.float32);scores[0,287]=0
        with self.assertRaises(FloatingPointError):constraint.processor(np)(np.array([1]),scores)

    def test_unknown_constraints_and_overlapping_oneof_fail_closed(self):
        for schema in ({'type':'string','pattern':'x'}, {'type':'array','items':{'const':'a'}},
            {'type':'array','items':{'const':'a'},'maxItems':33},
            {'type':'object','additionalProperties':False,'required':[],'properties':{'a':{'const':'a'}}},
            {'oneOf':[{'enum':['a','b']},{'const':'a'}]},
            {'type':'string','maxLength':1600}, {'type':'string','unknown':True}):
            with self.subTest(schema=schema),self.assertRaises(ValueError):self.constraint(schema)

    def test_array_hard_maximum(self):
        constraint=self.constraint({'type':'array','items':{'enum':['a','b']},'minItems':1,'maxItems':3})
        self.assertTrue(self.accepts(constraint,'["a","b","a"]'))
        self.assertFalse(self.accepts(constraint,'[]'))
        self.assertFalse(self.accepts(constraint,'["a","b","a","b"]'))

    def test_bounded_index_cache_reuses_immutable_graph_and_never_guide(self):
        cache=ConstraintCache(tokenizer_fixture(),vocab_size=288,eos=(256,257),maximum=2)
        first, miss=cache.get({'const':'a'},'literal')
        same, hit=cache.get({'const':'a'},'literal')
        self.assertIs(first,same)
        self.assertFalse(miss['cache_hit']);self.assertTrue(hit['cache_hit'])
        self.assertIsNot(first.processor(np).guide,same.processor(np).guide)
        cache.get({'const':'b'},'literal');cache.get({'const':'c'},'literal')
        replacement, miss_again=cache.get({'const':'a'},'literal')
        self.assertIsNot(first,replacement)
        self.assertFalse(miss_again['cache_hit'])
        self.assertEqual(miss_again['compiled_cache_size'],2)


if __name__=='__main__':unittest.main()
