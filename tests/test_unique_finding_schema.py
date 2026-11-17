"""Canonical finite finding pairs preserve sets and prevent duplicate IDs."""
import itertools
import json
import re
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
from medlens.generation import source_findings as s
from medlens.inference.structured import prepare_schema,required_order_regex,Constraint,byte_decoder,check_schema_budget
from test_source_findings import make,plan_data,select
from test_answers import FakeRuntime

def regex(schema):
 parent=ModuleType('outlines_core');child=ModuleType('outlines_core.json_schema')
 def unexpected(*args,**kwargs):raise AssertionError('No primitive fallback needed')
 child.build_regex_from_schema=unexpected
 with patch.dict(sys.modules,{'outlines_core':parent,'outlines_core.json_schema':child}):
  return re.compile(required_order_regex(schema,whitespace_pattern=''))

def fixture():
 return make([[('METHODS','Adult participants were assigned to agent R or placebo.'),
               ('RESULTS','Pain decreased with agent R. Response improved with agent R.')]])

def dumps(x):return json.dumps(x,separators=(',',':'))

class UniqueFindingSchema(unittest.TestCase):
 def test_every_set_of_zero_one_or_two_remains_representable(self):
  ids=['E1.Q1','E1.Q2','E1.Q10','E3.Q1'];grammar=regex(s._bounded_array(ids,2))
  for size in range(3):
   for values in itertools.combinations(ids,size):self.assertIsNotNone(grammar.fullmatch(dumps(values)))
  for qid in ids:self.assertIsNone(grammar.fullmatch(dumps([qid,qid])))
  self.assertIsNone(grammar.fullmatch(dumps(ids[:3])))
  self.assertIsNone(grammar.fullmatch(dumps(['unlisted'])))
 def test_pair_order_is_canonical_without_losing_any_pair(self):
  grammar=regex(s._bounded_array(['a','b','c'],2))
  self.assertIsNotNone(grammar.fullmatch('["a","c"]'))
  self.assertIsNone(grammar.fullmatch('["c","a"]'))
 def test_direct_minimum_stays_nonempty_and_singleton_can_finish(self):
  grammar=regex(s._bounded_array(['only'],2,1))
  self.assertIsNone(grammar.fullmatch('[]'));self.assertIsNotNone(grammar.fullmatch('["only"]'))
  self.assertIsNone(grammar.fullmatch('["only","only"]'))
 def test_no_finding_skip_stays_empty(self):
  grammar=regex(s._bounded_array([],2))
  self.assertIsNotNone(grammar.fullmatch('[]'));self.assertIsNone(grammar.fullmatch('["x"]'))
 def test_bad_catalog_or_unsupported_bounds_rejected(self):
  for args in ((['x','x'],2),(['x'],3),(['x'],2,2)):
   with self.subTest(args=args),self.assertRaises(ValueError):s._bounded_array(*args)
 def test_complete_selector_and_strict_postguard_preserve_distinct_roles(self):
  ctx,arts=fixture();idx,plans,checks=plan_data(ctx,arts);p=plans[0]
  scope=p['scope_quote_ids'][0];a,b=p['finding_quote_ids'][:2]
  schema,_=prepare_schema(s.selection_schema(plans),'literal',validate_regex=False);grammar=regex(schema)
  self.assertIsNotNone(grammar.fullmatch(dumps(select(scope=scope,findings=[a,b]))))
  self.assertIsNone(grammar.fullmatch(dumps(select(scope=scope,findings=[a,a]))))
  self.assertIsNone(grammar.fullmatch(dumps(select(scope=scope,findings=[scope]))))
  out=s.generate_source_findings_answer('Question',ctx,FakeRuntime([select(scope=scope,findings=[a,a])]),source_articles=arts)
  self.assertEqual(out['status'],'validation_failed');self.assertEqual(out['validation_error']['code'],'selection_quotes')
 def test_representative_three_article_catalog_stays_within_schema_budgets(self):
  plans=[]
  for i,(n,k)in enumerate(((11,4),(9,3),(8,0)),1):
   plans.append({'article_id':str(i),'scope_quote_ids':[f'E{i}.Q{j}'for j in range(k)],
    'finding_quote_ids':[f'E{i}.Q{j}'for j in range(n)],'scope_required':bool(k),'allow_direct':True})
  schema=s.selection_schema(plans);check_schema_budget(schema)
  self.assertLess(len(dumps(schema).encode()),32768)
  self.assertNotIn('uniqueItems',dumps(schema))

class NativeUniqueFindingSchema(unittest.TestCase):
 def test_token_dfa_rejects_repeats_and_accepts_unique_pairs(self):
  ctx,arts=fixture();idx,plans,checks=plan_data(ctx,arts);p=plans[0]
  scope=p['scope_quote_ids'][0];a,b=p['finding_quote_ids'][:2]
  schema,_=prepare_schema(s.selection_schema(plans),'literal')
  vocab={ch:i for ch,i in byte_decoder().items()};backend={'decoder':{'type':'ByteLevel'},'model':{'type':'BPE','vocab':vocab}}
  tok=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:dumps(backend)),get_vocab=lambda:dict(vocab),all_special_ids=[256,257])
  c=Constraint(schema,tok,vocab_size=288,eos=(256,257),preserve_required_order=True,whitespace_pattern='')
  from outlines_core import Guide
  def accepts(value):
   guide=Guide(c.index)
   try:
    for token in dumps(value).encode():guide.advance(token,return_tokens=False)
   except ValueError:return False
   return guide.is_finished()
  self.assertTrue(accepts(select(scope=scope,findings=[a,b])))
  self.assertTrue(accepts(select(scope=scope,findings=[a])))
  self.assertFalse(accepts(select(scope=scope,findings=[a,a])))
  self.assertFalse(accepts(select(scope=scope,findings=[b,a])))

if __name__=='__main__':unittest.main()
