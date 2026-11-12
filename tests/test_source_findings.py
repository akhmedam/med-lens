"""Fictional scope/identity/closed-schema checks without any model inference."""
from copy import deepcopy
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
import unittest,sys
import test_answers as fixtures
from medlens.generation.context import ContextConfig,build_context,digest,COVERAGE_POLICY
from medlens.generation.quotes import quote_catalog
from medlens.inference.structured import Constraint,byte_decoder,prepare_schema
from medlens.generation import source_findings as c
HAS_CORE=importlib.util.find_spec('outlines_core')is not None


def make(abstracts,title='Fictional study',coverage=True):
 articles=[];windows=[]
 for i,sections in enumerate(abstracts,1):
  pmid=str(i);provenance={'fixture':'scope-guards','record':i}
  source_sections=[]
  for j,(label,text)in enumerate(sections):
   section={'index':j,'label':label,'text':text,'source_path':f'abstract/{j}'};source_sections.append(section)
   windows.append({'pmid':pmid,'window_id':f'{i}-{j}','text_version_id':f'v{i}',
    'title':title or '','kind':'abstract','section_index':j,'section_label':label,'source_path':f'abstract/{j}',
    'char_start':0,'char_end':len(text),'text':text,'provenance':provenance})
  articles.append({'pmid':pmid,'title':title,'journal':None,'dates':[],
   'abstract':{'sections':source_sections},'text_version':{'text_version_id':f'v{i}'},'provenance':provenance})
 dm={'dataset_id':'fictional','articles_sha256':digest(articles)};wm={**dm,'artifact_id':'fictional-windows'}
 config=ContextConfig(selection_mode='results-methods'if coverage else'article-diversity')
 ctx=build_context(SimpleNamespace(manifest=dm,articles=articles),wm,windows,[a['pmid']for a in articles],[.9]*len(windows),fixtures.Tokenizer(),{},config=config)
 return ctx,{a['pmid']:a for a in articles}


def animal():
 return make([[('BACKGROUND','Prior experiments in animals may have relevance to humans.'),
  ('METHODS','In a mouse experiment, mice were treated with agent ZX.'),
  ('RESULTS','Signal fell by 30% in that model.')]])


def select(article_id='1',scope=None,findings=(),relation='direct'):
 return {'articles':[{'article_id':article_id,'scope_quote_id':scope,'finding_quote_ids':list(findings),'relation':relation}],'limitations':[]}


def plan_data(ctx,arts,q='What does the model show?'):
 idx=quote_catalog(ctx,fixtures.Tokenizer());plans,checks=c.scope_plans(q,ctx,idx,arts);return idx,plans,checks


class Guards(unittest.TestCase):
 def test_actual_animal_setup_is_forced_and_visible_before_result(self):
  ctx,arts=animal();before=deepcopy(ctx);idx,plans,checks=plan_data(ctx,arts)
  self.assertTrue(plans[0]['reported_animal_study']);self.assertTrue(plans[0]['scope_required'])
  scope=plans[0]['scope_quote_ids'][0];finding=next(q for q in plans[0]['eligible_quote_ids']if q!=scope)
  raw=json.dumps(select(scope=scope,findings=[finding]));rt=fixtures.FakeRuntime([raw])
  out=c.generate_source_findings_answer('What does the model show?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertEqual([cap for _,cap in rt.calls],[512])
  claim=out['claims'][0];self.assertTrue(claim['text'].startswith('In a mouse experiment'))
  self.assertEqual(claim['citations'][0]['quote_id'],scope);self.assertEqual(claim['source_title'],'Fictional study')
  self.assertEqual(claim['source_article_id'],'1');self.assertEqual(out['model_calls'][0]['text'],raw)
  self.assertFalse(out['semantic_support_verified']);self.assertTrue(out['quote_identity_verified']);self.assertEqual(ctx,before)

 def test_background_animals_do_not_make_a_human_study_animal_only(self):
  ctx,arts=make([[('BACKGROUND','Previous studies in mice were performed.'),
   ('METHODS','Adult participants were enrolled and assigned to agent Z.'),('RESULTS','The score fell.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertFalse(plans[0]['reported_animal_study']);self.assertTrue(plans[0]['scope_required'])
  quotes=[idx['catalog'][q]['quote']for q in plans[0]['scope_quote_ids']]
  self.assertEqual(quotes,['Adult participants were enrolled and assigned to agent Z.'])

 def test_missing_bounded_animal_context_cannot_be_direct(self):
  text='A'*520+' mice were treated with agent ZX.'
  ctx,arts=make([[('METHODS',text),('RESULTS','The score fell.')]])
  rt=fixtures.FakeRuntime([]);out=c.generate_source_findings_answer('What does the model show?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(rt.calls,[])
  self.assertEqual(out['scope_checks']['1']['policy_blocks'][0]['code'],'reported_animal_scope_not_in_bounded_catalog')

 def test_required_scope_missing_or_repeated_in_findings_is_rejected(self):
  ctx,arts=animal();idx,plans,checks=plan_data(ctx,arts);scope=plans[0]['scope_quote_ids'][0]
  result=next(q for q in plans[0]['eligible_quote_ids']if q!=scope)
  for value in (select(scope=None,findings=[result]),select(scope=scope,findings=[scope])):
   with self.assertRaises(c.AnswerError):c.materialize_selection(json.dumps(value),ctx,idx,plans,checks)

 def test_unknown_adult_eligibility_is_caution_not_inferred_population(self):
  ctx,arts=make([[(None,'Participants received agent ZX versus placebo. The score fell.')]])
  idx,plans,checks=plan_data(ctx,arts,'In adults, what happened?');self.assertFalse(plans[0]['scope_required'])
  qid=plans[0]['eligible_quote_ids'][-1];rt=fixtures.FakeRuntime([select(scope=None,findings=[qid])])
  out=c.generate_source_findings_answer('In adults, what happened?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertIn('population_unclear',out['limitations'])
  self.assertEqual(out['claims'][0]['scope_cautions'][0]['code'],'adult_eligibility_not_explicit')

 def test_explicit_scope_veto_only_demotes_and_can_abstain_before_model(self):
  title='Agent A as add-on to dose-adjusted B versus B in a trial'
  ctx,arts=make([[('METHODS','Adults were assigned to A plus adjusted B versus B.'),('RESULTS','The score fell.')]],title)
  question='How does agent A compare with agent B?';idx,plans,checks=plan_data(ctx,arts,question)
  self.assertTrue(checks['1']['blocks']);self.assertFalse(plans[0]['allow_direct'])
  scope=plans[0]['scope_quote_ids'][0];qid=next(q for q in plans[0]['eligible_quote_ids']if q!=scope)
  raw=select(scope=scope,findings=[qid]);value,draft,body,relations=c.materialize_selection(json.dumps(raw),ctx,idx,plans,checks)
  self.assertEqual(value,raw);self.assertEqual(relations[0]['raw_relation'],'direct');self.assertEqual(relations[0]['effective_relation'],'related')
  rt=fixtures.FakeRuntime([]);out=c.generate_source_findings_answer(question,ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(rt.calls,[])

 def test_rehashed_source_or_sentence_metadata_cannot_bypass_article_binding(self):
  ctx,arts=animal();mutated=deepcopy(arts);mutated['1']['abstract']['sections'][1]['text']+=' Changed.'
  out=c.generate_source_findings_answer('Question',ctx,fixtures.FakeRuntime([]),source_articles=mutated)
  self.assertEqual(out['status'],'validation_failed');self.assertEqual(out['model_calls'],[])
  changed=deepcopy(ctx);source=changed['sources'][0];source['quote_sentence_spans'][0]['char_end']-=1
  changed['sources_sha256']=digest(changed['sources'])
  keys=('schema_version','dataset_id','window_artifact_id','article_ranking','scores_sha256','config','selection_policy','tokenizer','sources_sha256','implementation_sha256')
  changed['context_id']=digest({k:changed[k]for k in keys})
  out=c.generate_source_findings_answer('Question',changed,fixtures.FakeRuntime([]),source_articles=arts)
  self.assertEqual(out['status'],'validation_failed');self.assertEqual(out['model_calls'],[])

 def test_missing_title_none_is_preserved_and_legacy_context_rejected(self):
  ctx,arts=make([[(None,'A source finding.')]],title=None);idx,plans,checks=plan_data(ctx,arts)
  out=c.generate_source_findings_answer('Question',ctx,fixtures.FakeRuntime([select(findings=[plans[0]['eligible_quote_ids'][0]])]),source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertIsNone(out['claims'][0]['source_title'])
  legacy,articles=make([[(None,'A source finding.')]],coverage=False)
  out=c.generate_source_findings_answer('Question',legacy,fixtures.FakeRuntime([]),source_articles=articles)
  self.assertEqual(out['validation_error']['code'],'coverage_policy')

 def test_empty_direct_and_capped_selection_fail_without_repair(self):
  ctx,arts=make([[(None,'A source finding.')]])
  out=c.generate_source_findings_answer('Question',ctx,fixtures.FakeRuntime([select(findings=[])]),source_articles=arts)
  self.assertEqual(out['validation_error']['code'],'selection_scope')
  class Limited(fixtures.FakeRuntime):
   def generate(self,prompt,*,max_new_tokens):return {**super().generate(prompt,max_new_tokens=max_new_tokens),'hit_output_limit':True}
  raw=json.dumps(select(findings=['E1.Q1']));out=c.generate_source_findings_answer('Question',ctx,Limited([raw]),source_articles=arts)
  self.assertEqual(out['validation_error']['code'],'output_limit');self.assertEqual(out['model_calls'][0]['text'],raw)

@unittest.skipUnless(HAS_CORE,'native outlines-core dependency absent')
class Native(unittest.TestCase):
 def make(self,ctx,arts):
  idx,plans,checks=plan_data(ctx,arts);schema,_=prepare_schema(c.selection_schema(plans),'literal')
  vocab={ch:i for ch,i in byte_decoder().items()};backend={'decoder':{'type':'ByteLevel'},'model':{'type':'BPE','vocab':vocab}}
  tok=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:json.dumps(backend)),get_vocab=lambda:dict(vocab),all_special_ids=[256,257])
  return Constraint(schema,tok,vocab_size=288,eos=(256,257),preserve_required_order=True,whitespace_pattern=''),plans
 def accepts(self,native,value):
  from outlines_core import Guide
  guide=Guide(native.index)
  try:
   for token in json.dumps(value,separators=(',',':')).encode():guide.advance(token,return_tokens=False)
  except ValueError:return False
  return guide.is_finished()
 def test_animal_context_is_structural_and_source_role_cannot_repeat(self):
  ctx,arts=animal();native,plans=self.make(ctx,arts);scope=plans[0]['scope_quote_ids'][0]
  finding=next(q for q in plans[0]['eligible_quote_ids']if q!=scope)
  self.assertTrue(self.accepts(native,select(scope=scope,findings=[finding])))
  self.assertTrue(self.accepts(native,select(scope=None,findings=[],relation='none')))
  self.assertFalse(self.accepts(native,select(scope=None,findings=[finding])))
  self.assertFalse(self.accepts(native,select(scope=scope,findings=[scope])))
  self.assertFalse(self.accepts(native,select(scope=scope,findings=[])))
 def test_optional_nonanimal_context_and_exact_slot_shape(self):
  ctx,arts=make([[(None,'A source finding.')]]);native,plans=self.make(ctx,arts)
  self.assertTrue(self.accepts(native,select(scope=None,findings=[plans[0]['eligible_quote_ids'][0]])))
  bad=select(findings=['E1.Q1']);bad['articles'][0]['text']='Invented';self.assertFalse(self.accepts(native,bad))
  bad=select(findings=['E1.Q1']);bad['articles'][0]['article_id']='2';self.assertFalse(self.accepts(native,bad))

if __name__=='__main__':unittest.main()
