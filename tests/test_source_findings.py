"""Fictional scope/identity/closed-schema checks without any model inference."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest,sys
import test_answers as fixtures
from medlens.generation.context import ContextConfig,build_context,digest,COVERAGE_POLICY
from medlens.generation.quotes import quote_catalog
from medlens.inference.structured import Constraint,byte_decoder,prepare_schema
from medlens.generation import source_findings as c


def make(abstracts,title='Fictional study',coverage=True):
 articles=[];windows=[]
 for i,sections in enumerate(abstracts,1):
  article_title=title[i-1]if isinstance(title,list)else title
  pmid=str(i);provenance={'fixture':'scope-guards','record':i}
  source_sections=[]
  for j,(label,text)in enumerate(sections):
   section={'index':j,'label':label,'text':text,'source_path':f'abstract/{j}'};source_sections.append(section)
   windows.append({'pmid':pmid,'window_id':f'{i}-{j}','text_version_id':f'v{i}',
    'title':article_title or '','kind':'abstract','section_index':j,'section_label':label,'source_path':f'abstract/{j}',
    'char_start':0,'char_end':len(text),'text':text,'provenance':provenance})
  articles.append({'pmid':pmid,'title':article_title,'journal':None,'dates':[],
   'abstract':{'sections':source_sections},'text_version':{'text_version_id':f'v{i}'},'provenance':provenance})
 dm={'dataset_id':'fictional','articles_sha256':digest(articles)};wm={**dm,'artifact_id':'fictional-windows'}
 config=ContextConfig(selection_mode='results-methods'if coverage else'article-diversity')
 ctx=build_context(SimpleNamespace(manifest=dm,articles=articles),wm,windows,[a['pmid']for a in articles],[.9]*len(windows),fixtures.Tokenizer(),{},config=config)
 return ctx,{a['pmid']:a for a in articles}


def animal():
 return make([[('BACKGROUND','Prior experiments in animals may have relevance to humans.'),
  ('METHODS','In a mouse experiment, mice were treated with agent ZX.'),
  ('RESULTS','Signal fell by 30% in that model.')]])


def select(article_id='1',scope=None,findings=(),relation='direct',applicability=None):
 decision=applicability or {'direct':'answers_question','related':'indirect_result','none':'unrelated','unclear':'missing_details'}[relation]
 return {'articles':[{'article_id':article_id,'scope_quote_id':scope,'finding_quote_ids':list(findings),
  'applicability':decision}],'limitations':[]}


def plan_data(ctx,arts,q='What does the model show?'):
 idx=quote_catalog(ctx,fixtures.Tokenizer());plans,checks=c.scope_plans(q,ctx,idx,arts);return idx,plans,checks


class Guards(unittest.TestCase):
 def test_actual_animal_setup_is_forced_and_visible_before_result(self):
  ctx,arts=animal();before=deepcopy(ctx);idx,plans,checks=plan_data(ctx,arts)
  self.assertTrue(plans[0]['reported_animal_study']);self.assertTrue(plans[0]['scope_required'])
  scope=plans[0]['scope_quote_ids'][0];finding=plans[0]['finding_quote_ids'][0]
  raw=json.dumps(select(scope=scope,findings=[finding]));rt=fixtures.FakeRuntime([raw])
  out=c.generate_source_findings_answer('What does the model show?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertEqual([cap for _,cap in rt.calls],[768])
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

 def test_missing_bounded_human_setup_cannot_be_omitted_from_card(self):
  text='A'*520+' adult patients were randomized to agent Z or placebo.'
  ctx,arts=make([[('METHODS',text),('RESULTS','Pain was lower with agent Z.')]])
  rt=fixtures.FakeRuntime([])
  out=c.generate_source_findings_answer('Does agent Z reduce pain?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(rt.calls,[])
  self.assertEqual(out['scope_checks']['1']['policy_blocks'][0]['code'],'reported_study_scope_not_in_bounded_catalog')

 def test_required_scope_missing_or_repeated_in_findings_is_rejected(self):
  ctx,arts=animal();idx,plans,checks=plan_data(ctx,arts);scope=plans[0]['scope_quote_ids'][0]
  result=plans[0]['finding_quote_ids'][0]
  for value in (select(scope=None,findings=[result]),select(scope=scope,findings=[scope])):
   with self.assertRaises(c.AnswerError):c.materialize_selection(json.dumps(value),ctx,idx,plans,checks)

 def test_unknown_adult_eligibility_is_caution_not_inferred_population(self):
  ctx,arts=make([[(None,'Participants received agent ZX versus placebo. The score fell.')]])
  idx,plans,checks=plan_data(ctx,arts,'In adults, what happened?');self.assertFalse(plans[0]['scope_required'])
  qid=plans[0]['eligible_quote_ids'][-1];rt=fixtures.FakeRuntime([select(scope=None,findings=[qid])])
  out=c.generate_source_findings_answer('In adults, what happened?',ctx,rt,source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertIn('population_unclear',out['limitations'])
  self.assertEqual(out['claims'],[])
  self.assertEqual(out['draft']['claims'][0]['scope_cautions'][0]['code'],'adult_eligibility_not_explicit')
  self.assertEqual(out['card_admission']['excluded_cards'][0]['reason_code'],'adult_eligibility_not_explicit')

 def test_explicit_scope_veto_only_demotes_and_can_abstain_before_model(self):
  title='Agent A as add-on to dose-adjusted B versus B in a trial'
  ctx,arts=make([[('METHODS','Adults were assigned to A plus adjusted B versus B.'),('RESULTS','The score fell.')]],title)
  question='How does agent A compare with agent B?';idx,plans,checks=plan_data(ctx,arts,question)
  self.assertTrue(checks['1']['blocks']);self.assertFalse(plans[0]['allow_direct'])
  scope=plans[0]['scope_quote_ids'][0];qid=plans[0]['finding_quote_ids'][0]
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

 def test_endpoint_list_cannot_replace_actual_review_population(self):
  ctx,arts=make([[('METHODS','We included trials of agent Z during the pre-exposure phase in participants at risk of the condition. Primary outcomes included symptom scores and incidence.'),('RESULTS','Symptom scores were lower with agent Z.')]])
  idx,plans,checks=plan_data(ctx,arts)
  pool=[idx['catalog'][qid]['quote']for qid in plans[0]['scope_quote_ids']]
  self.assertEqual(pool,['We included trials of agent Z during the pre-exposure phase in participants at risk of the condition.'])
  self.assertTrue(plans[0]['scope_required'])

 def test_background_and_methods_cannot_be_selected_as_results(self):
  ctx,arts=animal();idx,plans,checks=plan_data(ctx,arts)
  scope=plans[0]['scope_quote_ids'][0]
  background=scope
  with self.assertRaises(c.AnswerError)as caught:
   c.materialize_selection(json.dumps(select(scope=scope,findings=[background])),ctx,idx,plans,checks)
  self.assertEqual(caught.exception.code,'selection_result_role')

 def test_every_explicit_nonanswer_decision_stays_hidden_without_rewriting_source(self):
  ctx,arts=make([[(None,'The measured outcome improved.')]])
  idx,plans,checks=plan_data(ctx,arts);qid=plans[0]['finding_quote_ids'][0]
  for decision in c.APPLICABILITY[1:]:
   raw=select(findings=[qid],applicability=decision)
   value,draft,body,relations=c.materialize_selection(json.dumps(raw),ctx,idx,plans,checks)
   self.assertEqual(value,raw);self.assertNotEqual(relations[0]['effective_relation'],'direct')
   self.assertEqual(draft['claims'][0]['text'],'The measured outcome improved.')
   out=c.generate_source_findings_answer('Question',ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
   self.assertEqual(out['status'],'insufficient_evidence')

 def test_vetoed_first_article_is_audited_but_not_sent_to_selector(self):
  ctx,arts=make([[('METHODS','Adults were assigned to A plus reduced-dose B or B.'),('RESULTS','The score improved.')],
                [('METHODS','Adults were randomized to agent A or agent B.'),('RESULTS','Pain was reduced with A compared with B.')]],
               title=['A as add-on to dose-reduced B versus B','A versus B trial'])
  question='How does A compare with B?';idx,plans,checks=plan_data(ctx,arts,question)
  self.assertFalse(plans[0]['allow_direct']);self.assertTrue(plans[1]['allow_direct'])
  raw=select(article_id='2',scope=plans[1]['scope_quote_ids'][0],findings=plans[1]['finding_quote_ids'])
  runtime=fixtures.FakeRuntime([raw]);out=c.generate_source_findings_answer(question,ctx,runtime,source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertEqual(out['claims'][0]['source_article_id'],'2')
  self.assertEqual(out['selector_menu']['program_excluded_article_ids'],['1'])
  self.assertIn('1',out['scope_checks']);self.assertTrue(out['scope_checks']['1']['blocks'])
  self.assertEqual({source['pmid']for source in json.loads(out['generator_evidence_payload'])['sources']},{'2'})
  self.assertNotIn('A as add-on to dose-reduced B',runtime.calls[0][0]['messages'][1]['content'])

 def test_month_may_in_reported_date_does_not_hide_adult_scope(self):
  ctx,arts=make([[('METHODS',"Adults were enrolled from March to May 1998."),('RESULTS','Pain scores fell.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertEqual([idx['catalog'][qid]['quote']for qid in plans[0]['scope_quote_ids']],
                   ['Adults were enrolled from March to May 1998.'])

 def test_change_correlation_cannot_supply_a_symptom_effect_even_if_model_says_direct(self):
  ctx,arts=make([[('METHODS','Adult patients were randomized to agent Z or comparator X.'),
   ('RESULTS','Marker K improved more with Z. Change in marker K was not correlated with change in pain.')]])
  question='Does agent Z reduce pain in adults?';idx,plans,checks=plan_data(ctx,arts,question)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence')
  relation=out['source_relations'][0]
  self.assertEqual(relation['effective_relation'],'related')
  self.assertEqual(relation['selected_scope_checks']['blocks'][0]['code'],'selected_outcome_is_change_association')

 def test_unstructured_procedure_ids_are_excluded_while_actual_results_remain(self):
  ctx,arts=make([[(None,'The aim of this study was to assess accuracy in adults. Sensitivity and specificity were calculated against RX42 as gold standard. The assay had high sensitivity and low specificity.')]])
  idx,plans,checks=plan_data(ctx,arts,'What is diagnostic accuracy using RX42 as gold standard?')
  findings=[idx['catalog'][qid]['quote']for qid in plans[0]['finding_quote_ids']]
  self.assertEqual(findings,['The assay had high sensitivity and low specificity.'])
  raw=select(findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer('What is diagnostic accuracy using RX42 as gold standard?',ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'partial')
  method_id=next(qid for qid in plans[0]['eligible_quote_ids']if 'were calculated' in idx['catalog'][qid]['quote'])
  with self.assertRaises(c.AnswerError)as caught:
   c.materialize_selection(json.dumps(select(findings=[method_id])),ctx,idx,plans,checks)
  self.assertEqual(caught.exception.code,'selection_result_role')

 def test_all_procedure_source_abstains_before_model(self):
  ctx,arts=make([[(None,'The aim of this study was to assess accuracy. Sensitivity and specificity were calculated against RX42 as gold standard.')]])
  runtime=fixtures.FakeRuntime([])
  out=c.generate_source_findings_answer('What is diagnostic accuracy?',ctx,runtime,source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(runtime.calls,[])

 def test_mixed_review_trial_counts_do_not_become_treatment_effect(self):
  ctx,arts=make([[('METHODS','Trials included treatment or prevention during pregnancy or after birth.'),
   ('RESULTS','Seven trials including 521 participants met the inclusion criteria. Three randomized trials and four longitudinal studies were found.')]])
  question='Does therapy reduce symptoms in women with postpartum Quartz syndrome?'
  idx,plans,checks=plan_data(ctx,arts,question)
  self.assertTrue(plans[0]['allow_direct'])
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'insufficient_evidence')
  self.assertEqual(out['source_relations'][0]['selected_scope_checks']['blocks'][0]['code'],'selected_findings_are_study_accounting')

 def test_open_label_same_drug_then_placebo_is_not_new_post_modality_treatment(self):
  question='Does continuation treatment reduce relapse compared with discontinuation in adults?'
  for setup,expected in [('Adults who remitted during open-label AX17 were randomized to continue AX17 or placebo.','partial'),
                         ('After remission with another modality, adults were randomized to start AX17 or placebo.','insufficient_evidence'),
                         ('Adults who responded to AX17 were randomized to B-AX17 or placebo.','insufficient_evidence'),
                         ('Adults who responded to modality Q were randomized to agent Z or placebo.','insufficient_evidence')]:
   ctx,arts=make([[('METHODS',setup),('RESULTS','Relapse was lower with AX17 than placebo.')]])
   idx,plans,checks=plan_data(ctx,arts,question)
   outputs=[select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])]if expected=='partial'else[]
   runtime=fixtures.FakeRuntime(outputs)
   out=c.generate_source_findings_answer(question,ctx,runtime,source_articles=arts)
   self.assertEqual(out['status'],expected)
   self.assertEqual(len(runtime.calls),1 if expected=='partial'else 0)

 def test_actual_continuation_versus_stopping_result_remains_useful(self):
  ctx,arts=make([[('METHODS','Adults in remission were randomized to continuation medication or discontinuation with placebo.'),
   ('RESULTS','Relapse was lower with continued medication than placebo.')]])
  question='Does continuation treatment reduce relapse compared with discontinuation in adults?'
  idx,plans,checks=plan_data(ctx,arts,question)
  self.assertTrue(plans[0]['allow_direct'])
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertIn('continuation medication',out['claims'][0]['text'])

 def test_null_effect_and_unrestricted_animal_mechanism_remain_useful(self):
  ctx,arts=make([[('METHODS','Adults were randomized to agent Z or placebo.'),('RESULTS','Pain was not reduced by agent Z compared with placebo.')]])
  question='Does agent Z reduce pain in adults?';idx,plans,checks=plan_data(ctx,arts,question)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertIn('not reduced',out['claims'][0]['text'])
  ctx,arts=animal();idx,plans,checks=plan_data(ctx,arts)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=c.generate_source_findings_answer('What mechanism does the experiment show?',ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
  self.assertEqual(out['status'],'partial');self.assertIn('mouse experiment',out['claims'][0]['text'])

 def test_association_can_answer_association_question_but_not_treatment_effect(self):
  ctx,arts=make([[(None,'Change in marker K was associated with pain.')]])
  for question,status in [('Is marker K associated with pain?','partial'),('Does agent Z reduce pain?','insufficient_evidence')]:
   idx,plans,checks=plan_data(ctx,arts,question)
   raw=select(findings=plans[0]['finding_quote_ids'],applicability='answers_question'if status=='partial'else'indirect_result')
   out=c.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=arts)
   self.assertEqual(out['status'],status)

class Native(unittest.TestCase):
 def make(self,ctx,arts,blocked_first=False):
  idx,plans,checks=plan_data(ctx,arts)
  if blocked_first:plans[0]['allow_direct']=False
  schema,_=prepare_schema(c.selection_schema(plans),'literal')
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
  finding=plans[0]['finding_quote_ids'][0]
  self.assertTrue(self.accepts(native,select(scope=scope,findings=[finding])))
  self.assertTrue(self.accepts(native,select(scope=None,findings=[],relation='none')))
  self.assertFalse(self.accepts(native,select(scope=None,findings=[finding])))
  self.assertFalse(self.accepts(native,select(scope=scope,findings=[scope])))
  self.assertFalse(self.accepts(native,select(scope=scope,findings=[])))
  bad=select(scope=scope,findings=[finding]);bad['articles'][0]['applicability']='not_requested'
  self.assertFalse(self.accepts(native,bad))
 def test_blocked_first_slot_does_not_forbid_direct_later_slot(self):
  ctx,arts=make([[(None,'One measured result.')],[(None,'Another measured result.')]])
  native,plans=self.make(ctx,arts,blocked_first=True)
  first=select(findings=[],relation='none')['articles'][0]
  second=select(article_id='2',findings=plans[1]['finding_quote_ids'])['articles'][0]
  self.assertTrue(self.accepts(native,{'articles':[first,second],'limitations':[]}))
  first['applicability']='answers_question';first['finding_quote_ids']=plans[0]['finding_quote_ids']
  self.assertFalse(self.accepts(native,{'articles':[first,second],'limitations':[]}))
 def test_optional_nonanimal_context_and_exact_slot_shape(self):
  ctx,arts=make([[(None,'A source finding.')]]);native,plans=self.make(ctx,arts)
  self.assertTrue(self.accepts(native,select(scope=None,findings=[plans[0]['eligible_quote_ids'][0]])))
  bad=select(findings=['E1.Q1']);bad['articles'][0]['text']='Invented';self.assertFalse(self.accepts(native,bad))
  bad=select(findings=['E1.Q1']);bad['articles'][0]['article_id']='2';self.assertFalse(self.accepts(native,bad))

if __name__=='__main__':unittest.main()
