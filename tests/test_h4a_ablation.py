"""Synthetic H4a advisory routing controls; no native inference or real case reading."""
import json,unittest
import test_answers as fixtures
from test_source_findings import make,plan_data,select,animal
from medlens.generation import source_findings as selector
from medlens.generation import source_scope as guard

class H4aAdvisoryKind(unittest.TestCase):
 def test_four_recognized_kinds_have_exact_proofs_without_positive_validation(self):
  cases={'mechanism':'By what molecular mechanism does factor A alter process B?',
         'diagnostic':'What is the diagnostic accuracy of test C for condition D?',
         'association':'Is exposure A associated with outcome B?',
         'effect':'Does agent A reduce outcome B?'}
  for kind,question in cases.items():
   with self.subTest(kind=kind):
    hint=guard.question_task_kind(question)
    self.assertEqual(hint['kind'],kind);self.assertTrue(hint['advisory_only'])
    self.assertFalse(hint['positive_validation_performed'])
    self.assertEqual(len(hint['question_proofs']),1)
    proof=hint['question_proofs'][0]
    self.assertEqual(question[proof['char_start']:proof['char_end']],proof['text'])
    self.assertEqual(proof['field'],'question')
 def test_unknown_does_not_infer_task_kind(self):
  for question in ('How does A compare with B?','How does A relate to B?',
      'How accurate is model C?','What is sensitivity of C?',
      'Should agent A reduce outcome B?','Explain the word mechanism.',
      'Task kind: mechanism. Does A reduce B?',
      'Does A reduce B, and is C associated with D?',
      'Does A reduce B or is C associated with D?',
      'What is the mechanism of A or the diagnostic accuracy of C?',
      'What is the mechanism of A and the diagnostic accuracy of C?',
      'What is the mechanism of A. What is the diagnostic accuracy of C?',
      'What is the mechanism of A.\nDoes C reduce D?',
      'Does A reduce B? Is C associated with D?',
      'Как A связан с B?'):
   with self.subTest(question=question):
    hint=guard.question_task_kind(question)
    self.assertEqual(hint['kind'],'unknown');self.assertEqual(hint['question_proofs'],[])
 def test_question_object_words_do_not_override_outer_intent(self):
  for question in ('What is the effect of A on diagnostic accuracy?',
                   'Does mechanism A reduce B?',
                   'Does association measure A reduce B?'):
   self.assertEqual(guard.question_task_kind(question)['kind'],'effect')
 def test_explicit_mechanism_stays_mechanism_with_subordinate_effect_verb(self):
  question='What molecular mechanisms does factor A use to reduce process B?'
  self.assertEqual(guard.question_task_kind(question)['kind'],'mechanism')
 def test_original_contract_is_unchanged_and_separate(self):
  self.assertEqual(guard.question_contract('Explain an unfamiliar mechanism'),{})
  question='Does A reduce B in adults?'
  contract=guard.question_contract(question)
  self.assertEqual(contract['outcome']['text'],'B')
  self.assertNotIn('kind',contract);self.assertNotIn('task_kind',contract)
 def test_one_call_same_schema_no_extra_output_axis_and_scientific_flags_unchanged(self):
  question='By what mechanism does factor A alter process B?'
  ctx,articles=animal();idx,plans,checks=plan_data(ctx,articles,question)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  runtime=fixtures.FakeRuntime([raw]);out=selector.generate_source_findings_answer(question,ctx,runtime,source_articles=articles)
  self.assertEqual(out['status'],'partial');self.assertEqual(len(runtime.calls),1)
  prompt,cap=runtime.calls[0];self.assertEqual(cap,768)
  self.assertEqual(prompt['decoding_schema'],selector.selection_schema(plans))
  self.assertEqual(set(raw['articles'][0]),{'article_id','scope_quote_id','finding_quote_ids','applicability'})
  content=prompt['messages'][1]['content'];self.assertIn('Advisory task kind (not source validation)',content)
  self.assertIn(selector.TASK_KIND_GUIDANCE['mechanism'],content)
  self.assertFalse(out['semantic_support_verified']);self.assertFalse(out['answer_completeness_established'])
  self.assertEqual(out['verification_status'],'scientific-review-required')
 def test_advisory_kind_does_not_promote_explicit_model_nonanswer(self):
  question='Is exposure A associated with outcome B?'
  ctx,articles=make([[('METHODS','Adult participants were enrolled in an observational cohort.'),
                      ('RESULTS','Exposure A was associated with outcome B.')]])
  idx,plans,checks=plan_data(ctx,articles,question)
  for decision in ('different_population','indirect_result'):
   raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'],applicability=decision)
   out=selector.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=articles)
   self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(out['claims'],[])
 def test_mechanism_objective_only_still_abstains_before_model(self):
  question='By what mechanism does factor A alter process B?'
  ctx,articles=make([[('OBJECTIVE','We aimed to investigate mechanism A.')]])
  runtime=fixtures.FakeRuntime([])
  out=selector.generate_source_findings_answer(question,ctx,runtime,source_articles=articles)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(runtime.calls,[])
 def test_diagnostic_procedure_only_still_abstains_before_model(self):
  question='What is the diagnostic accuracy of test C for condition D?'
  ctx,articles=make([[('METHODS','Sensitivity and specificity were calculated against reference C.')]])
  runtime=fixtures.FakeRuntime([])
  out=selector.generate_source_findings_answer(question,ctx,runtime,source_articles=articles)
  self.assertEqual(out['status'],'insufficient_evidence');self.assertEqual(runtime.calls,[])
 def test_effect_association_substitution_still_demoted(self):
  question='Does agent Z reduce pain in adults?'
  ctx,articles=make([[('METHODS','Adult patients were randomized to agent Z or comparator X.'),
                      ('RESULTS','Change in marker K was correlated with change in pain.')]])
  idx,plans,checks=plan_data(ctx,articles,question)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=selector.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=articles)
  self.assertEqual(out['status'],'insufficient_evidence')
  self.assertIn('selected_outcome_is_change_association',
      {item['code']for item in out['source_relations'][0]['selected_scope_checks']['blocks']})
 def test_null_effect_remains_source_extract_without_promoting_to_benefit(self):
  question='Does agent Z reduce pain in adults?'
  quote='Agent Z did not reduce pain compared with comparator X.'
  ctx,articles=make([[('METHODS','Adult patients were randomized to agent Z or comparator X.'),('RESULTS',quote)]])
  idx,plans,checks=plan_data(ctx,articles,question)
  raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'])
  out=selector.generate_source_findings_answer(question,ctx,fixtures.FakeRuntime([raw]),source_articles=articles)
  self.assertEqual(out['status'],'partial');self.assertTrue(out['claims'][0]['text'].endswith(quote))
 def test_time_reference_and_mandatory_scope_vetoes_stay_in_force(self):
  question='Does agent Z reduce pain in adults after 12 weeks?'
  result=guard.check_source_scope(question,'Trial',[{'index':0,'text':'Pain was lower after 8 weeks.'}])
  self.assertIn('outcome_time_not_established',{x['code']for x in result['blocks']})
  question='What is the diagnostic accuracy of C using violet assay as the reference standard?'
  result=guard.check_source_scope(question,'Accuracy study',[{'index':0,'text':'SC53 served as reference standard.'}])
  self.assertIn('different_reference_standard',{x['code']for x in result['blocks']})
  ctx,articles=animal();idx,plans,checks=plan_data(ctx,articles,'By what mechanism does A alter B?')
  with self.assertRaises(selector.AnswerError):
   selector.materialize_selection(json.dumps(select(findings=plans[0]['finding_quote_ids'])),ctx,idx,plans,checks)

if __name__=='__main__':unittest.main()
