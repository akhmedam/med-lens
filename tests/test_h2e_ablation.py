import unittest
from medlens.generation.source_scope import check_selected_findings,check_source_scope

Q='What is the diagnostic accuracy of assay R for Quartz syndrome in adults using clinical diagnosis as the reference?'
CODE='selected_activity_instead_of_diagnostic_presence'
def evaluate(quote,question=Q,extra='',population='Patients with Quartz syndrome were enrolled.'):
 sections=[{'index':0,'label':'OBJECTIVE','text':'To assess clinical activity and disease severity in patients with Quartz syndrome.'},
           {'index':1,'label':'METHODS','text':population},
           {'index':2,'label':'RESULTS','text':quote+(' '+extra if extra else'')}]
 return check_selected_findings(question,'Assay R for disease activity',sections,[quote]),sections

def codes(result,bucket='blocks'):return {item['code']for item in result[bucket]}

class H2eSelectedDiagnosticTarget(unittest.TestCase):
 def test_established_disease_activity_is_not_diagnostic_presence(self):
  quote='Assay R had sensitivity 90% for endoscopic activity and specificity 80% for remission.'
  result,_=evaluate(quote)
  self.assertIn(CODE,codes(result));self.assertFalse(result['positive_validation_performed'])
 def test_active_disease_label_requires_actual_activity_goal_and_known_population(self):
  quote='Assay R showed sensitivity 85% and specificity 75% in diagnosing active disease.'
  result,_=evaluate(quote)
  self.assertIn(CODE,codes(result))
 def test_condition_acronym_before_patients_is_generic(self):
  quote='Sensitivity in diagnosing active disease was 85%.'
  result,sections=evaluate(quote,population='We included 180 QS patients.')
  self.assertIn(CODE,codes(result))
 def test_explicitly_requested_activity_is_allowed_including_null_result(self):
  quote='Assay R did not distinguish endoscopic activity from remission (AUC 0.50).'
  result,_=evaluate(quote,question='What is diagnostic accuracy of assay R for Quartz syndrome activity?')
  self.assertNotIn(CODE,codes(result))
 def test_active_infection_presence_is_not_activity_monitoring(self):
  sections=[{'index':0,'label':'OBJECTIVE','text':'To diagnose active infection.'},
            {'index':1,'label':'METHODS','text':'Patients with suspected active infection were assessed.'},
            {'index':2,'label':'RESULTS','text':'The assay diagnosed active infection with sensitivity 90% and specificity 85%.'}]
  result=check_selected_findings('What is diagnostic accuracy of assay R for active infection?',
       'Diagnosis of active infection',sections,[sections[2]['text']])
  self.assertNotIn(CODE,codes(result))
 def test_mixed_paper_is_judged_by_selected_result(self):
  presence='Assay R diagnosed Quartz syndrome versus healthy controls with sensitivity 90% and specificity 85%.'
  activity='Assay R detected endoscopic activity and remission with sensitivity 80%.'
  result,_=evaluate(presence,extra=activity)
  self.assertNotIn(CODE,codes(result))
  result,_=evaluate(activity,extra=presence)
  self.assertIn(CODE,codes(result))
 def test_null_real_presence_result_remains_allowed(self):
  quote='Assay R did not distinguish Quartz syndrome from healthy controls (AUC 0.50).'
  result,_=evaluate(quote)
  self.assertNotIn(CODE,codes(result))
 def test_suspected_population_does_not_prove_established_disease(self):
  quote='Assay R detected endoscopic activity and remission with sensitivity 80%.'
  result,_=evaluate(quote,population='Patients with suspected Quartz syndrome were enrolled.')
  # The objective itself still mentions known patients: remove it for this control.
  sections=[{'index':0,'label':'METHODS','text':'Patients with suspected Quartz syndrome were assessed for clinical activity.'},
            {'index':1,'label':'RESULTS','text':quote}]
  result=check_selected_findings(Q,'Assessment of clinical activity',sections,[quote])
  self.assertNotIn(CODE,codes(result))
 def test_explicit_absence_of_known_disease_is_not_established_population(self):
  quote='Assay R detected endoscopic activity and remission with sensitivity 80%.'
  for population in ('Patients with no evidence of Quartz syndrome were assessed for clinical activity.',
                     'Patients with Quartz syndrome were excluded; clinical activity was assessed.',
                     'No QS patients were enrolled; clinical activity was assessed.'):
   with self.subTest(population=population):
    sections=[{'index':0,'label':'METHODS','text':population},
              {'index':1,'label':'RESULTS','text':quote}]
    result=check_selected_findings(Q,'Assessment of clinical activity',sections,[quote])
    self.assertNotIn(CODE,codes(result))
 def test_all_proof_spans_are_exact(self):
  quote='Assay R detected endoscopic activity and remission with sensitivity 80%.'
  result,sections=evaluate(quote);fields={'title':'Assay R for disease activity',**{f'abstract:{s["index"]}':s['text']for s in sections}}
  for item in result['blocks']:
   for proof in item['source_proofs']:self.assertEqual(fields[proof['field']][proof['char_start']:proof['char_end']],proof['text'])
   p=item['question_proof'];self.assertEqual(Q[p['char_start']:p['char_end']],p['text'])
 def test_h2d_comparator_veto_still_composes(self):
  result=check_source_scope('Does agent R compared with placebo reduce events?','Randomized study',
    [{'index':0,'label':'METHODS','text':'Participants were randomly assigned to agent R immediately or after deferral for one year.'}])
  self.assertIn('different_actual_control_than_requested_placebo',codes(result))
if __name__=='__main__':unittest.main()
