import unittest
from medlens.generation import source_scope as guard

class H4a2ExplanationGrammar(unittest.TestCase):
 def test_explanatory_action_forms_are_mechanism_advisory_only(self):
  for question in ('How does exposure A affect process B?',
                   'How does excess factor A promote matrix B?',
                   'How do factors A and B contribute to process C?',
                   'How does factor A alter pathway B?'):
   with self.subTest(question=question):
    hint=guard.question_task_kind(question)
    self.assertEqual(hint['kind'],'mechanism')
    self.assertEqual(hint['recognition_rule'],'how-agent-action-explanation')
    self.assertTrue(hint['advisory_only']);self.assertFalse(hint['positive_validation_performed'])
    for proof in hint['question_proofs']:
     self.assertEqual(question[proof['char_start']:proof['char_end']],proof['text'])
 def test_how_to_comparison_and_unknown_action_are_not_mechanism(self):
  for question in ('How does treatment compare with comparator?',
                   'How does treatment compare with comparator and alter endpoint B?',
                   'How does one alter cells for a procedure?',
                   'How does one perform the assay procedure?',
                   'How do you alter process B?',
                   'How does factor A relate to process B?'):
   with self.subTest(question=question):
    hint=guard.question_task_kind(question)
    self.assertEqual(hint['kind'],'unknown');self.assertEqual(hint['question_proofs'],[])
 def test_human_restriction_stays_in_full_question_and_existing_source_guard(self):
  question='How does factor A alter process B in adults?'
  hint=guard.question_task_kind(question)
  self.assertEqual(hint['kind'],'mechanism')
  check=guard.check_source_scope(question,'Mouse experiment',
    [{'index':0,'label':'METHODS','text':'Mice were exposed to factor A or vehicle.'},
     {'index':1,'label':'RESULTS','text':'Process B was reduced in exposed mice.'}])
  self.assertFalse(check['positive_validation_performed'])
  self.assertIn('adult_eligibility_not_explicit',{c['code']for c in check['cautions']})
 def test_compound_request_still_unknown(self):
  hint=guard.question_task_kind('How does factor A alter process B and what is the diagnostic accuracy of C?')
  self.assertEqual(hint['kind'],'unknown')
if __name__=='__main__':unittest.main()
