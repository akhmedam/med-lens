import unittest
from medlens.generation.source_scope import check_source_scope

Q='Does agent R compared with placebo reduce the event risk?'
CODE='different_actual_control_than_requested_placebo'
def check(*sections,title='Randomized study'):
 return check_source_scope(Q,title,[{'index':i,'label':label,'text':text}for i,(label,text)in enumerate(sections)])
def codes(result,bucket='blocks'):return {item['code']for item in result[bucket]}

class H2dActualComparator(unittest.TestCase):
 def test_background_placebo_trials_do_not_replace_actual_deferred_control(self):
  result=check(('BACKGROUND','Previous placebo-controlled trials showed benefit.'),
     ('METHODS','Participants were randomly assigned to receive agent R either immediately or after a deferral period of one year.'),
     ('CONCLUSIONS','The effect was larger than in placebo-controlled trials.'))
  self.assertIn(CODE,codes(result));self.assertFalse(result['positive_validation_performed'])
 def test_waitlist_usual_care_and_no_intervention_are_explicit_other_controls(self):
  for control in ('a waiting-list control','usual care','standard care','no intervention','deferred treatment'):
   result=check(('METHODS','Adults were randomized to agent R or '+control+'.'))
   with self.subTest(control=control):self.assertIn(CODE,codes(result))
 def test_missing_placebo_mention_alone_is_not_veto(self):
  self.assertNotIn(CODE,codes(check(('METHODS','Adults received agent R and were followed for one year.'))))
 def test_actual_placebo_allocation_is_kept_despite_delayed_outcome_text(self):
  result=check(('METHODS','Patients were randomized to agent R or matching placebo.'),
               ('RESULTS','Delayed symptom resolution was observed in both groups.'))
  self.assertNotIn(CODE,codes(result))
 def test_placebo_in_delayed_arm_is_not_nonplacebo_control(self):
  result=check(('METHODS','Adults were randomized to immediate agent R or delayed treatment with placebo during the waiting period.'))
  self.assertNotIn(CODE,codes(result))
 def test_true_placebo_analysis_prevents_blanket_review_veto(self):
  result=check(('METHODS','One study randomized adults to immediate agent R or a waiting list.'),
     ('RESULTS','The pooled comparison included four trials of agent R versus placebo.'),title='Systematic review and meta-analysis')
  self.assertNotIn(CODE,codes(result));self.assertIn('multiple_actual_control_analyses',codes(result,'cautions'))
 def test_negated_alternative_allocation_does_not_block_placebo_question(self):
  result=check(('METHODS','Patients were not randomized to immediate treatment or a waiting list.'))
  self.assertNotIn(CODE,codes(result))
 def test_all_proofs_and_question_offsets_match_original_bytes(self):
  text='Adults were randomly assigned to agent R immediately or after deferral for six months.'
  result=check(('METHODS',text))
  for row in result['blocks']:
   for proof in row['source_proofs']:
    self.assertEqual(text[proof['char_start']:proof['char_end']],proof['text'])
   proof=row['question_proof'];self.assertEqual(Q[proof['char_start']:proof['char_end']],proof['text'])
if __name__=='__main__':unittest.main()
