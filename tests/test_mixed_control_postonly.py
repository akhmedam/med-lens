import json,unittest
from medlens.generation import source_findings as selector
from test_source_findings import make,plan_data,select
from test_answers import FakeRuntime

Q='Does agent R compared with placebo reduce events in adults?'
MIX='Adults were assigned to agent R or a control intervention, which constituted either usual care plus placebo or usual care only.'
CODE='selected_pooled_mixed_controls_instead_of_requested_placebo'

def evaluate(sections,question=Q):
 context,articles=make([sections]);indexed,plans,checks=plan_data(context,articles,question)
 raw=select(scope=plans[0]['scope_quote_ids'][0],findings=plans[0]['finding_quote_ids'][:2])
 value,draft,body,relations=selector.materialize_selection(json.dumps(raw),context,indexed,plans,checks)
 return value,draft,relations,checks,raw

class MixedControlPostselection(unittest.TestCase):
 def test_actual_mixed_controls_demote_pooled_result_with_exact_proof_and_no_rewrite(self):
  for result in ('Agent R reduced events versus pooled control (RR 0.74).',
                 'Compared with placebo or usual care only, agent R reduced events (RR 0.74).'):
   with self.subTest(result=result):
    value,draft,relations,checks,raw=evaluate([('METHODS',MIX),('RESULTS',result)])
    self.assertEqual(value,raw);self.assertEqual(relations[0]['raw_relation'],'direct')
    self.assertEqual(relations[0]['effective_relation'],'related')
    block=next(b for b in relations[0]['selected_scope_checks']['blocks']if b['code']==CODE)
    for proof in block['source_proofs']:
     text=checks['1']['abstract_sections'][int(proof['field'].split(':')[1])]['text']
     self.assertEqual(text[proof['char_start']:proof['char_end']],proof['text'])
    self.assertTrue(draft['claims'][0]['text'].endswith(result))
    self.assertFalse(relations[0]['selected_scope_checks']['positive_validation_performed'])
 def test_background_mixed_control_mentions_do_not_veto_actual_placebo_trial(self):
  for label in ('BACKGROUND','METHODS'):
   with self.subTest(label=label):
    prior='Earlier studies enrolled adults and controls consisted of placebo or usual care only.'
    setup='Adults were randomized to agent R or placebo.'
    sections=([('METHODS',setup+' '+prior)]if label=='METHODS'else[(label,prior),('METHODS',setup)])
    _,_,relations,_,_=evaluate(sections+[('RESULTS','Events were reduced with agent R.')])
    self.assertEqual(relations[0]['effective_relation'],'direct')
 def test_selected_distinct_placebo_comparison_in_mixed_article_is_retained(self):
  for result in ('Compared with placebo, agent R reduced events (RR 0.74).',
                 'In the placebo-controlled subgroup, agent R reduced events (RR 0.74).'):
   with self.subTest(result=result):
    _,_,relations,_,_=evaluate([('METHODS',MIX),('RESULTS',result)])
    self.assertEqual(relations[0]['effective_relation'],'direct')
 def test_missing_mixed_proof_or_nonplacebo_request_is_not_new_veto(self):
  for setup,question in [('Adults were assigned to agent R or an unspecified control.',Q),
      (MIX,'Does agent R compared with usual care reduce events in adults?')]:
   with self.subTest(question=question):
    _,_,relations,_,_=evaluate([('METHODS',setup),('RESULTS','Agent R reduced events.')],question)
    self.assertNotIn(CODE,{b['code']for b in relations[0]['selected_scope_checks']['blocks']})
 def test_bad_mixed_card_does_not_cancel_a_separate_placebo_card(self):
  context,articles=make([[('METHODS',MIX),('RESULTS','Agent R reduced events versus pooled control.')],
   [('METHODS','Adults were randomized to agent R or placebo.'),('RESULTS','Compared with placebo, agent R reduced events.')]])
  indexed,plans,checks=plan_data(context,articles,Q)
  raw={'articles':[select(article_id=p['article_id'],scope=p['scope_quote_ids'][0],findings=p['finding_quote_ids'])['articles'][0]for p in plans],'limitations':[]}
  out=selector.generate_source_findings_answer(Q,context,FakeRuntime([raw]),source_articles=articles)
  self.assertEqual([c['source_article_id']for c in out['claims']],['2'])
  self.assertEqual([r['effective_relation']for r in out['source_relations']],['related','direct'])
  self.assertEqual(out['selection'],raw)
if __name__=='__main__':unittest.main()
