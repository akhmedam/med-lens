import unittest
from medlens.generation import source_findings as c
from test_source_findings import make,plan_data
from test_answers import FakeRuntime

class H2bContentSections(unittest.TestCase):
 def test_nonstandard_content_heading_can_supply_observed_result(self):
  ctx,arts=make([[('METHODS','Samples from cell cultures were assigned to agent R or vehicle.'),
                 ('MATRIX RESPONSES','Agent R increased matrix synthesis in the cultured cells.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertEqual([idx['catalog'][qid]['quote']for qid in plans[0]['finding_quote_ids']],
                   ['Agent R increased matrix synthesis in the cultured cells.'])
 def test_explicit_background_does_not_become_new_results(self):
  ctx,arts=make([[('METHODS','Samples were assigned to agent R or vehicle.'),
                 ('BACKGROUND','Matrix synthesis is involved in tissue remodeling.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertFalse(plans[0]['finding_quote_ids'])
 def test_explicit_background_category_is_excluded_under_unknown_label(self):
  ctx,arts=make([[('METHODS','Samples were assigned to agent R or vehicle.'),
                 ('PATHOGENESIS','Matrix synthesis contributes to tissue remodeling.')]])
  arts['1']['abstract']['sections'][1]['nlm_category']='BACKGROUND'
  idx,plans,checks=plan_data(ctx,arts)
  self.assertFalse(plans[0]['finding_quote_ids'])
 def test_composite_background_heading_is_not_result_content(self):
  ctx,arts=make([[('METHODS','Samples were assigned to agent R or vehicle.'),
                 ('BACKGROUND AND STUDY AIMS','Matrix synthesis contributes to tissue remodeling.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertFalse(plans[0]['finding_quote_ids'])
 def test_nonstandard_procedure_only_content_is_still_excluded(self):
  ctx,arts=make([[('METHODS','Samples were assigned to agent R or vehicle.'),
                 ('MATRIX RESPONSES','Matrix proteins were measured in cell culture samples.')]])
  idx,plans,checks=plan_data(ctx,arts)
  self.assertFalse(plans[0]['finding_quote_ids'])
 def test_service_sections_are_not_admitted(self):
  ctx,arts=make([[('METHODS','Samples were assigned to agent R or vehicle.'),
                 ('FUNDING','The study was supported by the laboratory.')]])
  self.assertNotIn('FUNDING',[source['section']for source in ctx['sources']])
if __name__=='__main__':unittest.main()
