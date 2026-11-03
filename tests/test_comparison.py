"""Common-pool comparisons and exact uncertainty bounds must not invent judgments."""

import copy
import math
import unittest

from medlens.evaluation.comparison import compare,missing_label_sensitivity


class ComparisonTests(unittest.TestCase):
    def fixture(self):
        rows=[{'query_id':'q1','candidates':['1','2','3'],'rankings':{'a':['1','2'],'b':['2','1']}},
              {'query_id':'q2','candidates':['4','5'],'rankings':{'a':['4','5'],'b':['4','5']}}]
        qrels={'q1':{'1':2,'2':None,'3':0},'q2':{'4':2,'5':0}}
        return rows,qrels

    def test_common_subset_requires_judged_topk_for_every_primary_method(self):
        rows,qrels=self.fixture();result=compare(rows,qrels,['a','b'],[('b','a')])
        self.assertEqual(result['comparison_status'],'provisional-incomplete-judgments')
        self.assertEqual(result['common_topk_complete']['query_ids'],['q2'])
        self.assertEqual(result['common_topk_complete']['macro']['a']['ndcg@10']['mean'],1)
        delta=result['paired_deltas']['b-minus-a']['ndcg@10']
        self.assertAlmostEqual(delta['mean'],(1/math.log2(3)-1)/2)
        self.assertEqual((delta['wins'],delta['ties'],delta['losses']),(0,1,1))

    def test_missing_label_changes_idcg_and_exact_bounds_without_mutating_qrels(self):
        rows,qrels=self.fixture();before=copy.deepcopy(qrels)
        result=missing_label_sensitivity(rows,qrels,[{'query_id':'q1','pmid':'2'}],['a','b'],[('b','a')])
        self.assertEqual(result['scenario_count'],3)
        self.assertAlmostEqual(result['macro_bounds']['b']['ndcg@10']['min'],(1/math.log2(3)+1)/2)
        self.assertEqual(result['macro_bounds']['b']['ndcg@10']['max'],1)
        self.assertEqual(result['paired_delta_bounds']['b-minus-a']['ndcg@10']['max'],0)
        self.assertEqual(qrels,before)

    def test_two_unknowns_enumerate_all_nine_completions(self):
        rows,qrels=self.fixture();qrels['q2']['5']=None
        result=missing_label_sensitivity(rows,qrels,[{'query_id':'q1','pmid':'2'},
            {'query_id':'q2','pmid':'5'}],['a','b'])
        self.assertEqual(result['scenario_count'],9)
        self.assertEqual(result['macro_bounds']['a']['ndcg@10']['defined_scenarios'],9)

    def test_unpooled_candidates_are_not_silently_added_to_sensitivity(self):
        rows,qrels=self.fixture();qrels['q1'].pop('3')
        result=missing_label_sensitivity(rows,qrels,[{'query_id':'q1','pmid':'2'}],['a','b'])
        self.assertEqual(result['scenario_count'],3)
        self.assertNotIn('3',qrels['q1'])

    def test_undefined_results_remain_none(self):
        rows=[{'query_id':'q','candidates':['1'],'rankings':{'a':['1'],'b':[]}}]
        qrels={'q':{'1':None}}
        result=compare(rows,qrels,['a','b'])
        self.assertIsNone(result['macro']['a']['ndcg@10']['mean'])
        self.assertIsNone(result['common_topk_complete']['macro']['a']['ndcg@10']['mean'])
        bounds=missing_label_sensitivity(rows,qrels,[{'query_id':'q','pmid':'1'}],['a','b'])
        self.assertEqual(bounds['macro_bounds']['a']['ndcg@10']['defined_scenarios'],2)
        self.assertEqual(bounds['macro_bounds']['a']['ndcg@10']['min'],1)

    def test_invalid_query_membership_and_methods_fail(self):
        rows,qrels=self.fixture()
        for bad_rows,bad_qrels,primary in ((rows,{'q1':qrels['q1']},['a']),
            (rows+[rows[0]],qrels,['a']),([],{},['a']),(rows,qrels,['missing'])):
            with self.assertRaises(ValueError):compare(bad_rows,bad_qrels,primary)
        rows[1]['rankings'].pop('b')
        with self.assertRaises(ValueError):compare(rows,qrels,['a'])

    def test_invalid_or_excessive_sensitivity_scope_fails(self):
        rows,qrels=self.fixture()
        for scope in ([{'query_id':'q1','pmid':'1'}],[{'query_id':'q1','pmid':'99'}],
            [{'query_id':'q1','pmid':'2'}]*2):
            with self.assertRaises(ValueError):missing_label_sensitivity(rows,qrels,scope,['a','b'])
        with self.assertRaises(ValueError):missing_label_sensitivity(rows,qrels,
            [{'query_id':'q1','pmid':'2'}],['a','b'],max_scenarios=2)
