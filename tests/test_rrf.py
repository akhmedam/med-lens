"""Rank-only fusion math, deterministic ties, compatibility, and offline replay."""

import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from medlens.retrieval.rrf import FusionError, HybridIndex, reciprocal_rank_fusion
from medlens.evaluation.hybrid_evaluation import run_hybrid_evaluation
import test_dense


class FusionTests(unittest.TestCase):
    def test_manual_scores_one_based_ranks_and_missing_contribution(self):
        rows=reciprocal_rank_fusion({'bm25':['1','2'],'dense':['2','3']},['1','2','3'])
        self.assertEqual([r['pmid'] for r in rows],['2','1','3'])
        self.assertAlmostEqual(rows[0]['score'],1/62+1/61)
        self.assertEqual(rows[1]['source_ranks'],{'bm25':1,'dense':None})
        self.assertEqual(rows[1]['contributions']['dense'],0)
        self.assertAlmostEqual(rows[1]['score'],1/61)

    def test_ties_keep_candidate_order_independent_of_method_order(self):
        first=reciprocal_rank_fusion({'bm25':['1','2'],'dense':['2','1']},['2','1'])
        second=reciprocal_rank_fusion({'dense':['2','1'],'bm25':['1','2']},['2','1'])
        self.assertEqual([r['pmid'] for r in first],['2','1'])
        self.assertEqual(first,second)

    def test_empty_single_list_c_zero_and_top_k_prefix(self):
        self.assertEqual(reciprocal_rank_fusion({'bm25':[],'dense':[]},['1']),[])
        rows=reciprocal_rank_fusion({'bm25':[],'dense':['2','1']},['1','2'],c=0)
        self.assertEqual([r['score'] for r in rows],[1,.5])
        self.assertEqual(reciprocal_rank_fusion({'bm25':[],'dense':['2','1']},['1','2'],c=0,top_k=1),rows[:1])

    def test_duplicate_unknown_and_invalid_parameters_fail(self):
        for rankings,candidates in (({'a':['1','1']},['1']),({'a':['2']},['1']),({},['1','1'])):
            with self.assertRaises(FusionError):reciprocal_rank_fusion(rankings,candidates)
        for c in (-1,float('nan'),float('inf'),True):
            with self.assertRaises(FusionError):reciprocal_rank_fusion({},[],c=c)
        for k in (0,-1,True,1.5):
            with self.assertRaises(FusionError):reciprocal_rank_fusion({},[],top_k=k)


class HybridTests(unittest.TestCase):
    def fixture(self,root):
        return test_dense.DenseArtifactTests().fixture(root)

    def test_real_artifact_fusion_is_rank_only_and_has_source_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            index=HybridIndex(root/'parsed',root/'windows',root/'vectors')
            with patch('socket.socket',side_effect=AssertionError('offline')):
                first=index.run('anxiety',encoder,depth=3,top_k=2)
                second=index.run('anxiety',encoder,depth=3,top_k=2)
            self.assertEqual(first['run_id'],second['run_id'])
            self.assertEqual(first['results'][0]['pmid'],'101')
            hit=first['results'][0]
            self.assertIsNotNone(hit['best_window'])
            self.assertAlmostEqual(hit['score'],sum(hit['contributions'].values()))
            lexical=first['sources']['bm25']
            scaled=json.loads(json.dumps(lexical))
            for row in scaled['results']:row['score']=row['score']*1000+7
            with patch.object(index.bm25,'run',return_value=scaled):
                changed=index.run('anxiety',encoder,depth=3,top_k=2)
            self.assertEqual([h['pmid'] for h in changed['results']],[h['pmid'] for h in first['results']])
            self.assertEqual([h['score'] for h in changed['results']],[h['score'] for h in first['results']])
            self.assertNotEqual(first['run_id'],index.run('anxiety',encoder,depth=3,top_k=2,c=0)['run_id'])

    def test_source_identity_and_depth_errors_prevent_fusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            index=HybridIndex(root/'parsed',root/'windows',root/'vectors')
            for kwargs in ({'depth':0},{'depth':2,'top_k':3},{'c':-1}):
                with self.assertRaises(FusionError):index.run('anxiety',encoder,**kwargs)
            source=index.bm25.run('anxiety',3);source['dataset_id']='wrong'
            with patch.object(index.bm25,'run',return_value=source):
                with self.assertRaisesRegex(FusionError,'disagree'):index.run('anxiety',encoder,depth=3,top_k=2)

    def test_evaluator_four_methods_common_labels_and_unjudged_signal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            for qid in ('q1','q2'):
                shutil.copytree(root/'windows',root/'windows-root'/qid)
                shutil.copytree(root/'vectors',root/'vectors-root'/qid)
            labels=[json.loads(line) for line in (root/'qrels.jsonl').read_text().splitlines()]
            (root/'qrels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels if row['pmid']!='101'))
            with patch('socket.socket',side_effect=AssertionError('offline')):
                report=run_hybrid_evaluation(root/'queries.jsonl',root/'qrels.jsonl',root/'windows-root',root/'vectors-root',encoder,k=2,depth=3)
            self.assertEqual(set(report['macro']),{'pubmed','bm25','dense','rrf'})
            self.assertEqual(report['comparison_status'],'provisional-incomplete-judgments')
            self.assertTrue(report['unjudged_rrf_pairs'])
            self.assertEqual(report['per_query'][0]['rrf_run']['sources']['dense']['top_k'],3)

    def test_rrf_supplement_only_adds_unknown_pairs(self):
        from medlens.evaluation.annotation import build_packet
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            for qid in ('q1','q2'):
                shutil.copytree(root/'windows',root/'windows-root'/qid)
                shutil.copytree(root/'vectors',root/'vectors-root'/qid)
            labels=[json.loads(line) for line in (root/'qrels.jsonl').read_text().splitlines()]
            (root/'qrels.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels if row['pmid']!='101'))
            report=run_hybrid_evaluation(root/'queries.jsonl',root/'qrels.jsonl',root/'windows-root',root/'vectors-root',encoder,k=2,depth=3)
            (root/'report.json').write_text(json.dumps(report))
            (root/'plan.json').write_text(json.dumps({'queries':[{'query_id':qid,'relevance_intent':'Synthetic'} for qid in ('q1','q2')]}))
            build_packet(root/'queries.jsonl',root/'plan.json',root/'supplement',depth=2,
                dense_report=root/'report.json',qrels_path=root/'qrels.jsonl',supplement_method='rrf')
            pool=json.loads((root/'supplement/pool.json').read_bytes())
            self.assertTrue(pool['items'])
            self.assertTrue(all(item['pmid']=='101' for item in pool['items']))
            self.assertTrue(all('source_ranks' not in item and 'score' not in item for item in pool['items']))

    def test_cli_offline_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            script=Path(__file__).resolve().parents[1]/'scripts/hybrid_search.py'
            output=root/'run.json'
            args=[str(script),str(root/'parsed'),'--windows',str(root/'windows'),'--embeddings',str(root/'vectors'),
                '--query','anxiety','--top-k','2','--depth','3','--output',str(output)]
            def invoke():
                stdout,stderr=io.StringIO(),io.StringIO()
                with patch.object(sys,'argv',args),redirect_stdout(stdout),redirect_stderr(stderr),patch(
                    'medlens.retrieval.embeddings.BGEEncoder',return_value=encoder),patch('socket.socket',side_effect=AssertionError('offline')):
                    with self.assertRaises(SystemExit) as result:runpy.run_path(str(script),run_name='__main__')
                return result.exception.code,stdout.getvalue()
            code,body=invoke();self.assertEqual(code,0)
            self.assertEqual(json.loads(body),json.loads(output.read_bytes()))
            saved=output.read_bytes();self.assertEqual(invoke()[0],1);self.assertEqual(output.read_bytes(),saved)


if __name__=='__main__':unittest.main()
