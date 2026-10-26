"""Independent cosine checks, article aggregation, and artifact-safe evaluation."""

import io
import json
import math
import runpy
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from medlens.evaluation.dense_evaluation import run_dense_evaluation
from medlens.retrieval.dense import DenseError, DenseIndex
from medlens.retrieval.embeddings import build_embeddings, load_embeddings
import test_embeddings


def unit(x):
    v=np.zeros(768,np.float32);v[0]=x;v[1]=math.sqrt(1-x*x);return v


def small_index():
    index=DenseIndex.__new__(DenseIndex)
    index.vectors=np.stack([unit(.2),unit(.8),unit(.7),unit(-.5)])
    index.positions={'1':[0,1],'2':[2],'3':[3]}
    index.articles={pmid:{'title':f'Paper {pmid}'} for pmid in index.positions}
    index.windows=[dict(window_id=str(i),kind='abstract',section_index=i,section_label=None,
        source_path='/synthetic',char_start=0,char_end=4,text='Text',provenance={},text_version_id='v1') for i in range(4)]
    return index


def query_vector():
    result=np.zeros(768,np.float32);result[0]=1;return result


class DenseTests(unittest.TestCase):
    def test_exact_cosine_distance_and_max_not_mean_article_score(self):
        index=small_index();hits=index.search(query_vector())
        self.assertEqual([h['pmid'] for h in hits],['1','2','3'])
        self.assertAlmostEqual(hits[0]['score'],.8,places=6)
        self.assertAlmostEqual(hits[0]['cosine_distance'],.2,places=6)
        self.assertEqual(hits[0]['best_window']['window_id'],'1')
        self.assertAlmostEqual(hits[2]['cosine_distance'],1.5,places=6)
        self.assertEqual(len({h['pmid'] for h in hits}),3)

    def test_ties_keep_candidate_and_window_order(self):
        index=small_index();index.vectors[:3]=unit(.8)
        hits=index.search(query_vector(),2)
        self.assertEqual([h['pmid'] for h in hits],['1','2'])
        self.assertEqual(hits[0]['best_window']['window_id'],'0')

    def test_brute_force_reference_and_prefix_consistency(self):
        index=small_index()
        for x in (-1,-.4,.3,1):
            q=unit(x)
            reference=[]
            for order,(pmid,positions) in enumerate(index.positions.items()):
                value=max(float(np.dot(index.vectors[pos],q)) for pos in positions)
                reference.append((value,order,pmid))
            expected=[row[2] for row in sorted(reference,key=lambda row:(-row[0],row[1]))]
            hits=index.search(q)
            self.assertEqual([h['pmid'] for h in hits],expected)
            self.assertEqual(index.search(q,1),hits[:1])

    def test_empty_vectors_and_missing_articles_do_not_fabricate_scores(self):
        index=small_index();index.positions={'1':[]};index.vectors=np.empty((0,768),np.float32)
        self.assertEqual(index.search(query_vector()),[])

    def test_invalid_query_and_top_k(self):
        index=small_index()
        for q in (np.zeros(768),np.ones(3),np.full(768,np.nan),np.ones((1,768))):
            with self.assertRaises(DenseError):index.search(q)
        for k in (0,-1,True,1.5):
            with self.assertRaises(DenseError):index.search(query_vector(),k)


class DenseArtifactTests(unittest.TestCase):
    def fixture(self,root):
        encoder=test_embeddings.EmbeddingArtifactTests().fixture(root)
        build_embeddings(root/'windows',encoder,root/'vectors')
        encoder.encode_queries=lambda texts:(np.tile(np.eye(768,dtype=np.float32)[2],(len(texts),1)),{'encode_seconds':0.0})
        return encoder

    def test_verified_artifacts_run_identity_and_encoder_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            with patch('socket.socket',side_effect=AssertionError('offline')):
                index=DenseIndex(root/'parsed',root/'windows',root/'vectors')
                a=index.run('anxiety',encoder);b=index.run('anxiety',encoder)
            self.assertEqual(a['run_id'],b['run_id'])
            self.assertEqual(a['results'][0]['pmid'],'101')
            self.assertEqual(a['searchable_article_count'],3)
            self.assertNotEqual(a['run_id'],index.run('depression',encoder)['run_id'])
            encoder.info={**encoder.info,'device':'cpu','batch_size':1}
            index.verify_encoder(encoder)
            encoder.info['pooling']='wrong'
            with self.assertRaisesRegex(DenseError,'incompatible'):index.verify_encoder(encoder)

    def test_rejects_cross_artifact_mapping_and_candidate_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            m,rows,v=load_embeddings(root/'vectors')
            with patch('medlens.retrieval.dense.load_embeddings',return_value=(m,rows[::-1],v)):
                with self.assertRaisesRegex(DenseError,'mapping'):DenseIndex(root/'parsed',root/'windows',root/'vectors')
            from medlens.retrieval.windows import load_windows
            w,rows=load_windows(root/'windows');w['dataset_id']='wrong'
            with patch('medlens.retrieval.dense.load_windows',return_value=(w,rows)):
                with self.assertRaisesRegex(DenseError,'disagree'):DenseIndex(root/'parsed',root/'windows',root/'vectors')

    def test_evaluation_exposes_unjudged_dense_hits_and_keeps_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            for qid in ('q1','q2'):
                shutil.copytree(root/'windows',root/'windows-root'/qid)
                shutil.copytree(root/'vectors',root/'vectors-root'/qid)
            qrels=root/'qrels.jsonl'
            labels=[json.loads(line) for line in qrels.read_text().splitlines()]
            qrels.write_text(''.join(json.dumps(row)+'\n' for row in labels if row['pmid']!='101'))
            with patch('socket.socket',side_effect=AssertionError('offline')):
                report=run_dense_evaluation(root/'queries.jsonl',qrels,root/'windows-root',root/'vectors-root',encoder)
            self.assertEqual(set(report['macro']),{'pubmed','bm25','dense'})
            self.assertEqual(report['comparison_status'],'provisional-incomplete-judgments')
            self.assertEqual(len(report['unjudged_dense_pairs']),2)
            self.assertFalse(report['per_query'][0]['methods']['dense']['ranking'][0]['judged'])

    def test_cli_offline_output_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            script=Path(__file__).resolve().parents[1]/'scripts/dense_search.py'
            output=root/'search.json'
            args=[str(script),str(root/'parsed'),'--windows',str(root/'windows'),'--embeddings',str(root/'vectors'),
                '--query','anxiety','--output',str(output)]
            def invoke():
                stdout,stderr=io.StringIO(),io.StringIO()
                with patch.object(sys,'argv',args),redirect_stdout(stdout),redirect_stderr(stderr),patch(
                    'medlens.retrieval.embeddings.BGEEncoder',return_value=encoder),patch('socket.socket',side_effect=AssertionError('offline')):
                    with self.assertRaises(SystemExit) as result:runpy.run_path(str(script),run_name='__main__')
                return result.exception.code,stdout.getvalue()
            code,body=invoke();self.assertEqual(code,0)
            self.assertEqual(json.loads(body),json.loads(output.read_bytes()))
            saved=output.read_bytes();self.assertEqual(invoke()[0],1);self.assertEqual(output.read_bytes(),saved)


class SupplementTests(unittest.TestCase):
    def fixture(self,root):
        from medlens.evaluation.annotation import build_packet
        encoder=DenseArtifactTests().fixture(root)
        for qid in ('q1','q2'):
            shutil.copytree(root/'windows',root/'windows-root'/qid)
            shutil.copytree(root/'vectors',root/'vectors-root'/qid)
        qrels=root/'qrels.jsonl'
        rows=[json.loads(line) for line in qrels.read_text().splitlines()]
        qrels.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['pmid']!='101'))
        report=run_dense_evaluation(root/'queries.jsonl',qrels,root/'windows-root',root/'vectors-root',encoder)
        (root/'report.json').write_text(json.dumps(report))
        (root/'plan.json').write_text(json.dumps({'queries':[{'query_id':qid,'relevance_intent':'Synthetic'} for qid in ('q1','q2')]}))
        build_packet(root/'queries.jsonl',root/'plan.json',root/'supplement',
            dense_report=root/'report.json',qrels_path=qrels)
        return json.loads((root/'supplement/pool.json').read_bytes())

    def test_supplement_contains_only_missing_pairs_and_merges_without_relabeling(self):
        from medlens.evaluation.annotation import export_qrels,merge_supplement
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pool=self.fixture(root)
            self.assertEqual([(row['query_id'],row['pmid']) for row in pool['items']],[('q1','101'),('q2','101')])
            before=(root/'qrels.jsonl').read_bytes()
            labels={'schema_version':1,'pool_id':pool['pool_id'],'judgments':[
                {'query_id':r['query_id'],'pmid':r['pmid'],'relevance':2} for r in pool['items']]}
            (root/'labels.json').write_text(json.dumps(labels))
            export_qrels(root/'supplement',root/'labels.json',root/'extra.jsonl')
            result=merge_supplement(root/'queries.jsonl',root/'qrels.jsonl',root/'extra.jsonl',root/'supplement',root/'merged.jsonl')
            self.assertEqual(result['judgment_count'],7)
            self.assertEqual((root/'qrels.jsonl').read_bytes(),before)
            merged={(r['query_id'],r['pmid']):r['relevance'] for r in map(json.loads,(root/'merged.jsonl').read_text().splitlines())}
            for old in map(json.loads,before.splitlines()):
                self.assertEqual(merged[(old['query_id'],old['pmid'])],old['relevance'])
            with self.assertRaises(FileExistsError):
                merge_supplement(root/'queries.jsonl',root/'qrels.jsonl',root/'extra.jsonl',root/'supplement',root/'merged.jsonl')

    def test_partial_supplement_and_changed_base_cannot_be_merged(self):
        from medlens.evaluation.annotation import export_qrels,merge_supplement
        from medlens.evaluation.formats import EvaluationError
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);pool=self.fixture(root)
            labels={'schema_version':1,'pool_id':pool['pool_id'],'judgments':[
                {'query_id':r['query_id'],'pmid':r['pmid'],'relevance':None} for r in pool['items']]}
            (root/'labels.json').write_text(json.dumps(labels))
            export_qrels(root/'supplement',root/'labels.json',root/'partial.jsonl',allow_partial=True)
            with self.assertRaises(EvaluationError):
                merge_supplement(root/'queries.jsonl',root/'qrels.jsonl',root/'partial.jsonl',root/'supplement',root/'bad.jsonl')
            for row in labels['judgments']:row['relevance']=1
            (root/'labels.json').write_text(json.dumps(labels))
            export_qrels(root/'supplement',root/'labels.json',root/'full.jsonl')
            with (root/'qrels.jsonl').open('ab') as stream:stream.write(b'\n')
            with self.assertRaises(EvaluationError):
                merge_supplement(root/'queries.jsonl',root/'qrels.jsonl',root/'full.jsonl',root/'supplement',root/'bad.jsonl')
            self.assertFalse((root/'bad.jsonl').exists())


if __name__=='__main__':unittest.main()
