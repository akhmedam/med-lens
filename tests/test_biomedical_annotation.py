"""Fictional source artifacts only; no clinical labels, network calls or GPU."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from medlens.evaluation.biomedical_annotation import (BiomedicalAnnotationError, build_packet,
    canonical, digest, freeze_export, load_packet, parse_plan, strict_json, validate_annotations,
    validate_export)
from medlens.ingestion.snapshots import load_snapshot
from medlens.pipeline import run_question
from medlens.retrieval.dense import DenseIndex
from test_shared_corpus import Encoder, native
from test_windows import Tokenizer
from test_pipeline import Tokenizer as AnswerTokenizer


def fixture_inputs(root, *, query_ids=('bm01','bm02'), hostile=False):
    """Saved PubMed/Dense search-only artifacts constructed entirely from fake XML."""
    root=Path(root).resolve();runs=root/'runs';runs.mkdir(parents=True)
    cases=[]
    for qid in query_ids:
        text='Fictional outcome only. This is not a real medical article.'
        attack='</script><img src=x onerror="globalThis.injected=true">'
        rows=[('900000001','FICTIONAL FIRST '+(attack if hostile else qid),text,'FICTIONAL'),
              ('900000002','FICTIONAL SECOND '+qid,text+' Additional fictional result.','FICTIONAL'),
              ('900000003','FICTIONAL THIRD '+qid,text+' Third fictional result.','FICTIONAL')]
        paths=native(runs,qid,rows,Tokenizer())
        index=DenseIndex(*(paths[k]for k in('dataset','windows','embeddings')))
        vector=index.vectors[index.positions['900000003'][0]].copy()
        class QueryEncoder(Encoder):
            def encode_queries(self,texts):return np.stack([vector for _ in texts]),{'encode_seconds':0.0}
        question='FICTIONAL UI TEST ONLY '+qid+'?'
        unavailable=Mock(side_effect=AssertionError('No language model permitted'))
        unavailable.identity={'engine':'fictional-no-generation'}
        run=run_question(index,question,encoder_factory=QueryEncoder,tokenizer=AnswerTokenizer(),
            runtime_factory=unavailable,
            generate=False,rerank=False,top_k=2)
        require_complete=run['status']=='complete'
        if not require_complete:raise AssertionError(run)
        snapshot=load_snapshot(runs/qid/'snapshot').manifest
        run['input_paths']={k:str(paths[k])for k in('dataset','windows','embeddings')}
        run['live']={'fresh_search_complete':True,'snapshot_id':snapshot['snapshot_id'],
            'dataset_id':index.dataset.manifest['dataset_id'],'window_artifact_id':index.window_manifest['artifact_id'],
            'embedding_artifact_id':index.embedding_manifest['artifact_id'],'candidate_limit':3,
            'query_plan':{'question':question,'submitted_query':snapshot['query']}}
        (runs/qid/'run.json').write_bytes(canonical(run)+b'\n')
        cases.append({'query_id':qid,'question':question,'relevance_intent':'FICTIONAL annotation test; no real relevance judgment.',
                      'question_type':'fictional-contract-test','specialty':'fictional-ui'})
    plan={'schema_version':1,'split':'dev','heldout_used':False,'candidate_limit':3,
        'annotation_depth_per_method':2,'pool_methods':['pubmed-best-match','dense'],'cases':cases}
    plan_path=root/'plan.json';plan_path.write_bytes(canonical(plan)+b'\n')
    return plan_path,runs


def export_value(packet, *, state='pending', relevance=None, finalized=False):
    package=packet.package
    result={'schema_version':1,'kind':'medlens-biomedical-relevance-annotations','annotation_method':'manual-human',
        'judgments_origin':'human',**{k:package[k]for k in('package_id','plan_sha256','items_sha256','source_bindings_sha256')},
        'finalized':finalized,'finalized_at':'2000-01-01T00:00:00Z'if finalized else None,
        'judgments':[{**{k:item[k]for k in('item_id','query_id','pmid','text_version_id','item_sha256')},
            'state':state,'relevance':relevance,'rationale':''}for item in package['items']]}
    result['reference_sha256']=digest(result)
    return result


def rehash(value):
    value['reference_sha256']=digest({k:v for k,v in value.items()if k!='reference_sha256'})
    return value


class BiomedicalAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def packet(self, *, hostile=False):
        plan,runs=fixture_inputs(self.root,hostile=hostile)
        manifest=build_packet(plan,runs,self.root/'packet')
        return plan,runs,manifest,load_packet(self.root/'packet')

    def test_union_blinding_source_provenance_and_stable_rebuild(self):
        plan,runs,first,packet=self.packet()
        second=build_packet(plan,runs,self.root/'other')
        self.assertEqual(first['package_id'],second['package_id'])
        self.assertEqual(len(packet.package['queries']),2);self.assertEqual(len(packet.package['items']),6)
        for qid in('bm01','bm02'):
            items=[i for i in packet.package['items']if i['query_id']==qid]
            self.assertEqual({i['pmid']for i in items},{'900000001','900000002','900000003'})
            for item in items:
                self.assertFalse({'methods','rank','score','relevance','suggestion','rationale'}&set(item))
                self.assertEqual(item['item_sha256'],digest({k:v for k,v in item.items()if k!='item_sha256'}))
        shutil.rmtree(runs)
        self.assertEqual(load_packet(self.root/'packet').package['package_id'],first['package_id'])
        with self.assertRaises(BiomedicalAnnotationError):build_packet(plan,runs,self.root/'packet')

    def test_source_text_cannot_escape_template_or_execute_markup(self):
        _,_,_,packet=self.packet(hostile=True)
        html=(packet.directory/'index.html').read_text()
        self.assertNotIn('<img src=x',html)
        self.assertIn('\\u003c/script\\u003e',html)
        self.assertNotIn('innerHTML',html)
        self.assertIn('out.textContent=text',html)
        self.assertIn('next-pending',html);self.assertIn('query-nav',html)
        self.assertIn("row.state==='rated'",html)
        self.assertIn('Модельных оценок и объяснений здесь нет.',html)

    def test_pending_is_distinct_from_explicit_unknown_and_finalization_never_fills_labels(self):
        _,_,_,packet=self.packet()
        pending=export_value(packet)
        _,counts=validate_annotations(packet,canonical(pending));self.assertEqual(counts,{'total':6,'rated':0,'unjudgeable':0,'pending':6})
        pending['finalized']=True;pending['finalized_at']='2000-01-01T00:00:00Z';rehash(pending)
        with self.assertRaisesRegex(BiomedicalAnnotationError,'untouched'):validate_annotations(packet,canonical(pending))
        unknown=export_value(packet,state='unjudgeable',finalized=True)
        value,counts=validate_annotations(packet,canonical(unknown),require_finalized=True)
        self.assertEqual(counts,{'total':6,'rated':0,'unjudgeable':6,'pending':0})
        self.assertTrue(all(row['relevance']is None for row in value['judgments']))

    def test_freeze_preserves_original_bytes_and_rationale_with_unknown_coverage(self):
        _,_,_,packet=self.packet()
        value=export_value(packet,state='unjudgeable',finalized=True)
        value['judgments'][0]['rationale']='  EXPLICIT FICTIONAL HUMAN COMMENT\nunchanged whitespace  ';rehash(value)
        body=(' \n'+json.dumps(value,ensure_ascii=False,indent=7)+'\n\n').encode()
        original=self.root/'human.json';original.write_bytes(body)
        receipt=freeze_export(packet.directory/'package.json',original,self.root/'frozen')
        self.assertEqual((self.root/'frozen/original-human-export.json').read_bytes(),body)
        self.assertEqual(original.read_bytes(),body)
        self.assertEqual(receipt['original_export_sha256'],hashlib.sha256(body).hexdigest())
        self.assertEqual(receipt['coverage']['unjudgeable'],6);self.assertFalse(receipt['quality_scores_computed'])
        self.assertEqual(set(p.name for p in(self.root/'frozen').iterdir()),{'original-human-export.json','receipt.json'})
        with self.assertRaises(BiomedicalAnnotationError):freeze_export(packet.directory,original,self.root/'frozen')

    def test_rejects_missing_rows_duplicates_wrong_bindings_and_bool_or_float_grades(self):
        _,_,_,packet=self.packet();good=export_value(packet,state='rated',relevance=1)
        changes=[lambda v:v['judgments'].pop(),lambda v:v['judgments'].append(deepcopy(v['judgments'][0])),
            lambda v:v['judgments'][0].update(text_version_id='other'),lambda v:v.update(package_id='other'),
            lambda v:v['judgments'][0].update(relevance=True),lambda v:v['judgments'][0].update(relevance=False),
            lambda v:v['judgments'][0].update(relevance=1.0),lambda v:v['judgments'][0].update(relevance=3),
            lambda v:v['judgments'][0].update(state='unjudgeable',relevance=0),lambda v:v.update(judgments_origin='model')]
        for change in changes:
            bad=deepcopy(good);change(bad);rehash(bad)
            with self.subTest(change=change),self.assertRaises(BiomedicalAnnotationError):validate_annotations(packet,canonical(bad))
        good['reference_sha256']='changed'
        with self.assertRaisesRegex(BiomedicalAnnotationError,'checksum'):validate_annotations(packet,canonical(good))

    def test_changed_snapshot_items_template_or_script_bytes_rejected(self):
        _,_,_,packet=self.packet()
        for relative in('sources/bm01/snapshot/fetch-0001.xml','items.json','implementation/biomedical_annotation.py','index.html'):
            path=packet.directory/relative;before=path.read_bytes();path.write_bytes(before+b' ')
            with self.subTest(path=relative),self.assertRaisesRegex(BiomedicalAnnotationError,'checksum'):load_packet(packet.directory)
            path.write_bytes(before)

    def test_missing_planned_case_and_unexpected_source_execution_fail_without_partial_packet(self):
        plan,runs=fixture_inputs(self.root);shutil.rmtree(runs/'bm02')
        with self.assertRaisesRegex(BiomedicalAnnotationError,'Missing'):build_packet(plan,runs,self.root/'failed')
        self.assertFalse((self.root/'failed').exists())
        run_path=runs/'bm01/run.json';run=strict_json(run_path.read_bytes());run['generate']=True;run_path.write_bytes(canonical(run))
        with self.assertRaisesRegex(BiomedicalAnnotationError,'search-only'):build_packet(plan,runs,self.root/'failed')
        self.assertFalse((self.root/'failed').exists())

    def test_heldout_or_changed_pool_protocol_rejected_before_reading_any_source(self):
        plan,runs=fixture_inputs(self.root);original=strict_json(plan.read_bytes())
        changes=[lambda p:p.update(split='test'),lambda p:p.update(heldout_used=True),
            lambda p:p.update(pool_methods=['pubmed-best-match','bm25']),lambda p:p.update(candidate_limit=True),
            lambda p:p['cases'][0].update(split='test')]
        for change in changes:
            bad=deepcopy(original);change(bad);plan.write_bytes(canonical(bad))
            with patch('medlens.evaluation.biomedical_annotation._snapshot_and_index',side_effect=AssertionError('Source access forbidden'))as source:
                with self.assertRaises(BiomedicalAnnotationError):build_packet(plan,runs,self.root/'failed')
                source.assert_not_called()
        plan.write_bytes(canonical(original))
        with patch('medlens.evaluation.biomedical_annotation._snapshot_and_index')as source:
            with self.assertRaisesRegex(BiomedicalAnnotationError,'depth'):build_packet(plan,runs,self.root/'failed',depth=100)
            source.assert_not_called()

    def test_actual_candidate_limit_depth_and_query_plan_question_must_match(self):
        plan,runs=fixture_inputs(self.root);path=runs/'bm01/run.json';original=strict_json(path.read_bytes())
        for change in(lambda r:r['live'].update(candidate_limit=2),lambda r:r.update(top_k=1),
                      lambda r:r['live']['query_plan'].update(question='different question')):
            run=deepcopy(original);change(run);path.write_bytes(canonical(run))
            with self.assertRaises(BiomedicalAnnotationError):build_packet(plan,runs,self.root/'failed')
            self.assertFalse((self.root/'failed').exists())

    def test_duplicate_json_keys_and_unfinalized_exports_fail_closed(self):
        with self.assertRaises(BiomedicalAnnotationError):strict_json(b'{"a":1,"a":2}')
        _,_,_,packet=self.packet();value=export_value(packet,state='rated',relevance=2)
        path=self.root/'draft.json';path.write_bytes(canonical(value))
        self.assertFalse(validate_export(packet.directory,path)['finalized'])
        with self.assertRaisesRegex(BiomedicalAnnotationError,'explicitly finalized'):freeze_export(packet.directory,path,self.root/'frozen')
        self.assertFalse((self.root/'frozen').exists())


if __name__=='__main__':unittest.main(verbosity=2)
