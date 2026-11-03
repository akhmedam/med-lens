"""Frozen evaluation must reject source/ranking drift and preserve base judgments."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medlens.generation.context import digest
from medlens.reranking.medcpt import MODEL_ID,REVISION

spec=importlib.util.spec_from_file_location('medcpt_evaluation_cli',Path(__file__).resolve().parents[1]/'scripts/evaluate_medcpt.py')
# The CLI's historical replay helper lives beside it.
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class FrozenComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.profile={'schema_version':1,'split':'pilot','evaluation_version':'fixture',
            'primary_methods':['pubmed','dense','dense_medcpt','medcpt'],
            'unresolved_pairs':[{'query_id':'p01','pmid':'2'}],
            'conservative_direct_pairs':[{'query_id':'p01','pmid':'3'}],
            'annotation':{'pair_count':3,'judged_count':2,'unjudged_count':1}}
        query={'query_id':'p01','text':'Question','split':'pilot','group_id':'g','dataset':'dataset'}
        query_bytes=(json.dumps(query)+'\n').encode();(self.root/'queries.jsonl').write_bytes(query_bytes)
        old=[{'query_id':'p01','pmid':'1','relevance':2},{'query_id':'p01','pmid':'2','relevance':None}]
        self.qrels=old+[{'query_id':'p01','pmid':'3','relevance':2}]
        old_body=self.jsonl(old);(self.root/'base-qrels.jsonl').write_bytes(old_body)
        base={'queries':{'path':'queries.jsonl','sha256':hashlib.sha256(query_bytes).hexdigest()},
            'qrels':{'path':'base-qrels.jsonl','sha256':hashlib.sha256(old_body).hexdigest()}}
        self.entry('base_profile',json.dumps(base).encode())
        self.articles={p:{'pmid':p,'title':'Title '+p,'abstract':{'sections':[]},
            'text_version':{'text_version_id':'v'+p},'provenance':{}} for p in ('1','2','3')}
        self.windows=[{'window_id':'w'+p,'pmid':p,'text_version_id':'v'+p,'kind':'abstract',
            'section_index':0,'section_label':'RESULTS','source_path':'/source','char_start':0,'char_end':2,
            'text':'a'+p,'provenance':{}} for p in self.articles]
        self.ranker=SimpleNamespace(articles=self.articles,dataset=SimpleNamespace(manifest={'dataset_id':'d'}),
            window_manifest={'artifact_id':'w'},eligible_windows=lambda order:[w for w in self.windows if w['pmid'] in order])
        self.item={'query_id':'p01','pmid':'3','question':'Question','dataset_id':'d',
            'text_version_id':'v3','title':'Title 3','abstract':self.articles['3']['abstract'],'provenance':{}}
        self.review={'query_id':'p01','pmid':'3','relevance':2,'confidence':'medium','rationale':'Direct fixture evidence.',
            'dataset_id':'d','text_version_id':'v3','item_sha256':digest(self.item)}
        self.entry('qrels',self.jsonl(self.qrels));self.entry('supplement_items',self.jsonl([self.item]))
        self.entry('supplement_reviews',self.jsonl([self.review]))
        self.provenance={'base_qrels_sha256':base['qrels']['sha256'],'total_pair_count':3,
            'judged_count':2,'unjudged_count':1,'added_pair_count':1,'preserved_base_pair_count':2}
        self.bind_provenance()
        historical_methods={m:{'ranking':['1','2']} for m in ('dense','bm25','rrf','dense_qwen','rrf_qwen')}
        historical_methods['pubmed']={'ranking':['1','2','3']}
        self.historical={'per_query':[{'query_id':'p01','dataset_id':'d','methods':historical_methods}]}
        def result(order,run_id):
            windows=[w for w in self.windows if w['pmid'] in order]
            scores=[{'window_id':w['window_id'],'pmid':w['pmid'],'text_version_id':w['text_version_id'],
                     'input_tokens':7,'score':float(w['pmid'])} for w in windows]
            ordered=sorted(windows,key=lambda w:-float(w['pmid']))
            hits=[{'rank':i,'pmid':w['pmid'],'score':float(w['pmid']),'text_version_id':w['text_version_id'],
                'best_window':{k:w[k] for k in ('window_id','kind','section_index','section_label','source_path',
                    'char_start','char_end','text','provenance')}} for i,w in enumerate(ordered,1)]
            return {'run_id':run_id,'eligible_pmids':order,'dataset_id':'d','window_artifact_id':'w','query':'Question',
                'runtime':{'model_id':MODEL_ID,'revision':REVISION},'window_scores':scores,
                'scoring':{'pair_count':len(windows)},'results':hits}
        self.cached={'status':'complete','heldout_used':False,'relevance_metrics_evaluated':False,
            'queries_sha256':hashlib.sha256(query_bytes).hexdigest(),'device':'cpu','batch_size':4,
            'dense_depth':10,'top_k':10,'code_sha256':{},'per_query':[{'query_id':'p01','query':'Question',
                'dense':{'run_id':'dense-id','query':'Question','dataset_id':'d','window_artifact_id':'w',
                         'results':[{'pmid':'1'},{'pmid':'2'}]},
                'methods':{'dense_medcpt':result(['1','2'],'m1'),'medcpt':result(['1','2','3'],'m2')}}],
            'input_run_ids':{'dense':['dense-id'],'medcpt':[{'dense_medcpt':'m1','medcpt':'m2'}]}}
        identity={k:self.cached[k] for k in ('queries_sha256','device','batch_size','dense_depth','top_k','code_sha256','input_run_ids')}
        self.cached['run_id']=digest(identity)
        self.entry('rankings',json.dumps(self.cached).encode())

    def jsonl(self,rows):return ''.join(json.dumps(row)+'\n' for row in rows).encode()
    def entry(self,key,body):
        (self.root/key).write_bytes(body)
        self.profile[key]={'path':key,'sha256':hashlib.sha256(body).hexdigest()}
    def bind_provenance(self):
        for key,name in (('items_sha256','supplement_items'),('reviews_sha256','supplement_reviews'),('qrels_sha256','qrels')):
            self.provenance[key]=self.profile[name]['sha256']
        self.entry('provenance',json.dumps(self.provenance).encode())
    def load(self):
        path=self.root/'profile.json';path.write_text(json.dumps(self.profile))
        with patch.object(cli,'rescore',return_value=(self.historical,None,None)), \
             patch.object(cli,'MedCPTRanker',return_value=self.ranker):
            return cli.load_comparison(path,root=self.root)

    def test_valid_extension_keeps_old_labels_and_separates_sensitivity(self):
        profile,rows,qrels,conservative,sources,provenance=self.load()
        self.assertIsNone(qrels['p01']['2'])
        self.assertEqual(qrels['p01']['1'],2)
        self.assertEqual(qrels['p01']['3'],2)
        self.assertEqual(conservative['p01']['3'],1)
        self.assertEqual(rows[0]['rankings']['medcpt'],['3','2','1'])
        self.assertEqual(sources[0]['new_judgment_count'],1)

    def test_changed_base_label_rejected_even_with_updated_qrels_hash(self):
        changed=copy.deepcopy(self.qrels);changed[0]['relevance']=0
        self.entry('qrels',self.jsonl(changed));self.bind_provenance()
        with self.assertRaisesRegex(ValueError,'unchanged'):self.load()

    def test_corrupted_bytes_rejected_before_evaluation(self):
        (self.root/'rankings').write_text('changed')
        with self.assertRaisesRegex(ValueError,'checksum'):self.load()

    def test_inconsistent_cached_ranking_cannot_be_hidden_by_updated_hash(self):
        self.cached['per_query'][0]['methods']['medcpt']['results'].reverse()
        self.entry('rankings',json.dumps(self.cached).encode())
        with self.assertRaisesRegex(ValueError,'ranking does not replay'):self.load()

    def test_missing_best_window_fields_rejected(self):
        self.cached['per_query'][0]['methods']['medcpt']['results'][0]['best_window'].pop('window_id')
        self.entry('rankings',json.dumps(self.cached).encode())
        with self.assertRaisesRegex(ValueError,'ranking does not replay'):self.load()

    def test_source_text_and_label_mismatches_rejected(self):
        original=copy.deepcopy(self.review)
        self.review['relevance']=0
        self.entry('supplement_reviews',self.jsonl([self.review]));self.bind_provenance()
        with self.assertRaisesRegex(ValueError,'label, source or rationale'):self.load()
        self.review=original;self.entry('supplement_reviews',self.jsonl([self.review]));self.bind_provenance()
        self.articles['3']['title']='Different source title'
        with self.assertRaisesRegex(ValueError,'evidence differs'):self.load()

    def test_float_review_grade_is_not_an_integer_judgment(self):
        self.review['relevance']=2.0
        self.entry('supplement_reviews',self.jsonl([self.review]));self.bind_provenance()
        with self.assertRaisesRegex(ValueError,'label, source or rationale'):self.load()

    def test_unknowns_and_sensitivity_must_match_actual_judgments(self):
        self.profile['unresolved_pairs']=[]
        with self.assertRaisesRegex(ValueError,'Unresolved-pair'):self.load()
        self.profile['unresolved_pairs']=[{'query_id':'p01','pmid':'2'}]
        self.profile['conservative_direct_pairs']=[]
        with self.assertRaisesRegex(ValueError,'all new medium-confidence'):self.load()

    def test_misreported_annotation_counts_rejected(self):
        self.profile['annotation']['judged_count']=3
        with self.assertRaisesRegex(ValueError,'counts differ'):self.load()
