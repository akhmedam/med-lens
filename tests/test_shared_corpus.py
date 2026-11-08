"""Shared collections retain lineage and never expose stale or partial revisions."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock,patch
from xml.sax.saxutils import escape

import numpy as np

from medlens.corpus import CorpusError,build_collection,inspect_collection,load_pooled_dataset,select_records
from medlens.ingestion.snapshots import download_snapshot
from medlens.parsing.dataset import build_dataset,load_dataset
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import build_embeddings
from medlens.retrieval.windows import build_windows,WindowConfig
from pubmed_helpers import make_client,search_json
from test_windows import Tokenizer

INFO={'fixture':'fixed-word-tokenizer'}


class Encoder:
    def __init__(self,policy='one'):
        self.info={'fixture':policy,'device':'mps','batch_size':16};self.tokenizer_info=INFO;self.calls=[]
    def encode(self,rows):
        self.calls.append(rows);vectors=[]
        for ids in rows:
            v=np.zeros(768,np.float32);values=np.frombuffer(hashlib.sha256(json.dumps(ids).encode()).digest(),dtype=np.uint8).astype(np.float32)+1
            v[:32]=values;v/=np.linalg.norm(v);vectors.append(v)
        return np.stack(vectors) if vectors else np.empty((0,768),np.float32),{'encode_seconds':0.0}


def native(root,name,records,tokenizer,stamp='2026-11-01T00:00:00+00:00'):
    folder=root/name;folder.mkdir()
    pmids=[r[0] for r in records]
    xml=''.join('<PubmedArticle><MedlineCitation><PMID>'+p+'</PMID><Article><ArticleTitle>'+escape(title)+
        '</ArticleTitle><Journal><Title>'+escape(journal)+'</Title></Journal><Abstract><AbstractText>'+escape(text)+
        '</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>' for p,title,text,journal in records)
    client,_,_=make_client([search_json(pmids,len(pmids)),('<PubmedArticleSet>'+xml+'</PubmedArticleSet>').encode()])
    with patch('medlens.ingestion.pubmed.datetime') as clock:
        clock.now.return_value.isoformat.return_value=stamp
        download_snapshot(client,'synthetic shared collection',folder/'snapshot',limit=max(1,len(pmids)))
    build_dataset(folder/'snapshot',folder/'dataset');build_windows(folder/'dataset',tokenizer,INFO,folder/'windows')
    build_embeddings(folder/'windows',Encoder(),folder/'embeddings')
    return {'query_id':name,'dataset':folder/'dataset','windows':folder/'windows','embeddings':folder/'embeddings'}


class SharedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.tokenizer=Tokenizer();self.cache=self.root/'cache'
    def source(self,name,rows,stamp='2026-11-01T00:00:00+00:00'):return native(self.root,name,rows,self.tokenizer,stamp)
    def build(self,sources,name,**kwargs):
        return build_collection(sources,self.tokenizer,INFO,self.cache,self.root/name,**kwargs)
    def index(self,name):return DenseIndex(self.root/name/'dataset',self.root/name/'windows',self.root/name/'embeddings')

    def test_dedup_latest_metadata_and_self_contained_lineage(self):
        a=self.source('a',[('1','Title','alpha beta','Old'),('2','Other','gamma delta','J')])
        b=self.source('b',[('1','Title','alpha beta','New'),('3','Third','epsilon zeta','J')],'2026-11-02T00:00:00+00:00')
        factory=Mock(side_effect=AssertionError('cached build must not load encoder'))
        result=self.build([a,b],'pool',encoder_factory=factory)
        self.assertEqual(result['counts']['unique_pmids'],3)
        self.assertEqual(result['counts']['duplicate_memberships_removed'],1)
        self.assertEqual(result['measurements']['cache']['encoded_inputs'],0)
        self.assertEqual(self.index('pool').articles['1']['journal']['title'],'New')
        catalog=json.loads((self.root/'pool/dataset/catalog.json').read_bytes())
        self.assertEqual(len(catalog[0]['observations']),2)
        self.assertEqual(len({o['text_version_id'] for o in catalog[0]['observations']}),1)
        shutil.rmtree(self.root/'a');shutil.rmtree(self.root/'b')
        self.assertEqual(inspect_collection(self.root/'pool')['collection_id'],result['collection_id'])
        index=self.index('pool');self.assertEqual(len(index.search(index.vectors[0])),3)

    def test_changed_text_encodes_only_new_inputs_and_retains_old_version(self):
        a=self.source('a',[('1','Title','alpha beta','J')]);self.build([a],'old')
        b=self.source('b',[('1','Title','new gamma','J')],'2026-11-02T00:00:00+00:00')
        enc=Encoder();factory=Mock(return_value=enc)
        result=self.build([a,b],'new',seed=False,encoder_factory=factory)
        self.assertEqual(result['counts']['pmids_with_multiple_text_versions'],1)
        self.assertEqual(result['measurements']['cache']['encoded_inputs'],1)
        factory.assert_called_once();self.assertEqual(len(enc.calls[0]),1)
        old,new=self.index('old'),self.index('new')
        self.assertNotEqual(old.articles['1']['text_version'],new.articles['1']['text_version'])
        self.assertTrue(all(w['text']=='new gamma' for w in new.windows))
        self.assertTrue(all(w['text']=='alpha beta' for w in old.windows))

    def test_metadata_refresh_changes_citations_but_reuses_vector(self):
        a=self.source('a',[('1','Title','alpha beta','Old')]);self.build([a],'old')
        b=self.source('b',[('1','Title','alpha beta','New')],'2026-11-02T00:00:00+00:00')
        factory=Mock(side_effect=AssertionError('unchanged input'))
        m=self.build([a,b],'new',seed=False,encoder_factory=factory)
        old,new=self.index('old'),self.index('new')
        self.assertNotEqual(old.windows[0]['window_id'],new.windows[0]['window_id'])
        self.assertEqual(old.windows[0]['input_ids'],new.windows[0]['input_ids'])
        np.testing.assert_array_equal(old.vectors,new.vectors)
        self.assertEqual(new.articles['1']['journal']['title'],'New')
        self.assertEqual(m['measurements']['cache']['encoder_loads'],0)

    def test_growth_removal_order_and_unchanged_rebuild(self):
        a=self.source('a',[('1','First','alpha beta','J'),('2','Second','gamma delta','J')])
        b=self.source('b',[('2','Second','gamma delta','J'),('3','Third','epsilon zeta','J')],'2026-11-02T00:00:00+00:00')
        small=self.build([a],'small');grown=self.build([a,b],'grown');same=self.build([b,a],'same')
        self.assertEqual(grown['collection_id'],same['collection_id'])
        self.assertNotEqual(grown['manifest_id'],same['manifest_id'])
        self.assertEqual(self.build([a],'again')['collection_id'],small['collection_id'])
        self.build([b],'removed');index=self.index('removed')
        self.assertEqual(set(index.articles),{'2','3'})
        self.assertNotIn('1',[r['pmid'] for r in index.search(self.index('small').vectors[0])])

    def test_equal_time_conflict_requires_explicit_latest_resolution(self):
        a=self.source('a',[('1','Title','alpha','J')]);b=self.source('b',[('1','Title','beta','J')])
        with self.assertRaisesRegex(CorpusError,'Ambiguous'):self.build([a,b],'conflict')
        self.assertFalse((self.root/'conflict').exists());self.assertFalse(self.cache.exists())
        sid=load_dataset(b['dataset']).manifest['dataset_id']
        self.build([a,b],'resolved',resolutions={'1':sid})
        self.assertEqual(self.index('resolved').windows[0]['text'],'beta')
        with self.assertRaises(CorpusError):self.build([a,b],'unknown',resolutions={'9':sid})

    def test_equal_time_equivalent_records_ignore_source_location(self):
        a=self.source('a',[('1','Title','alpha','J'),('2','Other','beta','J')])
        b=self.source('b',[('2','Other','beta','J'),('1','Title','alpha','J')])
        x=self.build([a,b],'one');y=self.build([b,a],'two')
        self.assertEqual(x['collection_id'],y['collection_id'])
        self.assertEqual(x['counts']['pmids_with_multiple_text_versions'],0)

    def test_failed_update_is_unreadable_and_old_revision_is_unchanged(self):
        a=self.source('a',[('1','Title','alpha','J')]);old=self.build([a],'old')
        with self.assertRaisesRegex(RuntimeError,'encoder failed'):
            self.build([a],'broken',seed=False,encoder_info=Encoder('two').info,
                encoder_factory=Mock(side_effect=RuntimeError('encoder failed')))
        self.assertEqual(json.loads((self.root/'broken/manifest.json').read_bytes())['status'],'failed')
        with self.assertRaises(CorpusError):self.index('broken')
        self.assertEqual(inspect_collection(self.root/'old')['manifest_id'],old['manifest_id'])

    def test_policy_or_window_change_does_not_return_stale_inputs(self):
        a=self.source('a',[('1','Title','alpha beta gamma delta','J')]);self.build([a],'old')
        enc=Encoder();m=self.build([a],'new',seed=False,config=WindowConfig(body_tokens=2,overlap=0),encoder_factory=lambda:enc)
        self.assertEqual(m['measurements']['cache']['encoded_inputs'],2)
        self.assertEqual(len(self.index('new').windows),2)

    def test_previous_revision_can_seed_an_empty_cache(self):
        a=self.source('a',[('1','Title','alpha','J')]);old=self.build([a],'old');shutil.rmtree(self.cache)
        current=self.build([a],'new',seed=False,reuse=self.root/'old',encoder_factory=Mock(side_effect=AssertionError('no model')))
        self.assertEqual(current['collection_id'],old['collection_id'])
        self.assertEqual(current['measurements']['cache']['encoded_inputs'],0)

    def test_mismatched_profile_corruption_and_overwrite_fail_closed(self):
        a=self.source('a',[('1','Title','alpha','J')])
        with self.assertRaises(CorpusError):self.build([{**a,'embedding_artifact_id':'wrong'}],'bad')
        self.build([a],'pool')
        with self.assertRaises(FileExistsError):self.build([a],'pool')
        p=self.root/'pool/dataset/articles.jsonl';p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(CorpusError):self.index('pool')

    def test_disjoint_artifact_roots_and_source_version_integrity(self):
        a=self.source('a',[('1','Title','alpha','J')])
        with self.assertRaises(CorpusError):self.build([a],'a/dataset/nested')
        dataset=load_dataset(a['dataset']);bad=copy.deepcopy(dataset.articles[0]);bad['abstract']['sections'][0]['text']='forged'
        with self.assertRaisesRegex(CorpusError,'text-version'):
            select_records({'a':type(dataset)(dataset.manifest,(bad,))})
