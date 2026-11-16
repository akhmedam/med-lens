"""One-article generation keeps Dense top ten and exact source binding; no inference."""
from copy import deepcopy
import json,unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from medlens.generation.context import digest
from medlens.pipeline import run_question
from medlens.reporting import render_report
from medlens.retrieval.dense import DenseIndex
from test_answers import FakeRuntime
from test_pipeline import Tokenizer,Encoder


def index_fixture():
 articles=[];windows=[];vectors=[];positions={}
 for i in range(1,11):
  pmid=str(i);provenance={'fixture':'one-article','record':i};sections=[];positions[pmid]=[]
  for j,(label,text)in enumerate([('METHODS','Adult participants were randomized to agent R or placebo.'),
                                  ('RESULTS',f'Pain decreased by {i} units with agent R compared with placebo.')]):
   section={'index':j,'label':label,'text':text,'source_path':f'abstract/{j}'};sections.append(section)
   positions[pmid].append(len(windows))
   windows.append({'pmid':pmid,'window_id':f'{i}-{j}','text_version_id':'v'+pmid,'title':'Fictional study '+pmid,
      'kind':'abstract','section_index':j,'section_label':label,'source_path':section['source_path'],
      'char_start':0,'char_end':len(text),'text':text,'provenance':provenance})
   cosine=.99-(i-1)*.025-j*.002;vector=np.zeros(768,np.float32);vector[0]=cosine;vector[1]=np.sqrt(1-cosine*cosine);vectors.append(vector)
  articles.append({'pmid':pmid,'title':'Fictional study '+pmid,'journal':None,'dates':[],
    'text_version':{'text_version_id':'v'+pmid},'provenance':provenance,'abstract':{'sections':sections}})
 manifest={'dataset_id':'one-article-fixture','articles_sha256':digest(articles)}
 index=DenseIndex.__new__(DenseIndex);index.dataset=SimpleNamespace(manifest=manifest,articles=articles)
 index.window_manifest={**manifest,'artifact_id':'fixture-windows'};index.windows=windows
 index.articles={a['pmid']:a for a in articles};index.positions=positions;index.vectors=np.asarray(vectors)
 index.embedding_manifest={'artifact_id':'fixture-embeddings'};index.missing_pmids=[]
 index.verify_encoder=lambda encoder:None
 return index

class AutomaticSelector(FakeRuntime):
 def __init__(self):
  super().__init__([]);self.tokenizer=Tokenizer();self.metadata={'fixture':True};self.close=Mock()
 def generate(self,prompt,*,max_new_tokens):
  plans=json.loads(prompt['messages'][1]['content'].split('\nArticle scope plans: ',1)[1].split('\nEvidence catalog:\n',1)[0])
  selection={'articles':[{'article_id':p['article_id'],'scope_quote_id':p['scope_quote_ids'][0],
     'finding_quote_ids':[p['finding_quote_ids'][0]],'applicability':'answers_question'}for p in plans],'limitations':[]}
  self.outputs=iter([selection]);return super().generate(prompt,max_new_tokens=max_new_tokens)

class OneArticlePipelineIntegration(unittest.TestCase):
 def test_top10_stays_visible_and_one_bound_article_uses_one_selector_call(self):
  index=index_fixture();before=deepcopy(index.dataset.articles);windows_before=deepcopy(index.windows)
  runtime=AutomaticSelector();loads=[]
  def factory():loads.append(True);return runtime
  report=run_question(index,'Does agent R reduce pain in adults?',encoder_factory=Encoder,
    tokenizer=Tokenizer(),runtime_factory=factory,generate=True,rerank=False,top_k=10,answer_style='source-findings')
  expected=[str(i)for i in range(1,11)]
  self.assertEqual(report['status'],'complete');self.assertEqual(report['answer']['status'],'partial')
  self.assertEqual([h['pmid']for h in report['retrieval']['results']],expected)
  self.assertEqual(report['context']['article_ranking'],expected)
  self.assertEqual(report['reranking']['status'],'skipped')
  self.assertEqual({s['pmid']for s in report['context']['sources']},{'1'})
  self.assertLessEqual(len(report['context']['sources']),2)
  self.assertEqual(len(loads),1);self.assertEqual([cap for _,cap in runtime.calls],[768])
  self.assertEqual(report['runtime_load_count'],1);runtime.close.assert_called_once_with()
  self.assertTrue(report['runtime_closed']);self.assertTrue(report['answer']['quote_identity_verified'])
  self.assertFalse(report['answer']['semantic_support_verified'])
  for claim in report['answer']['claims']:
   self.assertEqual(claim['source_article_id'],'1')
   for citation in claim['citations']:
    self.assertEqual(citation['pmid'],'1');self.assertEqual(citation['text_version_id'],'v1')
    section=index.articles['1']['abstract']['sections'][citation['section_index']]
    for span in citation['matches']:
     self.assertEqual(section['text'][span['char_start']:span['char_end']],citation['quote'])
  page=render_report(report)
  for i in range(1,11):self.assertIn('Fictional study '+str(i),page)
  self.assertEqual(index.dataset.articles,before);self.assertEqual(index.windows,windows_before)
if __name__=='__main__':unittest.main()
