"""Article maximums, complete window coverage and a Dense-free direct path."""

from contextlib import redirect_stderr,redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medlens.reranking.articles import MedCPTRanker,aggregate_windows
from medlens.reranking.medcpt import PairBudgetError,prepare_pair
import test_evidence_context as evidence
import test_medcpt as pairs


def fixture(changes=None):
    dataset,manifest,windows=evidence.fixtures()
    manifest['empty_pmids']=[]
    if changes:changes(dataset,manifest,windows)
    with patch('medlens.reranking.articles.load_dataset',return_value=dataset), \
         patch('medlens.reranking.articles.load_windows',return_value=(manifest,windows)):
        return MedCPTRanker('dataset','windows')


class Scorer:
    metadata={'fixture':True,'device':'cpu','default_batch_size':4}
    def __init__(self,windows,scores):
        self.values={(w['title'],w['text']):score for w,score in zip(windows,scores)}
        self.calls=[]
    def score(self,triples):
        self.calls.append(triples)
        lengths=[len(prepare_pair(*triple,pairs.Tokenizer())['input_ids']) for triple in triples]
        return {'scores':[self.values[(title,text)] for _,title,text in triples],
            'pair_count':len(triples),'pair_lengths':lengths,'score_seconds':0.0}


class ArticleRankingTests(unittest.TestCase):
    def test_maximum_is_not_mean_and_best_window_can_differ_from_dense(self):
        ranker=fixture()
        scorer=Scorer(ranker.windows,[-10,8,-10, 5,5,5, -1,-1,-1])
        result=ranker.run('Question',scorer,eligible_pmids=['1','2'])
        self.assertEqual([h['pmid'] for h in result['results']],['1','2'])
        self.assertEqual(result['results'][0]['score'],8)
        self.assertEqual(result['results'][0]['best_window']['window_id'],'1-1')
        self.assertEqual(result['scoring']['pair_count'],6)
        self.assertEqual([r['window_id'] for r in result['window_scores']],['1-0','1-1','1-2','2-0','2-1','2-2'])
        self.assertEqual(len(scorer.calls),1)
        self.assertEqual(scorer.calls[0], [('Question',w['title'],w['text']) for w in ranker.windows[:6]])
        self.assertNotIn('cosine_distance',result['results'][0])

    def test_direct_scores_all_candidates_including_outside_dense_subset(self):
        ranker=fixture();scorer=Scorer(ranker.windows,[1]*6+[10]*3)
        direct=ranker.run('Question',scorer)
        limited=ranker.run('Question',scorer,eligible_pmids=['1','2'])
        self.assertEqual(direct['results'][0]['pmid'],'3')
        self.assertEqual(direct['scoring']['pair_count'],9)
        self.assertEqual(limited['scoring']['pair_count'],6)
        self.assertNotIn('3',[h['pmid'] for h in limited['results']])

    def test_negative_scores_and_ties_keep_eligible_and_source_order(self):
        ranker=fixture();scorer=Scorer(ranker.windows,[-4]*9)
        result=ranker.run('Question',scorer,eligible_pmids=['2','1'])
        self.assertEqual([h['pmid'] for h in result['results']],['2','1'])
        self.assertEqual(result['results'][0]['best_window']['window_id'],'2-0')
        self.assertEqual(result['results'][1]['score'],-4)
        self.assertEqual([h['pmid'] for h in ranker.run('Question',scorer)['results']],['1','2','3'])

    def test_results_are_prefix_consistent_and_preserve_source_offsets(self):
        ranker=fixture();scorer=Scorer(ranker.windows,list(range(9)))
        one=ranker.run('Question',scorer,top_k=1)
        all_results=ranker.run('Question',scorer)
        self.assertEqual(one['results'],all_results['results'][:1])
        best=one['results'][0]['best_window'];source=ranker.dataset.articles[2]['abstract']['sections'][0]['text']
        self.assertEqual(best['text'],source[best['char_start']:best['char_end']])
        self.assertEqual(best['provenance'],ranker.windows[-1]['provenance'])

    def test_missing_windows_get_no_fabricated_score_and_empty_subset_skips_scorer(self):
        def missing(dataset,manifest,windows):
            windows[:]=[w for w in windows if w['pmid']!='3'];manifest['empty_pmids']=['3']
        ranker=fixture(missing);scorer=Scorer(ranker.windows,[1]*6)
        result=ranker.run('Question',scorer)
        self.assertEqual(result['ranked_article_count'],2)
        self.assertEqual(result['missing_window_pmids'],['3'])
        empty=ranker.run('Question',None,eligible_pmids=[])
        self.assertEqual(empty['results'],[])
        self.assertEqual(empty['scoring']['pair_count'],0)
        self.assertIsNone(empty['runtime'])

    def test_bad_scores_lengths_and_pair_counts_rejected(self):
        ranker=fixture()
        good=Scorer(ranker.windows,[1]*9).score([('Question',w['title'],w['text']) for w in ranker.windows])
        for key,value in (('scores',[1]*8),('scores',[float('nan')]*9),('scores',[True]*9),
                          ('pair_count',8),('pair_lengths',[513]*9)):
            class Bad:
                metadata={}
                def score(self,triples):return {**good,key:value}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):ranker.run('Question',Bad())

    def test_unknown_duplicate_and_invalid_inputs_fail_before_scorer(self):
        ranker=fixture();scorer=Scorer(ranker.windows,[1]*9)
        for eligible in (['1','1'],['99'],'1',[True]):
            with self.assertRaises(ValueError):ranker.run('Question',scorer,eligible_pmids=eligible)
        for top_k in (0,11,True):
            with self.assertRaises(ValueError):ranker.run('Question',scorer,top_k=top_k)
        with self.assertRaises(ValueError):ranker.run(' ',scorer)
        self.assertEqual(scorer.calls,[])

    def test_bad_source_or_artifact_is_rejected_before_loading_model(self):
        for change in (
            lambda d,m,w:m.update(dataset_id='other'),
            lambda d,m,w:w[0].update(text='fabricated'),
            lambda d,m,w:w[0].update(text_version_id='stale'),
            lambda d,m,w:w[0].update(provenance={}),
            lambda d,m,w:m.update(empty_pmids=['1']),
        ):
            with self.assertRaises(ValueError):fixture(change)

    def test_run_identity_includes_model_runtime_configuration(self):
        ranker=fixture();scorer=Scorer(ranker.windows,[1]*9)
        first=ranker.run('Question',scorer)
        scorer.metadata={**scorer.metadata,'default_batch_size':8}
        changed=ranker.run('Question',scorer)
        self.assertNotEqual(first['run_id'],changed['run_id'])
        self.assertEqual(first['results'],changed['results'])

    def test_over_budget_does_not_silently_skip_window_or_fallback(self):
        class Overflow:
            metadata={}
            def score(self,triples):raise PairBudgetError('Too long')
        with self.assertRaises(PairBudgetError):fixture().run('Question',Overflow())

    def test_saved_scores_can_be_aggregated_without_inference(self):
        ranker=fixture();scorer=Scorer(ranker.windows,[-10,7,3, 4,5,0, 8,1,-1])
        result=ranker.run('Question',scorer)
        expected=aggregate_windows(ranker.windows,[r['score'] for r in result['window_scores']],list(ranker.articles))
        self.assertEqual([(p,s) for p,s,_ in expected],[(h['pmid'],h['score']) for h in result['results']])
        self.assertEqual(len(scorer.calls),1)


class SearchCLITests(unittest.TestCase):
    def cli(self):
        path=Path(__file__).resolve().parents[1]/'scripts/medcpt_search.py'
        spec=importlib.util.spec_from_file_location('medcpt_cli',path)
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        return cli

    def test_direct_mode_does_not_load_dense_or_read_embeddings(self):
        cli=self.cli();ranker=fixture();scorer=Scorer(ranker.windows,list(range(9)))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run.json'
            args=['dataset','--windows','windows','--mode','medcpt','--query','Question','--output',str(output)]
            with patch.object(cli,'MedCPTRanker',return_value=ranker), \
                 patch.object(cli,'MedCPTScorer',return_value=scorer), \
                 patch.object(cli,'dense_selection',side_effect=AssertionError('Dense must not run')), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args),0)
            report=json.loads(output.read_bytes())
            self.assertIsNone(report['dense_selection'])
            self.assertNotIn('embeddings',report['input_paths'])
            self.assertEqual(report['ranking']['eligible_count'],3)

    def test_dense_mode_uses_selection_but_scores_all_its_windows(self):
        cli=self.cli();ranker=fixture();scorer=Scorer(ranker.windows,list(range(9)))
        selection={'dataset_id':'d','window_artifact_id':'w','results':[{'pmid':'2'},{'pmid':'1'}]}
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run.json'
            with patch.object(cli,'MedCPTRanker',return_value=ranker), \
                 patch.object(cli,'MedCPTScorer',return_value=scorer), \
                 patch.object(cli,'dense_selection',return_value=selection),redirect_stdout(io.StringIO()):
                code=cli.main(['dataset','--windows','windows','--mode','dense-medcpt','--embeddings','vectors',
                    '--query','Question','--output',str(output)])
            self.assertEqual(code,0)
            report=json.loads(output.read_bytes())
            self.assertEqual(report['ranking']['eligible_pmids'],['2','1'])
            self.assertEqual(report['ranking']['scoring']['pair_count'],6)

    def test_mismatched_dense_sources_fail_before_medcpt_model_load(self):
        cli=self.cli();ranker=fixture()
        selection={'dataset_id':'other','window_artifact_id':'w','results':[]}
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run.json'
            with patch.object(cli,'MedCPTRanker',return_value=ranker), \
                 patch.object(cli,'dense_selection',return_value=selection), \
                 patch.object(cli,'MedCPTScorer',side_effect=AssertionError('Must not load')), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                code=cli.main(['dataset','--windows','windows','--mode','dense-medcpt','--embeddings','vectors',
                    '--query','Question','--output',str(output)])
            self.assertEqual(code,1)
            self.assertNotIn('ranking',json.loads(output.read_bytes()))

    def test_failed_score_is_checkpointed_and_existing_output_is_protected(self):
        cli=self.cli();ranker=fixture()
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'run.json'
            args=['dataset','--windows','windows','--mode','medcpt','--query','Question','--output',str(output)]
            with patch.object(cli,'MedCPTRanker',return_value=ranker), \
                 patch.object(cli,'MedCPTScorer',side_effect=RuntimeError('Unavailable')), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args),1)
                before=output.read_bytes()
                self.assertEqual(json.loads(before)['error']['stage'],'medcpt-scoring')
                with self.assertRaises(SystemExit):cli.main(args)
                self.assertEqual(output.read_bytes(),before)
