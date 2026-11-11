"""Development acquisition remains search-only and binds the frozen plan/checkpoint."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('prepare_biomedical_cases',
    Path(__file__).resolve().parents[1]/'scripts/prepare_biomedical_cases.py')
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class BiomedicalCaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = self.root/'plan.json'
        self.case = {'query_id':'fictional-1','question':'How does fictional protein Z act?',
            'specialty':'fictional','question_type':'mechanism','relevance_intent':'Fictional test only.'}
        self.plan.write_text(json.dumps({'schema_version':1,'split':'dev','heldout_used':False,
            'candidate_limit':100,'cases':[self.case]}))
        self.output = self.root/'sources'
        self.argv = ['--plan',str(self.plan),'--output',str(self.output),'--cache',str(self.root/'cache')]

    def checkpoint(self, command, **kwargs):
        from medlens.pubmed_query import plan_query
        self.assertIn('--search-only',command)
        self.assertEqual(command[command.index('--answer-model')+1],'4b')
        self.assertNotIn('--generate',command)
        target = Path(command[command.index('--output-dir')+1]);target.mkdir(parents=True)
        value = {'status':'complete','question':self.case['question'],'runtime_load_count':0,
            'generate':False,'live':{'fresh_search_complete':True,'fetched_article_count':1,
                'query_plan':plan_query(self.case['question'])},
            'answer':{'status':'not_requested'},'retrieval':{'results':[{'pmid':'1'}]}}
        if getattr(self,'tamper',False):value['question']='other question'
        (target/'run.json').write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)

    def test_fresh_cases_do_not_generate_or_grade_answers_and_freeze_plan_bytes(self):
        with patch.object(cli.subprocess,'run',side_effect=self.checkpoint):
            self.assertEqual(cli.main(self.argv),0)
        result = json.loads((self.output/'preparation.json').read_bytes())
        self.assertEqual(result['status'],'complete-awaiting-manual-human-annotation')
        self.assertFalse(result['quality_evaluation_performed'])
        self.assertFalse(result['answer_generation_performed'])
        self.assertEqual((self.output/'plan-original.json').read_bytes(),self.plan.read_bytes())

    def test_unbound_checkpoint_fails_and_preserves_evidence(self):
        self.tamper = True
        with patch.object(cli.subprocess,'run',side_effect=self.checkpoint):
            self.assertEqual(cli.main(self.argv),1)
        result = json.loads((self.output/'preparation.json').read_bytes())
        self.assertEqual(result['status'],'failed')
        self.assertTrue((self.output/'fictional-1/run.json').is_file())

    def test_no_overwrite_or_network_for_existing_output_and_unsafe_plan_ids(self):
        self.output.mkdir();sentinel=self.output/'keep';sentinel.write_text('keep')
        with patch.object(cli.subprocess,'run') as run, self.assertRaises(SystemExit):cli.main(self.argv)
        run.assert_not_called();self.assertEqual(sentinel.read_text(),'keep')
        value=json.loads(self.plan.read_bytes());value['cases'][0]['query_id']='../outside'
        self.plan.write_text(json.dumps(value))
        with self.assertRaises(ValueError):cli.load_plan(self.plan)

    def test_existing_heldout_split_is_rejected_without_reading_its_sources(self):
        value=json.loads(self.plan.read_bytes());value['split']='test';value['heldout_used']=True
        self.plan.write_text(json.dumps(value))
        with patch.object(cli.subprocess,'run') as run, self.assertRaises(SystemExit):
            cli.main(self.argv)
        run.assert_not_called();self.assertFalse(self.output.exists())


class RelativePathTests(unittest.TestCase):
    setUp = BiomedicalCaseTests.setUp
    checkpoint = BiomedicalCaseTests.checkpoint
    def test_relative_paths_keep_caller_location_when_child_uses_repo_cwd(self):
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        observed = []
        def run(command, **kwargs):
            observed.append(command)
            return self.checkpoint(command, **kwargs)
        with patch.object(cli.subprocess,'run',side_effect=run):
            self.assertEqual(cli.main(['--plan','plan.json','--output','sources','--cache','cache']),0)
        self.assertEqual(Path(observed[0][observed[0].index('--output-dir')+1]),(self.output/'fictional-1').resolve())
        self.assertEqual(Path(observed[0][observed[0].index('--cache')+1]),(self.root/'cache').resolve())
