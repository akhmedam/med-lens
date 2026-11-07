"""A changed draft, resource identity or model is not a paired review experiment."""

import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/compare_semantic_reviews.py'
spec=importlib.util.spec_from_file_location('semantic_comparison',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture():
    r={'status':'complete','heldout_used':False,'resource_scope':'rebuilt-pilot',
       'relevance_judgments_bound':False,'qrels_sha256':None,'context_report_sha256':'context',
       'profile_sha256':'profile','case_ids':['p01'],'prompt_version':'v6',
       'runtime':{'model_id':'pinned','revision':'pinned','device':'mps','quantization':{'bits':4},'packages':{'torch':'pinned'}},
       'per_case':[{'case_id':'p01','kind':'pilot','question':'Q','context_id':'c','expected':None,
           'model_calls':[{'stage':'draft','text':'frozen','max_new_tokens':768,'hit_output_limit':False,
                           'prompt':{'prompt_sha256':'prompt','messages':[]},'cache_reused':True},
                          {'stage':'support-review','max_new_tokens':512}]}]}
    return r


class SemanticComparisonTests(unittest.TestCase):
    def test_same_drafts_are_pairable_without_historical_judgments(self):
        tool.verify_frozen_drafts(fixture(),fixture())

    def test_changed_drafts_or_prompt_content_are_rejected(self):
        for change in (lambda r:r['per_case'][0]['model_calls'][0].update(text='new'),
                       lambda r:r['per_case'][0]['model_calls'][0].update(cache_reused=False),
                       lambda r:r['per_case'][0]['model_calls'][0]['prompt'].update(messages=['different']),
                       lambda r:r['per_case'][0]['model_calls'][1].update(max_new_tokens=768),
                       lambda r:r['per_case'][0]['model_calls'].append({'stage':'draft'})):
            r=fixture();change(r)
            with self.assertRaises(ValueError):tool.verify_frozen_drafts(fixture(),r)

    def test_changed_model_scope_or_membership_are_rejected(self):
        for change in (lambda r:r['runtime'].update(revision='other'),
                       lambda r:r.update(context_report_sha256='other'),
                       lambda r:r.update(heldout_used=True),
                       lambda r:r.update(relevance_judgments_bound=True),
                       lambda r:r.update(case_ids=['other']),
                       lambda r:r['per_case'][0].update(question='different')):
            r=fixture();change(r)
            with self.assertRaises(ValueError):tool.verify_frozen_drafts(fixture(),r)
