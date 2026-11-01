"""Exact pair budgets, mask-correct batching and raw cross-encoder scores."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from medlens.reranking.medcpt import (ASSETS,MODEL_ID,REVISION,MedCPTScorer,
    PairBudgetError,prepare_pair,score_encoded,verify_assets)
try:
    import torch
except ImportError:
    torch=None


class Tokenizer:
    def __call__(self,question,document,**kwargs):
        self.last=(question,document,kwargs)
        q=[10]*len(question.split());d=[20]*len(document.split())
        ids=[2,*q,3,*d,3]
        return {'input_ids':ids,'attention_mask':[1]*len(ids),
            'token_type_ids':[0]*(len(q)+2)+[1]*(len(d)+1)}


class PairTests(unittest.TestCase):
    def test_complete_pair_includes_question_title_body_and_three_specials(self):
        tokenizer=Tokenizer()
        row=prepare_pair('research question','Article title','Exact source text',tokenizer)
        self.assertEqual(len(row['input_ids']),10)
        self.assertEqual(tokenizer.last[:2],('research question','Article title\nExact source text'))
        self.assertFalse(tokenizer.last[2]['truncation'])
        self.assertFalse(tokenizer.last[2]['padding'])
        self.assertEqual(row['token_type_ids'],[0]*4+[1]*6)

    def test_512_accepted_513_rejected_without_shortening_any_field(self):
        tokenizer=Tokenizer()
        row=prepare_pair('question','title','word '*507,tokenizer)
        self.assertEqual(len(row['input_ids']),512)
        with self.assertRaisesRegex(PairBudgetError,'513'):
            prepare_pair('question','title','word '*508,tokenizer)
        self.assertEqual(len(tokenizer.last[1].split()),509)
        with self.assertRaises(PairBudgetError):prepare_pair('q '*510,'title','excerpt',tokenizer)

    def test_empty_fields_and_invalid_tokenizer_encodings_rejected(self):
        for pair in (('','',''),(' ','title','text'),('question','',' '),(None,'title','text')):
            with self.assertRaises(ValueError):prepare_pair(*pair,Tokenizer())
        # Title-only retrieval pairs are allowed; they are not abstract evidence.
        self.assertEqual(len(prepare_pair('question','title','',Tokenizer())['input_ids']),5)
        for key,value in (('input_ids',[2,True,3]),('attention_mask',[True]*6),
                           ('token_type_ids',[0,1])):
            def bad(*args,**kwargs):
                row=Tokenizer()('q','t x');row[key]=value;return row
            with self.subTest(key=key),self.assertRaises(ValueError):prepare_pair('q','t','x',bad)

    def test_manifest_requires_all_pinned_files_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ASSETS:(root/name).write_text(name)
            source={'schema_version':1,'model_id':MODEL_ID,'revision':REVISION,
                'files':{name:hashlib.sha256(name.encode()).hexdigest() for name in ASSETS}}
            (root/'model-source.json').write_text(json.dumps(source))
            self.assertEqual(verify_assets(root),source)
            for change in ('revision','omitted-file','corruption'):
                broken=copy.deepcopy(source)
                if change=='revision':broken['revision']='main'
                if change=='omitted-file':broken['files'].pop('pytorch_model.bin')
                if change=='corruption':(root/'config.json').write_text('changed')
                (root/'model-source.json').write_text(json.dumps(broken))
                with self.subTest(change=change),self.assertRaises(ValueError):verify_assets(root)


@unittest.skipIf(torch is None,'Install the reranking extra for tensor tests')
class ScoringTests(unittest.TestCase):
    def scorer(self):
        class Model:
            calls=0
            def eval(self):return self
            def __call__(self,input_ids,attention_mask,token_type_ids):
                self.calls+=1
                # A signed score sensitive to masks AND pair segment IDs.
                score=((input_ids*attention_mask).sum(dim=1)
                    - 100*(token_type_ids*attention_mask).sum(dim=1)).float()
                return SimpleNamespace(logits=score[:,None])
        scorer=MedCPTScorer.__new__(MedCPTScorer)
        scorer.tokenizer=Tokenizer();scorer.model=Model();scorer.device='cpu';scorer.batch_size=2
        return scorer

    def test_batch_sizes_padding_order_and_raw_negative_scores(self):
        scorer=self.scorer();pairs=[('q','title','word'),('q q','t','body '*7),('q','t','x '*50)]
        a=scorer.score(pairs,batch_size=1);b=scorer.score(pairs,batch_size=3)
        self.assertEqual(a['scores'],b['scores'])
        self.assertEqual(a['pairs_sha256'],b['pairs_sha256'])
        self.assertEqual(a['batch_count'],3);self.assertEqual(b['batch_count'],1)
        self.assertLess(a['scores'][-1],0)
        reversed_result=scorer.score(list(reversed(pairs)))
        self.assertEqual(reversed_result['scores'],list(reversed(a['scores'])))

    def test_validate_entire_request_before_any_forward(self):
        scorer=self.scorer()
        with self.assertRaises(PairBudgetError):
            scorer.score([('q','t','ok'),('q','t','word '*508)],batch_size=1)
        self.assertEqual(scorer.model.calls,0)
        self.assertEqual(scorer.score([])['scores'],[])
        for size in (0,-1,True,1.5):
            with self.assertRaises(ValueError):scorer.score([('q','t','text')],batch_size=size)

    def test_nonfinite_or_multiple_logits_fail(self):
        scorer=self.scorer();rows=[prepare_pair('q','t','x',scorer.tokenizer)]
        for logits in (torch.tensor([[float('nan')]]),torch.ones((1,2)),torch.ones(1)):
            class BadModel:
                def eval(self):return self
                def __call__(self,**kwargs):return SimpleNamespace(logits=logits)
            with self.assertRaises(ValueError):score_encoded(rows,BadModel(),device='cpu',batch_size=1)

    def test_encoded_budget_and_device_fail_before_forward(self):
        scorer=self.scorer();row=prepare_pair('q','t','x',scorer.tokenizer)
        row['input_ids'] += [20]*512
        with self.assertRaises(PairBudgetError):score_encoded([row],scorer.model,device='cpu',batch_size=1)
        with self.assertRaises(ValueError):score_encoded([],scorer.model,device='cuda',batch_size=1)
        self.assertEqual(scorer.model.calls,0)


class HeldoutTests(unittest.TestCase):
    def test_frozen_question_hash_split_and_no_duplicate_pilot_groups(self):
        from medlens.evaluation.formats import parse_queries
        root=Path(__file__).resolve().parents[1]
        path=root/'benchmarks/heldout-v1/queries.jsonl'
        body=path.read_bytes();plan=json.loads((path.parent/'candidate-plan.json').read_bytes())
        self.assertEqual(hashlib.sha256(body).hexdigest(),plan['queries_sha256'])
        heldout=parse_queries(body,path.parent)
        pilot_path=root/'benchmarks/pilot-v1/queries.jsonl'
        pilot=parse_queries(pilot_path.read_bytes(),pilot_path.parent)
        self.assertEqual(len(heldout),12)
        self.assertTrue(all(q.split=='test' for q in heldout))
        self.assertEqual([q.query_id for q in heldout],[q['query_id'] for q in plan['queries']])
        self.assertFalse({q.group_id for q in heldout}&{q.group_id for q in pilot})
        self.assertFalse({q.text.casefold() for q in heldout}&{q.text.casefold() for q in pilot})
