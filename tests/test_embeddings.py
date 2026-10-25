"""BGE pooling, masked padding, query policy, and vector-to-source mapping."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from medlens.retrieval.embeddings import (DIMENSION, QUERY_INSTRUCTION, EmbeddingError,
    build_embeddings, encode_sequences, load_embeddings, normalized_cls, query_input, validate_vectors)
from medlens.retrieval.windows import build_windows
from test_evaluation import synthetic_inputs
from test_windows import Tokenizer

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the embeddings extra for tensor tests")
class TensorTests(unittest.TestCase):
    def test_cls_pooling_not_mean_or_pooler_and_unit_normalization(self):
        hidden = torch.tensor([[[3.,4.],[100.,100.]]])
        actual = normalized_cls(hidden).numpy()
        np.testing.assert_allclose(actual, [[0.6,0.8]], atol=1e-7)
        for bad in (torch.zeros(1,2,3), torch.full((1,2,3),float('nan'))):
            with self.assertRaises(EmbeddingError): normalized_cls(bad)

    def model(self):
        class MaskedModel:
            def eval(self): return self
            def __call__(self, input_ids, attention_mask, token_type_ids):
                assert not token_type_ids.any()
                batch,length=input_ids.shape
                hidden=torch.zeros(batch,length,DIMENSION)
                hidden[:,0,0]=(input_ids*attention_mask).sum(dim=1)
                hidden[:,0,1]=attention_mask.sum(dim=1)
                return SimpleNamespace(last_hidden_state=hidden)
        return MaskedModel()

    def test_padding_batch_size_order_and_eval_inference(self):
        rows=[[101,10,102],[101,20,30,40,102],[101,7,102]]
        first,_=encode_sequences(rows,self.model(),device='cpu',batch_size=1)
        second,_=encode_sequences(rows,self.model(),device='cpu',batch_size=3)
        np.testing.assert_allclose(first,second,atol=1e-7)
        validate_vectors(first,3)
        self.assertFalse(np.array_equal(first[0],first[1]))
        empty,_=encode_sequences([],self.model(),device='cpu')
        self.assertEqual(empty.shape,(0,768))

    def test_invalid_tokens_lengths_device_and_batch_rejected(self):
        for ids in ([101,102]+[7]*511,[101,30522,102],[101,True,102],[5,102],[]):
            with self.assertRaises(EmbeddingError):encode_sequences([ids],self.model(),device='cpu')
        for kwargs in ({'device':'auto'},{'device':'cpu','batch_size':0}):
            with self.assertRaises(EmbeddingError):encode_sequences([],self.model(),**kwargs)


class EmbeddingArtifactTests(unittest.TestCase):
    def test_query_instruction_preserves_text_and_never_truncates(self):
        calls=[]
        def tokenizer(text,**kwargs):
            calls.append((text,kwargs));return {'input_ids':[101,10,102]}
        self.assertEqual(query_input(tokenizer,'  Does CBT help? '),[101,10,102])
        self.assertEqual(calls[0][0],QUERY_INSTRUCTION+'  Does CBT help? ')
        self.assertFalse(calls[0][1]['truncation'])
        with self.assertRaises(EmbeddingError):query_input(tokenizer,' ')
        with self.assertRaisesRegex(EmbeddingError,'512'):
            query_input(lambda *a,**kw:{'input_ids':[101]*513},'long')

    def fixture(self,root):
        synthetic_inputs(root)
        build_windows(root/'parsed',Tokenizer(),{'fixture':True},root/'windows')
        class Encoder:
            tokenizer_info={'fixture':True}
            info={'fixture':'deterministic-encoder'}
            def encode(self,rows):
                vectors=np.zeros((len(rows),768),dtype=np.float32)
                for i in range(len(rows)):vectors[i,i]=1
                return vectors,{'encode_seconds':0.0}
        return Encoder()

    def test_artifact_order_reproducibility_mapping_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            with patch('socket.socket',side_effect=AssertionError('offline')):
                a=build_embeddings(root/'windows',encoder,root/'a')
                b=build_embeddings(root/'windows',encoder,root/'b')
                manifest,rows,vectors=load_embeddings(root/'a')
            self.assertEqual(a['artifact_id'],b['artifact_id'])
            self.assertEqual([r['pmid'] for r in rows],['102','103','101'])
            self.assertEqual(vectors.shape,(3,768))
            self.assertEqual(manifest['row_count'],3)
            with self.assertRaises(FileExistsError):build_embeddings(root/'windows',encoder,root/'a')

    def test_corruption_and_incomplete_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            build_embeddings(root/'windows',encoder,root/'a')
            path=root/'a/rows.jsonl';body=path.read_bytes();path.write_bytes(body+b'{}\n')
            with self.assertRaisesRegex(EmbeddingError,'checksum'):load_embeddings(root/'a')
            path.write_bytes(body)
            path=root/'a/manifest.json';m=json.loads(path.read_bytes());m['status']='running';path.write_text(json.dumps(m))
            with self.assertRaisesRegex(EmbeddingError,'Incomplete'):load_embeddings(root/'a')

    def test_tokenizer_mismatch_and_failed_encoder_publish_no_partial_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);encoder=self.fixture(root)
            encoder.tokenizer_info={}
            with self.assertRaisesRegex(EmbeddingError,'tokenizers differ'):
                build_embeddings(root/'windows',encoder,root/'failed')
            self.assertFalse((root/'failed').exists())
            encoder.tokenizer_info={'fixture':True}
            encoder.encode=lambda rows:(np.zeros((len(rows),768),np.float32),{})
            with self.assertRaisesRegex(EmbeddingError,'unit L2'):
                build_embeddings(root/'windows',encoder,root/'failed')
            self.assertFalse((root/'failed').exists())

    def test_vector_validation_shape_dtype_finite_and_norm(self):
        for vectors in (np.zeros((1,768),np.float32),np.ones((1,7),np.float32),
                        np.ones((1,768),np.float64),np.full((1,768),np.nan,np.float32)):
            with self.assertRaises(EmbeddingError):validate_vectors(vectors,1)


if __name__=='__main__':unittest.main()
