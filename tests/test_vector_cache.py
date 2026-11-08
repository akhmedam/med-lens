"""Cache identity, immutable publication and corruption are checked independently."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

import numpy as np

from medlens.retrieval.vector_cache import VectorCache,CacheError


class VectorCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cache=VectorCache(self.root/'cache',{'model':'one'},{'tokenizer':'one'})
        self.ids=[101,1000,102];self.vector=np.zeros(768,np.float32);self.vector[0]=1

    def test_model_tokenizer_input_and_device_are_cache_boundaries(self):
        self.cache.put(self.ids,self.vector)
        for policy,tokenizer in [({'model':'two'},{'tokenizer':'one'}),({'model':'one','device':'cpu'},{'tokenizer':'one'}),({'model':'one'},{'tokenizer':'two'})]:
            self.assertIsNone(VectorCache(self.root/'cache',policy,tokenizer).get(self.ids))
        self.assertIsNone(self.cache.get([101,2000,102]))
        np.testing.assert_array_equal(self.cache.get(self.ids),self.vector)

    def test_corruption_is_not_silently_recomputed_or_overwritten(self):
        self.cache.put(self.ids,self.vector);key=self.cache.key(self.ids);path=self.root/'cache'/key[:2]/(key+'.bin')
        data=bytearray(path.read_bytes());data[-1]^=1;path.write_bytes(data)
        with self.assertRaises(CacheError):self.cache.get(self.ids)
        with self.assertRaises(CacheError):self.cache.put(self.ids,self.vector)
        self.assertEqual(path.read_bytes(),bytes(data))

    def test_small_numeric_variation_keeps_original_but_conflict_fails(self):
        self.cache.put(self.ids,self.vector);close=self.vector.copy();close[10]=1e-7
        made,delta=self.cache.put(self.ids,close);self.assertFalse(made);self.assertGreater(delta,0)
        np.testing.assert_array_equal(self.cache.get(self.ids),self.vector)
        other=np.roll(self.vector,1)
        with self.assertRaises(CacheError):self.cache.put(self.ids,other)
        np.testing.assert_array_equal(self.cache.get(self.ids),self.vector)

    def test_concurrent_equal_writers_publish_one_complete_entry(self):
        with ThreadPoolExecutor(2) as executor:
            results=list(executor.map(lambda _:self.cache.put(self.ids,self.vector),range(2)))
        self.assertEqual(sum(made for made,_ in results),1)
        np.testing.assert_array_equal(self.cache.get(self.ids),self.vector)
        self.assertEqual(len(list((self.root/'cache').rglob('*.bin'))),1)
        self.assertEqual(list((self.root/'cache').rglob('*.tmp')),[])

    def test_symlink_prefix_and_invalid_inputs_fail_without_writes(self):
        outside=self.root/'outside';outside.mkdir();key=self.cache.key(self.ids)
        (self.root/'cache'/key[:2]).symlink_to(outside)
        with self.assertRaises(CacheError):self.cache.put(self.ids,self.vector)
        self.assertEqual(list(outside.iterdir()),[])
        for ids in ([101,True,102],[101,40000,102],[1000,102]):
            with self.assertRaises(CacheError):self.cache.get(ids)
