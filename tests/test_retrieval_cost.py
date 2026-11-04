"""Pool views preserve exact window/vector mapping and pooled quality denominators."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

import test_dense

spec = importlib.util.spec_from_file_location('retrieval_cost',
    Path(__file__).resolve().parents[1]/'scripts/benchmark_retrieval_cost.py')
cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost)


class PoolViewTests(unittest.TestCase):
    def index(self):
        index = test_dense.small_index()
        for w, pmid in zip(index.windows, ['1', '1', '2', '3']):
            w['pmid'] = pmid
        return index

    def test_subset_cannot_leak_high_scoring_article_and_keeps_all_windows(self):
        source = self.index()
        view = cost.scoped_index(source, ['3', '1'])
        hits = view.search(test_dense.query_vector())
        self.assertEqual([h['pmid'] for h in hits], ['1', '3'])
        self.assertEqual(view.positions, {'3': [2], '1': [0, 1]})
        self.assertEqual(hits[0]['best_window']['window_id'], '1')
        np.testing.assert_array_equal(view.vectors, source.vectors[[0, 1, 3]])
        self.assertEqual(len(source.vectors), 4)
        self.assertEqual(list(source.articles), ['1', '2', '3'])

    def test_ties_follow_subset_order_without_reordering_source_windows(self):
        source = self.index()
        source.vectors[:] = test_dense.unit(.8)
        view = cost.scoped_index(source, ['3', '1'])
        hits = view.search(test_dense.query_vector())
        self.assertEqual([h['pmid'] for h in hits], ['3', '1'])
        self.assertEqual(hits[1]['best_window']['window_id'], '0')

    def test_fresh_vectors_replace_only_the_subset(self):
        source = self.index()
        vectors = np.stack([test_dense.unit(-.2)])
        view = cost.scoped_index(source, ['2'], vectors)
        self.assertAlmostEqual(view.search(test_dense.query_vector())[0]['score'], -.2)
        self.assertAlmostEqual(float(source.vectors[2, 0]), .7)
        for order in ([], ['missing'], ['1', '1']):
            with self.assertRaises(ValueError):
                cost.scoped_index(source, order)
        with self.assertRaises(ValueError):
            cost.scoped_index(source, ['1'], vectors)

    def test_quality_keeps_full_known_pool_denominator_and_missing_coverage(self):
        dense = {'rows': [{'query_id': 'q', 'pool': 10, 'pmids': ['1', '2'],
            'windows': 3, 'vector_bytes': 9216, 'runs': [
                {'ranked_pmids': ['2', '1'], 'cached_seconds': .2, 'fresh_total_seconds': .8},
                {'ranked_pmids': ['2', '1'], 'cached_seconds': .4, 'fresh_total_seconds': 1.2}]}]}
        med = {'rows': [{'query_id': 'q', 'pool': 10, 'depth': 10, 'method': 'dense_medcpt',
            'repeat': i, 'request_seconds': t, 'pair_count': 3, 'ranked_pmids': ['1', '2']}
            for i, t in enumerate([1., 2.])]}
        groups = cost.summarize(dense, med, {'q': {'1': 2, '2': None, 'outside': 2}})
        rerank = groups['dense_medcpt/pool-10/depth-10']
        self.assertEqual(rerank['macro']['recall@10']['mean'], .5)
        self.assertEqual(rerank['macro']['judged_coverage@10']['mean'], .5)
        self.assertEqual(rerank['fully_judged_queries'], [])
        self.assertAlmostEqual(rerank['cached_seconds']['median'], 1.8)
        self.assertAlmostEqual(rerank['fresh_seconds']['median'], 2.5)
        med['rows'][1]['ranked_pmids'] = ['2', '1']
        with self.assertRaises(ValueError):
            cost.summarize(dense, med, {'q': {'1': 2}})


if __name__ == '__main__':
    unittest.main()
