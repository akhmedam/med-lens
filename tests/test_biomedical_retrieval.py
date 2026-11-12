"""Fictional diagnostic guards; no biomedical labels or held-out inputs."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "biomedical_retrieval", Path(__file__).resolve().parents[1] / "scripts/evaluate_biomedical_retrieval.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_no_relevant_denominator_is_undefined(self):
        result = evaluation.metrics(["a", "b"], ["a", "b"], {"a": 0, "b": 0})
        self.assertIsNone(result["ndcg@10"])
        for threshold in ("useful", "direct"):
            self.assertIsNone(result[threshold]["pool_recall@10"])
            self.assertIsNone(result[threshold]["mrr@10"])
            self.assertEqual(result[threshold]["precision@10"], 0)

    def test_short_list_has_explicit_precision_denominators(self):
        result = evaluation.metrics(["a", "b"], ["a", "b"], {"a": 2, "b": 1})
        self.assertEqual(result["returned_count"], 2)
        self.assertEqual(result["useful"]["precision@10"], 0.2)
        self.assertEqual(result["useful"]["precision_among_returned"], 1)
        self.assertEqual(result["direct"]["precision@10"], 0.1)
        self.assertEqual(result["direct"]["precision_among_returned"], 0.5)

    def test_graded_ndcg_uses_union_pool_ideal(self):
        result = evaluation.metrics(["b"], ["a", "b"], {"a": 2, "b": 1})
        self.assertAlmostEqual(result["ndcg@10"], 1 / (3 + 1 / math.log2(3)))
        self.assertEqual(result["useful"]["pool_recall@10"], 0.5)
        self.assertEqual(result["direct"]["pool_recall@10"], 0)

    def test_missing_top_label_is_not_irrelevant(self):
        with self.assertRaisesRegex(evaluation.EvaluationError, "Unjudged"):
            evaluation.metrics(["a", "b"], ["a", "b"], {"a": 2})

    def test_unknown_label_is_not_zero(self):
        with self.assertRaisesRegex(evaluation.EvaluationError, "unknown"):
            evaluation.metrics(["a"], ["a"], {"a": None})

    def test_changed_source_version_is_rejected(self):
        item = {"query_id": "fictional", "pmid": "1", "item_id": "one",
                "text_version_id": "version-one", "item_sha256": "source-one"}
        row = {**item, "state": "rated", "relevance": 2}
        body = json.dumps({"query_id": "fictional", "pmid": "1", "relevance": 2}).encode()
        evaluation.bound_grades([item], [row], body)
        changed = {**row, "text_version_id": "version-two"}
        with self.assertRaisesRegex(evaluation.EvaluationError, "binding mismatch"):
            evaluation.bound_grades([item], [changed], body)

    def test_missing_label_and_duplicate_judgment_are_rejected(self):
        item = {"query_id": "fictional", "pmid": "1", "item_id": "one",
                "text_version_id": "v1", "item_sha256": "s1"}
        row = {**item, "state": "rated", "relevance": 1}
        body = json.dumps({"query_id": "fictional", "pmid": "1", "relevance": 1}).encode()
        for rows in ([], [row, deepcopy(row)]):
            with self.assertRaises(evaluation.EvaluationError):
                evaluation.bound_grades([item], rows, body)


if __name__ == "__main__":
    unittest.main()
