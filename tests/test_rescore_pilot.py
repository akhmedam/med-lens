"""Changing qrels must not change or silently substitute the frozen rankings."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medlens.evaluation.metrics import evaluate

spec = importlib.util.spec_from_file_location("rescore_pilot", Path(__file__).resolve().parents[1] / "scripts/rescore_pilot.py")
rescore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore)


class RescoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pmids = [str(i) for i in range(1, 11)]
        self.profile = {"schema_version": 1, "split": "pilot", "evaluation_version": "v2", "annotation": {"method": "test"}}
        query = {"query_id": "p01", "text": "question", "split": "pilot", "group_id": "g", "dataset": "dataset"}
        self.write("queries", (json.dumps(query) + "\n").encode())
        def qrels(null=False):
            return b"".join((json.dumps({"query_id": "p01", "pmid": p,
                "relevance": None if null and p == "1" else 2}) + "\n").encode() for p in self.pmids)
        self.write("source_qrels", qrels())
        self.write("qrels", qrels(True))
        metrics = evaluate(self.pmids, self.pmids, {p: 2 for p in self.pmids}, 10)
        methods = {m: {"ranking": [{"pmid": p} for p in self.pmids], "metrics": metrics}
                   for m in ("pubmed", "bm25", "dense", "rrf")}
        self.base = {"queries_sha256": self.profile["queries"]["sha256"],
            "qrels_sha256": self.profile["source_qrels"]["sha256"], "k": 10, "split": "pilot", "run_id": "source",
            "per_query": [{"query_id": "p01", "text": "question", "dataset_id": "dataset", "methods": methods}]}
        self.write("frozen_retrieval", json.dumps(self.base).encode())
        mapping = {f"D{i}": p for i, p in enumerate(self.pmids, 1)}
        rows = [{"query_id": "p01", "source": m, "dataset_id": "dataset", "before_ranking": self.pmids,
            "ranking": list(reversed(self.pmids)), "before": metrics, "after": metrics, "fallback": False,
            "raw_output": json.dumps(list(reversed(mapping))), "prompt": {"mapping": mapping},
            "generation": {"hit_output_limit": False}} for m in ("dense", "rrf")]
        self.llm = {"queries_sha256": self.base["queries_sha256"], "qrels_sha256": self.base["qrels_sha256"],
            "baseline_sha256": self.profile["frozen_retrieval"]["sha256"], "status": "complete", "per_query": rows}
        self.write("frozen_reranking", json.dumps(self.llm).encode())
        self.dataset = SimpleNamespace(manifest={"dataset_id": "dataset"}, articles=[{"pmid": p} for p in self.pmids])

    def write(self, name, body):
        (self.root / name).write_bytes(body)
        self.profile[name] = {"path": name, "sha256": hashlib.sha256(body).hexdigest()}

    def run_profile(self):
        path = self.root / "profile.json"
        path.write_text(json.dumps(self.profile))
        with patch.object(rescore, "load_dataset", return_value=self.dataset):
            return rescore.rescore(path, root=self.root)

    def test_new_metrics_keep_rankings_and_unknowns(self):
        report, retrieval, summary = self.run_profile()
        self.assertEqual(report["comparison_status"], "provisional-incomplete-judgments")
        self.assertEqual(report["per_query"][0]["methods"]["dense"]["ranking"], self.pmids)
        self.assertEqual(report["per_query"][0]["methods"]["dense_qwen"]["ranking"], list(reversed(self.pmids)))
        self.assertLess(report["macro"]["dense"]["ndcg@10"]["mean"], 1)
        self.assertIsNone(retrieval["per_query"][0]["methods"]["dense"]["ranking"][0]["relevance"])
        self.assertNotIn("ranking", summary["per_query"][0]["metrics"]["dense"])
        self.assertNotEqual(retrieval["run_id"], self.base["run_id"])

    def test_changed_bytes_rejected(self):
        (self.root / "qrels").write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_profile()

    def test_inconsistent_model_output_rejected_even_with_updated_file_hash(self):
        self.llm["per_query"][0]["raw_output"] = '["invented"]'
        self.write("frozen_reranking", json.dumps(self.llm).encode())
        with self.assertRaisesRegex(ValueError, "reranking output"):
            self.run_profile()
