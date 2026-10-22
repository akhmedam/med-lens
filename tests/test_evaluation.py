"""Hand-calculated metric cases and offline evaluation replay."""

import io
import json
import math
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from medlens.evaluation.formats import EvaluationError, parse_queries, parse_qrels
from medlens.evaluation.metrics import evaluate, macro_average
from medlens.evaluation.runner import run_evaluation
from medlens.ingestion.snapshots import download_snapshot
from medlens.parsing.dataset import build_dataset
from pubmed_helpers import make_client, search_json


def jsonl(rows):
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


def query(qid="q1", split="pilot", group="g1", dataset="parsed"):
    return {"query_id": qid, "text": "anxiety", "split": split,
            "group_id": group, "dataset": dataset}


def synthetic_inputs(root):
    """Fictional records and labels; not a biomedical relevance benchmark."""
    pmids = ["102", "103", "101"]
    titles = ["Depression trial", "Anxiety control", "Anxiety anxiety therapy"]
    records = "".join(f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
        f"<Article><ArticleTitle>{title}</ArticleTitle></Article>"
        "</MedlineCitation></PubmedArticle>" for pmid, title in zip(pmids, titles))
    client, _, _ = make_client([search_json(pmids, 3),
        f"<PubmedArticleSet>{records}</PubmedArticleSet>".encode()])
    download_snapshot(client, "synthetic fixture", root / "snapshot", limit=3)
    build_dataset(root / "snapshot", root / "parsed")
    rows = [query(), {**query("q2", group="g2"), "text": "depression"}]
    judgments = [{"query_id": "q1", "pmid": pmid, "relevance": grade}
        for pmid, grade in (("101", 2), ("103", 1), ("102", 0), ("999", 2))]
    judgments += [{"query_id": "q2", "pmid": pmid, "relevance": grade}
        for pmid, grade in (("101", 0), ("103", 0), ("102", 2))]
    (root / "queries.jsonl").write_bytes(jsonl(rows))
    (root / "qrels.jsonl").write_bytes(jsonl(judgments))
    return root / "queries.jsonl", root / "qrels.jsonl"


class MetricTests(unittest.TestCase):
    def test_graded_ndcg_binary_metrics_and_global_recall_denominator(self):
        result = evaluate(["2", "3", "1"], ["1", "2", "3"],
            {"1": 2, "2": 1, "3": 0, "4": 2}, k=3)
        dcg = 1 + 3 / math.log2(4)
        ideal = 3 + 3 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(result["ndcg@3"], dcg / ideal)
        self.assertEqual(result["precision@3"], 2 / 3)
        self.assertEqual(result["recall@3"], 2 / 3)
        self.assertEqual(result["mrr@3"], 1)
        self.assertEqual(result["candidate_coverage"], 2 / 3)

    def test_perfect_ranking_and_cutoff(self):
        result = evaluate(["1", "2", "3"], ["1", "2", "3"], {"1": 2, "2": 1, "3": 2}, k=1)
        self.assertEqual(result["ndcg@1"], 1)
        self.assertEqual(result["recall@1"], 1 / 3)
        self.assertEqual(result["returned_count"], 1)

    def test_unjudged_is_distinct_from_zero_and_coverage_uses_returned_slots(self):
        result = evaluate(["1", "2", "3"], ["1", "2", "3"], {"1": 0, "2": None}, k=10)
        self.assertEqual(result["precision@10"], 0)
        self.assertEqual(result["judged_coverage@10"], 1 / 3)
        self.assertEqual(result["unjudged_returned_count"], 2)
        self.assertIsNone(result["ndcg@10"])
        self.assertIsNone(result["recall@10"])
        self.assertIsNone(result["mrr@10"])
        empty = evaluate(["1"], ["1"], {"1": None})
        self.assertIsNone(empty["precision@10"])

    def test_short_list_precision_uses_k_and_mrr_uses_first_relevant_rank(self):
        result = evaluate(["1", "2"], ["1", "2"], {"1": 0, "2": 1})
        self.assertEqual(result["precision@10"], 0.1)
        self.assertEqual(result["mrr@10"], 0.5)
        self.assertEqual(result["recall@10"], 1)

    def test_empty_ranking_and_candidates(self):
        result = evaluate([], [], {"1": 2})
        self.assertEqual(result["ndcg@10"], 0)
        self.assertEqual(result["candidate_coverage"], 0)
        self.assertIsNone(result["judged_coverage@10"])
        self.assertIsNone(result["candidate_judged_coverage"])

    def test_rejects_duplicate_out_of_pool_rankings_and_invalid_grades(self):
        for ranking, candidates, qrels, k in ((["1", "1"], ["1"], {}, 10),
                (["2"], ["1"], {}, 10), ([], ["1", "1"], {}, 10),
                ([], [], {"1": True}, 10), ([], [], {}, 0)):
            with self.assertRaises(EvaluationError):
                evaluate(ranking, candidates, qrels, k)

    def test_macro_excludes_undefined_values_and_reports_denominators(self):
        rows = [evaluate(["1"], ["1"], {"1": 2}), evaluate([], [], {})]
        result = macro_average(rows, 10)
        self.assertEqual(result["ndcg@10"], {"mean": 1, "defined_queries": 1, "total_queries": 2})


class FormatTests(unittest.TestCase):
    def test_relative_dataset_path_and_nullable_grade(self):
        queries = parse_queries(jsonl([query()]), Path("/tmp/evaluation"))
        self.assertEqual(queries[0].dataset, Path("/tmp/evaluation/parsed").resolve())
        labels = parse_qrels(jsonl([{"query_id": "q1", "pmid": "1", "relevance": None}]), queries)
        self.assertEqual(labels, {"q1": {"1": None}})

    def test_duplicate_queries_and_group_leakage_fail(self):
        for rows in ([query(), query()], [query(), query("q2", split="test")],
                     [query(split="training")], [{**query(), "text": " "}], []):
            with self.assertRaises(EvaluationError):
                parse_queries(jsonl(rows), Path("."))

    def test_rejects_bad_labels_unknown_queries_and_duplicate_pairs(self):
        queries = parse_queries(jsonl([query()]), Path("."))
        valid = {"query_id": "q1", "pmid": "1", "relevance": 2}
        for rows in ([valid, valid], [{**valid, "query_id": "unknown"}],
                     [{**valid, "pmid": "abc"}], [{**valid, "pmid": "0"}],
                     [{**valid, "relevance": True}], [{**valid, "relevance": 1.0}],
                     [{**valid, "relevance": -1}], [{**valid, "extra": "typo"}]):
            with self.assertRaises(EvaluationError):
                parse_qrels(jsonl(rows), queries)

    def test_invalid_json_and_non_object_rows(self):
        for body in (b"{", b"[]", b"\xff"):
            with self.assertRaises(EvaluationError):
                parse_queries(body, Path("."))


class RunnerTests(unittest.TestCase):
    def test_replay_same_pool_stable_identity_and_outside_candidate_judgment(self):
        with tempfile.TemporaryDirectory() as temp:
            queries, qrels = synthetic_inputs(Path(temp))
            with patch("socket.socket", side_effect=AssertionError("offline")):
                first, second = run_evaluation(queries, qrels), run_evaluation(queries, qrels)
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["macro"], second["macro"])
            self.assertEqual(first["query_count"], 2)
            q1 = first["per_query"][0]
            self.assertEqual(q1["methods"]["pubmed"]["ranking"][0]["pmid"], "102")
            self.assertEqual(q1["methods"]["bm25"]["ranking"][0]["pmid"], "101")
            self.assertEqual(q1["methods"]["bm25"]["metrics"]["candidate_coverage"], 2 / 3)
            self.assertGreater(first["macro"]["bm25"]["ndcg@10"]["mean"],
                               first["macro"]["pubmed"]["ndcg@10"]["mean"])
            self.assertNotEqual(first["run_id"], run_evaluation(queries, qrels, k=1)["run_id"])
            qrels.write_bytes(b"")
            unlabeled = run_evaluation(queries, qrels)
            self.assertIsNone(unlabeled["macro"]["bm25"]["ndcg@10"]["mean"])
            self.assertFalse(unlabeled["per_query"][0]["methods"]["bm25"]["ranking"][0]["judged"])
            self.assertNotEqual(first["run_id"], unlabeled["run_id"])

    def test_split_selection_never_loads_held_out_dataset_and_empty_split_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            queries, qrels = synthetic_inputs(Path(temp))
            with queries.open("ab") as stream:
                stream.write(jsonl([query("heldout", "test", "heldout", "missing-dataset")]))
            self.assertEqual(run_evaluation(queries, qrels)["query_count"], 2)
            with self.assertRaisesRegex(EvaluationError, "No queries"):
                run_evaluation(queries, qrels, split="dev")

    def test_cli_saves_report_offline_refuses_overwrite_and_publishes_no_partial_result(self):
        script = Path(__file__).resolve().parents[1] / "scripts/evaluate_retrieval.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queries, qrels = synthetic_inputs(root)
            output = root / "result.json"
            args = [str(script), str(queries), "--qrels", str(qrels), "--output", str(output)]
            def invoke():
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", args), redirect_stdout(stdout), redirect_stderr(stderr):
                    with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                        with self.assertRaises(SystemExit) as result:
                            runpy.run_path(str(script), run_name="__main__")
                return result.exception.code, stdout.getvalue(), stderr.getvalue()
            code, stdout, _ = invoke()
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["run_id"], json.loads(output.read_text())["run_id"])
            original = output.read_bytes()
            self.assertEqual(invoke()[0], 1)
            self.assertEqual(output.read_bytes(), original)
            output.unlink()
            qrels.write_text("invalid")
            self.assertEqual(invoke()[0], 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
