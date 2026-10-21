"""Check ranking against hand-computed scores and replay through the CLI."""

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

from medlens.ingestion.snapshots import download_snapshot
from medlens.parsing.dataset import LoadedDataset, build_dataset
from medlens.retrieval.bm25 import BM25Config, BM25Index, article_text, tokenize
from pubmed_helpers import make_client, search_json, xml_records


def dataset(texts):
    rows = tuple({"pmid": str(20 - i), "title": text, "abstract": None,
        "text_version": {"text_version_id": f"version-{i}"}}
        for i, text in enumerate(texts))
    return LoadedDataset({"dataset_id": "synthetic", "articles_sha256": "synthetic"}, rows)


class BM25Tests(unittest.TestCase):
    def test_tokenization_preserves_negation_abbreviations_numbers_and_single_letters(self):
        self.assertEqual(tokenize("  NOT\tCBT vs. SSRI: 5 mg, B-12, e\u0301 "),
            ["not", "cbt", "vs", "ssri", "5", "mg", "b", "12", "é"])

    def test_only_title_and_primary_abstract_text_are_indexed_once(self):
        row = {"title": "Depression", "abstract": {"sections": [
            {"label": "METHODS", "text": "CBT trial"}, {"text": "No difference"}]},
            "other_abstracts": [{"text": "unwanted"}], "book_title": "unwanted"}
        self.assertEqual(article_text(row), "Depression\nCBT trial\nNo difference")
        self.assertEqual(article_text({}), "")

    def test_lucene_scores_match_independent_formula_including_empty_document(self):
        index = BM25Index(dataset(["anxiety anxiety", "anxiety control control control", "depression", ""]))
        hits = index.search("anxiety", 10)
        self.assertEqual([hit.pmid for hit in hits], ["20", "19"])
        idf = math.log(1 + (4 - 2 + 0.5) / (2 + 0.5))
        for hit, tf, length in zip(hits, [2, 1], [2, 4]):
            expected = idf * tf / (tf + 1.5 * (0.25 + 0.75 * length / (7 / 4)))
            self.assertAlmostEqual(hit.score, expected, places=7)

    def test_ties_use_candidate_order_and_top_k_never_pads_with_nonmatches(self):
        index = BM25Index(dataset(["cbt", "cbt", "unrelated"]))
        self.assertEqual([h.pmid for h in index.search("CBT", 10)], ["20", "19"])
        self.assertEqual([h.rank for h in index.search("cbt", 1)], [1])
        self.assertEqual(index.search("unknown"), ())
        self.assertEqual(index.search("  --- "), ())

    def test_empty_dataset_and_empty_vocabulary(self):
        for texts in ([], ["", "---"]):
            self.assertEqual(BM25Index(dataset(texts)).search("anxiety"), ())

    def test_parameters_change_length_penalty_and_validate_inputs(self):
        source = dataset(["cbt", "cbt control control"])
        hits = BM25Index(source, BM25Config(b=0)).search("cbt")
        self.assertAlmostEqual(hits[0].score, hits[1].score)
        hits = BM25Index(source, BM25Config(b=1)).search("cbt")
        self.assertGreater(hits[0].score, hits[1].score)
        for kwargs in ({"k1": 0}, {"k1": float("nan")}, {"b": -1}, {"b": float("inf")}):
            with self.assertRaises(ValueError):
                BM25Config(**kwargs)
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                BM25Index(source).search("cbt", k)

    def test_repeated_terms_add_scores_and_unknown_terms_do_not(self):
        index = BM25Index(dataset(["cbt", "control"]))
        score = index.search("cbt")[0].score
        self.assertAlmostEqual(index.search("cbt cbt")[0].score, 2 * score)
        self.assertEqual(index.search("cbt unknown"), index.search("cbt"))

    def test_run_identity_and_source_links(self):
        index = BM25Index(dataset(["cbt", "control"]))
        first, second = index.run(" CBT "), index.run(" CBT ")
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["results"], second["results"])
        self.assertNotEqual(first["run_id"], index.run("control")["run_id"])
        self.assertNotEqual(first["run_id"], BM25Index(index.dataset, BM25Config(b=0)).run(" CBT ")["run_id"])
        self.assertEqual(first["query"], " CBT ")
        self.assertEqual(first["normalized_query"], "cbt")
        self.assertEqual(first["results"][0]["url"], "https://pubmed.ncbi.nlm.nih.gov/20/")
        self.assertEqual(first["results"][0]["text_version_id"], "version-0")

    def test_cli_offline_json_output_no_overwrite_and_corruption_rejection(self):
        script = Path(__file__).resolve().parents[1] / "scripts/bm25_search.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client, _, _ = make_client([search_json(["1"], 1), xml_records(["1"])])
            download_snapshot(client, "depression", root / "snapshot", limit=1)
            build_dataset(root / "snapshot", root / "parsed")
            report = root / "runs/search.json"
            args = [str(script), str(root / "parsed"), "--query", "1", "--output", str(report)]
            def invoke(argv):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                    with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                        with self.assertRaises(SystemExit) as exit:
                            runpy.run_path(str(script), run_name="__main__")
                return exit.exception.code, stdout.getvalue(), stderr.getvalue()
            code, stdout, _ = invoke(args)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout), json.loads(report.read_text()))
            self.assertEqual(json.loads(stdout)["candidate_count"], 1)
            original = report.read_bytes()
            self.assertEqual(invoke(args)[0], 1)
            self.assertEqual(report.read_bytes(), original)
            (root / "parsed/articles.jsonl").write_text("corrupt")
            code, stdout, stderr = invoke(args[:-2])
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("checksum", stderr)


if __name__ == "__main__":
    unittest.main()
