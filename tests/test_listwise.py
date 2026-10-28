"""A failed rerank must preserve membership and the original order."""

import json
import unittest
from medlens.reranking.listwise import build_prompt, apply_response


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return "".join(tokens)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["return_dict"] is False
        return list(json.dumps(messages))


class ListwiseTests(unittest.TestCase):
    def test_exact_permutation_and_failures(self):
        mapping = {"D1": "123", "D2": "456"}
        result = apply_response('["D2","D1"]', mapping)
        self.assertEqual(result["ranking"], ["456", "123"])
        self.assertFalse(result["fallback"])
        for text in ('["D1","D1"]', '["D1"]', '["D2","D3"]', '[1,2]',
                     '{"D1":1}', 'null', '```json\n["D1","D2"]\n```', ''):
            with self.subTest(text=text):
                row = apply_response(text, mapping)
                self.assertTrue(row["fallback"])
                self.assertEqual(row["ranking"], ["123", "456"])
        self.assertTrue(apply_response('["D2","D1"]', mapping, hit_output_limit=True)["fallback"])

    def test_label_free_prompt_and_explicit_truncation(self):
        article = {"pmid": "123", "title": "AB", "abstract": {"sections": [{"text": "CDEFG"}]},
                   "relevance": 2, "score": 0.99, "other_abstracts": [{"text": "SECRET"}]}
        prompt = build_prompt("question", [article], CharacterTokenizer(), document_tokens=5)
        data = json.loads(prompt["messages"][1]["content"])
        self.assertEqual(data["papers"], [{"id": "D1", "text": "AB\nCD"}])
        self.assertEqual(prompt["documents"][0]["original_tokens"], 8)
        self.assertTrue(prompt["documents"][0]["truncated"])
        self.assertEqual(prompt["mapping"], {"D1": "123"})
        self.assertNotIn("relevance", prompt["messages"][1]["content"])
        self.assertNotIn("SECRET", prompt["messages"][1]["content"])
        with self.assertRaisesRegex(ValueError, "input budget"):
            build_prompt("question", [article], CharacterTokenizer(), input_tokens=1)

    def test_missing_abstract_and_invalid_candidates(self):
        article = {"pmid": "123", "title": None, "abstract": None}
        self.assertFalse(build_prompt("q", [article], CharacterTokenizer())["documents"][0]["truncated"])
        for articles in ([], [article, article], [article] * 11):
            with self.assertRaises(ValueError):
                build_prompt("q", articles, CharacterTokenizer())
        with self.assertRaises(ValueError):
            build_prompt(" ", [article], CharacterTokenizer())


class FrozenEvaluationTests(unittest.TestCase):
    def test_report_binding_and_metric_replay(self):
        import importlib.util
        import hashlib
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from medlens.evaluation.metrics import evaluate
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("evaluate_reranking", root / "scripts/evaluate_reranking.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            query_path, qrel_path, report_path = [base / f for f in ("queries.jsonl", "qrels.jsonl", "run.json")]
            query_path.write_text(json.dumps({"query_id": "p01", "text": "question", "split": "pilot",
                "group_id": "g01", "dataset": "dataset"}) + "\n")
            pmids = [str(i) for i in range(1, 11)]
            qrel_path.write_text("".join(json.dumps({"query_id": "p01", "pmid": p, "relevance": 2}) + "\n" for p in pmids))
            metrics = evaluate(pmids, pmids, {p: 2 for p in pmids}, 10)
            report = {"queries_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest(),
                "qrels_sha256": hashlib.sha256(qrel_path.read_bytes()).hexdigest(),
                "k": 10, "split": "pilot", "comparison_status": "fully-judged-top-k", "run_id": "run",
                "per_query": [{"query_id": "p01", "text": "question", "dataset_id": "dataset",
                    "methods": {m: {"ranking": [{"pmid": p} for p in pmids], "metrics": dict(metrics)}
                                for m in ("dense", "rrf")}}]}
            dataset = SimpleNamespace(manifest={"dataset_id": "dataset"}, articles=[{"pmid": p} for p in pmids])
            with patch.object(module, "load_dataset", return_value=dataset):
                report_path.write_text(json.dumps(report))
                jobs, _, _ = module.prepare(query_path, qrel_path, report_path)
                self.assertEqual(len(jobs), 2)
                report["per_query"][0]["methods"]["dense"]["metrics"]["ndcg@10"] = 0.5
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "metrics"):
                    module.prepare(query_path, qrel_path, report_path)
                report["queries_sha256"] = "wrong"
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, "hashes"):
                    module.prepare(query_path, qrel_path, report_path)
