"""Context limits and citations must refer to exact, unmodified source spans."""

import copy
import json
from types import SimpleNamespace
import unittest

from medlens.generation.context import ContextConfig, build_context, locate_quote, render_evidence


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def fixtures():
    articles, windows = [], []
    for pmid in ("1", "2", "3"):
        text = "alpha beta gamma delta alpha 🙂 final"
        provenance = {"source": "pubmed", "record_index": int(pmid)}
        articles.append({"pmid": pmid, "title": "Title " + pmid, "journal": None, "dates": [],
            "abstract": {"sections": [{"index": 0, "label": "RESULTS", "text": text, "source_path": "abstract/0"}]},
            "text_version": {"text_version_id": "v" + pmid}, "provenance": provenance})
        for n, (start, end) in enumerate(((0, 16), (6, 22), (23, len(text)))):
            windows.append({"pmid": pmid, "window_id": pmid + "-" + str(n), "text_version_id": "v" + pmid,
                "title": "Title " + pmid, "kind": "abstract", "section_index": 0,
                "section_label": "RESULTS", "source_path": "abstract/0", "char_start": start, "char_end": end,
                "text": text[start:end], "provenance": provenance})
    dataset = SimpleNamespace(manifest={"dataset_id": "d", "articles_sha256": "a"}, articles=articles)
    manifest = {"dataset_id": "d", "articles_sha256": "a", "artifact_id": "w"}
    return dataset, manifest, windows


class EvidenceTests(unittest.TestCase):
    def build(self, ranking=("2", "1"), config=None, windows=None, scores=None):
        dataset, manifest, original = fixtures()
        return build_context(dataset, manifest, windows if windows is not None else original, list(ranking),
            scores if scores is not None else [.9, .8, .7] * 3,
            CharacterTokenizer(), {"revision": "test"}, config=config)

    def test_diversity_then_second_nonoverlapping_window(self):
        r = self.build()
        self.assertEqual([s["pmid"] for s in r["sources"]], ["2", "1", "2", "1"])
        self.assertEqual([s["window_id"] for s in r["sources"]], ["2-0", "1-0", "2-2", "1-2"])
        self.assertTrue(any(s["reason"] == "overlapping-span" for s in r["skipped"]))
        payload = json.loads(r["evidence_payload"])
        self.assertNotIn("window_cosine", payload["sources"][0])
        self.assertNotIn("article_rank", payload["sources"][0])
        self.assertEqual(r["evidence_tokens"], len(r["evidence_payload"]))
        self.assertEqual(r["context_id"], self.build()["context_id"])

    def test_budget_skips_whole_excerpt_without_truncating(self):
        one = self.build(config=ContextConfig(max_fragments=1))
        budget = one["evidence_tokens"]
        r = self.build(config=ContextConfig(evidence_tokens=budget))
        self.assertEqual(len(r["sources"]), 1)
        self.assertEqual(r["sources"][0]["text"], "alpha beta gamma")
        self.assertLessEqual(r["evidence_tokens"], budget)
        self.assertTrue(any(s["reason"] == "token-budget" for s in r["skipped"]))
        self.assertEqual(len(self.build(config=ContextConfig(max_fragments=2))["sources"]), 2)
        self.assertEqual(len(self.build(config=ContextConfig(max_per_article=1))["sources"]), 2)

    def test_title_only_is_not_abstract_evidence(self):
        _, _, windows = fixtures()
        for w in windows:
            w.update(kind="title", text="")
        r = self.build(windows=windows)
        self.assertEqual(r["status"], "no-source-excerpts")
        self.assertEqual(r["evidence_payload"], render_evidence([]))

    def test_source_corruption_and_bad_ranking_rejected(self):
        for changes in ({"text": "fabricated"}, {"text_version_id": "old"}, {"char_start": True},
                        {"provenance": {}}, {"source_path": "other"}, {"section_index": 9}):
            _, _, windows = fixtures()
            windows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.build(windows=windows)
        for ranking in (("1", "1"), ("99",)):
            with self.assertRaises(ValueError):
                self.build(ranking=ranking)
        for scores in ([float('nan')] * 9, [1.1] * 9, [0.] * 8):
            with self.assertRaises(ValueError):
                self.build(scores=scores)
        with self.assertRaises(ValueError):
            self.build(config=ContextConfig(evidence_tokens=1))

    def test_quote_is_verbatim_and_bound_to_one_citation(self):
        r = self.build()
        source = r["sources"][0]
        q = locate_quote(r, source["citation_id"], "beta")
        self.assertEqual(q["pmid"], "2")
        self.assertEqual(q["matches"], [{"char_start": 6, "char_end": 10}])
        unicode_source = r["sources"][2]
        self.assertEqual(locate_quote(r, unicode_source["citation_id"], "🙂")["matches"][0]["char_end"],
            fixtures()[0].articles[0]["abstract"]["sections"][0]["text"].index("🙂") + 1)
        for cid, quote in (("E99", "beta"), ("E1", "BETA"), ("E1", "alpha  beta"), ("E1", " ")):
            with self.subTest(cid=cid, quote=quote), self.assertRaises(ValueError):
                locate_quote(r, cid, quote)
        altered = copy.deepcopy(r)
        altered["sources"][0]["text"] = "yes yes"
        match = locate_quote(altered, "E1", "yes")
        self.assertTrue(match["ambiguous"])
        self.assertEqual(len(match["matches"]), 2)

    def test_negative_scores_retained_and_ties_use_source_order(self):
        r = self.build(scores=[-.5] * 9)
        self.assertEqual(r["sources"][0]["window_id"], "2-0")
        self.assertEqual(r["sources"][0]["window_cosine"], -.5)
