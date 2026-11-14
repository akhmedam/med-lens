from copy import deepcopy
import unittest

from medlens.cli_output import render_evidence, render_result, safe_text


def job(answer_status="partial"):
    return {"request_id": "a" * 32, "status": "succeeded", "stage": "complete",
        "request": {"question": "How does treatment help?"},
        "result": {"request_seconds": 12.25, "independent_validation": False,
            "search": {"fetched_article_count": 10, "total_pubmed_matches": 120},
            "question": "How does treatment help?",
            "articles": [{"rank": 1, "pmid": "123", "title": "Study title",
                          "score": 0.8561, "url": "https://pubmed.ncbi.nlm.nih.gov/123/"}],
            "answer": {"status": answer_status, "claims": [{"claim_id": "C1", "text": "Finding text",
                "citations": [{"citation_id": "S1", "quote_id": "Q1", "quote": "Exact quotation",
                    "pmid": "123", "char_start": 12, "char_end": 27,
                    "text_version_id": "text-v1", "source_path": "/PubmedArticle/MedlineCitation/Abstract",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/123/"}],
                "scope_cautions": [{"code": "population_unclear", "message": "Eligibility is uncertain."}]}]},
            "evidence": [{"citation_id": "S1", "pmid": "123", "title": "Study title",
                "char_start": 0, "char_end": 70, "section": "RESULTS", "text": "Full source passage",
                "quote_id": "Q1", "text_version_id": "text-v1",
                "source_path": "/PubmedArticle/MedlineCitation/Abstract",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/"}]}}


class TerminalOutputTests(unittest.TestCase):
    def test_full_answer_simple_citations_pubmed_counts_and_ranked_articles(self):
        value = job()
        before = deepcopy(value)
        text = render_result(value)
        for expected in ["частичный исследовательский черновик", "Finding text", "Exact quotation",
                         "S1", "PMID 123", "PubMed: загружено 10, совпадений 120",
                         "1. Study title", "https://pubmed.ncbi.nlm.nih.gov/123/",
                         "Eligibility is uncertain.", "независимо не проверены", "12.25 с"]:
            self.assertIn(expected, text)
        for technical in ["Q1", "символы 12–27", "text-v1", "Раздел XML", "0.856", "косинусная"]:
            self.assertNotIn(technical, text)
        self.assertNotIn("Full source passage", text)
        self.assertEqual(value, before)

    def test_evidence_is_complete_with_simple_source_reference(self):
        value = job()
        passage = "evidence words " * 1000
        value["result"]["evidence"][0]["text"] = passage
        text = render_evidence(value)
        self.assertIn(passage, text)
        self.assertIn("RESULTS", text)
        self.assertIn("S1", text)
        self.assertIn("PMID 123", text)
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/123/", text)
        for technical in ["Q1", "символы 0–70", "text-v1", "Раздел XML", "/PubmedArticle"]:
            self.assertNotIn(technical, text)

    def test_pubmed_counts_accept_only_nonnegative_integers(self):
        for counts in [{"fetched_article_count": True, "total_pubmed_matches": float("inf")},
                       {"fetched_article_count": -1, "total_pubmed_matches": "120"}]:
            value = job()
            value["result"]["search"] = counts
            self.assertNotIn("PubMed:", render_result(value))
        value["result"]["search"] = {"fetched_article_count": 0}
        self.assertIn("PubMed: загружено 0", render_result(value))

    def test_claim_and_quotes_are_not_truncated(self):
        value = job()
        claim, quote = "claim " * 1000, "quotation " * 1000
        value["result"]["answer"]["claims"][0]["text"] = claim
        value["result"]["answer"]["claims"][0]["citations"][0]["quote"] = quote
        text = render_result(value)
        self.assertIn(claim, text)
        self.assertIn(quote, text)

    def test_validation_failure_suppresses_claims_but_keeps_articles(self):
        text = render_result(job("validation_failed"))
        self.assertIn("ответ скрыт", text)
        self.assertNotIn("Finding text", text)
        self.assertNotIn("Exact quotation", text)
        self.assertIn("Study title", text)

    def test_abstention_search_only_and_empty_results_are_distinct(self):
        for status, expected in [("insufficient_evidence", "недостаточно доказательств"),
                                 ("not_requested", "только поиск")]:
            text = render_result(job(status))
            self.assertIn(expected, text)
            self.assertNotIn("Finding text", text)
        value = job()
        value["result"]["articles"] = []
        value["result"]["evidence"] = []
        self.assertIn("Статьи отсутствуют", render_result(value))
        self.assertIn("Фрагменты отсутствуют", render_evidence(value))
        value["result"]["answer"].update(status="insufficient_evidence", reason="empty-pubmed-result")
        self.assertIn("PubMed не вернул статей", render_result(value))

    def test_failed_job_retains_partial_answer_and_sanitized_error(self):
        value = job()
        value.update(status="failed", error={"code": "inference_failed", "stage": "answer",
                                            "message": "The answer model stage failed."})
        text = render_result(value)
        self.assertIn("Выполнение: ошибка", text)
        self.assertIn("inference_failed", text)
        self.assertIn("The answer model stage failed.", text)
        self.assertIn("Exact quotation", text)

    def test_source_findings_show_quotes_and_verification_limits(self):
        value = job()
        answer = value["result"]["answer"]
        answer.update(answer_style="source-findings", quote_identity_verified=True,
                      semantic_support_verified=False, answer_completeness_established=False)
        answer["claims"][0]["source_title"] = "Original study"
        text = render_result(value)
        for expected in ["частичная подборка цитат", "Original study", "Exact quotation",
                         "точное совпадение", "Семантическая поддержка", "Полнота ответа не установлена"]:
            self.assertIn(expected, text)
        self.assertNotIn("Finding text", text)

    def test_all_terminal_controls_and_surrogates_are_escaped(self):
        controls = "".join(chr(code) for code in [*range(32), *range(127, 160)])
        value = "abc" + controls + "\ud800\udfff\u202e\u2066Ж"
        rendered = safe_text(value)
        self.assertTrue(all(ord(char) >= 32 and not 127 <= ord(char) <= 159
                            and not 0xD800 <= ord(char) <= 0xDFFF for char in rendered))
        self.assertIn("\\x1b", rendered)
        self.assertIn("\\x0d", rendered)
        self.assertIn("\\x9b", rendered)
        self.assertIn("\\ud800", rendered)
        self.assertIn("\\u202e", rendered)
        self.assertTrue(rendered.endswith("Ж"))
        rendered.encode("utf-8")

    def test_untrusted_fields_are_safe_and_private_fields_are_ignored(self):
        value = job()
        value["result"]["articles"][0]["title"] = "title\x1b[2J\r\ud800"
        value["result"]["answer"]["claims"][0]["citations"][0]["quote"] = "quote\x9b2J"
        value["result"]["evidence"][0]["text"] = "passage\x07"
        value.update(private_error="PRIVATE_SENTINEL", report_html="PRIVATE_SENTINEL")
        value["result"].update(model_calls=["PRIVATE_SENTINEL"], draft="PRIVATE_SENTINEL")
        for text in [render_result(value), render_evidence(value)]:
            self.assertNotIn("\x1b", text)
            self.assertNotIn("\r", text)
            self.assertNotIn("\x9b", text)
            self.assertNotIn("\x07", text)
            self.assertNotIn("\ud800", text)
            self.assertNotIn("PRIVATE_SENTINEL", text)

    def test_pending_or_missing_result_has_no_traceback(self):
        text = render_result({"request_id": "b" * 32, "status": "running", "result": None})
        self.assertIn("Выполнение: выполняется", text)
        self.assertIn("пока недоступен", text)
        self.assertIn("Фрагменты отсутствуют", render_evidence({}))


if __name__ == "__main__":
    unittest.main()
