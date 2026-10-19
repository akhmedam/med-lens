"""Protocol and failure cases; no live NCBI calls."""

import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import parse_qs

from medlens.ingestion.pubmed import ClientConfig, PubMedError, parse_search, validate_fetch
from pubmed_helpers import http_error, make_client, search_json, xml_records


class ClientTests(unittest.TestCase):
    def test_posts_query_and_batch_without_hidden_filters(self):
        client, _, transport = make_client([search_json(), xml_records(["31", "12"])])
        query = "depression AND CBT + anxiety"
        client.search(query, 30)
        client.fetch(["31", "12"])
        _, request, timeout = transport.calls[0]
        params = parse_qs(request.data.decode())
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(params["term"], [query])
        self.assertEqual(params["retmax"], ["30"])
        self.assertEqual(params["sort"], ["relevance"])
        self.assertNotIn("mindate", params)
        self.assertEqual(timeout, 20)
        self.assertEqual(parse_qs(transport.calls[1][1].data.decode())["id"], ["31,12"])

    def test_rate_limit_applies_across_search_and_fetch(self):
        client, _, transport = make_client([b"one", b"two", b"three"])
        client.search("depression")
        client.fetch(["1"])
        client.search("anxiety")
        times = [call[0] for call in transport.calls]
        self.assertAlmostEqual(times[1] - times[0], 0.4)
        self.assertAlmostEqual(times[2] - times[1], 0.4)

    def test_retries_rate_limit_and_respects_retry_after(self):
        client, clock, _ = make_client([http_error(429, "6"), search_json()])
        response = client.search("depression")
        self.assertEqual(response.attempts, 2)
        self.assertEqual(clock.sleeps, [6])

    def test_respects_retry_after_http_date(self):
        now = datetime(2026, 10, 19, 8, tzinfo=timezone.utc)
        client, clock, _ = make_client([http_error(429, format_datetime(now + timedelta(seconds=12))), search_json()])
        with patch("medlens.ingestion.pubmed.datetime") as wall_clock:
            wall_clock.now.return_value = now
            self.assertEqual(client.search("depression").attempts, 2)
        self.assertEqual(clock.sleeps, [12])

    def test_retries_timeout_and_server_failure_with_backoff(self):
        client, clock, _ = make_client([TimeoutError(), http_error(503), search_json()])
        self.assertEqual(client.search("depression").attempts, 3)
        self.assertEqual(clock.sleeps, [1, 2])

    def test_fails_after_transport_retry_budget(self):
        client, _, transport = make_client([URLError("fixture")] * 3)
        with self.assertRaisesRegex(PubMedError, "after 3"):
            client.search("depression")
        self.assertEqual(len(transport.calls), 3)

    def test_does_not_retry_permanent_http_errors(self):
        client, _, transport = make_client([http_error(400)])
        with self.assertRaisesRegex(PubMedError, "HTTP 400"):
            client.search("depression")
        self.assertEqual(len(transport.calls), 1)

    def test_does_not_retry_earlier_than_long_cooldown(self):
        client, clock, _ = make_client([http_error(429, "120")])
        with self.assertRaisesRegex(PubMedError, "try later"):
            client.search("depression")
        self.assertEqual(clock.sleeps, [])

    def test_rejects_invalid_requests_before_network(self):
        client, _, transport = make_client([])
        for invoke in (lambda: client.search(" "), lambda: client.search("x", 0),
                       lambda: client.fetch([]), lambda: client.fetch(["1", "1"]),
                       lambda: client.fetch(["-1"]), lambda: client.fetch(["１２"]),
                       lambda: ClientConfig(requests_per_second=4)):
            with self.subTest(invoke=invoke), self.assertRaises((PubMedError, ValueError)):
                invoke()
        self.assertEqual(transport.calls, [])


class ResponseTests(unittest.TestCase):
    def test_preserves_search_order_and_translation(self):
        result = parse_search(search_json(), 3)
        self.assertEqual(result.pmids, ("31", "12", "25"))
        self.assertEqual(result.total_count, 8)
        self.assertEqual(result.query_translation, "translated query")

    def test_empty_search_is_valid(self):
        self.assertEqual(parse_search(search_json((), 0), 30).pmids, ())

    def test_rejects_bad_or_partial_search_payloads(self):
        for body in (b"<html>error</html>", b"[]", b'{"error":"rate limit"}',
                     search_json(["1", "1"], 2), search_json(["1"], 8),
                     b'{"esearchresult":{"count":"1","idlist":["1"],"errorlist":{"fieldnotfound":["bad"]}}}'):
            with self.subTest(body=body), self.assertRaises(PubMedError):
                parse_search(body, 3)

    def test_checks_primary_pmids_without_assuming_fetch_order(self):
        self.assertEqual(validate_fetch(xml_records(["12", "31"]), ["31", "12"]), ("12", "31"))

    def test_accepts_book_records(self):
        body = b"<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>2</PMID></BookDocument></PubmedBookArticle></PubmedArticleSet>"
        self.assertEqual(validate_fetch(body, ["2"]), ("2",))

    def test_rejects_partial_duplicate_unexpected_and_malformed_fetch(self):
        for body in (xml_records(["31"]), xml_records(["31", "31"]),
                     xml_records(["31", "99"]), b"<PubmedArticleSet>",
                     b"<html>maintenance</html>", b"<ERROR>server error</ERROR>"):
            with self.subTest(body=body), self.assertRaises(PubMedError):
                validate_fetch(body, ["31", "12"])


if __name__ == "__main__":
    unittest.main()
