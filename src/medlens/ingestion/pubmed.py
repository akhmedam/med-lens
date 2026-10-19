"""Read-only NCBI E-Utilities client. Article parsing belongs elsewhere."""

from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class PubMedError(RuntimeError):
    """A request or response cannot be used as a valid candidate set."""


@dataclass(frozen=True)
class ClientConfig:
    timeout: float = 20.0
    requests_per_second: float = 2.5
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    max_retry_wait: float = 60.0

    def __post_init__(self) -> None:
        for name in ("timeout", "requests_per_second", "backoff_seconds", "max_retry_wait"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.requests_per_second > 3:
            raise ValueError("This client has no API key; requests_per_second must be <= 3")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")


@dataclass(frozen=True)
class RawResponse:
    endpoint: str
    parameters: dict[str, str]
    body: bytes
    retrieved_at: str
    attempts: int
    content_type: str


@dataclass(frozen=True)
class SearchResult:
    total_count: int
    pmids: tuple[str, ...]
    query_translation: str | None


def validate_pmids(pmids: list[str] | tuple[str, ...]) -> None:
    if not all(isinstance(p, str) and p.isascii() and p.isdigit() and int(p) > 0 for p in pmids):
        raise PubMedError("PMIDs must be positive decimal strings")
    if len(set(pmids)) != len(pmids):
        raise PubMedError("Duplicate PMIDs")


class PubMedClient:
    """Use one sequential client per job; its limiter is not shared across processes."""

    def __init__(
        self,
        config: ClientConfig | None = None,
        *,
        opener: Callable = urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or ClientConfig()
        self._opener, self._sleep, self._clock = opener, sleep, monotonic
        self._next_request = 0.0

    def search(self, query: str, limit: int = 30) -> RawResponse:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        if type(limit) is not int or not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        return self._request("esearch.fcgi", {
            "db": "pubmed", "term": query, "retmax": str(limit),
            "retstart": "0", "retmode": "json", "sort": "relevance", "tool": "medlens",
        })

    def fetch(self, pmids: list[str] | tuple[str, ...]) -> RawResponse:
        validate_pmids(pmids)
        if not 1 <= len(pmids) <= 200:
            raise ValueError("fetch requires 1–200 unique PMIDs")
        return self._request("efetch.fcgi", {
            "db": "pubmed", "id": ",".join(pmids),
            "retmode": "xml", "rettype": "abstract", "tool": "medlens",
        })

    def _request(self, endpoint: str, parameters: dict[str, str]) -> RawResponse:
        request = Request(BASE_URL + endpoint, data=urlencode(parameters).encode(), headers={
            "User-Agent": "medlens/0.0.1",
            "Content-Type": "application/x-www-form-urlencoded",
        })
        for attempt in range(1, self.config.max_attempts + 1):
            wait = self._next_request - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_request = self._clock() + 1 / self.config.requests_per_second
            retry_after = None
            try:
                with self._opener(request, timeout=self.config.timeout) as response:
                    body = response.read(MAX_RESPONSE_BYTES + 1)
                    if not body or len(body) > MAX_RESPONSE_BYTES:
                        raise PubMedError(f"{endpoint}: empty or oversized response")
                    return RawResponse(endpoint, parameters, body,
                        datetime.now(timezone.utc).isoformat(), attempt,
                        response.headers.get("Content-Type", ""))
            except HTTPError as exc:
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status not in RETRYABLE_STATUS or attempt == self.config.max_attempts:
                    raise PubMedError(f"{endpoint}: HTTP {status} after {attempt} attempt(s)") from exc
            except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                if attempt == self.config.max_attempts:
                    raise PubMedError(f"{endpoint}: transport failed after {attempt} attempt(s)") from exc
            delay = self.config.backoff_seconds * 2 ** (attempt - 1)
            if retry_after is not None:
                try:
                    seconds = float(retry_after)
                except ValueError:
                    try:
                        date = parsedate_to_datetime(retry_after)
                        seconds = (date - datetime.now(timezone.utc)).total_seconds()
                    except (TypeError, ValueError, OverflowError):
                        seconds = 0
                if math.isfinite(seconds):
                    delay = max(delay, seconds)
            if delay > self.config.max_retry_wait:
                raise PubMedError(f"{endpoint}: retry wait exceeds configured budget; try later")
            self._sleep(delay)
        raise AssertionError("unreachable")


def parse_search(body: bytes, limit: int) -> SearchResult:
    """Parse the protocol envelope, not the articles. Preserve the returned rank order."""
    try:
        payload = json.loads(body)
        if payload.get("error"):
            raise PubMedError("ESearch returned an API error")
        result = payload["esearchresult"]
        if result.get("errorlist"):
            raise PubMedError("ESearch rejected part of the query; inspect the raw response")
        count_text = result["count"]
        if not isinstance(count_text, str) or not count_text.isascii() or not count_text.isdigit():
            raise ValueError("invalid count")
        count = int(count_text)
        pmids = result["idlist"]
        if not isinstance(pmids, list):
            raise ValueError("invalid idlist")
        validate_pmids(pmids)
        if len(pmids) != min(count, limit) or str(result.get("retstart", "0")) != "0":
            raise PubMedError("ESearch returned an incomplete or unexpected result window")
        translation = result.get("querytranslation")
        if translation is not None and not isinstance(translation, str):
            raise ValueError("invalid query translation")
        return SearchResult(count, tuple(pmids), translation)
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PubMedError("Invalid ESearch JSON; inspect the saved raw response") from exc


def validate_fetch(body: bytes, expected: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Check delivery of primary records, without treating reference PMIDs as records."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise PubMedError("Invalid EFetch XML; inspect the saved raw response") from exc
    if root.tag != "PubmedArticleSet":
        raise PubMedError("EFetch did not return a PubmedArticleSet")
    found = []
    for record in root:
        paths = {"PubmedArticle": "./MedlineCitation/PMID",
                 "PubmedBookArticle": "./BookDocument/PMID"}
        if record.tag not in paths:
            raise PubMedError(f"Unexpected EFetch record type: {record.tag}")
        nodes = record.findall(paths[record.tag])
        if len(nodes) != 1 or nodes[0].text is None:
            raise PubMedError("EFetch record has no unambiguous primary PMID")
        found.append(nodes[0].text.strip())
    validate_pmids(found)
    if set(found) != set(expected):
        raise PubMedError("EFetch returned missing or unexpected PMIDs")
    return tuple(found)
