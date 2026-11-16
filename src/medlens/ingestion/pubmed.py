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

    def __init__(self, *args, transport_diagnostics=()):
        super().__init__(*args)
        # Private, bounded, typed metadata; str(error) and public projections keep
        # their existing sanitized messages and never include underlying text.
        self.transport_diagnostics = tuple(transport_diagnostics)


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
    transport_diagnostics: tuple[dict, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    total_count: int
    pmids: tuple[str, ...]
    query_translation: str | None


# Never serialize str(exc), URLError.reason text, URLs or arbitrary class names.
# Unknown/custom exception types remain explicitly OtherError.
_DIAGNOSTIC_CLASSES = frozenset({"HTTPError", "URLError", "TimeoutError", "ConnectionError",
    "ConnectionRefusedError", "ConnectionResetError", "ConnectionAbortedError", "BrokenPipeError",
    "PermissionError", "OSError", "gaierror", "herror", "SSLError", "SSLCertVerificationError",
    "SSLZeroReturnError", "SSLEOFError", "SSLWantReadError", "SSLWantWriteError", "IncompleteRead",
    "PubMedError", "str", "NoneType"})


def _diagnostic_class(value):
    name = type(value).__name__
    return name if name in _DIAGNOSTIC_CLASSES else "OtherError"


def _attempt_diagnostic(endpoint, attempt, stage, elapsed, *, error=None):
    row = {"endpoint": endpoint, "attempt": attempt, "stage": stage,
        "elapsed_seconds": max(0.0, elapsed), "outcome": "error" if error is not None else "success"}
    if error is not None:
        reason = getattr(error, "reason", None)
        row.update(exception_class=_diagnostic_class(error), reason_class=_diagnostic_class(reason))
        for name, value in (("errno", getattr(error, "errno", None)),
                            ("reason_errno", getattr(reason, "errno", None))):
            if type(value) is int:
                row[name] = value
        cause = getattr(error, "__cause__", None)
        if cause is not None:
            row["cause_class"] = _diagnostic_class(cause)
        status = getattr(error, "code", None)
        if type(status) is int and 100 <= status <= 599:
            row["http_status"] = status
    return row


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
        diagnostics = []
        for attempt in range(1, self.config.max_attempts + 1):
            wait = self._next_request - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_request = self._clock() + 1 / self.config.requests_per_second
            retry_after = None
            started = self._clock()
            stage = "open-response"
            try:
                with self._opener(request, timeout=self.config.timeout) as response:
                    stage = "read-response"
                    body = response.read(MAX_RESPONSE_BYTES + 1)
                    if not body or len(body) > MAX_RESPONSE_BYTES:
                        raise PubMedError(f"{endpoint}: empty or oversized response")
                    content_type = response.headers.get("Content-Type", "")
                diagnostics.append(_attempt_diagnostic(endpoint, attempt, stage, self._clock() - started))
                return RawResponse(endpoint, parameters, body,
                    datetime.now(timezone.utc).isoformat(), attempt, content_type, tuple(diagnostics))
            except HTTPError as exc:
                diagnostics.append(_attempt_diagnostic(endpoint, attempt, stage, self._clock() - started, error=exc))
                status = exc.code
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status not in RETRYABLE_STATUS or attempt == self.config.max_attempts:
                    raise PubMedError(f"{endpoint}: HTTP {status} after {attempt} attempt(s)",
                        transport_diagnostics=diagnostics) from exc
            except (URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
                diagnostics.append(_attempt_diagnostic(endpoint, attempt, stage, self._clock() - started, error=exc))
                if attempt == self.config.max_attempts:
                    raise PubMedError(f"{endpoint}: transport failed after {attempt} attempt(s)",
                        transport_diagnostics=diagnostics) from exc
            except PubMedError as exc:
                diagnostics.append(_attempt_diagnostic(endpoint, attempt, stage, self._clock() - started, error=exc))
                exc.transport_diagnostics = tuple(diagnostics)
                raise
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
            diagnostics[-1]["retry_delay_seconds"] = delay
            if delay > self.config.max_retry_wait:
                raise PubMedError(f"{endpoint}: retry wait exceeds configured budget; try later",
                    transport_diagnostics=diagnostics)
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
