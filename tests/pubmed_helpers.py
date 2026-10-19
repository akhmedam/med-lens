"""Synthetic PubMed responses and a clock-controlled fake transport."""

import io
import json
from email.message import Message
from urllib.error import HTTPError

from medlens.ingestion.pubmed import PubMedClient


def search_json(pmids=("31", "12", "25"), count=8):
    return json.dumps({"esearchresult": {"count": str(count), "retstart": "0",
        "idlist": list(pmids), "querytranslation": "translated query"}}).encode()


def xml_records(pmids):
    records = []
    for pmid in pmids:
        records.append(f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
            "<Article><ArticleTitle>Fixture only</ArticleTitle></Article></MedlineCitation>"
            "<PubmedData><ReferenceList><Reference><ArticleIdList>"
            "<ArticleId IdType='pubmed'>999999999</ArticleId>"
            "</ArticleIdList></Reference></ReferenceList></PubmedData></PubmedArticle>")
    return ("<PubmedArticleSet>" + "".join(records) + "</PubmedArticleSet>").encode()


class Response(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.headers = Message()
        self.headers["Content-Type"] = "application/xml"


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class Transport:
    def __init__(self, events, clock):
        self.events = iter(events)
        self.clock = clock
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((self.clock.now(), request, timeout))
        event = next(self.events)
        if isinstance(event, Exception):
            raise event
        return Response(event)


def make_client(events, config=None):
    clock = Clock()
    transport = Transport(events, clock)
    client = PubMedClient(config, opener=transport, sleep=clock.sleep, monotonic=clock.now)
    return client, clock, transport


def http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        status, "fixture", headers, io.BytesIO(b"error"))
