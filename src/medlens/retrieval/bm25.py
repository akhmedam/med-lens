"""Article-level BM25 over one verified PubMed candidate dataset."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import bm25s

from medlens.parsing.dataset import LoadedDataset


TOKEN_PATTERN = r"(?u)\b\w+\b"
PREPROCESSING_VERSION = "unicode-nfc-casefold-words-v1"


def normalize_query(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tokenize(text: str) -> list[str]:
    # Keep single characters, negations, numbers, and abbreviation components.
    return re.findall(TOKEN_PATTERN, normalize_query(text))


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self):
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be finite and between 0 and 1")


def article_text(article: dict) -> str:
    abstract = article.get("abstract") or {}
    return "\n".join([article.get("title") or "", *[
        section["text"] for section in abstract.get("sections", [])]])


@dataclass(frozen=True)
class SearchHit:
    rank: int
    pmid: str
    score: float
    title: str | None
    url: str
    text_version_id: str


class BM25Index:
    """Rebuild a small candidate index; never mix datasets or global caches."""

    def __init__(self, dataset: LoadedDataset, config: BM25Config | None = None):
        started = perf_counter()
        self.config = config or BM25Config()
        self.dataset = dataset
        self.tokens = [tokenize(article_text(row)) for row in dataset.articles]
        self.model = None
        # BM25S cannot build an empty vocabulary. Empty records still count in N.
        if any(self.tokens):
            self.model = bm25s.BM25(k1=self.config.k1, b=self.config.b,
                method="lucene", idf_method="lucene", backend="numpy", dtype="float64")
            self.model.index(self.tokens, show_progress=False)
        self.index_seconds = perf_counter() - started

    def search(self, query: str, top_k: int = 10) -> tuple[SearchHit, ...]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        tokens = tokenize(query)
        if not tokens or self.model is None:
            return ()
        scores = self.model.get_scores(tokens)
        # Stable ties follow the original PubMed candidate order, not PMID order.
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        hits = []
        for index in order:
            if scores[index] <= 0:
                continue
            row = self.dataset.articles[index]
            hits.append(SearchHit(len(hits) + 1, row["pmid"], float(scores[index]),
                row.get("title"), f"https://pubmed.ncbi.nlm.nih.gov/{row['pmid']}/",
                row["text_version"]["text_version_id"]))
            if len(hits) == top_k:
                break
        return tuple(hits)

    def run(self, query: str, top_k: int = 10) -> dict:
        started = perf_counter()
        hits = self.search(query, top_k)
        search_seconds = perf_counter() - started
        specification = {
            "dataset_id": self.dataset.manifest["dataset_id"],
            "articles_sha256": self.dataset.manifest["articles_sha256"],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bm25s_version": version("bm25s"), "numpy_version": version("numpy"),
            "method": "lucene", "idf_method": "lucene", "dtype": "float64",
            "parameters": asdict(self.config), "preprocessing": PREPROCESSING_VERSION,
            "token_pattern": TOKEN_PATTERN, "stopwords": False, "stemming": False,
            "fields": ["title", "abstract.sections.text"], "field_weighting": "concatenate-once",
            "ties": "source-candidate-order", "zero_scores": "omit",
            "query": query, "normalized_query": normalize_query(query),
            "query_tokens": tokenize(query), "top_k": top_k,
        }
        run_id = hashlib.sha256(json.dumps(specification, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return {"schema_version": 1, "run_id": run_id, **specification,
            "candidate_count": len(self.tokens), "token_count": sum(map(len, self.tokens)),
            "empty_document_count": sum(not tokens for tokens in self.tokens),
            "index_seconds": self.index_seconds, "search_seconds": search_seconds,
            "results": [asdict(hit) for hit in hits]}
