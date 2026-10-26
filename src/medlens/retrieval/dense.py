"""Exact NumPy cosine retrieval with maximum-window article scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from medlens.parsing.dataset import load_dataset
from .embeddings import DIMENSION, EmbeddingError, load_embeddings
from .windows import load_windows


class DenseError(ValueError):
    """Vectors, windows, candidates, or query encoder are incompatible."""


class DenseIndex:
    def __init__(self, dataset_dir: Path, windows_dir: Path, embeddings_dir: Path):
        started = perf_counter()
        self.dataset = load_dataset(dataset_dir)
        self.window_manifest, self.windows = load_windows(windows_dir)
        self.embedding_manifest, mapping, self.vectors = load_embeddings(embeddings_dir)
        if (self.window_manifest["dataset_id"] != self.dataset.manifest["dataset_id"]
            or self.window_manifest["articles_sha256"] != self.dataset.manifest["articles_sha256"]
            or self.embedding_manifest["source_window_artifact_id"] != self.window_manifest["artifact_id"]):
            raise DenseError("Dataset, window, and vector artifacts disagree")
        expected = [{key: row[key] for key in ("window_id", "pmid", "text_version_id")} for row in self.windows]
        if list(mapping) != expected:
            raise DenseError("Ordered vector-to-window mapping mismatch")
        self.articles = {row["pmid"]: row for row in self.dataset.articles}
        self.positions = {pmid: [] for pmid in self.articles}
        for i, row in enumerate(self.windows):
            article = self.articles.get(row["pmid"])
            if article is None or article["text_version"]["text_version_id"] != row["text_version_id"]:
                raise DenseError("Window does not belong to the candidate text version")
            self.positions[row["pmid"]].append(i)
        missing = [pmid for pmid, positions in self.positions.items() if not positions]
        if missing != self.window_manifest["empty_pmids"]:
            raise DenseError("Candidate window coverage mismatch")
        self.missing_pmids = missing
        self.index_seconds = perf_counter() - started

    def verify_encoder(self, encoder):
        # CPU and batch-size changes are allowed; the encoding policy must match.
        expected = {key: value for key, value in self.embedding_manifest["encoder"].items()
                    if key not in {"device", "batch_size"}}
        actual = {key: value for key, value in encoder.info.items()
                  if key not in {"device", "batch_size"}}
        if expected != actual or encoder.tokenizer_info != self.window_manifest["tokenizer"]:
            raise DenseError("Query and passage encoders are incompatible")

    def search(self, query_vector, top_k: int = 10) -> list[dict]:
        if type(top_k) is not int or top_k <= 0:
            raise DenseError("top_k must be a positive integer")
        query = np.asarray(query_vector, dtype=np.float32)
        if (query.shape != (DIMENSION,) or not np.isfinite(query).all()
            or not np.isclose(np.linalg.norm(query), 1, atol=1e-5, rtol=0)):
            raise DenseError("Expected a finite unit query vector with 768 components")
        scores = np.clip(self.vectors @ query, -1.0, 1.0)
        articles = []
        for order, (pmid, positions) in enumerate(self.positions.items()):
            if not positions:
                continue
            # argmax retains the first source window on an exact tie.
            best = positions[int(np.argmax(scores[positions]))]
            articles.append((float(scores[best]), order, pmid, best))
        articles.sort(key=lambda row: (-row[0], row[1]))
        hits = []
        for rank, (score, _, pmid, best) in enumerate(articles[:top_k], 1):
            article, window = self.articles[pmid], self.windows[best]
            hits.append({"rank": rank, "pmid": pmid, "score": score,
                "cosine_distance": 1 - score, "title": article.get("title"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "text_version_id": window["text_version_id"],
                "best_window": {key: window[key] for key in ("window_id", "kind", "section_index",
                    "section_label", "source_path", "char_start", "char_end", "text", "provenance")}})
        return hits

    def run(self, query: str, encoder, top_k: int = 10) -> dict:
        if type(top_k) is not int or top_k <= 0:
            raise DenseError("top_k must be a positive integer")
        self.verify_encoder(encoder)
        vectors, encoding = encoder.encode_queries([query])
        if vectors.shape != (1, DIMENSION):
            raise EmbeddingError("Expected exactly one query vector")
        started = perf_counter()
        hits = self.search(vectors[0], top_k)
        search_seconds = perf_counter() - started
        specification = {"dataset_id": self.dataset.manifest["dataset_id"],
            "window_artifact_id": self.window_manifest["artifact_id"],
            "embedding_artifact_id": self.embedding_manifest["artifact_id"],
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "query_encoder_implementation_sha256": hashlib.sha256(Path(__file__).with_name("embeddings.py").read_bytes()).hexdigest(),
            "query": query, "query_encoder": encoder.info, "top_k": top_k,
            "query_vector_sha256": hashlib.sha256(vectors[0].astype(np.float32).tobytes()).hexdigest(),
            "similarity": "cosine-dot-unit-vectors", "distance": "1-cosine",
            "article_score": "max-window", "article_ties": "candidate-order",
            "window_ties": "source-window-order", "negative_scores": "retain"}
        run_id = hashlib.sha256(json.dumps(specification, sort_keys=True,
            ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        return {"schema_version": 1, "run_id": run_id, **specification,
            "candidate_count": len(self.articles), "searchable_article_count": len(self.articles) - len(self.missing_pmids),
            "missing_vector_pmids": self.missing_pmids, "window_count": len(self.windows),
            "index_validation_seconds": self.index_seconds, "query_encoding": encoding,
            "search_seconds": search_seconds, "results": hits}
