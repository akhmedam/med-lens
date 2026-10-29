"""Pack exact source excerpts under the generator's evidence token budget."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class ContextConfig:
    max_fragments: int = 6
    max_per_article: int = 2
    evidence_tokens: int = 4000
    prefer_outcomes: bool = True

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.max_fragments, self.max_per_article, self.evidence_tokens)):
            raise ValueError("Context limits must be positive integers")
        if type(self.prefer_outcomes) is not bool:
            raise ValueError("prefer_outcomes must be boolean")


def source_section(article, window):
    if window["text_version_id"] != article["text_version"]["text_version_id"]:
        raise ValueError("Stale window text version")
    if window["provenance"] != article["provenance"]:
        raise ValueError("Window provenance differs from article")
    if window["kind"] == "title":
        if window["text"] or window["title"] != (article.get("title") or ""):
            raise ValueError("Invalid title-only window")
        return None
    if window["kind"] != "abstract":
        raise ValueError("Unsupported evidence source")
    sections = (article.get("abstract") or {}).get("sections", [])
    found = [s for s in sections if s["index"] == window["section_index"]]
    if len(found) != 1:
        raise ValueError("Window section is missing or ambiguous")
    section = found[0]
    start, end = window["char_start"], window["char_end"]
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(section["text"])
            or window["text"] != section["text"][start:end]
            or window["source_path"] != section["source_path"]
            or window["section_label"] != section.get("label")
            or window["title"] != (article.get("title") or "")):
        raise ValueError("Window does not match its exact source span")
    return section


SERVICE_LABELS = {"systematic review registration", "trial registration", "clinical trial registration",
    "registration", "funding", "funding sources", "funding source", "financial support"}


def normalized_label(text):
    return " ".join((text or "").casefold().replace("-", " ").replace("_", " ").split())


def section_priority(article, window):
    section = source_section(article, window)
    label = normalized_label(section.get("label"))
    category = normalized_label(section.get("nlm_category"))
    if category in {"results", "conclusions"} or label in {
            "results", "conclusion", "conclusions", "authors' conclusions", "conclusions and relevance"}:
        return 0
    return 1 if not label else 2


def render_evidence(sources):
    """This exact string is the evidence payload; generation must not reserialize it."""
    return canonical({"sources": [{k: s[k] for k in ("citation_id", "pmid", "title", "journal",
        "publication_date", "section", "text")} for s in sources]})


def build_context(dataset, window_manifest, windows, ranking, scores, tokenizer, tokenizer_info,
                  *, config=None):
    config = config or ContextConfig()
    articles = {a["pmid"]: a for a in dataset.articles}
    if (window_manifest["dataset_id"] != dataset.manifest["dataset_id"]
            or window_manifest["articles_sha256"] != dataset.manifest["articles_sha256"]):
        raise ValueError("Window/dataset identity mismatch")
    if len(set(ranking)) != len(ranking) or not set(ranking) <= set(articles):
        raise ValueError("Article ranking must have unique dataset PMIDs")
    if len(scores) != len(windows) or len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Window scores or IDs are inconsistent")
    grouped = {p: [] for p in ranking}
    skipped = []
    for order, (window, score) in enumerate(zip(windows, scores)):
        if isinstance(score, bool) or not math.isfinite(float(score)) or not -1 <= score <= 1:
            raise ValueError("Expected finite cosine window scores")
        if window["pmid"] not in articles:
            raise ValueError("Window PMID outside dataset")
        section = source_section(articles[window["pmid"]], window)
        if window["pmid"] not in grouped:
            continue
        if section is None or not window["text"].strip():
            skipped.append({"window_id": window["window_id"], "reason": "no-abstract-excerpt"})
        elif normalized_label(section.get("label")) in SERVICE_LABELS:
            skipped.append({"window_id": window["window_id"], "reason": "service-section"})
        else:
            grouped[window["pmid"]].append((float(score), order, window))
    for candidates in grouped.values():
        candidates.sort(key=lambda x: (section_priority(articles[x[2]["pmid"]], x[2])
            if config.prefer_outcomes else 0, -x[0], x[1]))
    sources = []
    empty_tokens = len(tokenizer.encode(render_evidence([]), add_special_tokens=False))
    if empty_tokens > config.evidence_tokens:
        raise ValueError("Evidence budget cannot hold an empty payload")
    # Round one favors article diversity; later rounds fill from non-overlapping windows.
    for round_number in range(config.max_per_article):
        for article_rank, pmid in enumerate(ranking, 1):
            if len(sources) >= config.max_fragments:
                break
            selected = [s for s in sources if s["pmid"] == pmid]
            if len(selected) != round_number:
                continue
            article = articles[pmid]
            for score, _, window in grouped[pmid]:
                if any(s["window_id"] == window["window_id"] for s in selected):
                    continue
                if any(s["section_index"] == window["section_index"]
                       and max(s["char_start"], window["char_start"]) < min(s["char_end"], window["char_end"])
                       for s in selected):
                    skipped.append({"window_id": window["window_id"], "reason": "overlapping-span"})
                    continue
                date = next((d["iso_date"] for d in article.get("dates", [])
                             if d["kind"] == "journal_issue" and d.get("iso_date")), None)
                source = {"citation_id": f"E{len(sources) + 1}", "pmid": pmid,
                    "title": article.get("title"), "journal": (article.get("journal") or {}).get("title"),
                    "publication_date": date, "section": window["section_label"],
                    **{k: window[k] for k in ("window_id", "text_version_id", "section_index",
                        "source_path", "char_start", "char_end", "text", "provenance")},
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "article_rank": article_rank,
                    "window_cosine": score}
                count = len(tokenizer.encode(render_evidence([*sources, source]), add_special_tokens=False))
                if count > config.evidence_tokens:
                    skipped.append({"window_id": window["window_id"], "reason": "token-budget"})
                    continue
                sources.append(source)
                break
    payload = render_evidence(sources)
    identity = {"schema_version": 1, "dataset_id": dataset.manifest["dataset_id"],
        "window_artifact_id": window_manifest["artifact_id"], "article_ranking": list(ranking),
        "scores_sha256": digest([float(s) for s in scores]), "config": asdict(config),
        "selection_policy": "article-round-robin-section-priority-window-cosine-no-overlap-v2",
        "tokenizer": tokenizer_info, "sources_sha256": digest(sources),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return {**identity, "context_id": digest(identity), "sources": sources, "evidence_payload": payload,
        "evidence_tokens": len(tokenizer.encode(payload, add_special_tokens=False)),
        "status": "source-excerpts-selected" if sources else "no-source-excerpts", "skipped": skipped}


def locate_quote(context, citation_id, quote):
    """Locate verbatim text, not entailment or scientific validity of a claim."""
    if (digest(context["sources"]) != context["sources_sha256"]
            or render_evidence(context["sources"]) != context["evidence_payload"]):
        raise ValueError("Context source mapping or payload was modified")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Quote must be a nonempty string")
    sources = [s for s in context["sources"] if s["citation_id"] == citation_id]
    if len(sources) != 1:
        raise ValueError("Unknown or ambiguous citation ID")
    source = sources[0]
    offsets, start = [], 0
    while (index := source["text"].find(quote, start)) >= 0:
        offsets.append({"char_start": source["char_start"] + index,
                        "char_end": source["char_start"] + index + len(quote)})
        start = index + 1
    if not offsets:
        raise ValueError("Quote is not present verbatim in the selected excerpt")
    return {"citation_id": citation_id, "pmid": source["pmid"], "window_id": source["window_id"],
        "text_version_id": source["text_version_id"], "source_path": source["source_path"],
        "section_index": source["section_index"], "matches": offsets,
        "ambiguous": len(offsets) > 1, "quote": quote}
