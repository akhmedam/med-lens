"""Pack exact source excerpts under the generator's evidence token budget."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
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
    selection_mode: str = "article-diversity"
    max_articles: int = 3

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.max_fragments, self.max_per_article, self.evidence_tokens)):
            raise ValueError("Context limits must be positive integers")
        if type(self.prefer_outcomes) is not bool:
            raise ValueError("prefer_outcomes must be boolean")
        if self.selection_mode not in {"article-diversity", "results-methods"}:
            raise ValueError("Unknown evidence selection mode")
        if type(self.max_articles) is not int or self.max_articles <= 0:
            raise ValueError("max_articles must be a positive integer")
        if self.selection_mode == "results-methods" and (
                self.max_fragments > 6 or self.max_per_article > 2 or self.evidence_tokens > 4000):
            raise ValueError("Coverage selection preserves six fragments, two per article and 4000 tokens")


def source_section(article, window):
    if window["text_version_id"] != article["text_version"]["text_version_id"]:
        raise ValueError("Stale window text version")
    if window["provenance"] != article["provenance"]:
        raise ValueError("Window provenance differs from article")
    if window["kind"] == "title":
        if window["text"] or window["title"] != (article.get("title") or ""):
            raise ValueError("Invalid title-only window")
        return None
    if window["kind"] not in {"abstract", "fulltext"}:
        raise ValueError("Unsupported evidence source")
    field = "full_text" if window["kind"] == "fulltext" else "abstract"
    sections = (article.get(field) or {}).get("sections", [])
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


def coverage_role(article, window):
    """Generic abstract-section roles, without biomedical entities or query labels."""
    section = source_section(article, window)
    label = normalized_label(section.get("label"))
    category = normalized_label(section.get("nlm_category"))
    if category == "results" or label in {
            "results", "main results", "findings", "results and discussion",
            "data synthesis", "synthesis of results"}:
        return "results"
    if category == "methods" or label in {
            "methods", "materials and methods", "patients and methods", "design",
            "participants", "study population", "patient population", "interventions",
            "design setting and participants", "design, setting, and participants",
            "selection criteria", "eligibility criteria", "included studies",
            "study selection", "study selection and data extraction",
            "search methods", "search strategy", "data sources", "data extraction",
            "data collection and analysis", "statistical analysis", "risk of bias",
            "synthesis methods"}:
        return "methods"
    if category == "conclusions" or label in {
            "conclusion", "conclusions", "authors' conclusions",
            "conclusions and relevance", "conclusions/interpretation"}:
        return "conclusions"
    return "unstructured" if not label else "other"


def companion_priority(article, window):
    role = coverage_role(article, window)
    if role == "methods":
        label = normalized_label(source_section(article, window).get("label"))
        boilerplate = {"search methods", "search strategy", "data sources", "data extraction",
            "data collection and analysis", "statistical analysis", "risk of bias",
            "synthesis methods", "study selection and data extraction"}
        return 1 if label in boilerplate else 0
    return {"results": 2, "unstructured": 3, "conclusions": 4, "other": 5}[role]


def token_budget_reason(sources, tokenizer, config, *, indexed):
    if len(tokenizer.encode(render_evidence(sources), add_special_tokens=False)) > config.evidence_tokens:
        return "token-budget"
    if indexed:
        # Import lazily: quote_catalog imports this module's source helpers.
        from .quotes import quote_catalog
        try:
            count = quote_catalog({"sources": sources}, tokenizer)["tokens"]
        except ValueError as exc:
            if str(exc) != "Quote-indexed evidence exceeds the 4000-token budget":
                raise
            return "quote-token-budget"
        if count > config.evidence_tokens:
            return "quote-token-budget"
    return None


COVERAGE_POLICY = "article-results-methods-complete-source-sentences-v4"
SENTENCE_POLICY = "source-section-sentence-spans-v1"
SENTENCE_FIELDS = ("quote_sentence_policy", "source_section_sha256", "source_section_length",
                   "quote_sentence_spans")


def section_sentence_spans(text):
    """Boundary heuristic on the complete source section, never on a token window.

    Section edges are boundaries, including an unpunctuated final source unit.
    Internal boundaries follow .!? + whitespace + uppercase/`(`. Abbreviations,
    missing whitespace and unusual casing remain limitations; this does not
    establish linguistic completeness, entailment or scientific validity.
    """
    boundaries = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])', text)], len(text)]
    spans = []
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append({"char_start": start, "char_end": end})
    return spans


def source_sentence_metadata(article, window):
    section = source_section(article, window)
    if section is None:
        raise ValueError("Sentence metadata requires a source text section")
    text = section["text"]
    spans = [span for span in section_sentence_spans(text)
        if window["char_start"] <= span["char_start"] < span["char_end"] <= window["char_end"]]
    return {"quote_sentence_policy": SENTENCE_POLICY,
            "source_section_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_section_length": len(text), "quote_sentence_spans": spans}


def source_quote_spans(source):
    """Validate metadata shape locally; full source verification is separate."""
    if not any(key in source for key in SENTENCE_FIELDS):
        return None
    if (not all(key in source for key in SENTENCE_FIELDS)
            or source["quote_sentence_policy"] != SENTENCE_POLICY
            or type(source["source_section_length"]) is not int or source["source_section_length"] <= 0
            or not isinstance(source["source_section_sha256"], str)
            or re.fullmatch(r'[0-9a-f]{64}', source["source_section_sha256"]) is None
            or not isinstance(source["quote_sentence_spans"], list)):
        raise ValueError("Invalid complete-source-sentence metadata")
    lower, upper = source["char_start"], source["char_end"]
    if (type(lower) is not int or type(upper) is not int
            or not 0 <= lower < upper <= source["source_section_length"]
            or upper - lower != len(source["text"])):
        raise ValueError("Invalid sentence window bounds")
    previous = lower
    for span in source["quote_sentence_spans"]:
        if (not isinstance(span, dict) or set(span) != {"char_start", "char_end"}
                or type(span["char_start"]) is not int or type(span["char_end"]) is not int
                or not lower <= span["char_start"] < span["char_end"] <= upper
                or span["char_start"] < previous):
            raise ValueError("Invalid or overlapping complete sentence span")
        text = source["text"][span["char_start"] - lower:span["char_end"] - lower]
        if not text or text != text.strip():
            raise ValueError("Sentence span contains empty or untrimmed source text")
        previous = span["char_end"]
    return source["quote_sentence_spans"]


def verify_source_sentence_spans(source, article, window, *, required=False):
    """Reject rehashed fabricated spans using the actual dataset section."""
    spans = source_quote_spans(source)
    if spans is None:
        if required:
            raise ValueError("Complete-source-sentence metadata is required")
        return
    expected = source_sentence_metadata(article, window)
    if any(source[key] != expected[key] for key in SENTENCE_FIELDS):
        raise ValueError("Sentence metadata differs from complete original source section")
    if not spans:
        raise ValueError("Selected source has no complete source sentence")


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
    coverage = config.selection_mode == "results-methods"
    if coverage and token_budget_reason([], tokenizer, config, indexed=True):
        raise ValueError("Evidence budget cannot hold an empty indexed payload")
    if coverage:
        schedule = []
        first_priority = {"results": 0, "unstructured": 1, "conclusions": 2, "methods": 3, "other": 4}
        for article_rank, pmid in enumerate(ranking[:config.max_articles], 1):
            article = articles[pmid]
            first = sorted(grouped[pmid], key=lambda x:
                (first_priority[coverage_role(article, x[2])], -x[0], x[1]))
            companions = sorted(grouped[pmid], key=lambda x:
                (companion_priority(article, x[2]), -x[0], x[1]))
            schedule.extend((article_rank, pmid, round_number, first if round_number == 0 else companions)
                for round_number in range(config.max_per_article))
    else:
        # Preserve the historical article-diversity ordering for existing callers.
        schedule = [(article_rank, pmid, round_number, grouped[pmid])
            for round_number in range(config.max_per_article)
            for article_rank, pmid in enumerate(ranking, 1)]
    for article_rank, pmid, round_number, candidates in schedule:
        if len(sources) >= config.max_fragments:
            break
        selected = [s for s in sources if s["pmid"] == pmid]
        if len(selected) != round_number:
            continue
        article = articles[pmid]
        for score, _, window in candidates:
            if any(s["window_id"] == window["window_id"] for s in selected):
                continue
            if any(s.get("source_kind", "abstract") == window["kind"]
                   and s["section_index"] == window["section_index"]
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
            if coverage:
                source.update(source_sentence_metadata(article, window))
                if not source["quote_sentence_spans"]:
                    skipped.append({"window_id": window["window_id"], "reason": "no-complete-source-sentence"})
                    continue
            if window["kind"] == "fulltext":
                source.update(source_kind="fulltext", pmcid=article["full_text"]["pmcid"],
                    license_uri=article["full_text"]["license_uri"])
                source["url"] = "https://pmc.ncbi.nlm.nih.gov/articles/" + source["pmcid"] + "/"
            reason = token_budget_reason([*sources, source], tokenizer, config, indexed=coverage)
            if reason:
                skipped.append({"window_id": window["window_id"], "reason": reason})
                continue
            sources.append(source)
            break
    payload = render_evidence(sources)
    identity = {"schema_version": 1, "dataset_id": dataset.manifest["dataset_id"],
        "window_artifact_id": window_manifest["artifact_id"], "article_ranking": list(ranking),
        "scores_sha256": digest([float(s) for s in scores]), "config": asdict(config),
        "selection_policy": (COVERAGE_POLICY if coverage
            else "article-round-robin-section-priority-window-cosine-no-overlap-v2"),
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
    return {**({k:source[k] for k in ("source_kind", "pmcid", "license_uri")}
        if source.get("source_kind")=="fulltext" else {}),
        "citation_id": citation_id, "pmid": source["pmid"], "window_id": source["window_id"],
        "text_version_id": source["text_version_id"], "source_path": source["source_path"],
        "section_index": source["section_index"], "matches": offsets,
        "ambiguous": len(offsets) > 1, "quote": quote}
