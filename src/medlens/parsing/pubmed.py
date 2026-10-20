"""Parse PubMed EFetch XML without downloading DTDs or inventing missing fields."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date

from .models import (
    Abstract, AbstractSection, Article, Author, AuthorList, Identifier, Journal,
    MeshHeading, Provenance, RecordDate, Term, TextVersion,
)

PARSER_VERSION = "pubmed-xml-v1"
MONTHS = {name: i for i, pair in enumerate((
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
    ("may", "may"), ("jun", "june"), ("jul", "july"), ("aug", "august"),
    ("sep", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december"),
), start=1) for name in pair}


class ParseError(ValueError):
    """A record cannot be represented without losing its identity or structure."""


def _raw(node: ET.Element | None) -> str | None:
    return None if node is None else "".join(node.itertext())


def _text(node: ET.Element | None) -> str | None:
    raw = _raw(node)
    return None if raw is None else " ".join(raw.split())


def _field(node: ET.Element, path: str) -> str | None:
    return _text(node.find(path))


def _abstract(node: ET.Element, path: str) -> Abstract:
    sections = tuple(AbstractSection(i, child.get("Label"), child.get("NlmCategory"),
        _text(child) or "", _raw(child) or "", f"{path}/AbstractText[{i + 1}]")
        for i, child in enumerate(node.findall("./AbstractText")))
    return Abstract(sections, _field(node, "./CopyrightInformation"), dict(node.attrib), path)


def _authors(node: ET.Element, path: str) -> tuple[AuthorList, ...]:
    groups = []
    for group_index, group in enumerate(node.findall("./AuthorList"), start=1):
        group_path = f"{path}/AuthorList[{group_index}]"
        authors = []
        for i, author in enumerate(group.findall("./Author"), start=1):
            author_path = f"{group_path}/Author[{i}]"
            identifiers = tuple(Identifier(item.get("Source"), _text(item) or "",
                f"{author_path}/Identifier[{j}]", dict(item.attrib))
                for j, item in enumerate(author.findall("./Identifier"), start=1))
            affiliations = tuple(_text(item) or "" for item in author.findall("./AffiliationInfo/Affiliation"))
            authors.append(Author(_field(author, "./LastName"), _field(author, "./ForeName"),
                _field(author, "./Initials"), _field(author, "./Suffix"),
                _field(author, "./CollectiveName"), affiliations, identifiers,
                dict(author.attrib), author_path))
        groups.append(AuthorList(tuple(authors), dict(group.attrib), group_path))
    return tuple(groups)


def _date(node: ET.Element, kind: str, path: str) -> RecordDate:
    raw = {child.tag: _raw(child) or "" for child in node}
    values = {key: value.strip() for key, value in raw.items()}
    iso, precision = None, "unknown"
    if values.get("MedlineDate"):
        precision = "free_text"
    elif values.get("Season"):
        precision = "season"
    elif values.get("Year"):
        try:
            year_text = values["Year"]
            if len(year_text) != 4 or not year_text.isascii() or not year_text.isdigit():
                raise ValueError("unparsed year")
            year = int(year_text)
            date(year, 1, 1)  # Validate the year; do not manufacture a day in the output.
            month_text, day_text = values.get("Month"), values.get("Day")
            if month_text:
                month = int(month_text) if month_text.isascii() and month_text.isdigit() else MONTHS.get(month_text.casefold())
                if month is None:
                    raise ValueError("unparsed month")
                date(year, month, 1)
                if day_text:
                    if not day_text.isascii() or not day_text.isdigit():
                        raise ValueError("unparsed day")
                    iso = date(year, month, int(day_text)).isoformat()
                    precision = "day"
                else:
                    iso, precision = f"{year:04d}-{month:02d}", "month"
            elif day_text:
                raise ValueError("day without month")
            else:
                iso, precision = f"{year:04d}", "year"
        except ValueError:
            precision = "unparsed"
    return RecordDate(kind, raw, dict(node.attrib), iso, precision, path)


def _term(node: ET.Element, path: str) -> Term:
    return Term(_text(node) or "", _raw(node) or "", dict(node.attrib), path)


def _ids(node: ET.Element, path: str) -> tuple[Identifier, ...]:
    return tuple(Identifier(child.get("IdType"), _text(child) or "",
        f"{path}/ArticleId[{i}]", dict(child.attrib)) for i, child in enumerate(node.findall("./ArticleId"), start=1))


def _text_version(pmid: str, title: str | None, vernacular: str | None,
                  book_title: str | None, abstract: Abstract | None,
                  other: tuple[Abstract, ...]) -> TextVersion:
    def projection(value: Abstract | None):
        if value is None:
            return None
        return {"attributes": value.attributes, "sections": [
            {"label": s.label, "nlm_category": s.nlm_category, "text": s.text}
            for s in value.sections]}

    content = {"title": title, "vernacular_title": vernacular, "book_title": book_title,
               "abstract": projection(abstract), "other_abstracts": [projection(a) for a in other]}
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    identity = hashlib.sha256(f"{pmid}:{PARSER_VERSION}:{digest}".encode()).hexdigest()
    return TextVersion(PARSER_VERSION, digest, f"pubmed:{pmid}:{identity}")


def _record(record: ET.Element, provenance: Provenance) -> Article:
    path = provenance.record_path
    is_book = record.tag == "PubmedBookArticle"
    primary_path = "./BookDocument" if is_book else "./MedlineCitation"
    primary = record.find(primary_path)
    if primary is None:
        raise ParseError(f"{path}: missing primary record")
    pmids = primary.findall("./PMID")
    if len(pmids) != 1:
        raise ParseError(f"{path}: expected one primary PMID")
    pmid = _text(pmids[0]) or ""
    if not pmid.isascii() or not pmid.isdigit() or int(pmid) <= 0:
        raise ParseError(f"{path}: invalid primary PMID")
    content = primary if is_book else primary.find("./Article")
    if content is None:
        raise ParseError(f"{path}: missing Article")
    primary_source = f"{path}/{primary_path.removeprefix('./')}"
    content_source = primary_source if is_book else f"{primary_source}/Article"
    data_path = "./PubmedBookData" if is_book else "./PubmedData"
    data = record.find(data_path)
    data_source = f"{path}/{data_path.removeprefix('./')}"
    title_node = content.find("./ArticleTitle")
    title = _text(title_node)
    vernacular = _field(content, "./VernacularTitle")
    book = content.find("./Book") if is_book else None
    book_title = _field(book, "./BookTitle") if book is not None else None
    abstract_nodes = content.findall("./Abstract")
    if len(abstract_nodes) > 1:
        raise ParseError(f"{path}: multiple primary abstracts")
    abstract = _abstract(abstract_nodes[0], f"{content_source}/Abstract") if abstract_nodes else None
    status = "missing" if abstract is None else (
        "present" if any(section.text for section in abstract.sections) else "empty")
    other = tuple(_abstract(node, f"{primary_source}/OtherAbstract[{i}]")
        for i, node in enumerate(primary.findall("./OtherAbstract"), start=1))
    groups = _authors(content, content_source)
    if book is not None:
        groups += _authors(book, f"{content_source}/Book")

    dates = []
    journal = None
    journal_node = content.find("./Journal")
    if journal_node is not None:
        issns = tuple(Identifier(node.get("IssnType"), _text(node) or "",
            f"{content_source}/Journal/ISSN[{i}]", dict(node.attrib))
            for i, node in enumerate(journal_node.findall("./ISSN"), start=1))
        journal = Journal(_field(journal_node, "./Title"), _field(journal_node, "./ISOAbbreviation"),
            _field(primary, "./MedlineJournalInfo/MedlineTA"), issns,
            _field(journal_node, "./JournalIssue/Volume"), _field(journal_node, "./JournalIssue/Issue"))
        for node in journal_node.findall("./JournalIssue/PubDate"):
            dates.append(_date(node, "journal_issue", f"{content_source}/Journal/JournalIssue/PubDate"))
    if book is not None:
        for tag in ("PubDate", "BeginningDate", "EndingDate"):
            for node in book.findall(f"./{tag}"):
                dates.append(_date(node, f"book:{tag}", f"{content_source}/Book/{tag}"))
    for i, node in enumerate(content.findall("./ArticleDate"), start=1):
        dates.append(_date(node, "article_electronic", f"{content_source}/ArticleDate[{i}]"))
    for tag in ("DateCompleted", "DateRevised", "ContributionDate"):
        for node in primary.findall(f"./{tag}"):
            dates.append(_date(node, f"record:{tag}", f"{primary_source}/{tag}"))
    if data is not None:
        for i, node in enumerate(data.findall("./History/PubMedPubDate"), start=1):
            dates.append(_date(node, f"history:{node.get('PubStatus', 'unknown')}",
                f"{data_source}/History/PubMedPubDate[{i}]"))

    identifiers = []
    if data is not None:
        for node in data.findall("./ArticleIdList"):
            identifiers.extend(_ids(node, f"{data_source}/ArticleIdList"))
    if is_book:
        for node in content.findall("./ArticleIdList"):
            identifiers.extend(_ids(node, f"{content_source}/ArticleIdList"))
    for i, node in enumerate(content.findall("./ELocationID"), start=1):
        identifiers.append(Identifier(node.get("EIdType"), _text(node) or "",
            f"{content_source}/ELocationID[{i}]", dict(node.attrib)))
    if any(item.kind == "pubmed" and item.value != pmid for item in identifiers):
        raise ParseError(f"{path}: conflicting primary PubMed identifiers")

    type_path = "PublicationType" if is_book else "PublicationTypeList/PublicationType"
    types = tuple(_term(node, f"{content_source}/{type_path}[{i}]")
        for i, node in enumerate(content.findall(f"./{type_path}"), start=1))
    mesh = []
    for i, node in enumerate(primary.findall("./MeshHeadingList/MeshHeading"), start=1):
        heading_path = f"{primary_source}/MeshHeadingList/MeshHeading[{i}]"
        descriptors = node.findall("./DescriptorName")
        if len(descriptors) != 1:
            raise ParseError(f"{path}: ambiguous MeSH descriptor")
        mesh.append(MeshHeading(_term(descriptors[0], f"{heading_path}/DescriptorName"),
            tuple(_term(q, f"{heading_path}/QualifierName[{j}]")
                for j, q in enumerate(node.findall("./QualifierName"), start=1))))

    warnings = []
    if not title:
        warnings.append("missing_title")
    if status != "present":
        warnings.append(f"abstract_{status}")
    if any(d.precision == "unparsed" for d in dates):
        warnings.append("unparsed_date")
    text_nodes = ([title_node] if title_node is not None else []) + list(abstract_nodes)
    text_nodes += list(primary.findall("./OtherAbstract"))
    if any(str(child.tag).startswith("{http://www.w3.org/1998/Math/MathML}")
           for node in text_nodes for child in node.iter()):
        warnings.append("mathml_plain_text_projection")
    if is_book and content.find("./Sections") is not None:
        warnings.append("book_full_text_not_parsed")
    return Article(
        pmid=pmid,
        pmid_version=pmids[0].get("Version"),
        record_type=record.tag,
        title=title,
        raw_title=_raw(title_node),
        vernacular_title=vernacular,
        book_title=book_title,
        journal=journal,
        author_lists=groups,
        abstract=abstract,
        abstract_status=status,
        other_abstracts=other,
        dates=tuple(dates),
        languages=tuple(_text(node) or "" for node in content.findall("./Language")),
        publication_types=types,
        mesh_headings=tuple(mesh),
        identifiers=tuple(identifiers),
        publication_status=_field(data, "./PublicationStatus") if data is not None else None,
        reuse_license=None,
        provenance=provenance,
        text_version=_text_version(pmid, title, vernacular, book_title, abstract, other),
        warnings=tuple(warnings),
    )


def parse_pubmed_xml(body: bytes, *, source_file: str | None = None,
                     snapshot_id: str | None = None, retrieved_at: str | None = None) -> tuple[Article, ...]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ParseError("Malformed PubMed XML") from exc
    if root.tag != "PubmedArticleSet":
        raise ParseError("Expected PubmedArticleSet")
    raw_hash = hashlib.sha256(body).hexdigest()
    articles, seen, counts = [], set(), Counter()
    for i, record in enumerate(root):
        if record.tag not in {"PubmedArticle", "PubmedBookArticle"}:
            raise ParseError(f"Unsupported record type: {record.tag}")
        counts[record.tag] += 1
        provenance = Provenance("pubmed", snapshot_id, source_file, retrieved_at,
            raw_hash, i, f"./{record.tag}[{counts[record.tag]}]")
        article = _record(record, provenance)
        if article.pmid in seen:
            raise ParseError(f"Duplicate PMID: {article.pmid}")
        seen.add(article.pmid)
        articles.append(article)
    return tuple(articles)
