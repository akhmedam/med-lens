"""The PubMed fields used by retrieval, display, and provenance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Identifier:
    kind: str | None
    value: str
    source_path: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class Author:
    last_name: str | None
    fore_name: str | None
    initials: str | None
    suffix: str | None
    collective_name: str | None
    affiliations: tuple[str, ...]
    identifiers: tuple[Identifier, ...]
    attributes: dict[str, str]
    source_path: str


@dataclass(frozen=True)
class AuthorList:
    authors: tuple[Author, ...]
    attributes: dict[str, str]
    source_path: str


@dataclass(frozen=True)
class AbstractSection:
    index: int
    label: str | None
    nlm_category: str | None
    text: str
    raw_text: str
    source_path: str


@dataclass(frozen=True)
class Abstract:
    sections: tuple[AbstractSection, ...]
    copyright: str | None
    attributes: dict[str, str]
    source_path: str


@dataclass(frozen=True)
class RecordDate:
    kind: str
    raw_components: dict[str, str]
    attributes: dict[str, str]
    iso_date: str | None
    precision: str
    source_path: str


@dataclass(frozen=True)
class Term:
    label: str
    raw_label: str
    attributes: dict[str, str]
    source_path: str


@dataclass(frozen=True)
class MeshHeading:
    descriptor: Term
    qualifiers: tuple[Term, ...]


@dataclass(frozen=True)
class Journal:
    title: str | None
    iso_abbreviation: str | None
    nlm_abbreviation: str | None
    issns: tuple[Identifier, ...]
    volume: str | None
    issue: str | None


@dataclass(frozen=True)
class Provenance:
    source: str
    snapshot_id: str | None
    source_file: str | None
    retrieved_at: str | None
    raw_response_sha256: str
    record_index: int
    record_path: str


@dataclass(frozen=True)
class TextVersion:
    parser_version: str
    text_sha256: str
    text_version_id: str


@dataclass(frozen=True)
class Article:
    pmid: str
    pmid_version: str | None
    record_type: str
    title: str | None
    raw_title: str | None
    vernacular_title: str | None
    book_title: str | None
    journal: Journal | None
    author_lists: tuple[AuthorList, ...]
    abstract: Abstract | None
    abstract_status: str
    other_abstracts: tuple[Abstract, ...]
    dates: tuple[RecordDate, ...]
    languages: tuple[str, ...]
    publication_types: tuple[Term, ...]
    mesh_headings: tuple[MeshHeading, ...]
    identifiers: tuple[Identifier, ...]
    publication_status: str | None
    reuse_license: str | None
    provenance: Provenance
    text_version: TextVersion
    warnings: tuple[str, ...]
