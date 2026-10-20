"""Field semantics and provenance, using synthetic XML only."""

import hashlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from medlens.parsing.pubmed import ParseError, parse_pubmed_xml

FIXTURE = Path(__file__).parent / "fixtures/pubmed/structured.xml"


def record(fragment, pmid="1"):
    return (f"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>{pmid}</PMID>"
            f"<Article>{fragment}</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>").encode()


class PubMedParsingTests(unittest.TestCase):
    def setUp(self):
        self.raw = FIXTURE.read_bytes()
        self.articles = parse_pubmed_xml(self.raw, source_file="fixture.xml", snapshot_id="fixture-snapshot",
                                         retrieved_at="2026-10-19T08:00:00+00:00")
        self.first, self.book, self.last = self.articles

    def test_preserves_inline_text_without_inserting_spaces_inside_words(self):
        self.assertEqual(self.first.title, "Treating depression without sedation.")
        self.assertEqual(self.first.abstract.sections[0].text, "An antidepressant trial & follow-up.")
        self.assertEqual(self.first.abstract.sections[1].text, "No benefit was observed at 12 weeks.")
        self.assertIn("\n", self.first.abstract.sections[1].raw_text)

    def test_keeps_section_order_and_original_headings(self):
        sections = self.first.abstract.sections
        self.assertEqual([s.index for s in sections], [0, 1])
        self.assertEqual([s.label for s in sections], ["OBJECTIVE", "RESULTS"])
        self.assertEqual(sections[1].nlm_category, "RESULTS")
        self.assertEqual(self.first.abstract_status, "present")

    def test_collective_authors_and_incomplete_author_lists(self):
        group = self.first.author_lists[0]
        self.assertEqual(group.attributes["CompleteYN"], "N")
        self.assertEqual(group.authors[0].suffix, "Jr")
        self.assertEqual(group.authors[0].identifiers[0].kind, "ORCID")
        self.assertEqual(group.authors[0].affiliations, ("Example University.",))
        self.assertEqual(group.authors[1].collective_name, "Example Research Group")
        self.assertIsNone(group.authors[1].last_name)

    def test_publication_types_and_mesh_attributes_are_not_quality_scores(self):
        self.assertEqual(self.first.publication_types[0].attributes["UI"], "D016449")
        heading = self.first.mesh_headings[0]
        self.assertEqual(heading.descriptor.attributes["MajorTopicYN"], "Y")
        self.assertEqual(heading.qualifiers[0].label, "drug therapy")

    def test_dates_keep_publication_and_processing_events_separate(self):
        dates = {d.kind: d for d in self.first.dates}
        self.assertEqual(dates["journal_issue"].iso_date, "2024-02-29")
        self.assertEqual(dates["article_electronic"].iso_date, "2024-01-15")
        self.assertEqual(dates["history:received"].iso_date, "2023-10-01")
        self.assertEqual(dates["history:entrez"].raw_components["Hour"], "6")
        self.assertEqual(dates["record:DateRevised"].iso_date, "2025-04-02")

    def test_partial_season_and_range_dates_do_not_invent_days(self):
        examples = [
            ("<Year>2020</Year>", "2020", "year"),
            ("<Year>2020</Year><Month>September</Month>", "2020-09", "month"),
            ("<Year>2020</Year><Season>Spring</Season>", None, "season"),
            ("<MedlineDate>2020 Dec-2021 Jan</MedlineDate>", None, "free_text"),
            ("<Year>2023</Year><Month>Feb</Month><Day>29</Day>", None, "unparsed"),
            ("<Year>2020</Year><Month>Jan-Feb</Month>", None, "unparsed"),
        ]
        for fields, iso, precision in examples:
            with self.subTest(fields=fields):
                parsed = parse_pubmed_xml(record(f"<Journal><JournalIssue><PubDate>{fields}</PubDate></JournalIssue></Journal>"))[0]
                self.assertEqual(parsed.dates[0].iso_date, iso)
                self.assertEqual(parsed.dates[0].precision, precision)
                self.assertTrue(parsed.dates[0].raw_components)

    def test_missing_primary_abstract_does_not_use_other_language_silently(self):
        self.assertEqual(self.last.abstract_status, "missing")
        self.assertIsNone(self.last.abstract)
        self.assertEqual(self.last.other_abstracts[0].attributes["Language"], "fre")
        self.assertEqual(self.last.languages, ())
        self.assertEqual(self.last.publication_types, ())

    def test_empty_and_unstructured_abstracts_are_distinct_from_missing(self):
        empty = parse_pubmed_xml(record("<Abstract><AbstractText> </AbstractText></Abstract>"))[0]
        self.assertEqual(empty.abstract_status, "empty")
        plain = parse_pubmed_xml(record("<Abstract><AbstractText>Plain text.</AbstractText></Abstract>"))[0]
        self.assertEqual(plain.abstract_status, "present")
        self.assertIsNone(plain.abstract.sections[0].label)

    def test_copyright_does_not_become_a_license(self):
        self.assertIn("copyright", self.first.abstract.copyright)
        self.assertIsNone(self.first.reuse_license)

    def test_identifiers_exclude_references_and_keep_validity(self):
        values = [identifier.value for identifier in self.first.identifiers]
        self.assertIn("10.0000/main-fixture", values)
        self.assertNotIn("10.0000/reference", values)
        self.assertNotIn("999", values)
        invalid = next(i for i in self.first.identifiers if i.value == "10.0000/invalid-fixture")
        self.assertEqual(invalid.attributes["ValidYN"], "N")
        self.assertEqual(self.first.pmid_version, "2")

    def test_books_keep_editor_and_chapter_author_scopes(self):
        self.assertEqual(self.book.record_type, "PubmedBookArticle")
        self.assertIsNone(self.book.journal)
        self.assertEqual(self.book.book_title, "Example Handbook")
        self.assertEqual([g.attributes["Type"] for g in self.book.author_lists], ["authors", "editors"])
        self.assertEqual(self.book.dates[0].precision, "free_text")
        self.assertIn("book_full_text_not_parsed", self.book.warnings)

    def test_provenance_paths_identify_the_actual_record_and_sections(self):
        root = ET.fromstring(self.raw)
        self.assertEqual(self.last.provenance.record_path, "./PubmedArticle[2]")
        self.assertEqual(self.last.provenance.record_index, 2)
        self.assertEqual(self.first.provenance.raw_response_sha256, hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(self.first.provenance.snapshot_id, "fixture-snapshot")
        for article in self.articles:
            self.assertIsNotNone(root.find(article.provenance.record_path))
            if article.abstract:
                for section in article.abstract.sections:
                    self.assertIsNotNone(root.find(section.source_path))

    def test_text_versions_are_stable_across_metadata_and_retrieval_time_changes(self):
        changed = self.raw.replace(b"<LastName>Smith</LastName>", b"<LastName>Other</LastName>")
        updated = parse_pubmed_xml(changed, retrieved_at="later")[0]
        self.assertEqual(updated.text_version, self.first.text_version)
        self.assertNotEqual(updated.provenance.raw_response_sha256, self.first.provenance.raw_response_sha256)
        new_text = parse_pubmed_xml(self.raw.replace(b"No benefit", b"Some benefit"))[0]
        self.assertNotEqual(new_text.text_version, self.first.text_version)

    def test_dtd_and_mathml_do_not_trigger_network_access(self):
        with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
            self.assertEqual(len(parse_pubmed_xml(self.raw)), 3)
            math = parse_pubmed_xml(record('<Abstract><AbstractText xmlns:m="http://www.w3.org/1998/Math/MathML"><m:math><m:mi>x</m:mi></m:math></AbstractText></Abstract>'))[0]
        self.assertIn("mathml_plain_text_projection", math.warnings)

    def test_rejects_malformed_xml_unknown_records_and_ambiguous_identity(self):
        bad = [b"<PubmedArticleSet>", b"<html/>", b"<PubmedArticleSet><DeleteCitation/></PubmedArticleSet>",
               record("", pmid="invalid"), self.raw.replace(b'<PMID Version="2">101</PMID>', b'<PMID>101</PMID><PMID>102</PMID>'),
               self.raw.replace(b'<ArticleId IdType="pubmed">101</ArticleId>', b'<ArticleId IdType="pubmed">999</ArticleId>')]
        for body in bad:
            with self.subTest(body=body[:40]), self.assertRaises(ParseError):
                parse_pubmed_xml(body)
        duplicate = b"<PubmedArticleSet>" + b"<PubmedArticle><MedlineCitation><PMID>1</PMID><Article/></MedlineCitation></PubmedArticle>" * 2 + b"</PubmedArticleSet>"
        with self.assertRaises(ParseError):
            parse_pubmed_xml(duplicate)


if __name__ == "__main__":
    unittest.main()
