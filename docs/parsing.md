# Parsed PubMed records

Build from a complete snapshot; this step makes no network requests:

```sh
python scripts/parse_pubmed_snapshot.py build data/snapshots/depression-cbt \
  --output data/parsed/depression-cbt
python scripts/parse_pubmed_snapshot.py inspect data/parsed/depression-cbt
```

The output directory must be new. It contains `articles.jsonl`, `manifest.json`,
and an exact copy of the source `manifest.json` as `source-manifest.json`.
Records stay in the original ESearch order, even if EFetch reordered a batch.
`load_dataset` checks hashes, record count/order, and source provenance.

## Fields and policies

- Titles and abstract sections keep adjacent inline text intact. Display text
  collapses whitespace; `raw_title` and section `raw_text` retain it. Labels,
  NLM categories, and section order are separate fields. Text formatting is not
  rendered. MathML gets a warning; its original structure remains in raw XML.
- A primary abstract can be present, empty, or missing. Other-language abstracts
  stay separate; they do not silently replace the primary one.
- Author lists retain their scope, order, and completeness attributes. Collective
  names stay collective names. Book/chapter author lists and book editors remain
  distinguishable by source path and list attributes.
- Dates keep their kind, raw components, source path, and precision. A known year
  becomes `YYYY`, a known month `YYYY-MM`, and a complete date `YYYY-MM-DD`.
  Seasons, ranges, and unparsed values get no fabricated ISO day. Electronic,
  journal-issue, received, accepted, and indexing dates are not interchangeable.
  This parser does not choose the date for a future freshness signal.
- Publication types and MeSH keep their labels, IDs, and attributes. They are not
  converted to a quality score. Article IDs are read from record-level fields,
  not references; identifier validity flags are retained.
- Missing metadata stays missing. A copyright statement is preserved separately
  and does not imply a reuse license. `reuse_license` is unknown for this input.

Each record points to a source file, retrieval time, snapshot ID, raw response
SHA-256, and record position/path. `text_sha256` describes the normalized titles
and abstract text/labels; `text_version_id` also includes PMID and parser version.
Metadata-only changes leave the text version unchanged, while the raw response
hash still exposes a changed source document.

The dataset manifest records source, output, and parser-code hashes. Its ID is
stable for the same source and implementation; creation time is excluded from
that identity. These are integrity checks, not signatures of authenticity.

## Check on the saved snapshot

On 2026-10-20, all 30 records from the previous day's snapshot were parsed and
reloaded in order. All had titles and primary abstracts; 17 had multiple abstract
sections. Journal dates included 12 complete dates, 14 year/month dates, one
year-only date, and three seasons. No parser warnings occurred in this sample.

The [machine-readable report](experiments/pubmed-parsing.json) contains the exact
dataset ID, hashes, and counts. Article text remains in the local data directory.
The 46 offline tests also cover missing fields, books, collective authors,
partial dates, corrupt artifacts, and duplicate/conflicting primary IDs.

This handles the selected PubMed citation fields, not arbitrary publisher XML,
full book chapters, or PMC full text. It does not validate the entire DTD or
render mathematical notation. The implementation follows the
[PubMed export DTD referenced by the saved XML](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/250101/index.html),
including [abstract sections](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/250101/el-AbstractText.html),
[publication dates](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/250101/el-PubDate.html),
and [history dates](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/250101/el-PubMedPubDate.html).
