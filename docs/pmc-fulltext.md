# PMC full-text feasibility

This experiment evaluates a small, reproducible **permitted-content** subset and
compares abstract-only windows with the same articles plus narrative JATS body
paragraphs. It measures parsing, embedding and search/context costs. It does not
measure relevance, clinical correctness or answer quality.

## Source and license policy

The snapshot begins from the rebuilt local 277-article development collection.
It visits PMCID-linked candidates in ascending PMID order, sequentially, stopping
at six eligible articles or 36 candidates. The six selected records have explicit
CC BY 2.0, CC BY 3.0 or CC BY 4.0 licenses. A license URI must be on the fixed
allowlist; missing, ambiguous, non-English, unsupported or unavailable OAI
records are rejected. It took 33 requests to find six allowed articles. Twenty-
seven were rejected: 18 lacked an allowlisted license URI, two had ambiguous
licenses and seven could not be disseminated in the requested OAI format.

The snapshot log keeps rejection reason, PMCID/PMID, retrieval time and a response
hash. Raw XML is saved only for accepted records. Rejected response XML from the
initial discovery run was deleted (3,098,920 bytes); six licensed XML files remain
in the ignored local `data/pmc/20261109/snapshot/`. No PMC article text is included
in Git reports.

## Paired artifacts and provenance

Both representations use the same six PMC records and the same JATS abstract.
The full-text representation adds eligible narrative body paragraphs and their
section hierarchy. The parser excludes tables, figures, references, formulas,
supplementary material, sub-articles and other non-narrative structures. Each
paragraph retains its JATS element path, section label and hierarchy, and source
normalization policy. Character offsets refer to the saved normalized paragraph,
not byte positions in the XML. Exact source-window mappings are checked against
the saved record before the artifacts are considered complete.

The PubMed observation stays attached to each article as metadata provenance.
The article text provenance separately identifies the PMC snapshot, response
hash, retrieval time, PMCID, license and OAI datestamp. PubMed publication dates
remain separate from PMC release/retrieval dates. Citations from narrative
passages link to the corresponding PMC article and retain the license/source path.

Publication date filters use only the PubMed `journal_issue` date. A year or
month becomes its full calendar interval, and bounds include overlapping partial
intervals. A bounded filter excludes unknown dates. OAI datestamps, record update
dates and PMC release dates never stand in for publication dates. For the six
records, inclusive lower bound `2016-08-27` selects three. Each known article's
own interval endpoints passed inclusive-boundary checks. This demonstrates
filter behavior on a small known-date sample, not the completeness of PubMed date
metadata. The existing request CLI can apply the same inclusive publication
bounds to this PMC collection, for example:

```sh
.venv/bin/python scripts/ask.py data/pmc/20261109/fulltext/dataset \
  --windows data/pmc/20261109/fulltext/windows \
  --embeddings data/pmc/20261109/fulltext/embeddings \
  --query "Which outcomes are reported for cognitive behavioral therapy?" \
  --published-from 2014-01-01 --published-until 2019-12-31 \
  --output-dir results/requests/pmc-date-example
```

## Cost and retrieval comparison

The [aggregate report](experiments/pmc-fulltext.json) records pinned model and
code identities, all ten fixed development query IDs, artifact hashes, timing
scopes and memory counters. It contains IDs and aggregate diagnostics but no
article text, generated claims or raw responses. Reproduce the fetch, pair and
search using fresh destinations:

```sh
.venv/bin/python scripts/pmc_pilot.py fetch \
  --base data/collections/shared-20261108/shared \
  --output data/pmc/pmc-new/snapshot --limit 6 --max-candidates 36
.venv/bin/python scripts/pmc_pilot.py build --snapshot data/pmc/pmc-new/snapshot \
  --output data/pmc/pmc-new/abstract --mode abstract
.venv/bin/python scripts/pmc_pilot.py build --snapshot data/pmc/pmc-new/snapshot \
  --output data/pmc/pmc-new/fulltext --mode fulltext
.venv/bin/python scripts/benchmark_pmc_pilot.py \
  --output docs/experiments/pmc-new.json --repeats 5
```

The checked-in run contains 14 abstract windows for six articles. Adding the
narrative body produces 288 body windows (302 total) across 272 paragraphs. The
abstract build took 1.646 s and encoded 14 new vectors. The full-text build took
12.322 s, reused all 14 abstract vectors, and encoded 288 new vectors. These
prepublication timers exclude Python startup and tokenizer loading. The full-text
matrix is 927,744 bytes versus 43,008 bytes for the abstract matrix. Sampled MPS
driver allocations during the build were 1.083 GiB for abstracts and 2.008 GiB
for full text; these are post-forward samples, not peak device memory.

Search used one resident BGE model, ten development queries and five paired
repetitions. Both indexes used the same query vector in each pair, and traversal
order alternated. Warm-up and model construction are reported separately. Median
exact article search was 0.055 ms for abstracts and 0.082 ms with full text.
Median BGE query encoding was 14.243 ms, measured once per paired repetition.
Computing window scores and packing source evidence took a median 9.741 ms and
9.923 ms, respectively. The selected six candidate IDs are identical by design;
article ordering varies. Median Kendall rank agreement across queries was 0.80,
with two to six of six articles remaining in the same rank position per query.
The top-ranked article stayed the same in seven of ten queries.

Evidence packing selected six abstract fragments in abstract-only mode. With full
text enabled, the median was five narrative fragments and one abstract fragment
per query; the six-fragment policy still applied. Median evidence size fell from
2,118 to 1,737 Qwen tokens in this particular sample. This describes the window
and packing behavior. It does not establish that those fragments better answer
the question.

## Decision and limits

Keep full text as a license-gated, opt-in research mode. It preserves exact
section provenance and supplies more narrative passages at a modest measured
search cost. The development pilot has no independent judgments for these six
new articles; the fixed query ranking shifts are not evidence of better retrieval.
Do not apply historical qrels to these candidates, change the primary abstract
workflow or claim answer improvement from this experiment. The retrieval sample
is small and old by publication date because the deterministic candidate order
starts at the lowest PMID. Recency does not establish evidence quality. Broader
PMC selection, relevance labels and independent answer review are needed before
considering a default change.
