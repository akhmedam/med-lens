# Ten-query pilot

The [query set](../benchmarks/pilot-v1/queries.jsonl) contains five depression and
five anxiety questions. The [plan](../benchmarks/pilot-v1/plan.json) fixes the
PubMed queries and per-question relevance intent before examining results.
These are authoring suggestions for the pilot, not a representative clinical
benchmark. All 161 pairs have now been judged; see the [baseline review](pilot-baseline.md).

## Saved candidates

Each query has its own 30-candidate PubMed snapshot and verified parsed dataset.
Candidate statistics are never shared across queries. On 2026-10-23, all ten
downloads completed: 300 candidate memberships, 275 unique PMIDs. The snapshots,
parsed text, annotation packets, and judgments stay in ignored local directories.
Only queries, code, and the [preparation report](experiments/pilot-preparation.json)
are committed.

Verify the existing artifacts offline, or add `--fetch` to allow missing snapshots
to be downloaded through NCBI:

```sh
python scripts/fetch_pilot.py benchmarks/pilot-v1/queries.jsonl \
  --plan benchmarks/pilot-v1/plan.json --snapshots data/snapshots/pilot-v1
```

Existing snapshots must match the plan's query and candidate limit. A failed or
changed snapshot is not silently replaced. For a fresh live corpus use a new
pilot version, dataset paths, and snapshot directory. Repeating a live search
does not recreate the exact historical candidate set.

## Blinded pool and local annotation

The pool is the unique PMID union of PubMed and BM25 top-10 **per query**. The
same PMID under two questions needs two independent judgments. Order is a
deterministic hash shuffle, independent of source rank. Dense and RRF will require
an expanded pool later; they are not part of this comparison yet.

```sh
python scripts/annotate_pilot.py build benchmarks/pilot-v1/queries.jsonl \
  --plan benchmarks/pilot-v1/plan.json --output data/annotation/pilot-v1-002
python -m http.server 8766 --bind 127.0.0.1 --directory data/annotation
```

Choose a new output directory if the packet already exists. Open
`http://127.0.0.1:8766/pilot-v1-002/` on the same computer. The HTML is standalone
and can also be opened as a local file, though browser storage behavior can vary.
The server is read-only and bound to localhost. Ctrl-C stops it.

The page presents the question, its relevance intent, title, primary abstract,
and a PubMed link. It contains no method names, retrieval scores, or source
ranks. A separate local manifest retains those details for audit; this is
annotation blinding, not access control. Avoid inspecting that manifest while
judging.

There are **161 cards**: 16, 15, 17, 18, 17, 15, 13, 16, 17, and 17 for p01–p10.
Start with p01's 16 cards if a full session is too long. Use the
[0/1/2 rubric](evaluation.md); absence of an abstract is not an automatic grade 0.
Seven candidate records lack a primary abstract, one lacks an article title, and
three are book records whose full text is outside the parser's scope. The page
shows missing text rather than fabricating it.

Scores persist in the browser under the packet ID. **Download annotations before
clearing browser storage or changing devices.** The JSON export can be reimported
into the same packet. Import replaces that packet's current browser labels, so
download the current state first if both versions matter. Nothing is uploaded.
The packet ID binds the labels to the questions and displayed text versions;
another pool's export is rejected.

## Export and evaluate

After clicking “Скачать разметку”, validate the downloaded file and convert it to
the evaluator's JSONL format:

```sh
python scripts/annotate_pilot.py export data/annotation/pilot-v1-002 \
  --annotations /path/to/medlens-annotations.json \
  --output data/evaluation/pilot-v1/human-001.jsonl
python scripts/evaluate_retrieval.py benchmarks/pilot-v1/queries.jsonl \
  --qrels data/evaluation/pilot-v1/human-001.jsonl --split pilot \
  --output results/evaluation/pilot-v1-human-001.json
```

Incomplete annotation requires `--allow-partial` on export. Such a report is
interim: inspect judged coverage and defined-query counts, not just the macro
mean. An adjacent receipt preserves the source pool and annotation-file hashes.
Keep that receipt with the qrels. The general evaluator also accepts standalone
qrels and does not enforce this receipt automatically.

No human labels have been created by the preparation code. A full unjudged run
was exercised through export and evaluation; quality metrics remain null. The preparation run was followed by human annotation and the [baseline review](pilot-baseline.md). Use pilot groups for
development; keep their paraphrases out of the held-out test set.

## Checks

All 74 offline tests pass. New tests cover pool union/deduplication, reproducible
blinding, source text escaping, pool binding, corrupted packets, partial export,
duplicate/invalid labels, and overwrite protection. In a separate synthetic UI
packet, setting a score survived reload, clearing it restored the unjudged count,
and navigation changed the displayed card. No real article was labeled in that
UI check.
