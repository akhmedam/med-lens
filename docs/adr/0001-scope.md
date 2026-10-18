# Scope and corpus

Status: accepted

## Context

This project is for learning and measuring biomedical retrieval on one laptop.
A full PubMed dump would add work before we have a useful evaluation.

## Decision

Search PubMed dynamically for English queries about depression and anxiety.
Start with 30–50 candidates, target 100–200. Keep snapshots so ranking experiments
can use identical inputs. Begin with titles and abstracts; add permitted PMC full
text as a separate experiment.

Use BM25S and BGE-base-en-v1.5, exact NumPy cosine search, then RRF. Long abstracts
use section-bounded windows starting at 384 tokens with overlap 64. Reserve space
for the title and special tokens within BGE's 512-token input limit. Article dense
score is the maximum over windows. Later evaluate chunk retrieval across all
PubMed candidates.

The first interface is a CLI. Queries and generated answers are in English.
Python modules and functions are enough; add interfaces where replacement is
actually needed.

## Alternatives

A local corpus improves offline availability but requires collection and updates.
An ANN service can help at larger scales; it is unnecessary for this candidate pool.

## Consequences

No downstream ranker can recover a paper missing from the PubMed candidates.
Report candidate coverage separately. Cache contents must not silently change
BM25 corpus statistics. Snapshots freeze experiments, not the live corpus.

Evaluation targets 50 manually judged queries, split 30 development / 20 held-out,
with grouped paraphrases. Pool top-10 PubMed, BM25, dense, and RRF by PMID. Relevance
is 0/1/2; unjudged remains distinct. Primary metric: nDCG@10. BioASQ is a separate
external check if a suitable subset is available.
