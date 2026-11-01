# Held-out questions v1

Frozen on 2026-11-01 before MedCPT ranking experiments. Twelve new English
questions remain within depression and anxiety; their intervention/population
intents differ from the ten pilot questions. They are research questions, not
claims of treatment effectiveness. Dataset paths are reserved; no candidates,
model scores or relevance judgments have been collected for this split.

## Protocol

- Pilot questions are development data. Do not run held-out ranking or generation
  to tune models, windows, aggregation, prompts or thresholds. Freeze the winner
  and its runtime configuration before the Day 26 confirmation run.
- `candidate-plan.json` pins question-file SHA256, PubMed expressions, Best Match
  ordering and a limit of thirty candidates per question. Save one complete
  snapshot per expression, including retrieval time, then share it across all
  methods. This assesses ordering within retrieved candidates, not PubMed-wide
  recall. API results may change before snapshot capture; only the saved snapshot
  becomes the final corpus reference.
- Score PubMed Best Match, BGE Dense, Dense top-10 articles → MedCPT, and direct
  MedCPT over every candidate. Use the same source windows and article maximum
  aggregation for both MedCPT modes. Freeze their exact text/input policies and
  model revisions on development data before inspecting test outputs.
- Judge all available candidate articles for each question on the existing
  0/1/2 relevance scale. Keep uncertain pairs unjudged rather than turning them
  into zero. Hide method identities, scores and ranks during assessment; record
  who or what produced the judgments and what was actually blinded. AI assessment
  is not independent expert validation.
- Primary metric: macro nDCG@10, gain `2**grade - 1`, logarithmic rank discount.
  Secondary: Recall@10 and MRR@10 with grades 1 and 2 counted as relevant. Use
  `medlens.evaluation.metrics` and preserve its undefined-metric handling. Report
  defined/total query counts and judgment coverage. Incomplete judgment sets are
  provisional; do not select a winner on apparent gains caused by missing labels.
- Report query-level results and paired differences, latency, scored pair counts,
  load time and the memory measurement scope. Separate cold preparation from
  cached retrieval. Day 26 results confirm the preselected pipeline; a change
  motivated by those results requires a new future test split for confirmation.
- For generation, inspect exact citations, population/severity/comparison scope,
  unsupported claims, completeness, abstention and contradictions. Keep format
  validity separate from scientific support. Record the assessor's provenance;
  the same Qwen instance is not an independent judge.

This is query-level holdout, not a disjoint-paper or disjoint-disease benchmark.
Some relevant publications or clinical concepts may overlap the pilot. Check and
disclose overlap when snapshots are collected. The question design was informed
by the project's domain, not by held-out model outputs. Twelve questions and AI
labels cannot establish broad biomedical or clinical validity.

Changes require a new version and a reason. Do not silently rewrite these
questions, candidate expressions or evaluation rules after seeing test results.
