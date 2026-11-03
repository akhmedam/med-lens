# MedCPT article ranking

Two modes now use the pinned [MedCPT scorer](medcpt.md) on the same verified
local datasets and source windows:

- **Dense → MedCPT:** BGE selects ten articles, then MedCPT scores every window
  of those articles. The cross-encoder is not restricted to BGE's best window.
- **Direct MedCPT:** MedCPT scores every window of every candidate article. No
  BGE model, query embedding or saved embedding artifact is loaded in this path.

Both modes use the maximum raw window logit as the article score. Sort scores
descending, retaining negative values. Exact article ties retain eligible input
order: Dense order for the subset, original candidate order for the direct path.
Exact window ties retain source-window order. Articles without windows receive no
invented score and are listed separately. Title-only windows can be ranked but
are not abstract evidence.

Every selected pair is checked against MedCPT's full 512-token budget. A bad
pair or invalid score fails the request; no window is silently truncated,
ignored or assigned a default relevance value. The saved record contains every
scored window ID, input length and raw score, plus each result's best source
window, original character offsets and provenance. This supports aggregation
replay without loading a model.

## Search commands

From the project root with the existing assets:

```sh
HF_HUB_OFFLINE=1 python scripts/medcpt_search.py data/parsed/pilot-v1/p01 \
  --windows data/windows/pilot-v1-002/p01 \
  --mode dense-medcpt \
  --embeddings data/embeddings/pilot-v1/p01 \
  --query "How does cognitive behavioral therapy compare with antidepressants for depressive symptoms in adults with major depression?" \
  --output results/ranking/dense-medcpt-example.json
```

For direct scoring, use the same dataset, windows and question, change the mode
to `--mode medcpt`, and omit `--embeddings`. The script prints PMID/title/score
rows and saves the full JSON. It never fetches new PubMed candidates. Use fresh
output paths. `--device cpu` is explicit; the default is MPS with batch size four.
`--top-k` limits returned articles to one through ten. Dense still selects up to
ten candidates before reranking, independently of that output limit. Changing
candidate depth is not part of this comparison.

Setup/model/scoring failures are checkpointed with their stage and return exit
code 1; an interrupt returns 130. Invalid CLI arguments or an existing output
return 2 before model inference. Success returns 0. A scoring failure does not
fall back to a different retrieval method or emit a partial MedCPT ranking.
In Dense mode the completed selection remains available in the failed report.

## Prepare the pilot comparison

```sh
HF_HUB_OFFLINE=1 python scripts/prepare_medcpt_rankings.py \
  --output results/ranking/pilot-v1-medcpt-example.json \
  --summary results/ranking/pilot-v1-medcpt-example-summary.json
```

This runner accepts pilot development queries only. It validates paired source
identities, loads BGE once for all ten selections, releases it, then loads
MedCPT once for both modes. The source text/window boundaries do not change.
Qwen, relevance labels and the held-out split are not used. Progress checkpoints
preserve completed methods if a later query fails.

The [saved preparation run](experiments/medcpt-ranking.json) completed all ten
questions and both modes on MPS, with thirty candidate articles per question:

| Mode | Articles scored per question | Total window pairs | Median ranking time per question | Range |
| --- | ---: | ---: | ---: | ---: |
| Dense → MedCPT | 10 | 350 | 0.476 s | 0.296–0.714 s |
| Direct MedCPT | 30 | 975 | 1.388 s | 1.048–3.144 s |

These are single sequential preparation calls with a reused model, not cold
request latencies. The ranking timings include tokenization, forward passes and
article aggregation, but exclude model loading, source/index validation and
Dense candidate selection. The raw report records those components separately.
No explicit warm-up call was made. Model load/filesystem cache and batching
must be accounted for in the broader Day 18 cost comparison. Process peak RSS
was 1.09 GiB across sequential BGE/MedCPT stages; it is not a device memory peak.

All shared-window scores between the two modes agree within absolute tolerance
1e-4; maximum observed drift was 0.00001336. Batch padding can cause small
floating-point differences, so this is not bitwise identity or a guarantee for
near-tied ordering. The [Day 17 evaluation](medcpt-evaluation.md) adds new judgments and compares
quality. No quality improvement is claimed by this preparation report.
The corresponding full local artifact is
`results/ranking/pilot-v1-medcpt-002.json`; raw reports stay outside Git.

An [offline audit](experiments/medcpt-ranking-audit.json) verified every one of
1,325 pair encodings against its stored source and replayed all twenty article
rankings independently from their saved scores. All 200 best-window references
matched their exact source fields and offsets. The standalone CLI also completed
both modes on p01; direct mode succeeded with nonexistent BGE asset paths and no
embedding input. An oversized 849-token pair produced a saved failure and no
partial ranking. These are integration/source checks, not relevance judgments.

## API

```python
from medlens.reranking.articles import MedCPTRanker
from medlens.reranking.medcpt import MedCPTScorer

ranker = MedCPTRanker('data/parsed/pilot-v1/p01', 'data/windows/pilot-v1-002/p01')
scorer = MedCPTScorer('.cache/models/medcpt-cross-encoder', device='mps', batch_size=4)
all_candidates = ranker.run(question, scorer)
# For Dense → MedCPT, pass the ordered PMIDs returned by the Dense top-10 stage.
subset = ranker.run(question, scorer, eligible_pmids=dense_pmids)
```

The input question and eligible list are caller-supplied in this example.
`aggregate_windows` can reconstruct article ordering from saved window scores;
input provenance should be checked against the matching stored artifacts first.

## Integration boundary

The current `ask.py --rerank` still uses Qwen. These commands provide the new
ranking experiments; primary-pipeline integration follows selection on Day 19.
MedCPT logits are not cosines or calibrated probabilities. The existing evidence
builder currently expects cosine scores; integration must explicitly support the
new score kind rather than clamping logits or labeling them cosine similarity.

Maximum pooling can favor an article with many windows; the
[Day 17 case analysis](medcpt-evaluation.md#source-based-error-analysis) inspects
selected-window and scope errors without claiming a causal max-pooling ablation. Ranking relevance also does not establish claim entailment.
The twelve [held-out questions](../benchmarks/heldout-v1/README.md), their protocol
and the active pilot qrels v2 remain unchanged and unscored here.
