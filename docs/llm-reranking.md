# Listwise Qwen reranking

Qwen receives a question and the title/primary abstract of ten papers. It returns
an ordered JSON array of temporary IDs (D1–D10). These IDs map back to the input
PMIDs; the model never creates article identifiers. Search scores and human
judgments are excluded from its prompt.

This experiment reorders the frozen dense top-10 and RRF top-10 independently.
It cannot recover a paper outside those lists. At k=10, their membership, pooled
recall, precision and judged coverage stay fixed; nDCG and MRR can change.

## Policy

- Qwen/Qwen3.5-9B at `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- One model instance, sequential calls, Transformers/MPS, Metal 4-bit/group 64.
- Greedy decoding, thinking disabled, maximum 128 output tokens.
- At most 768 Qwen tokens per paper, taking the beginning of title + primary
  abstract. No other-language abstract fallback. Missing text stays missing.
- Prompt limit: 12000 tokens including chat-template overhead. Excess input
  raises an error; it is not silently truncated. Each paper's original/shown
  token count and truncation flag are recorded.
- Complete JSON string-array permutations only. Missing, duplicated, invented
  or non-string IDs, wrappers, malformed JSON and output-limit hits preserve
  the original order, with an explicit fallback reason. No retry or repair.
- Runtime failures stop the experiment and leave a failed checkpoint. They
  are not reported as successful fallback rankings.

The system prompt asks for evidence matching the population, intervention,
comparator and outcomes in the question. Paper text is serialized as data and
marked untrusted; this instruction is not a general prompt-injection guarantee.
The prompt and its version/hash are retained locally. It was fixed before the
quality run; no prompt search was performed on these pilot labels.

## Run

Use the Python environment from the [local runtime check](experiments/runtime-smoke.md).
The pinned model weights must already be cached. Model/tokenizer loading uses
`local_files_only=True`. The `kernels` package may need network access to resolve
its Metal kernel version and perform the publisher trust check, so do not force
`HF_HUB_OFFLINE=1` unless that kernel version is also available offline.

```sh
python scripts/evaluate_reranking.py benchmarks/pilot-v1/queries.jsonl \
  --qrels data/evaluation/pilot-v1/pilot-qrels-v2.jsonl \
  --baseline results/evaluation/pilot-v1-v2-retrieval-001.json \
  --output results/evaluation/pilot-v1-v2-qwen-new-run-001.json
```

Prepare the retrieval baseline with the [v2 replay command](pilot-qrels-v2.md#replay-the-current-evaluation),
then choose a fresh output path. This command starts a new model run; rescoring
existing outputs needs no inference. Incomplete judgments remain provisional. Before model loading, the runner checks query/qrels
hashes, dataset identities, query order/text and the baseline metrics. It uses
all ten questions in the baseline's split. Each completed call is checkpointed
atomically. Local reports include raw prompts/responses, rankings, input
truncation, timings and environment/source hashes. Raw article text and individual
labels are excluded from Git.

The [current pilot results](pilot-qrels-v2.md) compare the added stage against
both source rankings and the strongest dense baseline. This development pilot
has ten questions and one annotator; it does not replace held-out evaluation.
