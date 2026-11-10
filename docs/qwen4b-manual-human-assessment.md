# Qwen 4B: manual human assessment

**The project owner manually supplied every human label and rationale in this evidence package.** These are MANUAL HUMAN ANNOTATIONS, not model-generated labels. The assistant preserved both original JSON exports byte for byte, checked their bindings, and computed the numeric audit. It did not rewrite rationales or regrade uncertain judgments.

The owner subsequently selected 4B for continuing development and authorized its pull request. This records a development choice; the pilot did **not** pass the safety and usability gates for unattended answers. Both 4B and 9B reviewers admitted human-ineligible claims. 9B is a comparison model, not ground truth.

## Scope and policy

The frozen development slice contains two independent depression/anxiety questions, six correlated source contexts, 30 unique claims, 32 claim occurrences, six paired answers, and one human annotator. Exact duplicates were retained with origin mappings; any/all acceptance bounds coincide because neither reviewer disagreed across those duplicates. These observations do not establish general biomedical quality, statistical superiority, or equivalence.

The manual comparison concerns the **frozen named axes-first, format-only reviewer arm** and its recombined answers. It does not independently validate outputs after changing prompts or enabling the new consistency-enforcing decoder. Enforced JSON consistency is not clinical truth.

## Claim-level results

The manual reference marks 24 claims eligible, five ineligible by the protocol, and one uncertain. All 30 factual-support labels are `supported`. In this particular reference, the five ineligible claims fail comparison alignment; they are not annotated as factually false. The owner's rationales capture the intended control comparison. Uncertain comparison remains uncertain.

| Unique-claim measure | 4B reviewer | 9B reviewer |
| --- | ---: | ---: |
| Accepted eligible claims | 16 / 24 | 15 / 24 |
| Accepted human-ineligible comparisons | 3 / 5 | 4 / 5 |
| Rejected eligible claims | 8 / 24 | 9 / 24 |
| Accepted uncertain claims | 1 / 1 | 0 / 1 |

The relative-recall threshold of at least 90% of 9B passed. The stricter requirement of zero observed accepted human-ineligible claims **failed for both reviewers**. The 4B reviewer also accepted the uncertain claim, which the protocol requires withholding. A passed recall or speed gate does not erase these failures.

Before filtering, 4B produced 18 claims across six parseable drafts: 14 eligible, three ineligible comparisons, and one uncertain. The five parseable 9B drafts contained 14 claims: 12 eligible and two ineligible comparisons; its sixth draft had invalid JSON and was not repaired. Two claims occur in both generator-origin groups, so these group totals are not 32 independent examples.

## Whole-answer manual results

| Owner's rating | 4B workflow | 9B workflow |
| --- | ---: | ---: |
| Overall usable | 5 / 6 | 4 / 6 |
| Adequately complete | 2 / 6 | 0 / 6 |
| Qualifiers preserved | 3 / 6 | 1 / 5 nonempty answers |

One 9B answer was empty after a format error. Its original human ratings remain unchanged; the `unknown` qualifier rating is treated as N/A for that empty answer. The formal paired-completeness check had zero adequate 9B answers, so its pass is not evidence that either workflow is complete. The candidate's overall-usability gate failed on one context.

## Timing and limitations

The paired median native reviewer-stage ratio was **2.27×** in favor of 4B on 11 identical frozen inputs. This excludes model loading, draft generation, retrieval, networking, and end-to-end request latency. It is not a live-search latency guarantee.

The compared model artifacts also differ in precision and quantization coverage: the pinned community 4B artifact retains BF16 parameters and FP32 `A_log`, with quantized tied embeddings/head; 9B uses the pinned official source, FP16 parameters and FP32 `A_log`, with embedding/head outside 4-bit quantization. The community artifact does not disclose an exact upstream source revision or converter commit. Model size alone cannot explain the observed results.

## Evidence and verification

The tracked [evidence package](../data/evaluation/qwen4b-manual-human-v1/) contains the two unchanged human exports, text-free prediction and blind-mapping audits, current provenance, and a standard-library verifier. The [compact assessment](experiments/qwen4b-manual-human-assessment.json) records counts, denominators, duplicate bounds, and gate outcomes.

```bash
python3 data/evaluation/qwen4b-manual-human-v1/verify.py
```

The verifier checks the exact original SHA256 values, the finalized claim-reference checksum, human/source/output bindings, and numeric totals. Claim labels were explicitly finalized in the form. The separate answer form had no finalize flag; the owner explicitly submitted it as completed, and its supplied values were checked without modification.

Source articles, generated claim/answer prose, model weights, and ignored local research reports are not included. Their existing hashes bind the compact evidence to the locally verified run. This package supports a numeric/provenance audit; independent semantic rejudgment still requires the original local source and answer artifacts. Historical local-only status in earlier research reports is preserved as history, not represented as the current PR authorization.
