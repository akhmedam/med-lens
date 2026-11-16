# Request isolation and source-selection limits

The current application policy is `medlens-mvp-source-findings-v2`, with selector
`source-findings-task-scoped-v5-candidate1`. The CLI and local API share these
rules. This page describes implementation and consumed development observations;
it does not replace a frozen confirmation record or establish clinical quality.

## One request budget

| Boundary | Policy |
| --- | --- |
| Start | HTTP ingress before body validation; CLI before validation/admission, or invocation for a one-shot command |
| Work | Up to 55 seconds from that start, including acquisition, preparation, model loading and inference |
| Cleanup | Three seconds reserved for terminating/reaping the owned request process group |
| Delivery | Two seconds reserved for terminal delivery, within the 60-second request target |

The parent supervises a fresh Python worker and its process group for each
accepted request. BGE and Qwen run sequentially inside that lifetime. The native
MLX worker inherits the owned group, so a stalled child does not outlive the
request deadline by keeping an independent model session alive. The supervisor
uses the same clock across stages; a stage does not receive a fresh budget.

Admission remains reserved through cleanup. If termination/reaping cannot be
confirmed, the service becomes unavailable rather than starting another job.
Closing an HTTP connection does not cancel work. Ctrl+C drains the accepted job;
the deadline remains active during shutdown. There is no manual cancel endpoint.

On timeout, job execution is `failed` with `search_timeout`, and the answer is
`not_completed`. It is not a completed `insufficient_evidence` selection. An
incomplete request must retain its failure in operational and usefulness counts;
it cannot be credited as an appropriate scientific abstention.

The service records terminal time after cleanup. CLI delivery is observed after
output flush; HTTP delivery is observed after the terminal ASGI body send. Client
polling, receipt and output consumption require separate observation. Reserved
time and supervisor enforcement do not prove that every client received a result
within 60 seconds, or that every accepted candidate limit completed successfully.

## Process memory and persistent artifacts

Pipeline/model process memory is released with the request process group. The
parent retains admission/progress control and reads completed public jobs from
disk. Immutable passage vectors remain reusable across requests; source snapshots,
text versions, private checkpoints and worker cleanup evidence remain local.
Fresh acquisition failures do not fall back to old sources.

Isolation bounds the model/pipeline lifetime. It is not proof of a whole-machine
memory ceiling, zero swap use or unlimited parent-session stability. Process-group
cleanup, parent RSS and client latency are distinct observations. Abrupt parent
termination, separate concurrent application instances and public deployment are
outside the supported lifecycle contract.

## Selecting actual source findings

The generator receives up to three articles, six excerpts and 4000 evidence
tokens, without enlarging the original Dense ranking. Source setup has priority
over endpoint/analysis boilerplate when choosing a companion excerpt. Exact
source sentence IDs and article/text-version bindings are preserved.

One bounded Qwen4B call chooses scope/finding IDs and one of eight applicability
decisions. Only the explicit `answers_question` decision can admit a card;
program checks can demote it. Recognized source-scope vetoes remove an article
from the selector menu while retaining its audit record. The program does not
promote an indirect or uncertain decision into an answer.

Generic checks distinguish actual findings from aims or procedures, a treatment
effect from correlation between endpoint changes, prevention from established-
condition treatment, and ongoing-drug withdrawal from another treatment started
against placebo. A placebo arm alone does not establish discontinuation. Null
findings and explicitly scoped animal/cell mechanisms remain eligible; numerical
results are not compulsory.

These checks are conservative recognition rules, not an entailment model or
independent scientific review. Absence of a veto does not prove applicability.
The same Qwen model can select useful sentences yet misclassify their relation.
All nonempty displays remain `partial`; exact quotations do not certify age,
subgroup, causal attribution, complete comparison coverage or clinical advice.

## What the consumed development cases showed

Earlier development outputs substituted another endpoint, a relapse-predictor
analysis or prenatal prevention for the requested treatment question. Repairs
were developed on the old six biomedical questions and already observed test
material, with their previous outputs retained separately.

A more verbose multi-facet selector produced unnecessary refusals, so the
current contract uses one applicability decision. A later setup-only diagnostic
card exposed that exact quotation identity alone was insufficient; explicit
objective/procedure checks were added. The final development behavior still
withheld applicable diagnostic results as indirect. That false abstention is a
remaining tradeoff, not a successful evidence refusal.

Adult eligibility is absent from several abstracts. Some sources describe a
broader population than the question, and one preserved source internally
conflicts about its population. Source cautions and full-abstract AI review do
not resolve those facts. A post-remission trial of a new treatment versus placebo
is not automatically an ongoing-antidepressant withdrawal trial. Abstaining on
an inadequate packed context also does not establish that no relevant evidence
exists elsewhere in the acquired corpus or PubMed.

These are tuned AI-origin development diagnostics, not new independent labels
or an acceptance score. Historical protocols, source-reference packages and raw
attempts retain their original provenance. Candidate source/prompt/package
identities must be frozen before the reserved confirmation pass, and that pass
must keep failures, uncertainty and incomplete requests visible.

## Engineering verification for the frozen candidate

These results bind to freeze-manifest `manifest_id`
`a870a964853396ee16b2163e06b5e3dbd357e5ff97985b539f9298bc447f054c`.
Detailed raw artifacts remain local under
`results/request-quality-fixes-20261116/`; they are not tracked in Git.

| Check | Observed result |
| --- | --- |
| Main regression suite | 651 tests: 640 passed, 11 optional skips, no failures |
| Native suite | 111 passed; overlaps the main suite and is not 111 additional unique cases |
| Dependency consistency | `pip check` passed |
| Built wheel | All 75 packaged `.py`/`.html` source files matched the checked checkout byte for byte |
| Stable-code stress v2 | 29/29 operational checks passed: eight CLI candidate/top-k boundary requests, twenty requests in one API process, and an empty-result probe |
| Maximum recorded stress request wall time | 49.34324724999897 seconds (49.34 seconds) |

The checked wheel SHA-256 is
`e501de886f2fab92ec4c369976820bf914cf51df06ea9cce179c51f03973cbec`.
These checks bind code, packaging, execution and cleanup; their success is not
an answer-usefulness or scientific-support score.

Across the twenty requests in the same API process, sampled parent RSS went from
52.36 to 54.64 MiB, an increase of 2,392,064 bytes. This is a short observed
history, not proof of no memory leak, unlimited session stability or a
whole-machine memory ceiling. Parent RSS excludes the separate model-process
peak; those measurements must retain their individual scopes.

The separate [actual twenty-question validation](request-quality-validation.md)
failed release acceptance. Timing and cleanup passed; usefulness, acquisition
reliability and displayed-card support did not. Source labels were reconciled
and frozen before the actual-output review; engineering checks never supplied
semantic grades. Sampled RSS peaks have access-denied gaps and are not upper
bounds. The separately recorded actual-300-article probe is a development
observation, not a universal source-size or latency guarantee.
