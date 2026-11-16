# Interactive CLI

The primary MedLens interface is a plain-text terminal session: enter an English
biomedical question, watch stage messages, then read source findings, citations
and ranked articles. It runs the [supported MVP](mvp.md) on macOS/Apple Silicon
without an HTTP server, FastAPI installation or extra CLI dependency.

## Start

Use [README setup](../README.md#setup), then run:

```sh
.venv/bin/medlens
```

From a checkout, `.venv/bin/python scripts/medlens_cli.py` is equivalent. Help and
an idle session do not load models. Accepted research jobs require the prepared
BGE/Qwen assets; missing dependencies or assets produce an explicit setup error.

The interface is Russian; supported questions are English biomedical research
questions. Language and topic are not classified or translated. Source text and
quotations remain in their original language. Each question searches PubMed
anew and can reuse unchanged passage vectors. The shared application policy is
`medlens-mvp-source-findings-v2`; exact quotations are selected by
`source-findings-task-scoped-v5-candidate1` in one bounded model call.

```text
Вопрос> How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?
```

The CLI prints the request ID, stage changes, execution and answer statuses,
source-findings cards with citations/cautions, and ranked PubMed articles. It
returns to `Вопрос>` only after the job and cleanup finish. Rejected input or a
recoverable job failure allows another question.

## Commands

| Command | Action |
| --- | --- |
| `/help` | Show the short command reference. |
| `/evidence` | Show all selected fragments from the latest completed job. |
| `/export [PATH]` | Export that job as public `.html` or `.json`; omit the path for HTML in its job directory. |
| `/quit` | Exit while idle; `/exit` is an alias. |

Blank lines are ignored. Commands needing a result explain when none exists.
An invalid question does not erase the last result available for evidence/export.

## Search settings

Defaults are up to 100 PubMed candidates, top-10 Dense-ranked articles and
source-findings generation. There is no public answer-style/model/backend
switch. The model selects exact quotations; it does not write factual synthesis.

```sh
.venv/bin/medlens --limit 30 --top-k 5
.venv/bin/medlens --search-only
.venv/bin/medlens --question 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?' --limit 10
```

`--question` processes one question and exits. `--limit` accepts 1–300 and
`--top-k` 1–10. `--search-only` returns articles/evidence without Qwen inference.
`--pubmed-query 'EXPRESSION'` submits that exact candidate-search expression for
the session; the original question still drives ranking and findings.

`--root` selects the project/resources. `--storage-dir` selects job storage
(default `ROOT/results/cli-requests`); `--cache` selects the embedding cache
(default `ROOT/.cache/embeddings/bge-input-v1`). They must be disjoint. Settings
remain fixed within the session; one service runs one accepted job at a time.

## Export and local artifacts

```text
Вопрос> /export
Вопрос> /export "/tmp/MedLens report.html"
Вопрос> /export /tmp/medlens-result.json
```

The default export is `results/cli-requests/REQUEST_ID/report-public.html`. HTML
contains the displayed result and all selected evidence; it is standalone. JSON
contains the public envelope, including source spans/versions, statuses and
timings. Choose an existing directory and quote paths containing spaces.
Existing files are never overwritten.

Exports use the same public projection as the API. They do not copy private
`run/report.html`, model prompts/calls, runtime paths or raw exceptions. Private
run artifacts stay in the job directory. `job.json` preserves the public result.
Terminal controls are escaped for display; stored quotations are not rewritten.

## Completion and exit

Ctrl+D at the prompt and `/quit` exit 0. Ctrl+C exits 130. During a request it
waits for that accepted job and runtime cleanup; it does not cancel the job.
Repeated Ctrl+C is ignored during this wait. The automatic request deadline
continues to apply; Ctrl+C is not a separate immediate-cancel command. Automatic
resumption after abrupt termination is outside this MVP.

For interactive input, the request budget begins before validation/admission
after the question is entered; time spent typing is excluded. A one-shot request
starts its clock at CLI invocation. The shared 60-second policy reserves
55 seconds for work, three for terminating/reaping the request process group
and two for delivery. The CLI records delivery after printing/flushing the
terminal result. Slow terminal output still needs an observed full-request
measurement; work-budget enforcement alone is not a latency pass.

For `--question`, a successful job exits 0, invalid input or answer-validation
failure exits 2, other job failure exits 1, and interruption exits 130. Partial
findings and insufficient evidence can be successful executions: read answer
status separately. Uncertain cleanup/storage stops admission instead of
starting another job. A deadline failure exits 1 in one-shot mode and reports
`search_timeout` / `not_completed`; it is not a successful insufficient-evidence
answer. See [request reliability](request-reliability.md).

Nonempty source-findings displays remain `partial`. Exact quotations do not
prove support for the requested comparison or completeness. Independent semantic
and clinical validation remain absent; the [known limits](biomedical-quality-improvement.md)
and human-reference gates are unchanged.

## Historical CLI integration evidence

The original CLI record predates the unified source-findings-only policy.
Its test counts and smoke requests describe that feature snapshot, not current
final acceptance. The [integration record](experiments/interactive-cli.json) binds the implementation
and local smoke artifacts. The full suite passed **535 tests with 9 optional
native skips (544 total)**, including **23 new CLI tests**. These cover the input
loop, recoverable failures, exit codes, stage-complete versus actual cleanup,
Ctrl+C and repeated SIGINT during drain, safe rendering and public exports.
An isolated Python launch with site packages disabled reached the prompt and
exited without importing FastAPI, numpy, torch, transformers or MLX.

A real terminal session retrieved ten fresh PubMed articles, displayed a partial
source-findings answer and evidence, exported HTML/JSON, rejected a query without
searchable terms, and refused to overwrite an export. A separate real Ctrl+C
check waited for the accepted Qwen job to finish with runtime cleanup confirmed
before exiting 130. Repeated-signal behavior is covered by the deterministic worker test.
A one-shot search-only request also completed without loading Qwen. These are
functional checks, not a new answer-quality evaluation or latency benchmark.
