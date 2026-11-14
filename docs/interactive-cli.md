# Interactive CLI

The primary MedLens interface is a plain-text terminal session: enter an English biomedical question, watch stage messages, then read the answer, citations and ranked articles. It runs the existing live PubMed → BGE → Qwen3.5-4B/MLX pipeline on macOS/Apple Silicon. No separate HTTP server, FastAPI installation or new CLI dependency is needed.

## Start

Use the prepared project environment and model assets from [setup](../README.md#setup). Refresh the editable package to install the console entry point:

```sh
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/medlens
```

From a checkout, `.venv/bin/python scripts/medlens_cli.py` is equivalent. Help and an idle session do not load models. Search requires the existing local BGE and Qwen assets; missing dependencies or assets produce a visible setup failure.

The interface is Russian; questions must be in English. The existing planner does not classify language or topic and does not translate inputs. Source text, answer text and quotations remain in their original language. A session keeps the same search settings for each question. Each accepted question performs a fresh bounded PubMed search and can reuse unchanged passage vectors.

```text
Вопрос> How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?
```

The CLI prints the request ID, progress when the stage changes, the execution and answer statuses, visible claims with citations, applicability cautions and ranked articles with PubMed links. It returns to `Вопрос>` after the job actually finishes, including model cleanup. A rejected input or a recoverable search failure allows another question.

## Commands

| Command | Action |
| --- | --- |
| `/help` | Show the short command reference. |
| `/evidence` | Show all selected source fragments from the latest completed job. |
| `/export [PATH]` | Export that job as `.html` or `.json`; omit the path for a public HTML report in its job directory. |
| `/quit` | Exit while idle; `/exit` is an alias. |

Blank lines are ignored. Commands needing a result explain when no job has completed yet. The latest result remains available for evidence display and export after an invalid question.

## Search settings

Run `medlens --help` for all options. Defaults match the API: up to **100** PubMed candidates, **top-10** Dense-ranked articles, generation enabled, and the existing **`draft`** answer mode. `source-findings` remains an explicit experimental option.

```sh
.venv/bin/medlens --limit 30 --top-k 5 --answer-style source-findings
.venv/bin/medlens --search-only
.venv/bin/medlens --question 'How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?' --limit 10
```

`--question` processes one question and exits instead of opening the interactive session. `--limit` accepts 1–300 and `--top-k` 1–10. `--search-only` returns articles and evidence without Qwen generation. `--pubmed-query 'EXPRESSION'` supplies an exact PubMed candidate-search expression for the session; the original question still drives ranking and answers.

`--root` selects the project and prepared assets. `--storage-dir` selects job storage (default `ROOT/results/cli-requests`); `--cache` selects the embedding cache (default `ROOT/.cache/embeddings/bge-input-v1`). Storage and cache must be disjoint directories. The CLI creates one local job service for the session, without a queue or an HTTP server.

## Export and local artifacts

```text
Вопрос> /export
Вопрос> /export "/tmp/MedLens report.html"
Вопрос> /export /tmp/medlens-result.json
```

The default export is `results/cli-requests/REQUEST_ID/report-public.html`. HTML contains the displayed result and all selected evidence; JSON contains the public job envelope, including structured citations, source spans, statuses and timings. HTML is standalone and requires no neighbouring files. Choose an existing destination directory and quote paths containing spaces. Existing files are never overwritten.

Exports use the same public result as the API. They do not copy the private diagnostic `run/report.html`, raw model prompts/calls, runtime paths or raw exception details. Private pipeline artifacts remain in the job's `run/` directory. `job.json` preserves the public result independently of the terminal session. Terminal control characters are escaped for display; stored source quotations are not rewritten.

## Completion and exit

`Ctrl+D` at the prompt and `/quit` exit with code 0. `Ctrl+C` exits with code 130. During an active request, `Ctrl+C` requests exit **after** the accepted job finishes and releases its runtime; it does not cancel the job. Further `Ctrl+C` signals are ignored during that shutdown wait. There is no hard-cancellation or automatic-resumption feature.

For `--question`, a successful job exits 0, invalid input or answer-validation failure exits 2, a failed job exits 1, and interrupted execution exits 130. Insufficient evidence or a partial answer can be a successful job: read the separately displayed answer status. If cleanup or storage becomes uncertain, the CLI stops accepting questions and exits rather than starting another job.

This interface preserves the existing model, generation policy and [answer-quality limitations](biomedical-quality-improvement.md). `independent_validation` remains false. Exact quotations and successful execution do not establish scientific support, completeness or clinical validity.

## Validation

The [integration record](experiments/interactive-cli.json) binds the implementation
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
