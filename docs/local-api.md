# Local research API

The local API exposes the existing live PubMed → BGE → Qwen3.5-4B/MLX workflow on macOS/Apple Silicon. Submission returns an ID immediately; clients poll progress and the structured result. One research job may run at a time. Generation prompts, evidence policies and scientific-verification limits are unchanged.

## Start

Use the prepared project environment and assets described in [setup](../README.md#setup). Install the API dependency lock and refresh the editable entry point:

```sh
.venv/bin/python -m pip install -r requirements-api.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/medlens-api
```

From the checkout, `.venv/bin/python scripts/serve_api.py` is equivalent. The default address is `http://127.0.0.1:8000`. `--port`, loopback `--host ::1`, `--root`, `--storage-dir` and `--cache` are available. The CLI starts one server process and offers no multiple-worker option. Public deployment and cross-platform model runtimes are outside the MVP.

Health is available without loading models. It describes service admission, not asset or scientific readiness; a request with missing assets fails explicitly during setup. The service uses the existing pinned BGE/MPS and Qwen4B/MLX paths without a model/backend fallback.

## Submit and poll

```sh
curl -sS http://127.0.0.1:8000/health

curl -sS http://127.0.0.1:8000/v1/requests \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?","limit":10,"answer_style":"source-findings"}'

# Replace REQUEST_ID with the returned request_id.
curl -sS http://127.0.0.1:8000/v1/requests/REQUEST_ID
```

POST returns `202 Accepted`, `request_id` and a `Location`/`links.self` polling URL. Pending jobs have `result: null`; progress is in `status` and `stage`. Poll until `succeeded`, `failed` or `interrupted`. The JSON result includes ranked `articles`, selected `evidence`, cited `answer`, fresh-search metadata, query transformations, timings and the original question.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `200` for ready/busy, `503` for stopping/unavailable/closed; no model load. |
| `POST /v1/requests` | `202` accepts one job; `409 service_busy` rejects concurrent work without queueing it. |
| `GET /v1/requests/{request_id}` | `200` returns current or persisted job; `404` for an unknown/invalid ID. |
| `GET /openapi.json` | Machine-readable request and response schemas. |

Request fields are strict; unknown fields and numeric strings are rejected. The JSON body must use UTF-8, unique keys and finite numbers, and be at most 256 KiB.

| Field | Default and limits |
| --- | --- |
| `question` | Required, nonempty single-line English biomedical question, at most 8000 characters; kept unchanged. |
| `pubmed_query` | Optional exact PubMed expression, at most 4000 characters. |
| `limit` | Integer 1–300, default 100 fresh candidates. |
| `top_k` | Integer 1–10, default 10 Dense-ranked articles. |
| `generate` | Boolean, default true. False returns search/evidence without Qwen inference. |
| `answer_style` | `draft` (unchanged default) or experimental `source-findings`. |

Invalid parameters return `422 invalid_request`; malformed JSON `400`, unsupported content type `415`, oversized body `413`. These failures do not allocate a job. `409` includes `Retry-After`; the client must retry explicitly after the active job ends. Stopping or unavailable services reject new work with `503`.

## Execution and answer status

Job `status` tracks execution. `result.status` preserves the pipeline status, and `result.answer.status` preserves the answer outcome. For example, a successful job may contain a partial research draft or an insufficient-evidence answer. An answer validation failure produces a failed job with `answer_validation_failed`, while available articles/evidence remain in `result`.

`independent_validation` remains false. Source quotations, scope cautions, text versions, spans and upstream verification markers are preserved. Exact quotations and a successful HTTP request do not establish scientific support or medical validity. The [quality assessment and remaining limitations](biomedical-quality-improvement.md) still apply; this API work does not reevaluate answers or satisfy independent human-reference gates.

Errors use stable `code`, `message` and `stage`. Expected source/model failures retain available partial results. Raw exceptions, model prompts/calls, private provenance and filesystem/runtime paths are excluded from public JSON. Fixed program abstention reasons remain visible. Full `run.json`, source snapshots and the diagnostic HTML report stay local under `results/api-requests/{request_id}/run/`; HTTP does not serve those raw diagnostic files. `job.json` is an atomic public job envelope. Storage and embedding cache roots must be disjoint.

## Lifetime and shutdown

Admission is reserved before scheduling a job, so there is no hidden backlog. The execution lane covers source retrieval, embedding work, generation and actual cleanup. A pipeline `complete` checkpoint can occur before cleanup; the job remains running until the runner returns. Health and status requests remain available during computation.

Each job creates a fresh runtime factory; the existing pipeline loads Qwen at most once and closes it in its `finally` block. Empty/search-only cases need no Qwen load. A lazy PubMed client is reused across sequential jobs so its 2.5 requests/second limiter is not reset by fast empty/search-only requests. Passage vectors for unchanged inputs are reused through the existing immutable cache. Completed result payloads are removed from memory and read from disk on demand.

Closing an HTTP connection does not cancel an accepted job. `Ctrl+C` stops admission and waits for the current pipeline to finish and release its runtime. Uncertain runtime cleanup or storage failure makes the service unavailable for new jobs; inspect local diagnostics and resolve the fault before restarting. Completed jobs remain readable after a restart. Abrupt process termination, automatic resumption and hard cancellation are outside this MVP; unfinished records may retain their last checkpoint.

## Validation

The [machine-readable integration record](experiments/local-api.json) binds the code and raw runs. The main suite passed 512 tests with nine optional native-grammar skips, 521 total; 49 API checks cover strict inputs, projection, job persistence, busy behavior, failures, memory history and shutdown through the HTTP lifespan.

A real HTTP request fetched ten fresh PubMed records and completed with Qwen4B/MLX in **18.74 seconds** of pipeline time, with three visible source findings and an explicit partial/unverified answer status. POST acceptance took about 6 ms. A simultaneous request received `409`, and health remained responsive. The native model PID was observed during execution and reaped before completion. A second fresh search-only request completed in **2.42 seconds** without loading Qwen; both requests reused all 14 unchanged passage inputs.

These are two warm-cache smoke requests, not a latency SLA or an answer-quality score. The first smoke controller incorrectly counted Python's persistent resource-tracker helper as a model leak; that harness check was corrected to identify the native MLX worker specifically, and both attempts were retained. Normal server shutdown exited cleanly. Raw artifacts remain local in `results/api-smoke-20261113-v3`.

A fresh API-only virtual environment also passed the locked install and console entry-point check. It imported the HTTP/core API and served health without numpy, torch, transformers or MLX. A job failed explicitly at setup with no model load or network access, and the service remained available afterward. This verifies API bootstrap and the missing-dependency error path; it is not the later full model clean-install release check.
