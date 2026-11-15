# Local research API

The loopback API exposes the [supported MVP](mvp.md): fresh PubMed titles and
abstracts → BGE/MPS Dense ranking → Qwen3.5-4B/MLX source-findings cards. It uses
the same policy as the terminal CLI. Submission returns an ID immediately;
clients poll progress and the structured result. One job may run at a time.

## Start

Use [README setup](../README.md#setup), then install the optional HTTP lock:

```sh
.venv/bin/python -m pip install -r requirements-api.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/medlens-api
```

From the checkout, `.venv/bin/python scripts/serve_api.py` is equivalent. The
default address is `http://127.0.0.1:8000`. `--port`, loopback `--host ::1`,
`--root`, `--storage-dir` and `--cache` configure the local service. It starts one
server process without a multiworker/reload option. Public deployment is outside
this MVP.

Startup and health do not load models. Health describes admission, not asset or
scientific readiness: missing dependencies/assets cause an explicit job setup
failure. There is no model, inference-backend or answer-format fallback.

## Submit and poll

```sh
curl -sS http://127.0.0.1:8000/health

curl -sS http://127.0.0.1:8000/v1/requests \
  -H 'Content-Type: application/json' \
  -d '{"question":"How does NLRP3 inflammasome activation contribute to pancreatic beta-cell dysfunction in type 2 diabetes?","limit":10}'

# Replace REQUEST_ID with the returned request_id.
curl -sS http://127.0.0.1:8000/v1/requests/REQUEST_ID
```

POST returns `202 Accepted`, `request_id` and a `Location`/`links.self` polling
URL. Pending jobs have `result: null`; progress is in `status` and `stage`. Poll
until `succeeded`, `failed` or `interrupted`. Results include ranked `articles`,
selected `evidence`, cited source findings, fresh-search metadata, query
transformations, timings and the unchanged question.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | `200` for ready/busy; `503` for stopping/unavailable/closed; no model load. |
| `POST /v1/requests` | `202` accepts one job; `409 service_busy` rejects concurrent work without queueing. |
| `GET /v1/requests/{request_id}` | `200` returns current/persisted job; `404` for an unknown or invalid ID. |
| `GET /openapi.json` | Machine-readable request and response schemas. |

Inputs are strict: unknown fields and numeric strings are rejected. JSON must
be UTF-8 with unique keys, finite numbers and a maximum size of 256 KiB.

| Field | Default and limits |
| --- | --- |
| `question` | Required, nonempty single-line research question, at most 8000 characters; kept unchanged. English biomedical questions are the supported scope. |
| `pubmed_query` | Optional exact PubMed expression, at most 4000 characters. |
| `limit` | Integer 1–300, default 100 fresh candidates. |
| `top_k` | Integer 1–10, default 10 Dense-ranked articles. |
| `generate` | Boolean, default true. False returns search/evidence without Qwen inference. |
| `answer_style` | Optional constant `source-findings`; omitted means the same policy. `draft` and other values are rejected. |

There is no model/backend selector. Language and topic are not classified and
no automatic translation is performed. An explicit PubMed expression is
submitted unchanged; no condition or disease filter is appended.

Invalid parameters return `422 invalid_request`, malformed JSON `400`, unsupported
content type `415`, and an oversized body `413`. These failures allocate no job.
`409` includes `Retry-After`; retry explicitly after the active job ends. A
stopping or unavailable service rejects new work with `503`.

Historical persisted jobs remain readable with their original request metadata,
including earlier draft jobs. Their absent MVP policy remains unknown; read-only
compatibility does not permit new draft requests or relabel old outputs.

## Results and errors

Job `status` tracks execution. `result.status` preserves pipeline status, while
`result.answer.status` describes the source-finding outcome. A successful job may
contain a `partial` quotation display, an insufficient-evidence answer or a
search-only result. A validation failure produces a failed job with
`answer_validation_failed`; available articles/evidence remain in `result`.

Qwen selects exact source quotations rather than writing factual synthesis.
Cards retain study context, findings, source titles, citations and applicability
cautions. Nonempty answers remain `partial`; quote identity does not certify
semantic support or completeness. `independent_validation` remains false. The
[known quality limits](biomedical-quality-improvement.md) and independent-human
reference gates are unchanged by successful HTTP execution.

Errors use stable `code`, `message` and `stage`. Expected source/model failures
retain available partial results. Raw exceptions, model prompts/calls, private
provenance and filesystem/runtime paths are excluded from public JSON. Full
`run.json`, source snapshots and diagnostic `report.html` stay local under
`results/api-requests/{request_id}/run/`; HTTP does not serve those files.
`job.json` is the atomic public envelope. Storage and embedding cache roots must
be disjoint.

## Lifetime and shutdown

Admission is reserved before scheduling. The lane covers source acquisition,
embedding work, generation and cleanup. A pipeline `complete` checkpoint can
precede cleanup; the job stays running until its runner returns. Health/status
remain available during computation.

Each job creates a fresh runtime factory. Empty/search-only cases need no Qwen
load. One lazy PubMed client preserves the 2.5 requests/second limiter across
sequential jobs; it is not an IP-wide coordinator. Unchanged passage vectors use
the immutable cache. Completed result payloads leave memory and are read from
disk on demand.

Closing an HTTP connection does not cancel an accepted job. Ctrl+C stops
admission and waits for the current job and runtime cleanup. Uncertain cleanup
or storage failure makes the service unavailable. Completed jobs remain readable
after restart; abrupt termination does not resume unfinished jobs automatically.
Hard cancellation and global coordination between separate application processes
are outside this MVP.

## Historical API integration evidence

The original API record predates the unified source-findings-only public policy.
Its counts and timings are historical integration evidence, not the current final
validation result. The [machine-readable integration record](experiments/local-api.json) binds the code and raw runs. The main suite passed 512 tests with nine optional native-grammar skips, 521 total; 49 API checks cover strict inputs, projection, job persistence, busy behavior, failures, memory history and shutdown through the HTTP lifespan.

A real HTTP request fetched ten fresh PubMed records and completed with Qwen4B/MLX in **18.74 seconds** of pipeline time, with three visible source findings and an explicit partial/unverified answer status. POST acceptance took about 6 ms. A simultaneous request received `409`, and health remained responsive. The native model PID was observed during execution and reaped before completion. A second fresh search-only request completed in **2.42 seconds** without loading Qwen; both requests reused all 14 unchanged passage inputs.

These are two warm-cache smoke requests, not a latency SLA or an answer-quality score. The first smoke controller incorrectly counted Python's persistent resource-tracker helper as a model leak; that harness check was corrected to identify the native MLX worker specifically, and both attempts were retained. Normal server shutdown exited cleanly. Raw artifacts remain local in `results/api-smoke-20261113-v3`.

A fresh API-only virtual environment also passed the locked install and console entry-point check. It imported the HTTP/core API and served health without numpy, torch, transformers or MLX. A job failed explicitly at setup with no model load or network access, and the service remained available afterward. This verifies API bootstrap and the missing-dependency error path; it is not the later full model clean-install release check.
