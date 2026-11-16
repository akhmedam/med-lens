"""Local HTTP transport for the bounded research-job service."""

from contextlib import asynccontextmanager
import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from .api_contract import RequestValidationError, SearchRequest
from .api_service import JobService, ServiceBusy, ServiceUnavailable, UnknownRequest
from .request_deadline import REQUEST_TIMEOUT_SECONDS, REQUEST_DELIVERY_RESERVE_SECONDS
from .pubmed_query import MAX_OVERRIDE_CHARS, MAX_QUESTION_CHARS
from .mvp import (ANSWER_STYLE, DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K,
                  MIN_CANDIDATE_LIMIT, MAX_CANDIDATE_LIMIT, MIN_TOP_K, MAX_TOP_K)

MAX_BODY_BYTES = 256 * 1024


class SearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS,
        description="English biomedical research question, preserved verbatim.")
    pubmed_query: str | None = Field(default=None, min_length=1, max_length=MAX_OVERRIDE_CHARS,
        description="Optional exact PubMed expression; no implicit topic filter.")
    limit: int = Field(default=DEFAULT_CANDIDATE_LIMIT, ge=MIN_CANDIDATE_LIMIT, le=MAX_CANDIDATE_LIMIT)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)
    generate: bool = True
    answer_style: Literal["source-findings"] = ANSWER_STYLE


class StoredSearchBody(SearchBody):
    """Read-only request metadata, including jobs accepted before this MVP policy.

    Incoming requests use SearchBody exclusively; this model cannot enable a
    historical answer mode on submission or relabel an old job as the new policy.
    """
    answer_style: Literal["draft", "source-findings"] | None = None


class JobError(BaseModel):
    code: str
    message: str
    stage: str | None = None


class JobResponse(BaseModel):
    schema_version: Literal[1] = 1
    request_id: str
    status: Literal["queued", "running", "succeeded", "failed", "interrupted"]
    stage: str
    created_at: str
    started_at: str | None = None
    updated_at: str
    finished_at: str | None = None
    request: StoredSearchBody
    mvp_policy: dict | None = Field(default=None,
        description="Frozen application policy for this job; null for jobs predating policy records.")
    result: dict | None = None
    error: JobError | None = None
    timing: dict | None = None
    links: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ready", "busy", "stopping", "unavailable", "closed"]
    accepting_requests: bool
    active_request_id: str | None = None
    max_active_requests: Literal[1] = 1


def _error(status, code, message, *, details=None, headers=None):
    value = {"error": {"code": code, "message": message}}
    if details is not None:
        value["error"]["details"] = details
    return JSONResponse(value, status_code=status,
        headers={"Cache-Control": "no-store", **(headers or {})})


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError("Duplicate JSON keys are not permitted")
        result[key] = value
    return result


def _nonfinite(_):
    raise ValueError("Nonfinite JSON numbers are not permitted")


async def _request_body(request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        return None, _error(415, "unsupported_media_type", "Use application/json.")
    length = request.headers.get("content-length")
    if length is not None:
        if not length.isascii() or not length.isdecimal():
            return None, _error(400, "invalid_content_length", "Invalid Content-Length.")
        try:
            declared_length = int(length)
        except ValueError:
            return None, _error(400, "invalid_content_length", "Invalid Content-Length.")
        if declared_length > MAX_BODY_BYTES:
            return None, _error(413, "request_too_large", "Request body exceeds 256 KiB.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BODY_BYTES:
            return None, _error(413, "request_too_large", "Request body exceeds 256 KiB.")
        data.extend(chunk)
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError):
        return None, _error(400, "invalid_json", "Use valid UTF-8 JSON with unique keys and finite numbers.")
    try:
        body = SearchBody.model_validate(value)
        return SearchRequest.from_payload(body.model_dump()), None
    except ValidationError as exc:
        return None, _error(422, "invalid_request", "Invalid search parameters.",
            details=exc.errors(include_url=False, include_input=False, include_context=False))
    except RequestValidationError as exc:
        return None, _error(422, "invalid_request", str(exc))


def _links(job):
    return {**job, "links": {"self": "/v1/requests/" + job["request_id"]}}


class _RequestTimingMiddleware:
    """Observe server ingress and completed terminal ASGI writes, without polling claims."""

    def __init__(self, app, service):
        self.app, self.service = app, service

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        state = scope.setdefault("state", {})
        state["medlens_submission_started"] = time.monotonic()
        async def observed_send(message):
            await send(message)
            request_id = state.get("medlens_terminal_delivery_id")
            observe = getattr(self.service, "observe_delivery", None)
            if (request_id and callable(observe) and message["type"] == "http.response.body"
                    and not message.get("more_body", False)):
                try:
                    await run_in_threadpool(observe, request_id, scope="http-asgi-body-sent")
                except (OSError, ValueError, UnknownRequest):
                    # A response already sent cannot be changed into a second
                    # response merely because its timing journal is unavailable.
                    pass
        await self.app(scope, receive, observed_send)


def create_app(service=None, *, root=None, storage_root=None, cache_root=None):
    """Create one process-local service; health and submission do not load models."""
    if service is None:
        from .api_backend import make_local_runner
        root = Path(root).resolve() if root else Path(__file__).resolve().parents[2]
        storage_root = Path(storage_root) if storage_root else root / "results/api-requests"
        cache_root = Path(cache_root) if cache_root else root / ".cache/embeddings/bge-input-v1"
        store, cache = storage_root.resolve(), cache_root.resolve()
        if store.is_relative_to(cache) or cache.is_relative_to(store):
            raise ValueError("API storage and embedding cache must be disjoint")
        service = JobService(storage_root, make_local_runner(root, cache_root))

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(title="MedLens local research API", version="0.1.0",
        description="Bounded live PubMed research. Answers retain their experimental verification status.",
        lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.job_service = service
    app.add_middleware(_RequestTimingMiddleware, service=service)

    @app.exception_handler(HTTPException)
    async def protocol_error(request, exc):
        code = "not_found" if exc.status_code == 404 else "method_not_allowed" if exc.status_code == 405 else "http_error"
        return _error(exc.status_code, code, "Endpoint not found." if exc.status_code == 404 else "HTTP request rejected.", headers=exc.headers)

    @app.get("/health", response_model=HealthResponse)
    async def health(response: Response):
        value = HealthResponse.model_validate(service.health()).model_dump()
        if value["status"] not in {"ready", "busy"}:
            return JSONResponse(value, status_code=503, headers={"Cache-Control": "no-store"})
        response.headers["Cache-Control"] = "no-store"
        return value

    @app.post("/v1/requests", status_code=202, response_model=JobResponse,
        openapi_extra={"requestBody": {"required": True, "content": {"application/json": {
            "schema": SearchBody.model_json_schema()}}}},
        responses={400: {"description": "Malformed JSON"}, 409: {"description": "One research request is already active"},
            413: {"description": "Body exceeds 256 KiB"}, 415: {"description": "Expected JSON"},
            422: {"description": "Invalid parameters"}, 503: {"description": "Service is stopping or unavailable"}})
    async def submit(request: Request, response: Response):
        submitted_at = request.state.medlens_submission_started
        try:
            # A slow/incomplete body cannot consume an unbounded pre-admission
            # interval and then reset the accepted request's work budget.
            async with asyncio.timeout(max(0.0, submitted_at + REQUEST_TIMEOUT_SECONDS
                    - REQUEST_DELIVERY_RESERVE_SECONDS - time.monotonic())):
                parsed, failure = await _request_body(request)
        except TimeoutError:
            return _error(408, "request_timeout", "The request body time budget was exceeded.")
        if failure is not None:
            return failure
        try:
            submit_started = getattr(service, "submit_started", None)
            job = (await run_in_threadpool(submit_started, parsed, submitted_at)
                if callable(submit_started) else await run_in_threadpool(service.submit, parsed))
        except ServiceBusy:
            return _error(409, "service_busy", "One research request is already active.", headers={"Retry-After": "1"})
        except ServiceUnavailable:
            return _error(503, "service_unavailable", "The service is stopping or requires a restart.")
        except OSError:
            return _error(503, "storage_unavailable", "The request could not be stored.")
        response.headers["Location"] = "/v1/requests/" + job["request_id"]
        response.headers["Cache-Control"] = "no-store"
        return _links(job)

    @app.get("/v1/requests/{request_id}", response_model=JobResponse)
    async def get_request(request_id: str, response: Response, request: Request):
        try:
            value = await run_in_threadpool(service.get, request_id)
        except UnknownRequest:
            return _error(404, "unknown_request", "Request not found.")
        except OSError:
            return _error(503, "storage_unavailable", "The stored request is unavailable.")
        response.headers["Cache-Control"] = "no-store"
        if value["status"] in {"succeeded", "failed", "interrupted"}:
            request.state.medlens_terminal_delivery_id = request_id
        return _links(value)

    return app
