"""Single-lane local research jobs over the existing live pipeline.

Transport code owns HTTP; this module owns admission, durable job envelopes and
actual computation lifetime. It imports no model, tensor or network runtime.
"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import time
from threading import RLock
from uuid import uuid4

from .api_contract import SearchRequest, public_report
from .mvp import policy_snapshot
from .request_deadline import RequestBudget


class UnknownRequest(LookupError):
    """The supplied ID does not name a readable job owned by this service."""


class ServiceBusy(RuntimeError):
    """One accepted search still owns the execution lane."""


class ServiceUnavailable(RuntimeError):
    """The service is closing or its cleanup/storage state is uncertain."""


_SUBMISSION_ORIGIN = ContextVar("medlens_submission_origin", default=None)

_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
_STATUSES = {"queued", "running", "succeeded", "failed", "interrupted"}
_STAGES = {"queued", "setup", "query-validation", "pubmed-download", "parsing",
    "windowing", "passage-embeddings", "index-validation", "retrieval",
    "retrieval-complete", "context", "context-complete", "reranking",
    "reranking-complete", "answer", "complete", "runtime-cleanup", "unknown"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _stage(value):
    return value if isinstance(value, str) and value in _STAGES else "unknown"


def _error(code, message, stage):
    return {"code": code, "message": message, "stage": _stage(stage)}


def _pipeline_error(report):
    if report.get("progress_persistence_failed") is True:
        return _error("storage_error", "Search progress state could not be saved.", report.get("stage"))
    if report.get("deadline_exceeded") is True:
        return _error("search_timeout", "The complete-request time budget was exceeded.", report.get("stage"))
    raw_error = report.get("error")
    raw_error = raw_error if isinstance(raw_error, dict) else {}
    stage = _stage(raw_error.get("stage", report.get("stage")))
    if raw_error.get("type") == "RequestWorkerError":
        return _error("search_worker_failed", "The isolated search worker failed.", stage)
    if report.get("status") == "interrupted":
        return _error("search_interrupted", "Search execution was interrupted.", stage)
    codes = {
        "setup": ("search_setup_failed", "Search setup failed."),
        "query-validation": ("invalid_query", "The question cannot be processed by the search pipeline."),
        "pubmed-download": ("pubmed_error", "Fresh PubMed retrieval failed."),
        "parsing": ("source_preparation_failed", "Downloaded sources could not be prepared."),
        "windowing": ("source_preparation_failed", "Downloaded sources could not be prepared."),
        "index-validation": ("source_preparation_failed", "Source artifacts failed verification."),
        "passage-embeddings": ("retrieval_failed", "Article embeddings could not be prepared."),
        "retrieval": ("retrieval_failed", "Article ranking failed."),
        "context": ("source_preparation_failed", "Source evidence could not be prepared."),
        "reranking": ("inference_failed", "The model stage failed."),
        "answer": ("inference_failed", "The answer model stage failed."),
        "runtime-cleanup": ("runtime_cleanup_failed", "Model cleanup could not be confirmed."),
    }
    code, message = codes.get(stage, ("search_failed", "Search execution failed."))
    return _error(code, message, stage)


def _uncertain_cleanup(report):
    native = report.get("native_worker_cleanup")
    if native is not None:
        return (not isinstance(native, dict) or native.get("confirmed") is not True
            or native.get("worker_reaped") is not True or native.get("group_absent") is not True)
    count = report.get("runtime_load_count", 0)
    if type(count) is not int or count < 0:
        return True
    return (report.get("runtime_closed") is False
        or (count > 0 and report.get("runtime_closed") is not True)
        or ("cleanup_error" in report and report["cleanup_error"] is not None))


class JobService:
    """One accepted job at a time; a disconnect never cancels its worker.

    The runner must return only after its pipeline/runtime cleanup finishes.
    A raised exception provides no cleanup proof and makes this instance
    unavailable. Server shutdown must call close from outside the job worker.
    """

    def __init__(self, storage_root, runner):
        if not callable(runner):
            raise TypeError("runner must be callable")
        self.storage_root = Path(storage_root).resolve()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._runner = runner
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medlens-search")
        self._jobs = {}
        self._active_id = None
        self._future = None
        self._closing = False
        self._closed = False
        self._unavailable = False
        self._instance_id = uuid4().hex

    def _directory(self, request_id):
        if type(request_id) is not str or _REQUEST_ID.fullmatch(request_id) is None:
            raise UnknownRequest("Unknown request.")
        directory = self.storage_root / request_id
        if directory.is_symlink():
            raise UnknownRequest("Unknown request.")
        return directory

    def _persist(self, job):
        directory = self._directory(job["request_id"])
        body = json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".job-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / "job.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def submit_started(self, request, submitted_at):
        """Carry ingress time while preserving injectable/overridden submit methods."""
        token = _SUBMISSION_ORIGIN.set(submitted_at)
        try:
            return self.submit(request)
        finally:
            _SUBMISSION_ORIGIN.reset(token)

    def submit(self, request):
        origin = _SUBMISSION_ORIGIN.get()
        budget = RequestBudget(time.monotonic() if origin is None else origin)
        if not isinstance(request, SearchRequest):
            raise TypeError("request must be a SearchRequest")
        # Direct dataclass construction must not bypass the public input contract.
        request = SearchRequest.from_payload(asdict(request))
        with self._lock:
            if self._closing or self._closed or self._unavailable:
                raise ServiceUnavailable("Search service is unavailable.")
            if self._active_id is not None:
                raise ServiceBusy("Another search request is already running.")
            request_id = uuid4().hex
            self._active_id = request_id
            created = _now()
            job = {"schema_version": 1, "mvp_policy": policy_snapshot(), "request_id": request_id, "status": "queued",
                "stage": "queued", "created_at": created, "started_at": None,
                "updated_at": created, "finished_at": None, "request": asdict(request),
                "result": None, "error": None,
                "timing": {"timeout_seconds": budget.seconds,
                    "scope": "transport-ingress-through-confirmed-cleanup",
                    "terminal_after_cleanup_seconds": None}}
            self._jobs[request_id] = job
            try:
                self._directory(request_id).mkdir(parents=False, exist_ok=False)
                # Monotonic origins are private and valid only in this service
                # instance; persisted history does not grow a resident map.
                (self._directory(request_id) / "submission-clock.json").write_text(
                    json.dumps({"instance_id": self._instance_id, "started": budget.started}),
                    encoding="utf-8")
                self._persist(job)
                # Reserve the lane before enqueueing; no second job may enter it.
                self._future = self._executor.submit(self._run, request_id, request, budget)
            except Exception:
                self._active_id = None
                self._unavailable = True
                job.update(status="failed", finished_at=_now(), updated_at=_now(),
                    error=_error("service_unavailable", "Search service could not start this request.", "setup"))
                raise ServiceUnavailable("Search service could not start this request.") from None
            return deepcopy(job)

    def _progress(self, request_id, stage, status):
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job["status"] != "running":
                return
            # A pipeline 'complete' checkpoint can precede runtime.close().
            # Job state and lane ownership remain running until runner returns.
            job.update(stage=_stage(stage), updated_at=_now())
            self._persist(job)

    def _run(self, request_id, request, budget):
        report = None
        terminal_report = False
        result = None
        failed_cleanup = False
        failed_progress = False
        status, error, stage = "failed", None, "setup"
        try:
            with self._lock:
                job = self._jobs[request_id]
                job.update(status="running", stage="setup", started_at=_now(), updated_at=_now())
                self._persist(job)
            output = self._directory(request_id) / "run"
            progress = lambda stage, status: self._progress(request_id, stage, status)
            bounded = getattr(self._runner, "run_with_budget", None)
            report = (bounded(request, output, progress, budget) if callable(bounded)
                else self._runner(request, output, progress))
            if not isinstance(report, dict) or report.get("status") not in {"complete", "failed", "interrupted"}:
                raise ValueError("Runner did not return a terminal report")
            terminal_report = True
            failed_cleanup = _uncertain_cleanup(report)
            failed_progress = report.get("progress_persistence_failed") is True
            stage = _stage(report.get("stage"))
            result = public_report(report)
            if report["status"] in {"failed", "interrupted"}:
                status = "interrupted" if report["status"] == "interrupted" else "failed"
                error = _pipeline_error(report)
            elif (report.get("answer") or {}).get("status") == "validation_failed":
                status = "failed"
                error = _error("answer_validation_failed", "The generated answer failed validation.", "answer")
            elif failed_cleanup:
                status = "failed"
                error = _error("runtime_cleanup_failed", "Model cleanup could not be confirmed.", "runtime-cleanup")
            else:
                status = "succeeded"
        except BaseException as exc:
            # Exception messages can contain model directories, source paths or
            # upstream internals; none enters durable/public job metadata.
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            failed_cleanup = not terminal_report or _uncertain_cleanup(report)
            stage = _stage(self._jobs[request_id].get("stage"))
            error = _error("search_interrupted" if status == "interrupted" else "internal_error",
                "Search execution was interrupted." if status == "interrupted" else "Search execution failed unexpectedly.", stage)
        finally:
            with self._lock:
                job = self._jobs[request_id]
                job.update(status=status, stage=stage, result=result, error=error,
                    finished_at=_now(), updated_at=_now())
                job["timing"].update(terminal_after_cleanup_seconds=budget.elapsed(),
                    cleanup_confirmed=not failed_cleanup)
                self._unavailable = self._unavailable or failed_cleanup or failed_progress
                try:
                    self._persist(job)
                except Exception:
                    self._unavailable = True
                    job.update(status="failed", error=_error("storage_error", "Search state could not be saved.", stage))
                else:
                    # Terminal envelopes are durable; retain no growing in-memory
                    # history of answer/source payloads. get() reads them on demand.
                    self._jobs.pop(request_id, None)
                self._active_id = None

    def get(self, request_id):
        with self._lock:
            directory = self._directory(request_id)
            if request_id in self._jobs:
                return deepcopy(self._jobs[request_id])
            path = directory / "job.json"
            if path.is_symlink() or not path.is_file():
                raise UnknownRequest("Unknown request.")
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                        or payload.get("request_id") != request_id or payload.get("status") not in _STATUSES):
                    raise ValueError("Invalid job envelope")
                # Reject non-JSON values before handing them to the transport.
                json.dumps(payload, allow_nan=False)
            except (OSError, ValueError, TypeError, RecursionError):
                raise UnknownRequest("Unknown request.") from None
            return deepcopy(payload)

    def observe_delivery(self, request_id, submitted_at=None, *, scope):
        """Record an observed output boundary; never infer arbitrary client receipt.

        CLI calls after flush. HTTP calls after ASGI terminal-body send returns;
        client-side receive and polling latency still require external measurement.
        """
        if scope not in {"cli-output-flushed", "http-asgi-body-sent"}:
            raise ValueError("Unknown delivery observation scope")
        with self._lock:
            job = self.get(request_id)
            if job["status"] not in {"succeeded", "failed", "interrupted"}:
                return
            if submitted_at is None:
                path = self._directory(request_id) / "submission-clock.json"
                if path.is_symlink():
                    return
                clock = json.loads(path.read_text(encoding="utf-8"))
                if clock.get("instance_id") != self._instance_id:
                    return
                submitted_at = clock["started"]
            timing = job.setdefault("timing", {})
            # Keep the first terminal delivery, not an arbitrarily late reread.
            if "delivery_observation" not in timing:
                timing["delivery_observation"] = {
                    "seconds": max(0.0, time.monotonic() - submitted_at), "scope": scope}
                self._persist(job)

    def report_path(self, request_id):
        self.get(request_id)  # Require a known valid job, not merely a directory.
        directory = self._directory(request_id)
        run = directory / "run"
        path = run / "report.html"
        if run.is_symlink() or not run.is_dir() or path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Report is not available.")
        return path

    def health(self):
        with self._lock:
            status = ("closed" if self._closed else "stopping" if self._closing
                else "unavailable" if self._unavailable else "busy" if self._active_id else "ready")
            return {"status": status, "accepting_requests": status == "ready",
                "active_request_id": self._active_id, "max_active_requests": 1,
                "unavailable": self._unavailable}

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closing = True
        # Do not cancel accepted work or release its lane on HTTP/server timeout.
        self._executor.shutdown(wait=True)
        with self._lock:
            self._closed = True
