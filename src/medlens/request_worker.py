"""Short-lived request processes, with an owned process group and hard deadline.

Only the child imports the live/model pipeline. Protocol state is kept in private
atomic files so arbitrary dependency logging cannot corrupt the control channel.
The parent retains neither tensor runtimes nor historical result payloads.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time

from .request_deadline import RequestBudget

WORKER_PROTOCOL = 1
_REPORT_STAGES = frozenset({"setup", "query-validation", "pubmed-download", "parsing",
    "windowing", "passage-embeddings", "index-validation", "retrieval", "retrieval-complete",
    "context", "context-complete", "reranking", "reranking-complete", "answer", "complete",
    "runtime-cleanup"})
_ANSWER_STATUSES = frozenset({"answered", "partial", "insufficient_evidence", "validation_failed",
    "not_requested", "not_completed"})
POLL_SECONDS = 0.025
MAX_REPORT_BYTES = 32 * 1024 * 1024


class _ProgressFailure(RuntimeError):
    """A service callback failed; do not hide lost durable job progress."""


def _atomic_json(path, value):
    body = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".worker-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    if path.is_symlink() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Invalid request-worker artifact")
    return json.loads(path.read_text(encoding="utf-8"))


def _request_digest(request):
    body = json.dumps(asdict(request), sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _worker_envelope(request, nonce, report):
    return {"protocol": WORKER_PROTOCOL, "nonce": nonce,
        "request_sha256": _request_digest(request), "report": report}


def _validated_worker_report(envelope, request, nonce):
    """Reject stale, foreign or incomplete IPC results before declaring success."""
    if (not isinstance(envelope, dict) or type(envelope.get("protocol")) is not int
            or envelope["protocol"] != WORKER_PROTOCOL or envelope.get("nonce") != nonce
            or envelope.get("request_sha256") != _request_digest(request)):
        raise ValueError("Invalid request-worker identity")
    report = envelope.get("report")
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or report.get("question") != request.question
            or report.get("answer_style") != request.answer_style
            or report.get("status") not in {"complete", "failed", "interrupted"}
            or report.get("stage") not in _REPORT_STAGES):
        raise ValueError("Invalid request-worker report")
    answer = report.get("answer")
    if (not isinstance(answer, dict) or answer.get("status") not in _ANSWER_STATUSES
            or not isinstance(answer.get("claims"), list)
            or any(not isinstance(row, dict) for row in answer["claims"])):
        raise ValueError("Invalid request-worker answer")
    for name in ("generate", "top_k"):
        if name in report and (type(report[name]) is not type(getattr(request, name))
                or report[name] != getattr(request, name)):
            raise ValueError("Request-worker parameters changed")
    if report["status"] == "complete":
        expected = ({"answered", "partial", "insufficient_evidence", "validation_failed"}
            if request.generate else {"not_requested"})
        if report["stage"] != "complete" or answer["status"] not in expected:
            raise ValueError("Incomplete request-worker terminal state")
    return report


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _cleanup(process, deadline, *, forced):
    """Reap our direct child and verify the entire owned group disappeared.

    An unreaped orphan/unknown survivor fails this proof and closes admission;
    the lane is never reported ready merely because the coordinator exited.
    """
    started = time.monotonic()
    if forced or _group_exists(process.pid):
        _signal_group(process.pid, signal.SIGTERM)
    term_until = min(deadline, started + 0.3)
    while time.monotonic() < term_until:
        process.poll()
        if not _group_exists(process.pid):
            break
        time.sleep(min(POLL_SECONDS, max(0.0, term_until - time.monotonic())))
    if _group_exists(process.pid):
        _signal_group(process.pid, signal.SIGKILL)
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process.pid):
            break
        time.sleep(min(POLL_SECONDS, max(0.0, deadline - time.monotonic())))
    reaped = process.poll() is not None
    absent = not _group_exists(process.pid)
    return {"confirmed": reaped and absent, "worker_pid": process.pid,
        "process_group": process.pid, "worker_reaped": reaped, "group_absent": absent,
        "forced": forced, "cleanup_seconds": time.monotonic() - started}


def _failure(request, stage, kind):
    return {"schema_version": 1, "question": request.question,
        "answer_style": request.answer_style, "status": "failed", "stage": stage,
        # A killed worker's model-load count is unknown, not zero.
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "answer": {"status": "not_completed", "claims": [], "limitations": []},
        "error": {"type": kind, "stage": stage, "message": "Request execution did not complete."}}


class IsolatedRunner:
    """Supervise one fresh Python process per job; do not import model modules."""

    def __init__(self, root, cache_root, *, worker_script=None):
        self.root, self.cache_root = Path(root).resolve(), Path(cache_root).resolve()
        self.worker_script = Path(worker_script) if worker_script else Path(__file__).with_name("request_worker_main.py")
        # Conservative cross-process PubMed spacing, including very short jobs.
        self._pubmed_not_before = 0.0

    def __call__(self, request, output, progress=None):
        return self.run_with_budget(request, output, progress, RequestBudget(time.monotonic()))

    def run_with_budget(self, request, output, progress, budget):
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a fresh output directory")
        directory = output.parent
        directory.mkdir(parents=True, exist_ok=True)
        payload = directory / "worker-request.json"
        result_path = directory / "worker-result.json"
        progress_path = directory / "worker-progress.json"
        for path in (payload, result_path, progress_path):
            if path.exists() or path.is_symlink():
                raise FileExistsError("Choose fresh request-worker artifacts")
        nonce = secrets.token_hex(24)
        _atomic_json(payload, {"protocol": WORKER_PROTOCOL, "nonce": nonce,
            "request_sha256": _request_digest(request), "request": asdict(request), "root": str(self.root),
            "cache_root": str(self.cache_root), "output": str(output),
            "work_deadline": budget.work_deadline, "pubmed_not_before": self._pubmed_not_before})
        process = None
        report = None
        timed_out = False
        stage = "setup"
        last_progress = None
        cleanup = {"confirmed": True, "worker_reaped": True, "group_absent": True,
            "forced": False, "cleanup_seconds": 0.0}
        log_path = directory / "worker.log"
        try:
            if time.monotonic() >= budget.work_deadline:
                timed_out = True
            else:
                with log_path.open("xb") as log:
                    # A new session owns every native descendant. MLX checks the
                    # explicit group identity before inheriting this group.
                    process = subprocess.Popen([sys.executable, "-I", "-u", str(self.worker_script), str(payload)],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
                while process.poll() is None:
                    if time.monotonic() >= budget.work_deadline:
                        timed_out = True
                        break
                    value = None
                    try:
                        checkpoint = _read_json(progress_path)
                        if (not isinstance(checkpoint, dict) or checkpoint.get("stage") not in _REPORT_STAGES
                                or checkpoint.get("status") not in {"running", "complete", "failed", "interrupted"}):
                            raise ValueError("Invalid request-worker progress")
                        value = (checkpoint["stage"], checkpoint["status"])
                    except (OSError, ValueError, KeyError, TypeError):
                        # Missing/atomic-replacement checkpoints are normal while
                        # the child is starting. Callback failures are not.
                        pass
                    if value is not None and value != last_progress:
                        stage = value[0]
                        if progress:
                            try:
                                progress(*value)
                            except Exception as exc:
                                raise _ProgressFailure("Durable request progress failed") from exc
                        last_progress = value
                    time.sleep(min(POLL_SECONDS, max(0.0, budget.work_deadline - time.monotonic())))
                if process.poll() is None:
                    timed_out = True
                elif time.monotonic() > budget.work_deadline:
                    timed_out = True
                elif process.returncode == 0:
                    report = _validated_worker_report(_read_json(result_path), request, nonce)
        except _ProgressFailure:
            report = _failure(request, stage, "RequestProgressError")
            report["progress_persistence_failed"] = True
        except Exception:
            report = _failure(request, stage, "RequestWorkerError")
        finally:
            if process is not None:
                cleanup = _cleanup(process, budget.cleanup_deadline, forced=timed_out or process.poll() is None)
            self._pubmed_not_before = time.monotonic() + 0.4
        if timed_out:
            report = _failure(request, stage, "RequestDeadlineExceeded")
            report["deadline_exceeded"] = True
        elif report is None:
            report = _failure(request, stage, "RequestWorkerError")
        report["native_worker_cleanup"] = cleanup
        report["request_timing"] = {"timeout_seconds": budget.seconds,
            "work_allowance_seconds": budget.seconds - budget.cleanup_reserve - budget.delivery_reserve,
            "terminal_after_cleanup_seconds": budget.elapsed()}
        # This separate supervisor result never overwrites an interrupted live
        # checkpoint. Native cleanup evidence remains durable on every outcome.
        _atomic_json(directory / "worker-outcome.json", report)
        if output.is_dir() and not output.is_symlink():
            # A successful live report also receives the independent cleanup proof.
            # Timeout keeps its partial run.json intact for diagnosis.
            if not timed_out and (output / "run.json").is_file():
                _atomic_json(output / "run.json", report)
            _atomic_json(output / "native-worker-cleanup.json", cleanup)
        return report
