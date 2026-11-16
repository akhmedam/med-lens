"""Private -I child entry point; intentionally not imported by the API parent."""

import json
import os
from pathlib import Path
import sys
import signal
import time

# Isolated mode excludes the script directory and any untrusted PYTHONPATH.
# Use this installed/source package's exact sibling directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from medlens.api_backend import make_inline_runner
    from medlens.api_contract import SearchRequest
    from medlens.request_worker import (_atomic_json, _request_digest, _worker_envelope,
                                       WORKER_PROTOCOL)
    path = Path(sys.argv[1])
    value = json.loads(path.read_text(encoding="utf-8"))
    def terminate(signum, frame):
        # Let Python finally/MLX.close reap the nested worker when the main
        # thread is responsive. The supervisor still KILLs the group after a
        # short grace period when native code cannot service this signal.
        raise KeyboardInterrupt("Request supervisor requested termination")
    signal.signal(signal.SIGTERM, terminate)
    os.environ["MEDLENS_REQUEST_PROCESS_GROUP"] = str(os.getpgrp())
    os.environ["MEDLENS_REQUEST_WORK_DEADLINE"] = str(value["work_deadline"])
    wait = value["pubmed_not_before"] - time.monotonic()
    if wait > 0:
        time.sleep(min(wait, max(0.0, value["work_deadline"] - time.monotonic())))
    request = SearchRequest.from_payload(value["request"])
    if (type(value.get("protocol")) is not int or value["protocol"] != WORKER_PROTOCOL
            or value.get("request_sha256") != _request_digest(request)
            or not isinstance(value.get("nonce"), str) or len(value["nonce"]) != 48):
        raise ValueError("Invalid request-worker submission identity")
    def progress(stage, status):
        _atomic_json(path.with_name("worker-progress.json"), {"stage": stage, "status": status})
    report = make_inline_runner(value["root"], value["cache_root"])(request, value["output"], progress)
    # Early setup/network failures may precede answer construction. Give these
    # failures an explicit uncompleted answer without repairing malformed success.
    if isinstance(report, dict) and report.get("status") in {"failed", "interrupted"} and "answer" not in report:
        report = {**report, "answer": {"status": "not_completed", "claims": []}}
    _atomic_json(path.with_name("worker-result.json"), _worker_envelope(request, value["nonce"], report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
