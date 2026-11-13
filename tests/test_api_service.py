"""Real worker-thread lifecycle checks with fake, controllable search runners."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

from medlens.api_contract import SearchRequest
from medlens.api_service import JobService, ServiceBusy, ServiceUnavailable, UnknownRequest


def complete(request, answer_status="insufficient_evidence"):
    return {"status": "complete", "stage": "complete", "question": request.question,
        "runtime_load_count": 0,
        "answer": {"status": answer_status, "claims": [], "limitations": []},
        "retrieval": {"results": [{"pmid": "123", "title": "Retrieved study", "score": .8}]}}


class JobServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.number = 0
        self.request = SearchRequest(question="Does intervention A reduce outcome B?")

    def service(self, runner, root=None):
        self.number += 1
        service = JobService(root or self.root / str(self.number), runner)
        self.addCleanup(service.close)
        return service

    def finish(self, service):
        # Synchronize on actual executor completion, rather than sleep/poll a job.
        service._future.result(timeout=5)

    def test_one_lane_stays_owned_through_pipeline_complete_and_cleanup(self):
        started, finish_work, cleanup_started, finish_cleanup = (Event() for _ in range(4))
        def runner(request, output, progress):
            self.assertFalse(output.exists())
            started.set()
            if not finish_work.wait(5):raise TimeoutError("test work was not released")
            progress("complete", "complete")
            cleanup_started.set()
            if not finish_cleanup.wait(5):raise TimeoutError("test cleanup was not released")
            return complete(request)
        service = self.service(runner)
        self.addCleanup(finish_work.set)
        self.addCleanup(finish_cleanup.set)
        submitted = service.submit(self.request)
        self.assertTrue(started.wait(2))
        self.assertEqual(submitted["status"], "queued")
        response_ready, values = Event(), {}
        def read_status():
            values["health"] = service.health()
            values["job"] = service.get(submitted["request_id"])
            response_ready.set()
        reader = Thread(target=read_status)
        reader.start()
        self.assertTrue(response_ready.wait(2), "health/get waited behind the search")
        reader.join(2)
        self.assertEqual(values["health"]["status"], "busy")
        self.assertEqual(values["job"]["status"], "running")
        with self.assertRaises(ServiceBusy):service.submit(self.request)
        # A client abandoning its await cannot cancel an already-running job.
        self.assertFalse(service._future.cancel())
        finish_work.set()
        self.assertTrue(cleanup_started.wait(2))
        self.assertEqual(service.get(submitted["request_id"])["stage"], "complete")
        self.assertEqual(service.get(submitted["request_id"])["status"], "running")
        with self.assertRaises(ServiceBusy):service.submit(self.request)
        finish_cleanup.set()
        self.finish(service)
        self.assertEqual(service.get(submitted["request_id"])["status"], "succeeded")
        self.assertEqual(service.health()["status"], "ready")

    def test_jobs_have_distinct_server_owned_paths_and_reports_survive_restart(self):
        paths = []
        def runner(request, output, progress):
            self.assertFalse(output.exists())
            paths.append(output)
            output.mkdir()
            (output / "report.html").write_text("<p>Research report</p>")
            return complete(request)
        service = self.service(runner)
        first = service.submit(self.request)
        self.finish(service)
        second = service.submit(self.request)
        self.finish(service)
        self.assertNotEqual(first["request_id"], second["request_id"])
        self.assertEqual(paths, [service.storage_root / first["request_id"] / "run",
                                 service.storage_root / second["request_id"] / "run"])
        for job in [first, second]:
            path = service.report_path(job["request_id"])
            self.assertEqual(path.read_text(), "<p>Research report</p>")
            self.assertTrue((path.parent.parent / "job.json").is_file())
        expected = service.get(first["request_id"])
        service.close()
        restored = self.service(runner, root=service.storage_root)
        self.assertEqual(restored.get(first["request_id"]), expected)
        changed = restored.get(first["request_id"])
        changed["request"]["question"] = "Mutated client copy"
        self.assertEqual(restored.get(first["request_id"]), expected)

    def test_reported_upstream_failure_allows_retry_without_leaking_error_path(self):
        calls = []
        def runner(request, output, progress):
            calls.append(output)
            if len(calls) == 1:
                report = complete(request)
                report.update(status="failed", stage="pubmed-download",
                    error={"type": "OSError", "message": "/Users/private/secret-source.xml", "stage": "pubmed-download"})
                return report
            return complete(request)
        service = self.service(runner)
        first = service.submit(self.request)
        self.finish(service)
        result = service.get(first["request_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "pubmed_error")
        self.assertNotIn("/Users/private", json.dumps(result))
        self.assertEqual(service.health()["status"], "ready")
        second = service.submit(self.request)
        self.finish(service)
        self.assertEqual(service.get(second["request_id"])["status"], "succeeded")

    def test_answer_abstention_partial_and_validation_failure_are_distinct(self):
        reports = iter(["partial", "insufficient_evidence", "validation_failed"])
        def runner(request, output, progress):
            report = complete(request, next(reports))
            report.update(runtime_load_count=1, runtime_closed=True)
            return report
        service = self.service(runner)
        for answer_status, expected in [("partial", "succeeded"), ("insufficient_evidence", "succeeded"),
                                         ("validation_failed", "failed")]:
            job = service.submit(self.request)
            self.finish(service)
            result = service.get(job["request_id"])
            self.assertEqual(result["status"], expected)
            self.assertEqual(result["result"]["status"], "complete")
            self.assertEqual(result["result"]["answer"]["status"], answer_status)
            self.assertEqual(result["result"]["articles"][0]["pmid"], "123")
            if expected == "failed":
                self.assertEqual(result["error"]["code"], "answer_validation_failed")
        self.assertEqual(service.health()["status"], "ready")

    def test_uncertain_cleanup_makes_service_unavailable(self):
        variants = [dict(runtime_load_count=1, runtime_closed=False),
                    dict(runtime_load_count=1),
                    dict(runtime_load_count=0, cleanup_error={"message": "/Users/private/worker.log"})]
        for fields in variants:
            with self.subTest(fields=fields):
                def runner(request, output, progress):
                    return {**complete(request), **fields}
                service = self.service(runner)
                job = service.submit(self.request)
                self.finish(service)
                result = service.get(job["request_id"])
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["code"], "runtime_cleanup_failed")
                self.assertNotIn("/Users/private", json.dumps(result))
                self.assertEqual(service.health()["status"], "unavailable")
                with self.assertRaises(ServiceUnavailable):service.submit(self.request)

    def test_escaped_exception_has_no_cleanup_proof_and_no_raw_message(self):
        def runner(request, output, progress):
            raise RuntimeError("/Users/private/model/config.json secret")
        service = self.service(runner)
        job = service.submit(self.request)
        self.finish(service)
        result = service.get(job["request_id"])
        self.assertEqual(result["error"]["code"], "internal_error")
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("/Users/private", json.dumps(result))
        self.assertEqual(service.health()["status"], "unavailable")
        with self.assertRaises(ServiceUnavailable):service.submit(self.request)

    def test_shutdown_waits_for_accepted_work_and_stops_admission(self):
        started, release, shutdown_entered = Event(), Event(), Event()
        def runner(request, output, progress):
            started.set()
            if not release.wait(5):raise TimeoutError("test shutdown was not released")
            return complete(request)
        service = self.service(runner)
        self.addCleanup(release.set)
        service.submit(self.request)
        self.assertTrue(started.wait(2))
        real_shutdown = service._executor.shutdown
        def shutdown(*args, **kwargs):
            shutdown_entered.set()
            return real_shutdown(*args, **kwargs)
        with patch.object(service._executor, "shutdown", side_effect=shutdown):
            closer = Thread(target=service.close)
            closer.start()
            self.assertTrue(shutdown_entered.wait(2))
            self.assertTrue(closer.is_alive(), "close returned before active work/cleanup")
            self.assertEqual(service.health()["status"], "stopping")
            with self.assertRaises(ServiceUnavailable):service.submit(self.request)
            release.set()
            closer.join(2)
            self.assertFalse(closer.is_alive())
        self.assertEqual(service.health()["status"], "closed")

    def test_unknown_paths_and_symlink_reports_cannot_escape_storage(self):
        service = self.service(lambda request, output, progress: complete(request))
        for value in ["../outside", "/etc/passwd", "a" * 31, "a" * 33, "A" * 32, None]:
            with self.subTest(value=value), self.assertRaises(UnknownRequest):service.get(value)
        outside = self.root / "outside"
        outside.mkdir()
        fake_id = "f" * 32
        (outside / "job.json").write_text(json.dumps({"schema_version": 1,
            "request_id": fake_id, "status": "succeeded", "secret": "outside"}))
        (service.storage_root / fake_id).symlink_to(outside, target_is_directory=True)
        with self.assertRaises(UnknownRequest):service.get(fake_id)
        with self.assertRaises(UnknownRequest):service.report_path(fake_id)
        job = service.submit(self.request)
        self.finish(service)
        with self.assertRaises(FileNotFoundError):service.report_path(job["request_id"])
        run = service.storage_root / job["request_id"] / "run"
        run.mkdir()
        secret = outside / "secret.html"
        secret.write_text("outside")
        (run / "report.html").symlink_to(secret)
        with self.assertRaises(FileNotFoundError):service.report_path(job["request_id"])

    def test_completed_payload_history_is_read_from_disk_without_memory_growth(self):
        service = self.service(lambda request, output, progress: complete(request))
        ids = []
        for _ in range(12):
            job = service.submit(self.request)
            ids.append(job["request_id"])
            self.finish(service)
            self.assertFalse(service._jobs)
        for request_id in ids:
            result = service.get(request_id)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["result"]["articles"][0]["pmid"], "123")
        self.assertFalse(service._jobs, "Reading history must not refill an unbounded cache")

    def test_failed_terminal_persistence_retains_one_readable_failure_and_stops_admission(self):
        service = self.service(lambda request, output, progress: complete(request))
        persist = service._persist
        def fail_terminal(job):
            if job["status"] in {"succeeded", "failed", "interrupted"}:
                raise OSError("/Users/private/full-disk")
            return persist(job)
        with patch.object(service, "_persist", side_effect=fail_terminal):
            job = service.submit(self.request)
            self.finish(service)
        result = service.get(job["request_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "storage_error")
        self.assertNotIn("full-disk", json.dumps(result))
        self.assertEqual(len(service._jobs), 1)
        self.assertEqual(service.health()["status"], "unavailable")
        with self.assertRaises(ServiceUnavailable):service.submit(self.request)

    def test_interruption_is_terminal_but_confirmed_cleanup_permits_next_job(self):
        calls = []
        def runner(request, output, progress):
            calls.append(output)
            report = complete(request)
            if len(calls) == 1:
                report.update(status="interrupted", stage="answer", runtime_load_count=1,
                    runtime_closed=True, error={"stage": "answer", "message": "/Users/private"})
            return report
        service = self.service(runner)
        first = service.submit(self.request)
        self.finish(service)
        self.assertEqual(service.get(first["request_id"])["status"], "interrupted")
        self.assertEqual(service.get(first["request_id"])["error"]["code"], "search_interrupted")
        second = service.submit(self.request)
        self.finish(service)
        self.assertEqual(service.get(second["request_id"])["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
