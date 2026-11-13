"""HTTP contract checks; real job lifetime is exercised by test_api_service."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from io import StringIO
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

from medlens.api_service import JobService, ServiceBusy, ServiceUnavailable, UnknownRequest
from medlens.api_server import main

HAS_HTTP = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "httpx"))
if HAS_HTTP:
    from fastapi.testclient import TestClient
    from medlens.api import MAX_BODY_BYTES, create_app


class FakeService:
    def __init__(self):
        self.calls = []
        self.jobs = {}
        self.busy = False
        self.unavailable = False
        self.closed = 0

    def submit(self, request):
        if self.unavailable:
            raise ServiceUnavailable("/Users/private/model/.secret-internal")
        if self.busy:
            raise ServiceBusy()
        self.calls.append(request)
        job = {"schema_version": 1, "request_id": "a" * 32, "status": "queued", "stage": "queued",
            "created_at": "2026-11-13T00:00:00+00:00", "updated_at": "2026-11-13T00:00:00+00:00",
            "started_at": None, "finished_at": None, "request": asdict(request), "result": None, "error": None}
        self.jobs[job["request_id"]] = job
        return deepcopy(job)

    def get(self, request_id):
        if request_id not in self.jobs:
            raise UnknownRequest()
        return deepcopy(self.jobs[request_id])

    def health(self):
        status = "closed" if self.closed else "unavailable" if self.unavailable else "busy" if self.busy else "ready"
        return {"status": status, "accepting_requests": status == "ready", "active_request_id": None,
            "max_active_requests": 1, "private_field": "/Users/private/model"}

    def close(self):
        self.closed += 1


@unittest.skipUnless(HAS_HTTP, "Install API and HTTP-test extras for transport checks")
class HTTPTests(unittest.TestCase):
    def test_accepts_async_job_with_unchanged_question_and_explicit_defaults(self):
        service = FakeService()
        with TestClient(create_app(service)) as client:
            response = client.post("/v1/requests", json={"question": " Does agent A reduce pain? "})
            self.assertEqual(response.status_code, 202)
            job = response.json()
            self.assertEqual(job["status"], "queued")
            self.assertIsNone(job["result"])
            self.assertEqual(service.calls[0].question, " Does agent A reduce pain? ")
            self.assertEqual(job["request"]["limit"], 100)
            self.assertEqual(job["request"]["answer_style"], "draft")
            self.assertTrue(job["request"]["generate"])
            self.assertEqual(response.headers["location"], job["links"]["self"])
            self.assertEqual(client.get(job["links"]["self"]).json(), job)
        self.assertEqual(service.closed, 1)

    def test_experimental_mode_and_search_only_are_explicit(self):
        service = FakeService()
        with TestClient(create_app(service)) as client:
            r = client.post("/v1/requests", json={"question": "Agent X effects?", "generate": False,
                "answer_style": "source-findings", "pubmed_query": "agent X[Title]", "limit": 7, "top_k": 2})
            self.assertEqual(r.status_code, 202)
            self.assertFalse(service.calls[0].generate)
            self.assertEqual(service.calls[0].pubmed_query, "agent X[Title]")
            self.assertEqual(service.calls[0].answer_style, "source-findings")

    def test_invalid_inputs_never_reach_admission(self):
        service = FakeService()
        invalid = [{}, {"question": " "}, {"question": "X\nY"}, {"question": 1},
            {"question": "X", "limit": True}, {"question": "X", "limit": 301},
            {"question": "X", "top_k": "2"}, {"question": "X", "generate": 1},
            {"question": "X", "answer_style": "other"}, {"question": "X", "output_dir": "/tmp/private"},
            {"question": "X", "pubmed_query": ""}, {"question": "X" * 8001}]
        with TestClient(create_app(service)) as client:
            for payload in invalid:
                with self.subTest(keys=list(payload)):
                    r = client.post("/v1/requests", json=payload)
                    self.assertEqual(r.status_code, 422)
                    self.assertEqual(r.json()["error"]["code"], "invalid_request")
                    self.assertNotIn("input", r.json()["error"])
        self.assertEqual(service.calls, [])

    def test_duplicate_nonfinite_and_non_utf8_json_rejected(self):
        service = FakeService()
        with TestClient(create_app(service)) as client:
            for data in (b'{"question":"X","question":"Y"}', b'{"question":"X","limit":NaN}', b'\xff', b'{'):
                with self.subTest(data=data):
                    r = client.post("/v1/requests", content=data, headers={"Content-Type": "application/json"})
                    self.assertEqual(r.status_code, 400)
                    self.assertEqual(r.json()["error"]["code"], "invalid_json")
            self.assertEqual(client.post("/v1/requests", json=[]).status_code, 422)
        self.assertEqual(service.calls, [])

    def test_media_type_and_body_budget_apply_before_job_creation(self):
        service = FakeService()
        with TestClient(create_app(service)) as client:
            self.assertEqual(client.post("/v1/requests", content='{"question":"X"}').status_code, 415)
            r = client.post("/v1/requests", content=b" " * (MAX_BODY_BYTES + 1),
                headers={"Content-Type": "application/json"})
            self.assertEqual(r.status_code, 413)
            r = client.post("/v1/requests", content=(chunk for chunk in (b" " * MAX_BODY_BYTES, b" ")),
                headers={"Content-Type": "application/json"})
            self.assertEqual(r.status_code, 413)
            r = client.post("/v1/requests", content='{}',
                headers={"Content-Type": "application/json", "Content-Length": "-1"})
            self.assertEqual(r.status_code, 400)
        self.assertEqual(service.calls, [])

    def test_busy_returns_immediately_and_health_remains_available(self):
        service = FakeService()
        service.busy = True
        with TestClient(create_app(service)) as client:
            r = client.post("/v1/requests", json={"question": "Agent X effects?"})
            self.assertEqual(r.status_code, 409)
            self.assertEqual(r.headers["retry-after"], "1")
            self.assertEqual(r.json()["error"]["code"], "service_busy")
            self.assertEqual(client.get("/health").json()["status"], "busy")
            self.assertNotIn("private_field", client.get("/health").json())

    def test_unavailable_error_does_not_echo_internal_exception(self):
        service = FakeService()
        service.unavailable = True
        with TestClient(create_app(service)) as client:
            r = client.post("/v1/requests", json={"question": "Agent X effects?"})
            self.assertEqual(r.status_code, 503)
            self.assertNotIn("/Users", r.text)
            self.assertEqual(client.get("/health").status_code, 503)

    def test_unknown_ids_and_methods_have_structured_errors(self):
        with TestClient(create_app(FakeService())) as client:
            self.assertEqual(client.get("/v1/requests/unknown").status_code, 404)
            self.assertEqual(client.get("/v1/requests/" + "0" * 32).json()["error"]["code"], "unknown_request")
            r = client.put("/v1/requests", json={})
            self.assertEqual(r.status_code, 405)
            self.assertIn("POST", r.headers["allow"])
            self.assertEqual(client.get("/missing").json()["error"]["code"], "not_found")

    def test_openapi_describes_strict_inputs_and_separate_job_states(self):
        with TestClient(create_app(FakeService())) as client:
            schema = client.get("/openapi.json").json()
            body = schema["paths"]["/v1/requests"]["post"]["requestBody"]["content"]["application/json"]["schema"]
            self.assertFalse(body["additionalProperties"])
            self.assertEqual(body["required"], ["question"])
            self.assertEqual(body["properties"]["answer_style"]["default"], "draft")
            self.assertNotIn("model_dir", body["properties"])
            self.assertEqual(schema["components"]["schemas"]["JobResponse"]["properties"]["status"]["enum"],
                ["queued", "running", "succeeded", "failed", "interrupted"])

    def test_health_needs_no_assets_or_model_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("medlens.api_backend.make_local_runner") as factory:
                app = create_app(root=root)
                with TestClient(app) as client:
                    self.assertEqual(client.get("/health").json()["status"], "ready")
                    factory.return_value.assert_not_called()
            self.assertEqual(list((root / "results/api-requests").iterdir()), [])

    def test_http_lifespan_waits_for_the_real_job_worker_before_closing(self):
        started, release, closing, exited = Event(), Event(), Event(), Event()
        observations = {}
        def runner(request, output, progress):
            started.set()
            if not release.wait(5):
                raise RuntimeError("Test runner was not released")
            return {"status": "complete", "stage": "complete", "runtime_load_count": 0,
                "answer": {"status": "not_requested", "claims": []}}
        class Service(JobService):
            def close(self):
                closing.set()
                super().close()
        def unblock():
            observations["close_called"] = closing.wait(5)
            observations["exited_while_worker_blocked"] = exited.wait(.05)
            release.set()
        with tempfile.TemporaryDirectory() as directory:
            service = Service(directory, runner)
            controller = Thread(target=unblock)
            try:
                with TestClient(create_app(service)) as client:
                    response = client.post("/v1/requests", json={"question": "Agent X effects?", "generate": False})
                    self.assertEqual(response.status_code, 202)
                    request_id = response.json()["request_id"]
                    self.assertTrue(started.wait(5))
                    controller.start()
                exited.set()
                controller.join(5)
                self.assertTrue(observations["close_called"])
                self.assertFalse(observations["exited_while_worker_blocked"])
                self.assertEqual(service.get(request_id)["status"], "succeeded")
                self.assertEqual(service.health()["status"], "closed")
            finally:
                release.set()
                if controller.ident is not None:
                    controller.join(5)
                service.close()

    def test_storage_cache_overlap_rejected_before_runner_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("medlens.api_backend.make_local_runner") as factory:
                with self.assertRaises(ValueError):
                    create_app(root=directory, storage_root=Path(directory)/"shared", cache_root=Path(directory)/"shared/cache")
                factory.assert_not_called()


class ServerCLITests(unittest.TestCase):
    def test_help_and_invalid_bind_parameters_need_no_models(self):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as result:
                main(["--help"])
            self.assertEqual(result.exception.code, 0)
            for args in (["--host", "0.0.0.0"], ["--port", "0"], ["--port", "65536"], ["--workers", "2"]):
                with self.subTest(args=args), self.assertRaises(SystemExit) as result:
                    main(args)
                self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
