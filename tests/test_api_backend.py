"""API backend boundaries use the existing pipeline without model/network work."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from medlens import api_backend


@dataclass(frozen=True)
class Request:
    question: str = "What does the fictional study show?"
    pubmed_query: str | None = None
    limit: int = 100
    top_k: int = 10
    generate: bool = True
    answer_style: str = "draft"


class BackendTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"HF_HOME": "/fixture/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1"})
        environment.start()
        self.addCleanup(environment.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "cache"
        self.deps = SimpleNamespace(
            runtime_factory=Mock(return_value=Mock()),
            PubMedClient=Mock(return_value=object()),
            local_encoder_policy=Mock(return_value={"fixture": "pinned-policy"}),
            run_live_question=Mock(return_value={"status": "complete", "runtime_closed": True}),
            load_tokenizer=Mock(return_value=("bge-tokenizer", {"fixture": True})),
            BGEEncoder=Mock(return_value="bge-encoder"),
            load_answer_tokenizer=Mock(return_value="answer-tokenizer"))

    def test_module_import_and_runner_creation_do_not_import_ml_or_read_assets(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = ("import sys; sys.path.insert(0, " + repr(source) + "); "
            "from medlens.api_backend import make_local_runner; "
            "runner = make_local_runner('/fixture/root', '/fixture/cache'); "
            "assert callable(runner); "
            "assert not any(n == p or n.startswith(p + '.') for n in sys.modules "
            "for p in ('torch', 'transformers', 'numpy', 'mlx', 'mlx_lm', "
            "'medlens.live', 'medlens.inference', 'medlens.retrieval')); print('lazy')")
        result = subprocess.run([sys.executable, "-I", "-S", "-c", code],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "lazy")
        with patch.object(api_backend, "_load_pipeline") as load:
            api_backend.make_local_runner(self.root, self.cache)
        load.assert_not_called()

    def test_parameters_assets_callbacks_and_default_model_are_forwarded(self):
        request = Request(pubmed_query='"fictional condition"[Title]', limit=17,
            top_k=4, generate=False, answer_style="source-findings")
        output = self.root / "job" / "run"
        progress = Mock()
        release = Mock()
        def live(question, actual_output, cache, **kwargs):
            self.assertEqual(question, request.question)
            self.assertEqual(actual_output, output)
            self.assertEqual(cache, self.cache)
            self.assertFalse(output.exists())
            self.assertEqual(kwargs["encoder_info"], {"fixture": "pinned-policy"})
            self.assertEqual(kwargs["tokenizer_loader"](), ("bge-tokenizer", {"fixture": True}))
            self.assertEqual(kwargs["encoder_factory"](), "bge-encoder")
            self.assertEqual(kwargs["answer_tokenizer_loader"](), "answer-tokenizer")
            self.assertIs(kwargs["runtime_factory"], self.deps.runtime_factory.return_value)
            self.assertIs(kwargs["client"], self.deps.PubMedClient.return_value)
            self.assertIs(kwargs["progress"], progress)
            self.assertIs(kwargs["release_accelerator"], release)
            self.assertEqual({k: kwargs[k] for k in ("limit", "pubmed_query", "generate", "top_k", "answer_style")},
                {"limit": 17, "pubmed_query": request.pubmed_query, "generate": False,
                 "top_k": 4, "answer_style": "source-findings"})
            self.assertNotIn("rerank", kwargs)
            kwargs["release_accelerator"]()
            return {"status": "complete"}
        self.deps.run_live_question.side_effect = live
        with patch.object(api_backend, "_load_pipeline", return_value=self.deps), \
             patch.object(api_backend, "_release_mps", release):
            result = api_backend.make_local_runner(self.root, self.cache)(request, output, progress)
        self.assertEqual(result, {"status": "complete"})
        self.deps.runtime_factory.assert_called_once_with("mlx", model="4b", root=self.root)
        self.deps.PubMedClient.assert_called_once_with()
        self.deps.local_encoder_policy.assert_called_once_with(self.root / ".cache/models/bge-base-en-v1.5", device="mps")
        self.deps.load_tokenizer.assert_called_once_with(self.root / ".cache/tokenizers/bge-base-en-v1.5")
        self.deps.BGEEncoder.assert_called_once_with(self.root / ".cache/models/bge-base-en-v1.5",
            self.root / ".cache/tokenizers/bge-base-en-v1.5", device="mps")
        self.deps.load_answer_tokenizer.assert_called_once_with(self.root, "4b")
        release.assert_called_once_with()

    def test_consecutive_jobs_create_distinct_factories_and_runtime_instances(self):
        factories, runtimes = [], []
        def create_factory(*args, **kwargs):
            def factory():
                runtime = SimpleNamespace(close=Mock())
                runtimes.append(runtime)
                return runtime
            factories.append(factory)
            return factory
        self.deps.runtime_factory.side_effect = create_factory
        def live(*args, **kwargs):
            runtime = kwargs["runtime_factory"]()
            runtime.close()
            return {"status": "complete", "runtime_closed": True, "runtime_load_count": 1}
        self.deps.run_live_question.side_effect = live
        runner = api_backend.make_local_runner(self.root, self.cache)
        with patch.object(api_backend, "_load_pipeline", return_value=self.deps):
            runner(Request(), self.root / "first" / "run")
            runner(Request(), self.root / "second" / "run")
        self.assertEqual(len(factories), 2)
        self.assertEqual(len(runtimes), 2)
        self.assertIsNot(factories[0], factories[1])
        self.assertIsNot(runtimes[0], runtimes[1])
        for runtime in runtimes:
            runtime.close.assert_called_once_with()

    def test_pubmed_limiter_is_shared_across_immediate_search_only_jobs(self):
        from medlens.ingestion.pubmed import PubMedClient, parse_search
        from pubmed_helpers import Clock, Transport, search_json
        clock = Clock()
        transport = Transport([search_json([], 0), search_json([], 0)], clock)
        self.deps.PubMedClient = Mock(side_effect=lambda: PubMedClient(
            opener=transport, sleep=clock.sleep, monotonic=clock.now))
        clients = []
        def live(question, output, cache, **kwargs):
            clients.append(kwargs['client'])
            response = kwargs['client'].search(question, limit=1)
            self.assertEqual(parse_search(response.body, 1).total_count, 0)
            return {'status': 'complete', 'stage': 'complete', 'runtime_load_count': 0,
                'answer': {'status': 'not_requested', 'claims': []}}
        self.deps.run_live_question.side_effect = live
        runner = api_backend.make_local_runner(self.root, self.cache)
        self.deps.PubMedClient.assert_not_called()
        with patch.object(api_backend, '_load_pipeline', return_value=self.deps):
            first = runner(Request(generate=False), self.root / 'first' / 'run')
            second = runner(Request(generate=False), self.root / 'second' / 'run')
        self.assertEqual((first['status'], second['status']), ('complete', 'complete'))
        self.deps.PubMedClient.assert_called_once_with()
        self.assertIs(clients[0], clients[1])
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual([call[0] for call in transport.calls], [0.0, 0.4])
        self.assertEqual(clock.sleeps, [0.4])
        self.assertEqual(self.deps.runtime_factory.call_count, 2)
        self.deps.runtime_factory.return_value.assert_not_called()

    def test_existing_output_or_dangling_link_is_rejected_before_imports(self):
        output = self.root / "prior"
        output.mkdir()
        sentinel = output / "user-data.txt"
        sentinel.write_text("preserved")
        link = self.root / "dangling"
        link.symlink_to(self.root / "missing-target", target_is_directory=True)
        with patch.object(api_backend, "_load_pipeline") as load:
            runner = api_backend.make_local_runner(self.root, self.cache)
            for path in (output, link):
                with self.subTest(path=path.name), self.assertRaises(FileExistsError):
                    runner(Request(), path)
        load.assert_not_called()
        self.assertEqual(sentinel.read_text(), "preserved")
        self.assertTrue(link.is_symlink())

    def test_missing_setup_assets_return_private_failed_artifacts_without_native_load(self):
        self.deps.local_encoder_policy.side_effect = FileNotFoundError("fixture missing BGE manifest")
        progress = Mock()
        output = self.root / "job" / "run"
        with patch.object(api_backend, "_load_pipeline", return_value=self.deps):
            result = api_backend.make_local_runner(self.root, self.cache)(Request(), output, progress)
        self.assertEqual((result["status"], result["stage"], result["runtime_load_count"]), ("failed", "setup", 0))
        self.assertEqual(result["error"], {"type": "FileNotFoundError", "message": "fixture missing BGE manifest", "stage": "setup"})
        self.assertEqual(json.loads((output / "run.json").read_text()), result)
        self.assertIn("Run failed at setup", (output / "report.html").read_text())
        self.deps.run_live_question.assert_not_called()
        self.deps.runtime_factory.return_value.assert_not_called()
        self.deps.PubMedClient.assert_not_called()
        progress.assert_called_once_with("setup", "failed")

    def test_missing_dependency_is_a_setup_failure_and_artifact_io_is_optional(self):
        with patch.object(api_backend, "_load_pipeline", side_effect=ImportError("fixture dependency unavailable")):
            runner = api_backend.make_local_runner(self.root, self.cache)
            output = self.root / "import-failure" / "run"
            result = runner(Request(), output)
            self.assertEqual(result["error"]["type"], "ImportError")
            self.assertEqual(result["runtime_load_count"], 0)
            self.assertTrue((output / "run.json").is_file())
            with patch.object(Path, "mkdir", side_effect=PermissionError("fixture disk unavailable")):
                unavailable = runner(Request(), self.root / "no-artifacts" / "run")
            self.assertEqual(unavailable["status"], "failed")
            self.assertEqual(unavailable["runtime_load_count"], 0)
            self.assertEqual(unavailable["error"]["type"], "ImportError")
            self.assertEqual(unavailable["artifact_error"]["type"], "PermissionError")

    def test_live_report_and_cleanup_failure_flags_are_preserved(self):
        report = {"status": "failed", "stage": "answer", "runtime_load_count": 1,
            "runtime_closed": False, "cleanup_error": {"message": "fixture cleanup failed"}}
        self.deps.run_live_question.return_value = report
        with patch.object(api_backend, "_load_pipeline", return_value=self.deps), \
             patch.object(api_backend, "_persist_setup_report") as persist:
            result = api_backend.make_local_runner(self.root, self.cache)(Request(), self.root / "run")
        self.assertIs(result, report)
        persist.assert_not_called()

    def test_unexpected_live_exception_propagates_without_overwriting_artifacts(self):
        output = self.root / "job" / "run"
        def live(*args, **kwargs):
            output.mkdir(parents=True)
            (output / "run.json").write_text("partial live checkpoint")
            raise RuntimeError("fixture unexpected live failure")
        self.deps.run_live_question.side_effect = live
        with patch.object(api_backend, "_load_pipeline", return_value=self.deps), \
             patch.object(api_backend, "_persist_setup_report") as persist:
            with self.assertRaisesRegex(RuntimeError, "unexpected live failure"):
                api_backend.make_local_runner(self.root, self.cache)(Request(), output)
        persist.assert_not_called()
        self.assertEqual((output / "run.json").read_text(), "partial live checkpoint")

    def test_real_live_search_only_path_uses_fake_pubmed_and_never_loads_qwen(self):
        from medlens import live
        from pubmed_helpers import make_client, search_json
        from test_live import Encoder, QUESTION, xml
        from test_shared_corpus import INFO
        from test_windows import Tokenizer as BGTokenizer
        from test_pipeline import Tokenizer as AnswerTokenizer
        client, _, transport = make_client([search_json(['1'], 1), xml()])
        native = Mock(side_effect=AssertionError("Search-only must not instantiate Qwen"))
        native.identity = {"engine": "mlx", "answer_model": "4b", "answer_policy": "compact-v1"}
        deps = SimpleNamespace(runtime_factory=Mock(return_value=native),
            PubMedClient=Mock(return_value=client),
            local_encoder_policy=Mock(return_value=Encoder().info),
            load_tokenizer=lambda path: (BGTokenizer(), INFO),
            BGEEncoder=lambda *args, **kwargs: Encoder(),
            load_answer_tokenizer=lambda *args: AnswerTokenizer(),
            run_live_question=live.run_live_question)
        release, progress = Mock(), Mock()
        output = self.root / "integration" / "run"
        with patch.object(api_backend, "_load_pipeline", return_value=deps), \
             patch.object(api_backend, "_release_mps", release), \
             patch.object(live, "sample_memory", side_effect=lambda stage: {"stage": stage, "process_peak_rss_bytes": 0}):
            result = api_backend.make_local_runner(self.root, self.cache)(
                Request(question=QUESTION, generate=False), output, progress)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['live']['fresh_search_complete'])
        self.assertEqual(result['runtime_load_count'], 0)
        self.assertEqual(result['answer']['status'], 'not_requested')
        self.assertEqual([row['pmid'] for row in result['retrieval']['results']], ['1'])
        self.assertEqual(len(transport.calls), 2)
        self.assertTrue((output / 'run.json').is_file())
        self.assertTrue((output / 'report.html').is_file())
        self.assertGreaterEqual(release.call_count, 3)
        self.assertIn(('pubmed-download', 'running'), [call.args for call in progress.call_args_list])
        native.assert_not_called()
        deps.runtime_factory.assert_called_once_with('mlx', model='4b', root=self.root)

    def test_huggingface_environment_matches_cli_and_is_set_only_at_job_start(self):
        with patch.dict(os.environ, {"HF_HOME": "/fixture/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0"}):
            runner = api_backend.make_local_runner(self.root, self.cache)
            self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "0")
            with patch.object(api_backend, "_load_pipeline", return_value=self.deps):
                runner(Request(), self.root / "first")
            self.assertEqual(os.environ["HF_HOME"], "/fixture/hf")
            self.assertEqual(os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HOME", None)
            with patch.object(api_backend, "_load_pipeline", return_value=self.deps):
                api_backend.make_local_runner(self.root, self.cache)(Request(), self.root / "second")
            self.assertEqual(os.environ["HF_HOME"], str(self.root / ".cache/huggingface"))


if __name__ == "__main__":
    unittest.main()
