"""Fixed-source replay measures fresh public pipeline calls without quality labels."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medlens.generation.context import canonical
from medlens.pipeline import run_question
import test_answers as answers
import test_pipeline as helpers


SPEC = importlib.util.spec_from_file_location("benchmark_runtime", Path(__file__).resolve().parents[1]
    / "scripts/benchmark_runtime.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class Runtime(answers.FakeRuntime):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.tokenizer = helpers.Tokenizer()
        self.metadata = {"runtime": "fixture", "revision": "pinned"}
        self.closed = False
        self.close_calls = 0
        self._process = SimpleNamespace(poll=lambda: 0 if self.closed else None)

    def generate(self, prompt, *, max_new_tokens):
        result = super().generate(prompt, max_new_tokens=max_new_tokens)
        ids = self.tokenizer.apply_chat_template(prompt["messages"])
        result["input_ids_sha256"] = cli.sha_bytes(canonical(ids).encode())
        return result

    def close(self):
        self.close_calls += 1
        self.closed = True


class RuntimeBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = helpers.index_fixture()
        self.question = "PRIVATE QUESTION FOR FIXED SOURCE REPLAY"
        self.source = run_question(self.index, self.question, encoder_factory=helpers.Encoder,
            tokenizer=helpers.Tokenizer(), runtime_factory=lambda: None, generate=False, rerank=False)
        self.source["input_paths"] = {key: str(self.root / key) for key in ("dataset", "windows", "embeddings")}
        self.source_path = self.root / "explicit-source.json"
        self.source_path.write_text(json.dumps(self.source))
        self.runtimes = []
        self.outputs = [answers.draft(), [{"claim_id": "C1", "verdict": "supported",
            "population_match": "match", "comparison_match": "not_requested", "outcome_match": "match",
            "reason": "supported_by_excerpt"}]]
        self.outputs[0]["claims"][0]["text"] = "PRIVATE GENERATED FINDING"
        self.pipeline = cli.run_question

    def execute(self, *, outputs=None, source=None, change_pipeline=None, repeat=False, compact=False):
        def factory(*args, **kwargs):
            def make():
                runtime = Runtime(copy.deepcopy(self.outputs if outputs is None else outputs))
                if compact:
                    runtime.answer_policy = 'compact-v1'
                    runtime.cache_identity = {'model_profile':'4b','artifact_id':'fixture'}
                self.runtimes.append(runtime)
                return runtime
            make.identity = {"engine": "fixture"}
            return make
        def pipeline(*args, **kwargs):
            result = self.pipeline(*args, **kwargs)
            if change_pipeline:
                change_pipeline(result)
            return result
        output = self.root / "results/replay"
        summary = self.root / "compact.json"
        argv = ["--source-run", str(source or self.source_path), "--engine", "mlx",
            "--output", str(output), "--summary", str(summary)]
        if repeat:
            argv.extend(["--source-run", str(source or self.source_path)])
        read_bytes = Path.read_bytes
        def guard_reads(path):
            if "/benchmarks/" in str(path) or "/data/evaluation/" in str(path):
                raise AssertionError("Benchmark must not discover question or relevance-label files")
            return read_bytes(path)
        with (patch.object(cli, "ROOT", self.root),
              patch.object(cli, "code_identity", return_value={"fixture.py": "code-hash"}),
              patch.object(cli, "DenseIndex", return_value=self.index),
              patch.object(cli, "load_answer_tokenizer", side_effect=lambda *a, **k: helpers.Tokenizer()),
              patch.object(cli, "BGEEncoder", return_value=helpers.Encoder()),
              patch.object(cli, "runtime_factory", side_effect=factory),
              patch.object(cli, "release_accelerator"),
              patch.object(cli, "run_question", side_effect=pipeline),
              patch.object(Path, "read_bytes", guard_reads)):
            code = cli.main(argv)
        return code, json.loads(summary.read_text()), output

    def test_fresh_public_pipeline_replays_explicit_sources_and_summary_contains_no_text(self):
        code, summary, output = self.execute(repeat=True)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(self.runtimes), 2)
        self.assertEqual(len(summary["runs"]), 2)
        for runtime, row in zip(self.runtimes, summary["runs"]):
            self.assertTrue(runtime.closed)
            self.assertEqual(runtime.close_calls, 1)
            self.assertIn("PRIVATE GENERATED FINDING", runtime.calls[1][0]["messages"][1]["content"])
            self.assertEqual(row["runtime_load_count"], 1)
            self.assertTrue(row["worker_reaped"])
            self.assertTrue(row["owned_runtimes_closed"])
            self.assertTrue(row["all_calls_fresh"])
            self.assertTrue(all(row[key] for key in ("same_sources", "same_evidence_payload", "same_dense_ranking")))
            self.assertEqual([call["max_new_tokens"] for call in row["model_calls"]], [768, 512])
            self.assertEqual([call["cache_reused"] for call in row["model_calls"]], [False, False])
            self.assertEqual(len(row["model_calls"][0]["input_ids_sha256"]), 64)
            self.assertTrue((output / row["run_directory"] / "report.html").is_file())
            raw = json.loads((output / row["run_directory"] / "run.json").read_text())
            self.assertEqual(raw["question"], self.question)
            self.assertEqual(raw["context"]["sources"], self.source["context"]["sources"])
        body = json.dumps(summary)
        for private_text in (self.question, "alpha beta gamma", "PRIVATE GENERATED FINDING"):
            self.assertNotIn(private_text, body)
        self.assertFalse(summary["semantic_quality_evaluated"])
        self.assertFalse(summary["qrels_read"])


    def test_replay_runtime_wrapper_preserves_compact4b_schema_policy(self):
        code, summary, output = self.execute(compact=True)
        self.assertEqual(code,0)
        row = summary['runs'][0]
        self.assertEqual(row['prompt_version'],'cited-answer-v8')
        runtime = self.runtimes[0]
        self.assertEqual([p['constraint_kind'] for p,_ in runtime.calls],['draft','literal'])
        self.assertTrue(all('decoding_schema' in p for p,_ in runtime.calls))
        self.assertTrue(runtime.closed)

    def test_validation_failed_is_recorded_separately_and_does_not_hide_sources(self):
        code, summary, output = self.execute(outputs=["not JSON"])
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "validation_failed")
        row = summary["runs"][0]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["answer_status"], "validation_failed")
        self.assertEqual(row["validation_error"], {"stage": "draft", "code": "invalid_json"})
        self.assertEqual(row["claim_count"], 0)
        self.assertTrue(row["worker_reaped"])
        self.assertIn("Selected source excerpts", (output / "run-001/report.html").read_text())

    def test_bad_source_is_bound_to_its_hash_and_fails_before_inference(self):
        self.source["dataset_id"] = "changed"
        self.source_path.write_text(json.dumps(self.source))
        code, summary, output = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["runs"][0]["error"]["type"], "ValueError")
        self.assertEqual(summary["runs"][0]["source_run_sha256"], cli.sha_bytes(self.source_path.read_bytes()))
        self.assertEqual(self.runtimes, [])
        self.assertTrue((output / "run-001/run.json").is_file())

    def test_cached_answers_are_not_successful_measurements(self):
        def cached(report):
            report["answer"]["model_calls"][0]["cache_reused"] = True
        code, summary, _ = self.execute(change_pipeline=cached)
        self.assertEqual(code, 1)
        self.assertEqual(summary["runs"][0]["status"], "failed")
        self.assertTrue(self.runtimes[0].closed)

    def test_native_failure_stays_failed_and_runtime_is_reaped(self):
        with patch.object(Runtime, "generate", side_effect=RuntimeError("PRIVATE NATIVE FAILURE")):
            code, summary, output = self.execute()
        self.assertEqual(code, 1)
        self.assertEqual(summary["runs"][0]["status"], "failed")
        self.assertTrue(summary["runs"][0]["worker_reaped"])
        self.assertNotIn("PRIVATE NATIVE FAILURE", json.dumps(summary))
        raw = json.loads((output / "run-001/run.json").read_text())
        self.assertEqual(raw["context"]["sources"], self.source["context"]["sources"])

    def test_runtime_cannot_report_a_different_rendered_input_hash(self):
        def tamper(report):
            report["answer"]["model_calls"][0]["input_ids_sha256"] = "0" * 64
        code, summary, _ = self.execute(change_pipeline=tamper)
        self.assertEqual(code, 1)
        self.assertEqual(summary["runs"][0]["status"], "failed")

    def test_changed_source_context_is_a_failed_replay(self):
        def tamper(report):
            report["context"]["evidence_payload"] += " changed"
        code, summary, _ = self.execute(change_pipeline=tamper)
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["runs"][0]["worker_reaped"])

    def test_existing_outputs_are_never_overwritten(self):
        output = self.root / "results/existing"
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with patch.object(cli, "ROOT", self.root), self.assertRaises(SystemExit) as caught:
            cli.main(["--source-run", str(self.source_path), "--engine", "mlx", "--output", str(output),
                "--summary", str(self.root / "compact.json")])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
