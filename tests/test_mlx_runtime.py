"""Native-runtime contracts without model downloads or GPU imports."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from medlens.inference import mlx as runtime_module
from medlens.inference.mlx import MlxRuntime, _digest
from medlens.inference.mlx_worker import _generate, _parameter_dtypes, _validate_request, require_finite_scores
from medlens.inference.qwen import MODEL_ID, REVISION


FAKE_WORKER = '''import argparse, hashlib, json, os, sys, time
p = argparse.ArgumentParser()
p.add_argument('--model-dir'); p.add_argument('--nonce')
a = p.parse_args()
root = a.model_dir
mode = json.load(open(root + '/mode.json'))
open(root + '/worker.pid', 'w').write(str(os.getpid()))
def send(x):
    print(json.dumps(x), flush=True)
ready = dict(op='ready', protocol=1, nonce=a.nonce,
    packages={n:'0.32.2' for n in ('mlx','mlx-lm','mlx-metal')},
    vocab_size=248320, eos_token_ids=[248044,248046], load_seconds=0.01,
    parameter_dtypes=['float16','float32','uint32'],
    parameter_dtype_policy='float16-except-A_log-float32-packed-uint32')
ready.update(mode.get('ready_override', {}))
if mode.get('behavior') == 'load_timeout': time.sleep(30)
send(ready)
for line in sys.stdin:
    q = json.loads(line)
    if q['op'] == 'close':
        send(dict(op='closed', nonce=a.nonce)); break
    open(root + '/request.json', 'w').write(json.dumps(q))
    behavior = mode.get('behavior')
    if behavior == 'timeout': time.sleep(30)
    if behavior == 'partial':
        sys.stdout.write('{'); sys.stdout.flush(); time.sleep(30)
    if behavior == 'eof': break
    if behavior == 'oversized':
        sys.stdout.write('x' * 131073); sys.stdout.flush(); time.sleep(30)
    if behavior == 'invalid_json':
        print('{bad json}', flush=True); continue
    if behavior == 'duplicate_json':
        print('{"op":"result","op":"result"}', flush=True); continue
    if behavior == 'native_error':
        send(dict(op='error', nonce=a.nonce, request_id=q['request_id'], error_type='FloatingPointError')); break
    output = mode.get('output_ids', [12,248046])
    r = dict(op='result', nonce=a.nonce, request_id=q['request_id'],
        output_ids=output, new_tokens=len(output), input_tokens=len(q['input_ids']),
        input_ids_sha256=hashlib.sha256(json.dumps(q['input_ids'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
        max_new_tokens=q['max_new_tokens'], stopped_on_eos=output[-1] in (248044,248046),
        hit_output_limit=len(output)>=q['max_new_tokens'], seconds=0.03,
        ttft_seconds=0.02, decode_seconds=0.01, peak_memory_bytes=100, rss_peak_bytes=200)
    r.update(mode.get('result_override', {})); send(r)
'''


class MlxFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.worker = self.root / "worker.py"
        self.worker.write_text(FAKE_WORKER)
        self.ids = [100, 200, 300]
        self.tokenizer = SimpleNamespace(
            apply_chat_template=Mock(side_effect=lambda *a, **k: list(self.ids)),
            decode=Mock(return_value="  original tokenizer output  "))
        self.mode({})
        config = {"text_config": {"vocab_size": 248320, "eos_token_id": 248044},
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}
        (self.model / "config.json").write_text(json.dumps(config))
        (self.model / "model.safetensors").write_bytes(b"fake-weights")
        self.manifest = {"schema_version": 1, "kind": "medlens-mlx-model", "status": "complete",
            "model": {"model_id": MODEL_ID, "revision": REVISION},
            "quantization": {"method": "mlx-affine", "bits": 4, "group_size": 64,
                "coverage": "all-linear-and-embedding"}, "dtype": "float16", "fp32_exceptions": ["A_log"],
            "files": {name: {"bytes": (self.model / name).stat().st_size,
                "sha256": hashlib.sha256((self.model / name).read_bytes()).hexdigest()}
                for name in ("config.json", "model.safetensors")}}
        self.write_manifest()
        tokenizer_patch = patch.object(runtime_module, "_load_tokenizer", return_value=self.tokenizer)
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)
        worker_patch = patch.object(runtime_module, "WORKER_SCRIPT", self.worker)
        worker_patch.start()
        self.addCleanup(worker_patch.stop)

    def write_manifest(self):
        self.manifest["artifact_id"] = _digest({k: v for k, v in self.manifest.items() if k != "artifact_id"})
        (self.model / "medlens-source.json").write_text(json.dumps(self.manifest))

    def mode(self, value):
        (self.model / "mode.json").write_text(json.dumps(value))

    def runtime(self, **kwargs):
        result = MlxRuntime(self.model, sys.executable, root=self.root,
            stderr_path=self.root / "stderr.log", **kwargs)
        self.addCleanup(result.close)
        return result

    def prompt(self):
        messages = [{"role": "user", "content": "question"}]
        return {"messages": messages, "input_tokens": len(self.ids), "prompt_sha256": _digest(messages)}


class MlxProtocolTests(MlxFixture):
    def test_worker_isolated_mode_excludes_sibling_native_package_shadow(self):
        # A script-directory import path would execute this instead of native mlx.
        (self.root / "mlx.py").write_text("raise RuntimeError('sibling module was imported')")
        self.worker.write_text(FAKE_WORKER.replace("root = a.model_dir",
            "assert sys.flags.isolated == 1\nassert os.path.dirname(os.path.abspath(__file__)) not in sys.path\nroot = a.model_dir"))
        with patch.object(runtime_module.subprocess, "Popen", wraps=runtime_module.subprocess.Popen) as launch:
            with self.runtime() as runtime:
                runtime.generate(self.prompt())
            self.assertEqual(launch.call_args.args[0][1:3], ["-I", "-u"])

    def test_original_ids_and_tokenizer_are_used_without_prompt_cache(self):
        with self.runtime() as runtime:
            first = runtime.generate(self.prompt(), max_new_tokens=2)
            second = runtime.generate(self.prompt(), max_new_tokens=2)
            request = json.loads((self.model / "request.json").read_text())
            self.assertEqual(request["request_id"], "2")
            self.assertEqual(request["input_ids"], self.ids)
            self.assertNotIn("messages", request)
            self.assertEqual(request["eos_token_ids"], [248044, 248046])
            self.assertEqual(first["text"], "original tokenizer output")
            self.assertEqual(first["output_ids"], [12, 248046])
            self.assertEqual(first["new_tokens"], 2)
            self.assertTrue(first["hit_output_limit"])
            self.assertEqual(first["prompt_sha256"], self.prompt()["prompt_sha256"])
            self.assertEqual(first["input_ids_sha256"], _digest(self.ids))
            self.assertFalse(runtime.metadata["prompt_cache_reuse"])
            self.assertEqual(runtime.metadata["runtime"], "mlx-lm-subprocess")
            self.assertEqual(runtime.metadata["artifact_validation"],
                "manifest-identity-all-file-sizes-config-sha256")
            self.tokenizer.apply_chat_template.assert_called_with(self.prompt()["messages"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            self.tokenizer.decode.assert_called_with([12, 248046], skip_special_tokens=True)
            self.assertEqual(second["output_ids"], first["output_ids"])
        self.assertIsNotNone(runtime._process.poll())
        self.assertTrue(runtime._process.stdout.closed)
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.generate(self.prompt())

    def test_both_stop_ids_are_terminal_and_included(self):
        for eos in (248044, 248046):
            with self.subTest(eos=eos):
                self.mode({"output_ids": [eos]})
                with self.runtime() as runtime:
                    result = runtime.generate(self.prompt(), max_new_tokens=3)
                    self.assertEqual(result["new_tokens"], 1)
                    self.assertTrue(result["stopped_on_eos"])
                    self.assertFalse(result["hit_output_limit"])

    def test_missing_optional_prompt_hash_is_computed(self):
        prompt = self.prompt()
        expected = prompt.pop("prompt_sha256")
        self.assertEqual(self.runtime().generate(prompt)["prompt_sha256"], expected)

    def test_prompt_hash_count_and_input_id_validation_precede_ipc(self):
        runtime = self.runtime()
        for changes in ({"prompt_sha256": "wrong"}, {"input_tokens": 4}, {"input_tokens": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runtime.generate(self.prompt() | changes)
        for ids in ([True], [-1], [248320], [1] * 12001, []):
            self.ids = ids
            with self.subTest(ids_length=len(ids)), self.assertRaises(ValueError):
                runtime.generate(self.prompt())
        self.assertFalse((self.model / "request.json").exists())
        self.assertIsNone(runtime._process.poll())

    def test_output_caps_reject_booleans_and_out_of_range(self):
        runtime = self.runtime()
        for cap in (True, 0, -1, 1025, 1.2):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                runtime.generate(self.prompt(), max_new_tokens=cap)
        self.assertFalse((self.model / "request.json").exists())

    def test_maximum_input_can_cross_pipe_buffer_without_blocking(self):
        self.ids = [248319] * 12000
        with self.runtime(timeout=2) as runtime:
            result = runtime.generate(self.prompt())
        self.assertEqual(result["input_tokens"], 12000)

    def test_invalid_protocol_closes_and_reaps_worker(self):
        cases = [
            {"result_override": {"nonce": "wrong"}},
            {"result_override": {"request_id": "wrong"}},
            {"result_override": {"output_ids": [True, 248046]}},
            {"result_override": {"output_ids": [-1, 248046]}},
            {"result_override": {"output_ids": [248320, 248046]}},
            {"result_override": {"new_tokens": True}},
            {"result_override": {"new_tokens": 1}},
            {"result_override": {"max_new_tokens": 3}},
            {"result_override": {"input_tokens": 99}},
            {"result_override": {"input_ids_sha256": "wrong"}},
            {"result_override": {"stopped_on_eos": False}},
            {"result_override": {"hit_output_limit": False}},
            {"result_override": {"seconds": float("nan")}},
            {"result_override": {"ttft_seconds": 1}},
            {"result_override": {"peak_memory_bytes": -1}},
            {"output_ids": [248046, 12]},
            {"output_ids": [12]},
            {"output_ids": [12, 13, 248046]},
            {"behavior": "invalid_json"}, {"behavior": "duplicate_json"},
            {"behavior": "native_error"}, {"behavior": "eof"}, {"behavior": "oversized"},
        ]
        for mode in cases:
            with self.subTest(mode=mode):
                self.mode(mode)
                runtime = self.runtime(timeout=2)
                with self.assertRaises((RuntimeError, ValueError)):
                    runtime.generate(self.prompt(), max_new_tokens=2)
                self.assertIsNotNone(runtime._process.poll())
                self.assertTrue(runtime._process.stdout.closed)

    def test_partial_line_does_not_defeat_timeout_and_worker_is_reaped(self):
        for behavior in ("timeout", "partial"):
            with self.subTest(behavior=behavior):
                self.mode({"behavior": behavior})
                runtime = self.runtime(timeout=0.1)
                with self.assertRaises(TimeoutError):
                    runtime.generate(self.prompt())
                self.assertIsNotNone(runtime._process.poll())
                self.assertTrue(runtime._process.stdin.closed)

    def test_constructor_handshake_failure_and_timeout_reap_worker(self):
        for mode in ({"ready_override": {"nonce": "wrong"}},
                     {"ready_override": {"packages": {"mlx": "0.0"}}},
                     {"behavior": "load_timeout"}):
            with self.subTest(mode=mode):
                self.mode(mode)
                with self.assertRaises((RuntimeError, TimeoutError)):
                    self.runtime(load_timeout=0.15)
                pid = int((self.model / "worker.pid").read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)


class ManifestTests(MlxFixture):
    def test_identity_and_model_quantization_are_pinned(self):
        cases = [("status", "partial"), ("model", {"model_id": MODEL_ID, "revision": "changed"}),
            ("dtype", "bfloat16"), ("fp32_exceptions", []),
            ("quantization", {"method": "mlx-affine", "bits": 8, "group_size": 64,
                "coverage": "all-linear-and-embedding"})]
        original = dict(self.manifest)
        for key, value in cases:
            self.manifest = original | {key: value}
            self.write_manifest()
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.runtime()
        self.assertFalse((self.model / "worker.pid").exists())

    def test_manifest_hash_config_hash_and_weight_size_are_verified(self):
        (self.model / "config.json").write_bytes(b"x" * self.manifest["files"]["config.json"]["bytes"])
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.runtime()
        (self.model / "model.safetensors").write_bytes(b"x")
        self.manifest["files"].pop("config.json")
        self.write_manifest()
        with self.assertRaises(RuntimeError):
            self.runtime()

    def test_weight_hashes_are_optional_and_scope_is_reported(self):
        (self.model / "model.safetensors").write_bytes(b"bad--weights")
        with self.runtime() as runtime:
            self.assertIn("sizes", runtime.metadata["artifact_validation"])
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.runtime(verify_weights=True)

    def test_path_traversal_symlinks_and_unmanifested_weights_are_rejected(self):
        self.manifest["files"]["../elsewhere"] = {"bytes": 0, "sha256": "0" * 64}
        self.write_manifest()
        with self.assertRaisesRegex(RuntimeError, "file entry"):
            self.runtime()
        del self.manifest["files"]["../elsewhere"]
        self.write_manifest()
        path = self.model / "model.safetensors"
        target = self.root / "weights"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, "type changed"):
            self.runtime()
        path.unlink()
        target.rename(path)
        (self.model / "model-foreign.safetensors").write_bytes(b"foreign")
        with self.assertRaisesRegex(RuntimeError, "Unmanifested"):
            self.runtime()

    def test_manifest_artifact_id_cannot_be_stale(self):
        value = json.loads((self.model / "medlens-source.json").read_text())
        value["extra"] = "changed"
        (self.model / "medlens-source.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "artifact identity"):
            self.runtime()


class NativeWorkerContractTests(unittest.TestCase):
    def test_parameter_dtype_contract_checks_exception_and_packed_weights(self):
        parameters = [("layer.A_log", np.zeros(1, dtype=np.float32)),
            ("layer.scales", np.zeros(1, dtype=np.float16)),
            ("layer.weight", np.zeros(1, dtype=np.uint32))]
        self.assertEqual(_parameter_dtypes(np, parameters), ["float16", "float32", "uint32"])
        for changed in ([*parameters, ("other", np.zeros(1, dtype=np.float32))],
                        [("layer.A_log", np.zeros(1, dtype=np.float16)), *parameters[1:]],
                        parameters[1:]):
            with self.assertRaises(RuntimeError):
                _parameter_dtypes(np, changed)

    def test_score_guard_preserves_usable_logits_and_negative_infinity_masks(self):
        for scores in (np.array([[1., 2., -3.]]), np.array([[-np.inf, 2., 1.]])):
            self.assertIs(require_finite_scores(np, scores), scores)

    def test_score_guard_rejects_nan_posinf_and_any_fully_masked_row(self):
        for row in ([np.nan, 0.], [np.inf, 0.], [-np.inf, -np.inf]):
            with self.subTest(row=row), self.assertRaises(FloatingPointError):
                require_finite_scores(np, np.array([[1., 2.], row]))

    def test_worker_request_ids_caps_eos_and_token_ids_are_strict(self):
        request = {"op": "generate", "nonce": "session", "request_id": "1",
            "input_ids": [1, 2], "max_new_tokens": 2, "eos_token_ids": [248044, 248046]}
        self.assertIs(_validate_request(request, "session"), request)
        for update in ({"nonce": "other"}, {"request_id": True}, {"max_new_tokens": True},
                       {"input_ids": [True]}, {"input_ids": [248320]},
                       {"input_ids": [1] * 12001}, {"eos_token_ids": [248044]}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                _validate_request(request | update, "session")

    def test_generation_owns_fresh_cache_and_closes_on_eos_and_native_error(self):
        for fail in (False, True):
            closed = []
            mx = SimpleNamespace(array=np.array, int32=np.int32, synchronize=Mock(),
                reset_peak_memory=Mock(), clear_cache=Mock(), get_peak_memory=lambda: 123)
            def step(ids, model, **kwargs):
                self.assertIsNone(kwargs["prompt_cache"])
                self.assertIsNone(kwargs["kv_bits"])
                self.assertEqual(kwargs["prefill_step_size"], 2048)
                self.assertNotIn("sampler", kwargs)
                self.assertEqual(kwargs["max_tokens"], 2)
                try:
                    if fail:
                        raise FloatingPointError("native error")
                    yield 248046, None
                    self.fail("Worker generated past EOS")
                finally:
                    closed.append(True)
            request = {"input_ids": [1, 2], "max_new_tokens": 2, "nonce": "session", "request_id": "1"}
            with self.subTest(fail=fail):
                if fail:
                    with self.assertRaises(FloatingPointError):
                        _generate(mx, step, object(), request)
                else:
                    result = _generate(mx, step, object(), request)
                    self.assertEqual(result["output_ids"], [248046])
                    self.assertFalse(result["hit_output_limit"])
                self.assertEqual(closed, [True])
                mx.clear_cache.assert_called_once()
                self.assertEqual(mx.synchronize.call_count, 2)

    def test_sync_failure_still_releases_unused_native_cache(self):
        mx = SimpleNamespace(array=np.array, int32=np.int32,
            synchronize=Mock(side_effect=[None, RuntimeError("device error")]),
            reset_peak_memory=Mock(), clear_cache=Mock())
        request = {"input_ids": [1], "max_new_tokens": 1, "nonce": "session", "request_id": "1"}
        with self.assertRaisesRegex(RuntimeError, "device error"):
            _generate(mx, lambda *a, **k: (pair for pair in [(248046, None)]), object(), request)
        mx.clear_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
