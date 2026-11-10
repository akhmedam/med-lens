"""Bounded MLX subprocess runtime using the original pinned HF tokenizer.

The parent process never imports MLX. A dedicated interpreter owns the native
model and weights; each request creates and releases its own unquantized KV cache.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import signal
import stat
import subprocess
import tempfile
import threading
import time

from .qwen import MODEL_ID, REVISION
from .profiles import DEFAULT_MODEL, get_profile, load_answer_tokenizer
from .structured import prepare_schema

PROTOCOL = 2
MAX_LINE_BYTES = 131072
MAX_INPUT_TOKENS = 12000
VOCAB_SIZE = 248320
EOS_IDS = (248044, 248046)
PACKAGES = {name: "0.32.2" for name in ("mlx", "mlx-lm", "mlx-metal")}
PACKAGES["outlines-core"] = "0.2.14"
WORKER_SCRIPT = Path(__file__).with_name("mlx_worker.py")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite JSON value")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise RuntimeError("Invalid MLX JSON protocol or manifest") from exc


def _file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _validate_manifest(model_dir, *, verify_weights, model="9b"):
    profile = get_profile(model)
    manifest_path = model_dir / "medlens-source.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("Missing regular MLX source manifest")
    manifest = _strict_json(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise RuntimeError("Invalid MLX manifest")
    identity = {key: value for key, value in manifest.items() if key != "artifact_id"}
    quantization = manifest.get("quantization")
    if (manifest.get("schema_version") != 1 or type(manifest.get("schema_version")) is not int
            or manifest.get("kind") != "medlens-mlx-model" or manifest.get("status") != "complete"
            or manifest.get("artifact_id") != _digest(identity)
            or manifest.get("model") != {key: profile[key] for key in ("model_id", "revision")}
            or not isinstance(quantization, dict)
            or quantization.get("method") != "mlx-affine"
            or type(quantization.get("bits")) is not int or quantization["bits"] != 4
            or type(quantization.get("group_size")) is not int or quantization["group_size"] != 64
            or quantization != profile["quantization"]
            or manifest.get("dtype") != profile["dtype"]
            or manifest.get("fp32_exceptions") != profile["fp32_exceptions"]
            or (model == "4b" and manifest.get("tie_word_embeddings") is not True)):
        raise RuntimeError("Unrecognized or altered MLX artifact identity")
    files = manifest.get("files")
    if (not isinstance(files, dict) or "config.json" not in files
            or not any(name.startswith("model") and name.endswith(".safetensors") for name in files)):
        raise RuntimeError("MLX manifest does not declare config and weights")
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in {".", ".."} or "\\" in name or not isinstance(expected, dict)
                or type(expected.get("bytes")) is not int or expected["bytes"] < 0
                or not isinstance(expected.get("sha256"), str)
                or len(expected["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in expected["sha256"])):
            raise RuntimeError("Invalid MLX manifest file entry")
        path = model_dir / name
        try:
            info = path.lstat()
        except OSError as exc:
            raise RuntimeError("Missing MLX artifact file") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected["bytes"]:
            raise RuntimeError("MLX artifact file size or type changed")
        if (verify_weights or name == "config.json") and _file_hash(path) != expected["sha256"]:
            raise RuntimeError("MLX artifact checksum mismatch")
    # Prevent load_model's glob from discovering unmanifested additional weights.
    actual_weights = {path.name for path in model_dir.glob("model*.safetensors")}
    expected_weights = {name for name in files if name.startswith("model") and name.endswith(".safetensors")}
    if actual_weights != expected_weights:
        raise RuntimeError("Unmanifested MLX weight files")
    config = _strict_json((model_dir / "config.json").read_bytes())
    if not isinstance(config, dict) or "model_file" in config:
        raise RuntimeError("Invalid MLX model config")
    text = config.get("text_config", config)
    quant = config.get("quantization", config.get("quantization_config"))
    if (not isinstance(text, dict) or text.get("vocab_size") != VOCAB_SIZE
            or type(text.get("vocab_size")) is not int
            or text.get("eos_token_id") != EOS_IDS[0]
            or text.get("hidden_size") != profile["hidden_size"]
            or text.get("dtype") != profile["dtype"]
            or config.get("tie_word_embeddings", text.get("tie_word_embeddings")) is not profile["tie_word_embeddings"]
            or not isinstance(quant, dict)
            or quant.get("bits") != 4 or quant.get("group_size") != 64
            or quant.get("mode", "affine") != "affine"):
        raise RuntimeError("MLX model config differs from pinned generation contract")
    return manifest


def _load_tokenizer(root, model=DEFAULT_MODEL):
    return load_answer_tokenizer(root, model)


def _ids(value, *, maximum, label):
    if (not isinstance(value, list) or not 1 <= len(value) <= maximum
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in value)):
        raise ValueError(f"Invalid {label} token IDs or token budget")
    return value


def _seconds(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


class MlxRuntime:
    def __init__(self, model_dir=None, worker_python=None, *, root=None, timeout=180.0,
                 load_timeout=180.0, stderr_path=None, verify_weights=False, model=DEFAULT_MODEL):
        self._lock = threading.RLock()
        self._process = None
        self._stderr = None
        self._closed = False
        self._buffer = bytearray()
        self._nonce = secrets.token_hex(24)
        self._request_number = 0
        if not _seconds(timeout) or timeout == 0 or not _seconds(load_timeout) or load_timeout == 0:
            raise ValueError("MLX timeouts must be positive finite seconds")
        if type(verify_weights) is not bool:
            raise ValueError("verify_weights must be boolean")
        self.timeout = timeout
        profile = get_profile(model)
        self.model_profile = model
        self.answer_policy = profile["answer_policy"]
        root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
        model_dir = Path(model_dir) if model_dir else root / profile["directory"]
        worker_python = Path(worker_python) if worker_python else root / ".cache/venvs/mlx/bin/python"
        manifest = _validate_manifest(model_dir, verify_weights=verify_weights, model=model)
        self.tokenizer = _load_tokenizer(root, model)
        self._tokenizer_sha256 = hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        self.cache_identity = {"model_id": profile["model_id"], "revision": profile["revision"],
            "artifact_id": manifest["artifact_id"], "answer_policy": self.answer_policy}
        if stderr_path is None:
            descriptor, name = tempfile.mkstemp(prefix="medlens-mlx-", suffix=".stderr.log")
            self.stderr_path = Path(name)
            self._stderr = os.fdopen(descriptor, "wb")
        else:
            self.stderr_path = Path(stderr_path)
            self._stderr = self.stderr_path.open("wb")
        try:
            started = time.perf_counter()
            # Isolated mode excludes this script directory: sibling mlx.py must
            # never shadow the native third-party mlx package inside the worker.
            self._process = subprocess.Popen([str(worker_python), "-I", "-u", str(WORKER_SCRIPT),
                "--model-dir", str(model_dir.resolve()), "--nonce", self._nonce,
                "--root", str(root), "--model", model, "--artifact-id", manifest["artifact_id"]],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                bufsize=0, start_new_session=True)
            os.set_blocking(self._process.stdout.fileno(), False)
            os.set_blocking(self._process.stdin.fileno(), False)
            ready = self._read(time.monotonic() + load_timeout)
            if (not isinstance(ready, dict) or ready.get("op") != "ready"
                    or ready.get("protocol") != PROTOCOL or type(ready.get("protocol")) is not int
                    or ready.get("nonce") != self._nonce or ready.get("packages") != PACKAGES
                    or ready.get("vocab_size") != VOCAB_SIZE
                    or ready.get("eos_token_ids") != list(EOS_IDS)
                    or ready.get("model_profile") != model or ready.get("runtime_identity") != self.cache_identity
                    or ready.get("tokenizer_backend_sha256") != self._tokenizer_sha256
                    or not _seconds(ready.get("load_seconds"))
                    or not isinstance(ready.get("parameter_dtypes"), list)
                    or any(not isinstance(value, str) for value in ready["parameter_dtypes"])
                    or set(ready["parameter_dtypes"]) != {profile["dtype"], "float32", "uint32"}
                    or ready.get("parameter_dtype_policy") != profile["dtype"] + "-except-A_log-float32-packed-uint32"):
                raise RuntimeError("Invalid MLX worker readiness handshake")
            self.metadata = {"model_id": profile["model_id"], "revision": profile["revision"],
                "model_profile": model, "answer_policy": self.answer_policy,
                "runtime": "mlx-lm-subprocess", "device": "metal",
                "artifact_id": manifest["artifact_id"], "model_dir": str(model_dir.resolve()),
                "quantization": manifest["quantization"], "dtype": manifest["dtype"],
                "fp32_exceptions": manifest["fp32_exceptions"],
                "parameter_dtypes": ready["parameter_dtypes"],
                "parameter_dtype_policy": ready["parameter_dtype_policy"], "packages": ready["packages"],
                "load_seconds": ready["load_seconds"],
                "worker_startup_seconds": time.perf_counter() - started,
                "generation": {"do_sample": False, "max_new_tokens": 128,
                    "enable_thinking": False, "input_token_limit": MAX_INPUT_TOKENS,
                    "eos_token_ids": list(EOS_IDS), "stop_policy": "source-eos-and-chat-end-union",
                    "include_terminal_eos_in_token_count": True},
                "score_validation": "reject-nan-positive-infinity-and-fully-masked-rows",
                "artifact_validation": "all-file-sha256" if verify_weights else
                    "manifest-identity-all-file-sizes-config-sha256",
                "tokenizer": getattr(self.tokenizer, "_medlens_tokenizer_source", profile["tokenizer"]),
                "tokenizer_backend_sha256": self._tokenizer_sha256,
                "structured_decoding": {"library": "outlines-core", "version": "0.2.14",
                    "compiled_index_cache_maximum": 8, "fresh_guide_per_call": True,
                    "raw_and_post_mask_score_validation": True},
                "prompt_cache_reuse": False, "kv_quantization": None, "prefill_step_size": 2048,
                "release_cache_after_generation": True,
                "ttft_scope": "first-generate-step-yield-including-native-one-step-lookahead",
                "stderr_path": str(self.stderr_path)}
        except BaseException:
            self._shutdown()
            raise

    def _read(self, deadline):
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > MAX_LINE_BYTES:
                    raise RuntimeError("Oversized MLX protocol message")
                line = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return _strict_json(line)
            if len(self._buffer) >= MAX_LINE_BYTES:
                raise RuntimeError("Oversized MLX protocol message")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MLX worker response timed out")
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                if not selector.select(remaining):
                    raise TimeoutError("MLX worker response timed out")
            chunk = os.read(self._process.stdout.fileno(), min(65536, MAX_LINE_BYTES - len(self._buffer)))
            if not chunk:
                raise RuntimeError("MLX worker exited without a complete response")
            self._buffer.extend(chunk)

    def _write(self, value, deadline):
        payload = (_canonical(value) + "\n").encode()
        if len(payload) > MAX_LINE_BYTES:
            raise RuntimeError("Oversized MLX request")
        remaining = memoryview(payload)
        while remaining:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise TimeoutError("MLX worker request timed out")
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdin, selectors.EVENT_WRITE)
                if not selector.select(wait):
                    raise TimeoutError("MLX worker request timed out")
            try:
                count = os.write(self._process.stdin.fileno(), remaining)
            except BlockingIOError:
                continue
            if count <= 0:
                raise RuntimeError("MLX worker input closed")
            remaining = remaining[count:]

    def generate(self, prompt, *, max_new_tokens=128):
        with self._lock:
            if self._closed:
                raise RuntimeError("MLX runtime is closed")
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
                raise ValueError("max_new_tokens must be an integer from 1 to 1024")
            if not isinstance(prompt, dict) or not isinstance(prompt.get("messages"), list):
                raise ValueError("Prompt messages are required")
            prompt_hash = _digest(prompt["messages"])
            if prompt.get("prompt_sha256", prompt_hash) != prompt_hash:
                raise ValueError("Prompt hash changed")
            ids = _ids(self.tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False),
                maximum=MAX_INPUT_TOKENS, label="input")
            if type(prompt.get("input_tokens")) is not int or len(ids) != prompt["input_tokens"]:
                raise ValueError("Prompt token count changed")
            schema = prompt.get("decoding_schema")
            kind = prompt.get("constraint_kind")
            constraint_expected = None
            if (schema is None) != (kind is None):
                raise ValueError("Supply both decoding_schema and constraint_kind")
            if schema is not None:
                _, constraint_expected = prepare_schema(schema, kind, validate_regex=False)
            request = {"op": "generate", "nonce": self._nonce,
                "input_ids": ids, "max_new_tokens": max_new_tokens, "eos_token_ids": list(EOS_IDS)}
            if schema is not None:
                request.update(decoding_schema=schema, constraint_kind=kind)
            self._request_number += 1
            request_id = str(self._request_number)
            request["request_id"] = request_id
            started = time.perf_counter()
            deadline = time.monotonic() + self.timeout
            try:
                self._write(request, deadline)
                result = self._read(deadline)
                if (not isinstance(result, dict) or result.get("op") != "result"
                        or result.get("nonce") != self._nonce or result.get("request_id") != request_id):
                    raise RuntimeError("Invalid MLX response identity or native generation failure")
                if (result.get("runtime_identity") != self.cache_identity
                        or result.get("tokenizer_backend_sha256") != self._tokenizer_sha256):
                    raise RuntimeError("MLX model or tokenizer response identity changed")
                try:
                    output = _ids(result.get("output_ids"), maximum=max_new_tokens, label="output")
                except ValueError as exc:
                    raise RuntimeError("Invalid MLX output token IDs") from exc
                stopped = output[-1] in EOS_IDS
                if (type(result.get("new_tokens")) is not int or result["new_tokens"] != len(output)
                        or type(result.get("input_tokens")) is not int or result["input_tokens"] != len(ids)
                        or result.get("input_ids_sha256") != _digest(ids)
                        or type(result.get("max_new_tokens")) is not int
                        or result["max_new_tokens"] != max_new_tokens
                        or type(result.get("stopped_on_eos")) is not bool
                        or result["stopped_on_eos"] != stopped
                        or type(result.get("hit_output_limit")) is not bool
                        or result["hit_output_limit"] != (len(output) >= max_new_tokens)
                        or any(token in EOS_IDS for token in output[:-1])
                        or (len(output) < max_new_tokens and not stopped)):
                    raise RuntimeError("Invalid MLX output stop or token-count contract")
                for name in ("seconds", "ttft_seconds", "decode_seconds", "peak_memory_bytes", "rss_peak_bytes"):
                    if not _seconds(result.get(name)):
                        raise RuntimeError("Invalid MLX generation metrics")
                if result["ttft_seconds"] > result["seconds"] or result["decode_seconds"] > result["seconds"]:
                    raise RuntimeError("Inconsistent MLX generation timing")
                constraint = result.get("constraint")
                stats = result.get("constraint_stats")
                if constraint_expected is None:
                    if constraint is not None or stats is not None:
                        raise RuntimeError("Unexpected decoding constraint response")
                else:
                    if (not isinstance(constraint, dict) or not isinstance(stats, dict)
                            or any(constraint.get(key) != constraint_expected[key] for key in
                                ("original_schema_sha256", "lowered_schema_sha256", "kind", "decode_relaxations"))
                            or type(constraint.get("cache_hit")) is not bool
                            or not _seconds(constraint.get("compile_seconds"))
                            or type(constraint.get("compiled_cache_size")) is not int
                            or not 1 <= constraint["compiled_cache_size"] <= 8
                            or constraint.get("compiled_cache_maximum") != 8
                            or not isinstance(constraint.get("regex_sha256"), str)
                            or len(constraint["regex_sha256"]) != 64
                            or constraint.get("whitespace_pattern") != ""
                            or constraint.get("object_property_order") != "required-list"
                            or stats.get("accepting") is not True or stats.get("eos_accepted") is not True
                            or stats.get("fresh_guide") is not True
                            or type(stats.get("accepted_generated_tokens")) is not int
                            or stats["accepted_generated_tokens"] != len(output)
                            or result["hit_output_limit"]):
                        raise RuntimeError("Invalid constrained-generation identity or terminal state")
                    for name in ("mask_seconds", "advance_seconds", "raw_validation_seconds", "post_mask_validation_seconds"):
                        if not _seconds(stats.get(name)):
                            raise RuntimeError("Invalid constrained-generation timing")
                wall = time.perf_counter() - started
                text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                return {key: value for key, value in result.items() if key not in {"op", "nonce", "request_id"}} | {
                    "text": text, "prompt_sha256": prompt_hash, "worker_wall_seconds": wall}
            except BaseException:
                self._shutdown()
                raise

    def _shutdown(self):
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            else:
                process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        if self._stderr is not None:
            self._stderr.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                deadline = time.monotonic() + 0.3
                self._write({"op": "close", "nonce": self._nonce}, deadline)
                result = self._read(deadline)
                if result != {"op": "closed", "nonce": self._nonce}:
                    raise RuntimeError("Invalid MLX close acknowledgment")
                self._process.wait(timeout=0.3)
            except (OSError, RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                pass
            finally:
                self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
