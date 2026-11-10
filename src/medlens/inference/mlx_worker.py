"""Private, bounded JSON-lines worker; native MLX imports stay in main()."""

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

PROTOCOL = 2
MAX_LINE_BYTES = 131072
MAX_INPUT_TOKENS = 12000
VOCAB_SIZE = 248320
EOS_IDS = (248044, 248046)
PACKAGES = {name: "0.32.2" for name in ("mlx", "mlx-lm", "mlx-metal")}
PACKAGES["outlines-core"] = "0.2.14"


def _sibling(name):
    """Load only an explicit project file; never add this directory to sys.path."""
    spec = importlib.util.spec_from_file_location("_medlens_worker_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_finite_scores(mx, logits):
    """Reject invalid raw scores before sampling; preserve allowed -inf masks."""
    invalid = (mx.any(mx.isnan(logits))
        | mx.any(mx.isinf(logits) & (logits > 0))
        | mx.any(mx.all(mx.isinf(logits) & (logits < 0), axis=-1)))
    if invalid.item():
        raise FloatingPointError("Nonfinite or fully masked native generation scores")
    return logits


def _parameter_dtypes(mx, parameters, dtype="float16"):
    floating = getattr(mx, dtype)
    observed = set()
    found_exception = False
    for name, value in parameters:
        if name.rsplit(".", 1)[-1] == "A_log":
            if value.dtype != mx.float32:
                raise RuntimeError("Native A_log must preserve float32")
            found_exception = True
            observed.add("float32")
        elif value.dtype == floating:
            observed.add(dtype)
        elif value.dtype == mx.uint32:
            observed.add("uint32")
        else:
            raise RuntimeError("Native parameter dtype differs from artifact contract")
    if not found_exception or observed != {dtype, "float32", "uint32"}:
        raise RuntimeError("Native parameter dtype coverage differs from artifact contract")
    return sorted(observed)


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate protocol key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Nonfinite protocol value")
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("Protocol object required")
    return value


def _validate_request(value, nonce):
    if value.get("nonce") != nonce:
        raise ValueError("Session mismatch")
    if value.get("op") == "close":
        if set(value) != {"op", "nonce"}:
            raise ValueError("Unexpected close fields")
        return value
    required = {"op", "nonce", "request_id", "input_ids", "max_new_tokens", "eos_token_ids"}
    optional = {"decoding_schema", "constraint_kind"}
    if (not required.issubset(value) or set(value) - required - optional
            or value.get("op") != "generate"
            or not isinstance(value.get("request_id"), str)
            or not value["request_id"].isascii() or not value["request_id"].isdecimal()
            or len(value["request_id"]) > 20
            or type(value.get("max_new_tokens")) is not int
            or not 1 <= value["max_new_tokens"] <= 1024
            or value.get("eos_token_ids") != list(EOS_IDS)):
        raise ValueError("Invalid generation request contract")
    ids = value.get("input_ids")
    if (not isinstance(ids, list) or not 1 <= len(ids) <= MAX_INPUT_TOKENS
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in ids)):
        raise ValueError("Invalid input token IDs or budget")
    if bool(optional.intersection(value)):
        if (not optional.issubset(value) or not isinstance(value.get("decoding_schema"), dict)
                or value.get("constraint_kind") not in {"draft", "review", "literal"}
                or len(json.dumps(value["decoding_schema"], ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False).encode()) > 32768):
            raise ValueError("Invalid or oversized decoding schema")
    return value


def _rss_peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _generate(mx, generate_step, model, request, *, constraint_cache=None):
    constraint = processor = None
    if "decoding_schema" in request:
        if constraint_cache is None:
            raise ValueError("Structured decoding requires a compiled constraint cache")
        compiled, constraint = constraint_cache.get(request["decoding_schema"], request["constraint_kind"])
        processor = compiled.processor(mx)
    mx.synchronize()
    mx.reset_peak_memory()
    started = time.perf_counter()
    first_token = None
    output = []
    generator = None
    try:
        generator = generate_step(mx.array(request["input_ids"], dtype=mx.int32), model,
            max_tokens=request["max_new_tokens"],
            logits_processors=[processor or (lambda _, logits: require_finite_scores(mx, logits))],
            prompt_cache=None, kv_bits=None, prefill_step_size=2048)
        for token, logprobs in generator:
            del logprobs
            if type(token) is not int or not 0 <= token < VOCAB_SIZE:
                raise ValueError("Invalid native output token")
            if first_token is None:
                first_token = time.perf_counter() - started
            output.append(token)
            if token in EOS_IDS or len(output) >= request["max_new_tokens"]:
                break
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            try:
                mx.synchronize()
            finally:
                generator = None
                gc.collect()
                mx.clear_cache()
    elapsed = time.perf_counter() - started
    if not output or first_token is None:
        raise RuntimeError("Native model produced no output")
    stats = processor.finish(output, hit_output_limit=len(output) >= request["max_new_tokens"]) if processor else None
    metrics = {"seconds": elapsed, "ttft_seconds": first_token,
        "decode_seconds": elapsed - first_token,
        "peak_memory_bytes": int(mx.get_peak_memory()), "rss_peak_bytes": _rss_peak_bytes()}
    if any(not math.isfinite(value) or value < 0 for value in metrics.values()):
        raise RuntimeError("Invalid native generation metrics")
    return {"op": "result", "nonce": request["nonce"], "request_id": request["request_id"],
        "output_ids": output, "new_tokens": len(output),
        "input_tokens": len(request["input_ids"]),
        "input_ids_sha256": hashlib.sha256(json.dumps(request["input_ids"],
            sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "max_new_tokens": request["max_new_tokens"],
        "stopped_on_eos": output[-1] in EOS_IDS,
        "hit_output_limit": len(output) >= request["max_new_tokens"],
        "constraint": constraint, "constraint_stats": stats, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--model", choices=("4b", "9b"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    args = parser.parse_args()
    # Keep a private protocol descriptor; redirect both Python and native import
    # diagnostics away from the protocol pipe before loading native libraries.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    def send(value):
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(payload) > MAX_LINE_BYTES:
            raise RuntimeError("Native response exceeds protocol budget")
        remaining = memoryview(payload)
        while remaining:
            written = protocol.write(remaining)
            if not written:
                raise BrokenPipeError("Native protocol pipe closed")
            remaining = remaining[written:]

    request_id = None
    try:
        started = time.perf_counter()
        versions = {name: importlib.metadata.version(name) for name in PACKAGES}
        if versions != PACKAGES:
            raise RuntimeError("Native package versions differ from pinned runtime")
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.generate import generate_step
        from mlx_lm.utils import load_model
        profiles, structured = _sibling("profiles"), _sibling("structured")
        profile = profiles.get_profile(args.model)
        manifest = _strict_json((args.model_dir / "medlens-source.json").read_bytes())
        if (manifest.get("artifact_id") != args.artifact_id
                or manifest["artifact_id"] != _digest({key: value for key, value in manifest.items() if key != "artifact_id"})
                or manifest.get("model") != {key: profile[key] for key in ("model_id", "revision")}):
            raise RuntimeError("Native model manifest changed before loading")
        config_body = (args.model_dir / "config.json").read_bytes()
        expected_config = manifest["files"]["config.json"]
        if (len(config_body) != expected_config["bytes"]
                or hashlib.sha256(config_body).hexdigest() != expected_config["sha256"]):
            raise RuntimeError("Native model config changed before loading")
        tokenizer = profiles.load_answer_tokenizer(args.root, args.model)
        tokenizer_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        runtime_identity = {"model_id": profile["model_id"], "revision": profile["revision"],
            "artifact_id": args.artifact_id, "answer_policy": profile["answer_policy"]}
        constraints = structured.ConstraintCache(tokenizer, vocab_size=VOCAB_SIZE, eos=EOS_IDS)
        if not mx.metal.is_available():
            raise RuntimeError("Native Metal device is required")
        mx.set_default_device(mx.gpu)
        model, config = load_model(args.model_dir, lazy=False, strict=True)
        text = config.get("text_config", config)
        if (text.get("vocab_size") != VOCAB_SIZE or text.get("eos_token_id") != EOS_IDS[0]
                or text.get("hidden_size") != profile["hidden_size"]
                or text.get("dtype") != profile["dtype"]
                or config.get("tie_word_embeddings", text.get("tie_word_embeddings")) is not profile["tie_word_embeddings"]):
            raise RuntimeError("Native model vocabulary or stop config changed")
        mx.synchronize()
        send({"op": "ready", "protocol": PROTOCOL, "nonce": args.nonce,
            "packages": versions, "vocab_size": VOCAB_SIZE, "eos_token_ids": list(EOS_IDS),
            "load_seconds": time.perf_counter() - started,
            "model_profile": args.model, "runtime_identity": runtime_identity,
            "tokenizer_backend_sha256": tokenizer_sha,
            "parameter_dtypes": _parameter_dtypes(mx, tree_flatten(model.parameters()), profile["dtype"]),
            "parameter_dtype_policy": profile["dtype"] + "-except-A_log-float32-packed-uint32"})
        last_request_number = 0
        while True:
            line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
            if not line:
                return 0
            if len(line) > MAX_LINE_BYTES or not line.endswith(b"\n"):
                raise ValueError("Oversized or incomplete request")
            request = _validate_request(_strict_json(line), args.nonce)
            if request["op"] == "close":
                send({"op": "closed", "nonce": args.nonce})
                return 0
            request_id = request["request_id"]
            request_number = int(request_id)
            if request_number <= last_request_number:
                raise ValueError("Repeated request identity")
            last_request_number = request_number
            result = _generate(mx, generate_step, model, request, constraint_cache=constraints)
            result.update(runtime_identity=runtime_identity, tokenizer_backend_sha256=tokenizer_sha)
            send(result)
    except Exception as exc:
        # Only the error class crosses the boundary; never echo prompt contents.
        try:
            send({"op": "error", "nonce": args.nonce, "request_id": request_id,
                "error_type": type(exc).__name__})
        except (OSError, ValueError, RuntimeError):
            pass
        print(f"MLX worker stopped: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        protocol.close()


if __name__ == "__main__":
    raise SystemExit(main())
