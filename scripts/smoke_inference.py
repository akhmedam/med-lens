"""Check Qwen/Transformers/MPS compatibility; not a quality benchmark."""

from __future__ import annotations

import argparse
import importlib.metadata
import hashlib
import json
import os
import platform
import resource
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def trim_user_agent(request) -> None:
    # kernels 0.17.2 + hub 2.2.0 can emit a trailing space when telemetry is off.
    # Keep the default client's TLS, auth, and publisher checks unchanged.
    value = request.headers.get("user-agent")
    if value is not None:
        request.headers["user-agent"] = value.rstrip()


def valid_permutation(text: str, expected: list[str]) -> bool:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return False
    return (
        isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        and len(value) == len(expected)
        and set(value) == set(expected)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Download weights and run inference")
    parser.add_argument("--revision", default="main", help="Resolved to a commit before loading")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new report path")
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "inference" if args.run else "preflight",
        "model_id": "Qwen/Qwen3.5-9B",
        "requested_revision": args.revision,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "quantization": {"method": "metal", "bits": 4, "group_size": 64},
        "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False},
        "status": "running",
        "stage": "imports",
        "packages": {},
        "cases": [],
    }
    stop = threading.Event()
    sampler = None
    peaks = {}
    started = time.perf_counter()
    try:
        for name in ("torch", "transformers", "accelerate", "kernels", "psutil", "huggingface-hub"):
            report["packages"][name] = importlib.metadata.version(name)
        import psutil
        import torch
        from transformers import AutoTokenizer, MetalConfig, Qwen3_5ForCausalLM

        report["mps_built"] = torch.backends.mps.is_built()
        report["mps_available"] = torch.backends.mps.is_available()
        report["mps_cpu_fallback_env"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")
        report["memory_total_bytes"] = psutil.virtual_memory().total
        report["memory_available_before_bytes"] = psutil.virtual_memory().available
        report["stage"] = "mps_preflight"
        if not report["mps_available"]:
            raise RuntimeError("MPS unavailable; target device was not tested")
        if report["mps_cpu_fallback_env"] == "1":
            raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to test without implicit CPU fallback")
        x = torch.ones(8, device="mps")
        if (x + x).sum().item() != 16:
            raise RuntimeError("MPS arithmetic check failed")
        del x
        config = MetalConfig(bits=4, group_size=64)
        report["metal_config_available"] = True
        if not args.run:
            report["status"] = "preflight_passed"
        else:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import get_session

            get_session().event_hooks["request"].append(trim_user_agent)
            report["http_workaround"] = "Strip trailing User-Agent whitespace (kernels 0.17.2 / hub 2.2.0)"
            from transformers.integrations.metal_quantization import MetalLinear

            report["stage"] = "metal_kernel_probe"
            probe = MetalLinear(64, 64, bits=4, group_size=64).to("mps")
            with torch.inference_mode():
                probe_output = probe(torch.ones(1, 64, device="mps", dtype=torch.float16))
            torch.mps.synchronize()
            if probe_output.shape != (1, 64) or not torch.isfinite(probe_output).all().item():
                raise RuntimeError("Metal kernel probe returned an invalid result")
            report["metal_kernel_probe_passed"] = True
            del probe, probe_output
            peaks = {"sampled_process_rss_bytes": 0, "sampled_mps_allocated_bytes": 0,
                     "sampled_mps_driver_bytes": 0}
            process = psutil.Process()

            def sample() -> None:
                while not stop.is_set():
                    peaks["sampled_process_rss_bytes"] = max(peaks["sampled_process_rss_bytes"], process.memory_info().rss)
                    peaks["sampled_mps_allocated_bytes"] = max(peaks["sampled_mps_allocated_bytes"], torch.mps.current_allocated_memory())
                    peaks["sampled_mps_driver_bytes"] = max(peaks["sampled_mps_driver_bytes"], torch.mps.driver_allocated_memory())
                    stop.wait(0.2)

            report["stage"] = "resolve_revision"
            info = HfApi(token=False).model_info(report["model_id"], revision=args.revision)
            report["resolved_revision"] = info.sha
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            report["stage"] = "load_tokenizer"
            tokenizer = AutoTokenizer.from_pretrained(report["model_id"], revision=info.sha, token=False, trust_remote_code=False)
            report["stage"] = "load_model"
            load_started = time.perf_counter()
            model = Qwen3_5ForCausalLM.from_pretrained(
                report["model_id"], revision=info.sha, token=False,
                device_map="mps", dtype=torch.float16,
                quantization_config=config, trust_remote_code=False,
                attn_implementation="sdpa",
            ).eval()
            torch.mps.synchronize()
            report["load_seconds_including_download"] = time.perf_counter() - load_started
            report["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
            report["module_types"] = sorted({f"{type(m).__module__}.{type(m).__name__}" for m in model.modules() if "metal" in type(m).__module__.lower()})
            if not report["module_types"]:
                raise RuntimeError("No Metal quantized modules found; refusing an unquantized success")
            model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            torch.manual_seed(0)
            ids = [f"D{i}" for i in range(1, 11)]
            candidates = [
                ("D1", "A study of walking speed in healthy adults."),
                ("D2", "A randomized trial of cognitive behavioral therapy for adult depression."),
                ("D3", "An experiment on plant growth under artificial light."),
                ("D4", "A review of psychotherapies for adult depressive disorders."),
                ("D5", "A study of pediatric anxiety, excluding depression."),
                ("D6", "A report on computer network congestion."),
                ("D7", "A trial of exercise for adult depressive symptoms."),
                ("D8", "A study of chronic pain without mental health outcomes."),
                ("D9", "An observational study of psychotherapy use in adults with depression."),
                ("D10", "A study of ocean temperatures."),
            ]
            rerank_prompt = (
                "Rank these synthetic candidates for the query: psychotherapy for adult depression. "
                "Return only a JSON array containing exactly TEN strings, most relevant first. "
                "Include ALL TEN candidates, even irrelevant ones; reorder, do not filter. "
                "Every allowed ID must appear exactly once. "
                "Copy IDs exactly, including the D prefix; do not return integer positions. "
                f"Allowed IDs: {json.dumps(ids)}.\n"
                + "\n".join(f"{key}: {text}" for key, text in candidates)
            )
            cases = [("short_generation", "Write one short sentence explaining what a literature search does."),
                     ("listwise_ten", rerank_prompt)]
            for name, prompt in cases:
                report["stage"] = name
                messages = [{"role": "user", "content": prompt}]
                inputs = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt").to("mps")
                torch.mps.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                torch.mps.synchronize()
                seconds = time.perf_counter() - t0
                generated = output[0, inputs["input_ids"].shape[-1]:]
                text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                row = {"name": name, "prompt": prompt, "output": text,
                       "input_tokens": inputs["input_ids"].shape[-1],
                       "new_tokens": len(generated), "wall_seconds": seconds,
                       "tokens_per_second_including_prefill": len(generated) / seconds,
                       "hit_output_limit": len(generated) == 128}
                if name == "listwise_ten":
                    row["valid_permutation"] = valid_permutation(text, ids)
                report["cases"].append(row)
                print(f"{name}: {seconds:.2f}s; {text}", flush=True)
            report["status"] = (
                "passed" if all(c["output"] and not c["hit_output_limit"] and c.get("valid_permutation", True) for c in report["cases"])
                else "inference_ran_output_validation_failed"
            )
            report["memory_samples"] = peaks
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=2)
        if peaks:
            report["memory_samples"] = peaks
        report["elapsed_seconds"] = time.perf_counter() - started
        # ru_maxrss is bytes on macOS, KiB on Linux.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["process_peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "stage": report["stage"], "report": str(args.output), "error": report.get("error")}, indent=2), flush=True)
    return 0 if report["status"] in ("passed", "preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
