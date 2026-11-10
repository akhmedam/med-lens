"""One cached Qwen instance shared by reranking, synthesis and support review."""

import hashlib
import json
import importlib.metadata
import os
import time

from .metal import RequireFiniteScores, bounded_metal_loading

MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
# The pinned tokenizer uses im_end; text config uses endoftext. Stop on either.
STOP_TOKEN_IDS = (248044, 248046)


class QwenRuntime:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, MetalConfig, Qwen3_5ForCausalLM
        if not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("MPS is required without implicit CPU fallback")
        # Same narrowly scoped compatibility fix as the verified Day 1 runtime.
        from huggingface_hub.utils import get_session
        def trim_user_agent(request):
            value = request.headers.get("user-agent")
            if value is not None:
                request.headers["user-agent"] = value.rstrip()
        get_session().event_hooks["request"].append(trim_user_agent)
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
            local_files_only=True, token=False, trust_remote_code=False)
        with bounded_metal_loading() as loading:
            self.model = Qwen3_5ForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
                local_files_only=True, token=False, trust_remote_code=False, device_map="mps",
                dtype=torch.float16, quantization_config=MetalConfig(bits=4, group_size=64),
                attn_implementation="sdpa").eval()
        from transformers.integrations.metal_quantization import MetalLinear
        if not any(isinstance(m, MetalLinear) for m in self.model.modules()):
            raise RuntimeError("No quantized Metal layers found")
        if {p.device.type for p in self.model.parameters()} != {"mps"}:
            raise RuntimeError("Model has parameters outside MPS")
        torch.mps.synchronize()
        torch.manual_seed(0)
        self.metadata = {"model_id": MODEL_ID, "revision": REVISION, "device": "mps",
            "quantization": {"method": "metal", "bits": 4, "group_size": 64},
            "engine": "transformers-metal",
            "generation": {"do_sample": False, "max_new_tokens": 128, "enable_thinking": False,
                           "eos_token_ids": list(STOP_TOKEN_IDS)},
            "load_seconds": time.perf_counter() - start, "loading": loading,
            "score_validation": "reject-nan-positive-infinity-and-fully-masked-rows",
            "release_cache_after_generation": True,
            "packages": {n: importlib.metadata.version(n) for n in
                         ("torch", "transformers", "accelerate", "kernels", "huggingface-hub")}}

    def generate(self, prompt, *, max_new_tokens=128):
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be an integer from 1 to 1024")
        if self.model is None:
            raise RuntimeError("Runtime is closed")
        import torch
        inputs = self.tokenizer.apply_chat_template(prompt["messages"], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=True,
            return_tensors="pt").to("mps")
        if inputs["input_ids"].shape[-1] != prompt["input_tokens"]:
            raise ValueError("Prompt token count changed")
        torch.mps.synchronize()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    logits_processor=[RequireFiniteScores()], eos_token_id=list(STOP_TOKEN_IDS),
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id)
            torch.mps.synchronize()
            generated = output[0, inputs["input_ids"].shape[-1]:]
            return {"text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
                    "new_tokens": len(generated), "hit_output_limit": len(generated) >= max_new_tokens,
                    "max_new_tokens": max_new_tokens,
                    "input_ids_sha256": hashlib.sha256(json.dumps(
                        inputs["input_ids"][0].tolist(), separators=(",", ":")).encode()).hexdigest(),
                    "seconds": time.perf_counter() - start}
        finally:
            torch.mps.empty_cache()

    def close(self):
        """Release model weights deterministically; safe to call more than once."""
        import gc
        import torch
        self.model = None
        gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
