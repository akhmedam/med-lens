"""Bound conversion cache growth and reject invalid generation scores on MPS."""

from contextlib import contextmanager
from threading import Lock

_LOAD_LOCK = Lock()


@contextmanager
def bounded_metal_loading():
    """Scope the pinned Metal converter; never change its arithmetic or device.

    Quantizing successive matrices retains large unused MPS allocations. Release
    those allocations after each completed conversion, while its results remain
    live. Restore the converter even if model construction fails. The lock makes
    concurrent MedLens model loads serialize around the temporary class wrapper.
    """
    import torch
    from transformers.integrations.metal_quantization import MetalQuantize

    with _LOAD_LOCK:
        original = MetalQuantize.convert
        stats = {"strategy": "synchronize-and-release-conversion-cache",
                 "conversion_count": 0, "sampled_conversion_driver_peak_bytes": 0}

        def convert(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            torch.mps.synchronize()
            stats["sampled_conversion_driver_peak_bytes"] = max(
                stats["sampled_conversion_driver_peak_bytes"], torch.mps.driver_allocated_memory())
            torch.mps.empty_cache()
            stats["conversion_count"] += 1
            return result

        MetalQuantize.convert = convert
        try:
            yield stats
        finally:
            MetalQuantize.convert = original


class RequireFiniteScores:
    """Stop NaN/positive infinity from silently decoding as token zero.

    Negative infinity is permitted for intentionally suppressed vocabulary items;
    a row with no usable token is still an error. No scores are repaired.
    """

    def __call__(self, input_ids, scores):
        import torch

        if (torch.isnan(scores).any().item() or torch.isposinf(scores).any().item()
                or torch.isneginf(scores).all(dim=-1).any().item()):
            raise FloatingPointError("Invalid model generation scores; output withheld")
        return scores
