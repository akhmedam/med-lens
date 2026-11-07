"""Cache lifecycle and fail-closed numerical checks without model downloads."""

import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch
from transformers.integrations.metal_quantization import MetalQuantize

from medlens.inference.metal import RequireFiniteScores, bounded_metal_loading
from medlens.inference.qwen import QwenRuntime


class MetalLoadingTests(unittest.TestCase):
    def test_conversion_preserves_results_and_releases_cache_after_completion(self):
        events = []
        output = {"weight": object()}
        original = Mock(side_effect=lambda *a, **k: (events.append("convert"), output)[1])
        with (patch.object(MetalQuantize, "convert", original),
              patch("torch.mps.synchronize", side_effect=lambda: events.append("sync")),
              patch("torch.mps.driver_allocated_memory", side_effect=[100, 80]),
              patch("torch.mps.empty_cache", side_effect=lambda: events.append("release"))):
            with bounded_metal_loading() as stats:
                converter = MetalQuantize.__new__(MetalQuantize)
                self.assertIs(converter.convert({"input": 1}, marker=True), output)
                self.assertIs(converter.convert({"input": 2}), output)
            self.assertIs(MetalQuantize.convert, original)
            self.assertEqual(stats["conversion_count"], 2)
            self.assertEqual(stats["sampled_conversion_driver_peak_bytes"], 100)
            self.assertEqual(events, ["convert", "sync", "release"] * 2)
            self.assertEqual(original.call_args_list[0].kwargs, {"marker": True})

    def test_original_converter_restored_on_load_failure(self):
        original = MetalQuantize.convert
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            with bounded_metal_loading():
                raise RuntimeError("load failed")
        self.assertIs(MetalQuantize.convert, original)

    def test_conversion_error_is_not_hidden_and_wrapper_is_restored(self):
        original = Mock(side_effect=ValueError("bad weights"))
        with patch.object(MetalQuantize, "convert", original), patch("torch.mps.empty_cache") as release:
            with self.assertRaisesRegex(ValueError, "bad weights"):
                with bounded_metal_loading():
                    MetalQuantize.__new__(MetalQuantize).convert({})
            self.assertIs(MetalQuantize.convert, original)
            release.assert_not_called()

    def test_synchronization_error_restores_wrapper(self):
        original = Mock(return_value={})
        with (patch.object(MetalQuantize, "convert", original),
              patch("torch.mps.synchronize", side_effect=RuntimeError("device error"))):
            with self.assertRaisesRegex(RuntimeError, "device error"):
                with bounded_metal_loading():
                    MetalQuantize.__new__(MetalQuantize).convert({})
            self.assertIs(MetalQuantize.convert, original)


class ScoreGuardTests(unittest.TestCase):
    def test_valid_scores_and_suppressed_tokens_are_unchanged(self):
        for scores in (torch.tensor([[1., 2., -3.]]), torch.tensor([[-float("inf"), 2., 1.]])):
            self.assertIs(RequireFiniteScores()(None, scores), scores)

    def test_invalid_row_is_rejected_even_when_another_row_is_valid(self):
        for row in ([float("nan"), 0.], [float("inf"), 0.], [-float("inf"), -float("inf")]):
            scores = torch.tensor([[1., 2.], row])
            with self.subTest(row=row), self.assertRaises(FloatingPointError):
                RequireFiniteScores()(None, scores)


class QwenGenerateTests(unittest.TestCase):
    def runtime(self):
        class Batch(dict):
            def to(self, device):
                if device != 'mps': raise AssertionError('Unexpected inference device')
                return self
        runtime = QwenRuntime.__new__(QwenRuntime)
        runtime.tokenizer = SimpleNamespace(pad_token_id=1, eos_token_id=2,
            apply_chat_template=lambda *a, **k: Batch(input_ids=torch.tensor([[1, 2]])),
            decode=lambda *a, **k: 'accepted output')
        runtime.model = SimpleNamespace(generate=Mock(return_value=torch.tensor([[1, 2, 3]])))
        return runtime

    def test_generation_keeps_budget_and_installs_score_guard(self):
        runtime = self.runtime()
        with patch('torch.mps.synchronize'), patch('torch.mps.empty_cache') as release:
            result = runtime.generate({'messages': [], 'input_tokens': 2}, max_new_tokens=768)
        kwargs = runtime.model.generate.call_args.kwargs
        self.assertEqual(kwargs['max_new_tokens'], 768)
        self.assertIsInstance(kwargs['logits_processor'][0], RequireFiniteScores)
        self.assertFalse(result['hit_output_limit'])
        release.assert_called_once()

    def test_generation_failure_still_releases_unused_cache(self):
        runtime = self.runtime()
        runtime.model.generate.side_effect = FloatingPointError('invalid scores')
        with patch('torch.mps.synchronize'), patch('torch.mps.empty_cache') as release:
            with self.assertRaises(FloatingPointError):
                runtime.generate({'messages': [], 'input_tokens': 2})
        release.assert_called_once()


class QwenConstructionTests(unittest.TestCase):
    def test_model_construction_is_inside_bounded_loading_scope(self):
        from contextlib import contextmanager
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        active = []
        @contextmanager
        def bounded():
            active.append(True)
            try: yield {"conversion_count": 248}
            finally: active.pop()
        model = Mock()
        model.modules.return_value = [object()]
        model.parameters.return_value = [SimpleNamespace(device=SimpleNamespace(type="mps"))]
        def construct(*args, **kwargs):
            self.assertEqual(active, [True])
            self.assertEqual(kwargs["device_map"], "mps")
            return SimpleNamespace(eval=lambda: model)
        with (patch("medlens.inference.qwen.bounded_metal_loading", bounded),
              patch("torch.backends.mps.is_available", return_value=True),
              patch("torch.mps.synchronize"), patch("torch.manual_seed"),
              patch("transformers.AutoTokenizer.from_pretrained", return_value=object()),
              patch("transformers.Qwen3_5ForCausalLM.from_pretrained", side_effect=construct),
              patch("transformers.integrations.metal_quantization.MetalLinear", object),
              patch("huggingface_hub.utils.get_session", return_value=SimpleNamespace(event_hooks={"request": []}))):
            runtime = QwenRuntime()
        self.assertEqual(runtime.metadata["loading"]["conversion_count"], 248)
        self.assertEqual(active, [])
