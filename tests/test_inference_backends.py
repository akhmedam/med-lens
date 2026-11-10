"""Backend policy identity, lazy availability and explicit fallback selection."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from medlens.inference.backends import runtime_factory


class BackendTests(unittest.TestCase):
    def test_missing_native_assets_do_not_prevent_search_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            make = runtime_factory("mlx", root=directory)
            self.assertEqual(make.identity["engine"], "mlx")
            self.assertIsNone(make.identity["artifact_manifest_sha256"])
            with self.assertRaises(FileNotFoundError):make()

    def test_native_manifest_identity_is_checked_again_at_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "medlens-source.json"
            path.write_text("original")
            make = runtime_factory("mlx", model_dir=directory)
            with patch("medlens.inference.mlx.MlxRuntime", return_value="native") as runtime:
                # The factory captures the constructor at selection time.
                fresh = runtime_factory("mlx", model_dir=directory)
                self.assertEqual(fresh(), "native")
                runtime.assert_called_once()
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "manifest changed"):make()

    def test_fallback_is_always_explicit_and_rejects_native_paths(self):
        with patch("medlens.inference.qwen.QwenRuntime", return_value="torch") as runtime:
            make = runtime_factory("transformers")
            self.assertEqual(make(), "torch")
            self.assertEqual(make.identity["eos_token_ids"], [248044, 248046])
            runtime.assert_called_once()
        for engine, options in [("unknown", {}), ("transformers", {"model_dir": "unused"}),
                                 ("transformers", {"worker_python": "unused"})]:
            with self.assertRaises(ValueError):runtime_factory(engine, **options)
