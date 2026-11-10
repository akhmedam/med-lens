"""Explicit backend selection: dependency isolation never means silent fallback."""

import hashlib
from pathlib import Path

from .qwen import STOP_TOKEN_IDS
from .profiles import DEFAULT_MODEL, get_profile


def runtime_factory(engine, *, model=DEFAULT_MODEL, root=None, model_dir=None, worker_python=None):
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    profile = get_profile(model)
    policy = {"engine": engine, "model_id": profile["model_id"], "revision": profile["revision"],
              "answer_model": model, "answer_policy": profile["answer_policy"],
              "eos_token_ids": list(STOP_TOKEN_IDS), "do_sample": False}
    if engine == "mlx":
        from .mlx import MlxRuntime
        model_dir = Path(model_dir) if model_dir else root / profile["directory"]
        # This identity lookup does not require model availability. Source retrieval
        # and empty/search-only requests must still work before MLX is prepared.
        try:
            policy["artifact_manifest_sha256"] = hashlib.sha256(
                (model_dir / "medlens-source.json").read_bytes()).hexdigest()
        except OSError:
            policy["artifact_manifest_sha256"] = None
        def factory():
            # Reject an asset change between request identification and model use.
            current = hashlib.sha256((model_dir / "medlens-source.json").read_bytes()).hexdigest()
            if current != policy["artifact_manifest_sha256"]:
                raise ValueError("MLX artifact manifest changed during request")
            return MlxRuntime(model=model, root=root, model_dir=model_dir, worker_python=worker_python)
    elif engine == "transformers":
        if model != "9b":
            raise ValueError("Transformers comparison requires --answer-model 9b")
        if model_dir is not None or worker_python is not None:
            raise ValueError("MLX paths require --engine mlx")
        from .qwen import QwenRuntime
        factory = QwenRuntime
    else:
        raise ValueError("Unknown inference engine: " + str(engine))
    # Avoid mutating QwenRuntime's class attributes across independently configured factories.
    make = lambda: factory()
    make.identity = policy
    return make


def add_runtime_arguments(parser):
    parser.add_argument("--answer-model", choices=("4b", "9b"), default=DEFAULT_MODEL,
        help="Answer model profile (default: 4b; 9b is an explicit comparison)")
    parser.add_argument("--engine", choices=("mlx", "transformers"), default="mlx",
        help="Qwen backend; MLX requires the separately prepared native assets")
    parser.add_argument("--mlx-model-dir", type=Path,
        help="Prepared MLX artifact directory (default: selected answer-model profile)")
    parser.add_argument("--mlx-python", type=Path,
        help="Isolated MLX Python (default: .cache/venvs/mlx/bin/python)")
