"""Compatibility import for the shared Qwen runtime (reranking defaults unchanged)."""

from medlens.inference.qwen import MODEL_ID, REVISION, QwenRuntime

QwenReranker = QwenRuntime
