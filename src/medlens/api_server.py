"""Serve the MedLens research API on loopback with one worker process."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
        help="Prepared MedLens project containing the pinned local assets")
    parser.add_argument("--storage-dir", type=Path, help="Request artifacts; default ROOT/results/api-requests")
    parser.add_argument("--cache", type=Path, help="Shared embedding cache; default ROOT/.cache/embeddings/bge-input-v1")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        import uvicorn
        from .api import create_app
    except ImportError:
        parser.error("Install the API dependencies: python -m pip install -e '.[api]'")
    try:
        app = create_app(root=args.root, storage_root=args.storage_dir, cache_root=args.cache)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False)
    finally:
        app.state.job_service.close()
    return 0
