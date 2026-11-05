"""Check restored pilot inputs, pinned assets and environment without model inference."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh report path')
    report = {'status': 'running', 'platform': platform.platform(), 'machine': platform.machine(),
        'python': platform.python_version(), 'model_inference': False, 'heldout_used': False, 'checks': []}
    try:
        if sys.version_info[:2] != (3, 11):
            raise ValueError('Use Python 3.11 for the recorded environment')
        lock = ROOT/'requirements-macos-arm64-day19.lock'
        report['package_mismatches'] = []
        for line in lock.read_text().splitlines():
            if not line or line.startswith('#'): continue
            name, expected = line.split('==')
            try: actual = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: actual = None
            if actual != expected:
                report['package_mismatches'].append({'name': name, 'expected': expected, 'actual': actual})
        if report['package_mismatches']:
            raise ValueError('Restore the recorded package versions before reproducing artifacts')
        from medlens.transfer import sha_file
        annotation = json.loads((ROOT/'benchmarks/pilot-v1/annotation/manifest.json').read_bytes())
        for name, checksum in annotation['files'].items():
            if sha_file(ROOT/name) != checksum: raise ValueError('Annotation checksum differs: '+name)
        report['checks'].append('tracked annotation identities')
        source_path = ROOT/'.cache/models/bge-base-en-v1.5'
        source = json.loads((source_path/'model-source.json').read_bytes())
        for name, checksum in source['files'].items():
            if sha_file(source_path/name) != checksum: raise ValueError('BGE asset checksum differs')
        from medlens.reranking.medcpt import verify_assets
        verify_assets(ROOT/'.cache/models/medcpt-cross-encoder')
        report['checks'].append('BGE and MedCPT asset checksums')
        from medlens.evaluation.formats import parse_queries
        from medlens.retrieval.dense import DenseIndex
        from medlens.reranking.articles import MedCPTRanker
        from medlens.inference.qwen import MODEL_ID, REVISION
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, token=False,
            trust_remote_code=False, local_files_only=True)
        if not tokenizer.is_fast: raise ValueError('Expected cached fast Qwen tokenizer')
        report['checks'].append('local Qwen tokenizer; no Qwen model loaded')
        # Every shard referenced by the pinned snapshot index must resolve locally.
        snapshot = ROOT/'.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots'/REVISION
        weight_map = json.loads((snapshot/'model.safetensors.index.json').read_bytes())['weight_map']
        for name in set(weight_map.values()):
            if not (snapshot/name).is_file(): raise ValueError('Missing Qwen weight shard: '+name)
        report['checks'].append('Qwen shard symlinks resolve')
        path = ROOT/'benchmarks/pilot-v1/queries.jsonl'
        queries = parse_queries(path.read_bytes(), path.parent)
        report['collections'] = []
        for query in queries:
            if query.split != 'pilot': raise ValueError('Only development pilot inputs permitted')
            windows = ROOT/'data/windows/pilot-v1-002'/query.query_id
            index = DenseIndex(query.dataset, windows, ROOT/'data/embeddings/pilot-v1'/query.query_id)
            ranker = MedCPTRanker(query.dataset, windows)
            if index.window_manifest != ranker.window_manifest: raise ValueError('Window identities differ')
            report['collections'].append({'query_id': query.query_id,
                'dataset_id': index.dataset.manifest['dataset_id'], 'windows': len(index.windows),
                'candidates': len(index.articles)})
        import torch
        import psutil
        report.update(mps_available=torch.backends.mps.is_available(), memory_bytes=psutil.virtual_memory().total)
        report['status'] = 'inputs_ready'
        report['qwen_inference_on_this_machine'] = 'not-tested'
    except Exception as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'inputs_ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
