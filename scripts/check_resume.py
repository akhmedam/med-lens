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
    parser.add_argument('--collections', type=Path, help='Explicit rebuilt-pilot resource profile; leaves frozen profiles unchanged')
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
        if args.collections:
            from medlens.resources import load_collection_profile
            profile, checksum, collections = load_collection_profile(ROOT, args.collections)
            report.update(collection_profile_sha256=checksum, resource_scope=profile['kind'],
                corpus_version=profile['version'], historical_snapshot_restore=False)
        else:
            path = ROOT/'benchmarks/pilot-v1/queries.jsonl'
            queries = parse_queries(path.read_bytes(), path.parent)
            if any(q.split != 'pilot' for q in queries):
                raise ValueError('Only development pilot inputs permitted')
            collections = [{'query_id': q.query_id, 'dataset': q.dataset,
                'windows': ROOT/'data/windows/pilot-v1-002'/q.query_id,
                'embeddings': ROOT/'data/embeddings/pilot-v1'/q.query_id} for q in queries]
            report['resource_scope'] = 'original-frozen-pilot'
        report['collections'] = []
        for row in collections:
            index = DenseIndex(row['dataset'], row['windows'], row['embeddings'])
            ranker = MedCPTRanker(row['dataset'], row['windows'])
            if index.window_manifest != ranker.window_manifest:
                raise ValueError('Window identities differ')
            actual = {'dataset_id': index.dataset.manifest['dataset_id'],
                'window_artifact_id': index.window_manifest['artifact_id'],
                'embedding_artifact_id': index.embedding_manifest['artifact_id']}
            if args.collections and any(row[key] != value for key, value in actual.items()):
                raise ValueError('Collection identity differs: '+row['query_id'])
            report['collections'].append({'query_id': row['query_id'], **actual,
                'windows': len(index.windows), 'candidates': len(index.articles)})
        import torch
        import psutil
        report.update(mps_available=torch.backends.mps.is_available(), memory_bytes=psutil.virtual_memory().total)
        report['status'] = 'inputs_ready'
        report['qwen_inference_on_this_machine'] = 'not-tested-by-this-check'
    except Exception as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'inputs_ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
