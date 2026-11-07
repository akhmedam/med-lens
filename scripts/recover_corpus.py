"""Build or verify a separate pilot corpus; never replace frozen pilot-v1 inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

from medlens.evaluation.formats import parse_queries
from medlens.ingestion.pubmed import PubMedClient
from medlens.ingestion.snapshots import download_snapshot, load_snapshot
from medlens.parsing.dataset import build_dataset, load_dataset
from medlens.retrieval.bge_tokenizer import load_tokenizer
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.embeddings import BGEEncoder, build_embeddings
from medlens.retrieval.windows import build_windows, load_windows


def corpus_version(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value) or value.startswith('pilot-v1'):
        raise argparse.ArgumentTypeError('Use a new safe version, not the frozen pilot-v1 namespace')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True, type=corpus_version)
    parser.add_argument('--fetch', action='store_true', help='Allow new PubMed requests for missing snapshots')
    parser.add_argument('--output', type=Path, required=True, help='Fresh collection-profile path')
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error('Choose a fresh profile path')
    query_path = ROOT/'benchmarks/pilot-v1/queries.jsonl'
    queries = parse_queries(query_path.read_bytes(), query_path.parent)
    plan = json.loads((ROOT/'benchmarks/pilot-v1/plan.json').read_bytes())
    if any(q.split != 'pilot' for q in queries) or [q.query_id for q in queries] != [r['query_id'] for r in plan['queries']]:
        raise ValueError('Pilot query order and plan differ')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', q.query_id) for q in queries):
        raise ValueError('Unsafe pilot query ID')
    tokenizer, info = load_tokenizer(ROOT/'.cache/tokenizers/bge-base-en-v1.5')
    client, encoder, rows = PubMedClient(), None, []
    for query, entry in zip(queries, plan['queries']):
        snapshot = ROOT/'data/snapshots'/args.version/query.query_id
        dataset = ROOT/'data/parsed'/args.version/query.query_id
        windows = ROOT/'data/windows'/args.version/query.query_id
        vectors = ROOT/'data/embeddings'/args.version/query.query_id
        for path in (snapshot, dataset, windows, vectors):
            if path.resolve() != ROOT.resolve()/path.relative_to(ROOT):
                raise ValueError('Corpus paths cannot contain symlinks')
        if not snapshot.exists():
            if not args.fetch: raise ValueError('Missing snapshot; --fetch is required for a new live corpus')
            download_snapshot(client, entry['pubmed_query'], snapshot, limit=plan['candidate_limit'])
        source = load_snapshot(snapshot).manifest
        if source['query'] != entry['pubmed_query'] or source['limit'] != plan['candidate_limit']:
            raise ValueError('Existing snapshot differs from the requested plan')
        if not dataset.exists(): build_dataset(snapshot, dataset)
        parsed = load_dataset(dataset)
        if parsed.manifest['source_snapshot_id'] != source['snapshot_id']:
            raise ValueError('Parsed dataset differs from snapshot')
        if not windows.exists(): build_windows(dataset, tokenizer, info, windows)
        window_manifest, _ = load_windows(windows)
        if window_manifest['dataset_id'] != parsed.manifest['dataset_id']:
            raise ValueError('Windows differ from parsed dataset')
        if not vectors.exists():
            if encoder is None:
                encoder = BGEEncoder(ROOT/'.cache/models/bge-base-en-v1.5',
                    ROOT/'.cache/tokenizers/bge-base-en-v1.5', device='mps')
            build_embeddings(windows, encoder, vectors)
        index = DenseIndex(dataset, windows, vectors)
        if encoder is not None: index.verify_encoder(encoder)
        rows.append({'query_id': query.query_id, 'dataset': str(dataset.relative_to(ROOT)),
            'windows': str(windows.relative_to(ROOT)), 'embeddings': str(vectors.relative_to(ROOT)),
            'dataset_id': index.dataset.manifest['dataset_id'],
            'window_artifact_id': index.window_manifest['artifact_id'],
            'embedding_artifact_id': index.embedding_manifest['artifact_id']})
        print(query.query_id, 'verified', len(index.articles), len(index.windows), flush=True)
    profile = {'schema_version': 1, 'kind': 'rebuilt-pilot', 'version': args.version,
        'historical_snapshot_restore': False,
        'queries': {'path': str(query_path.relative_to(ROOT)), 'sha256': hashlib.sha256(query_path.read_bytes()).hexdigest()},
        'collections': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:stream.write(json.dumps(profile, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
