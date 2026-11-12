"""Load primary Dense contexts from verified saved development requests."""

import hashlib
import json
from pathlib import Path
import re

from medlens.evaluation.formats import parse_queries
from medlens.resources import load_collection_profile
from medlens.retrieval.dense import DenseIndex
from .answers import check_context
from .context import COVERAGE_POLICY, canonical, source_section, verify_source_sentence_spans


def verify_context_sources(context, index):
    check_context(context)
    if (context['dataset_id'] != index.dataset.manifest['dataset_id']
            or context['window_artifact_id'] != index.window_manifest['artifact_id']):
        raise ValueError('Context source identities differ')
    by_window = {w['window_id']: w for w in index.windows}
    for source in context['sources']:
        window = by_window[source['window_id']]
        if any(source[key] != window[key] for key in ('pmid', 'text_version_id', 'text',
                'char_start', 'char_end', 'source_path', 'section_index', 'provenance')):
            raise ValueError('Context differs from saved source window')
        article = index.articles[source['pmid']]
        source_section(article, window)
        verify_source_sentence_spans(source, article, window,
            required=context.get('selection_policy') == COVERAGE_POLICY)
        date = next((d['iso_date'] for d in article.get('dates', [])
                     if d['kind'] == 'journal_issue' and d.get('iso_date')), None)
        if (source['title'] != article.get('title') or source['section'] != window['section_label']
                or source['journal'] != (article.get('journal') or {}).get('title')
                or source['publication_date'] != date):
            raise ValueError('Context metadata differs from source article')


def load_request_contexts(root, directory, profile_path):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    profile, profile_hash, collections = load_collection_profile(root, profile_path)
    query_path = root/profile['queries']['path']
    query_bytes = query_path.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != profile['queries']['sha256']:
        raise ValueError('Pilot questions changed during context loading')
    queries = parse_queries(query_bytes, query_path.parent)
    rows, sources = [], []
    for query, spec in zip(queries, collections):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', query.query_id):
            raise ValueError('Unsafe pilot query ID')
        path = directory/query.query_id/'run.json'
        if not path.resolve().is_relative_to(directory):
            raise ValueError('Request path escapes directory')
        body = path.read_bytes()
        run = json.loads(body)
        if (run.get('status') != 'complete' or run.get('generate') is not False
                or run.get('rerank') is not False or run.get('question') != query.text):
            raise ValueError('Require a completed primary Dense search request: '+query.query_id)
        index = DenseIndex(spec['dataset'], spec['windows'], spec['embeddings'])
        actual = {'dataset_id': index.dataset.manifest['dataset_id'],
            'window_artifact_id': index.window_manifest['artifact_id'],
            'embedding_artifact_id': index.embedding_manifest['artifact_id']}
        if any(spec[key] != value or run.get(key) != value for key, value in actual.items()):
            raise ValueError('Request resource identities differ')
        context = run['context']
        ranking = [hit['pmid'] for hit in run['retrieval']['results']]
        if (context['article_ranking'] != ranking or len(set(ranking)) != len(ranking)
                or not set(ranking) <= set(index.articles)):
            raise ValueError('Context and saved Dense ranking differ')
        verify_context_sources(context, index)
        rows.append({'query_id': query.query_id, 'query': query.text, 'context': context})
        sources.append({'path': str(path.relative_to(directory)), 'sha256': hashlib.sha256(body).hexdigest()})
    artifact = {'schema_version': 1, 'method': 'dense', 'resource_scope': 'rebuilt-pilot',
        'relevance_judgments_bound': False, 'qrels_sha256': None, 'profile_sha256': profile_hash,
        'queries_sha256': profile['queries']['sha256'], 'source_requests': sources, 'per_query': rows}
    return artifact, hashlib.sha256(canonical(artifact).encode()).hexdigest()
