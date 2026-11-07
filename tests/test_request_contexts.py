"""Rebuilt contexts must match actual sources, even after internal rehashing."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from medlens.generation.context import build_context, digest, render_evidence
from medlens.generation.request_contexts import load_request_contexts, verify_context_sources
from test_evidence_context import CharacterTokenizer, fixtures


def source_fixture():
    dataset, manifest, windows = fixtures()
    context = build_context(dataset, manifest, windows, ['1', '2'], [.8] * len(windows),
                            CharacterTokenizer(), {})
    index = SimpleNamespace(dataset=dataset, window_manifest=manifest, windows=windows,
        embedding_manifest={'artifact_id': 'e'}, articles={a['pmid']: a for a in dataset.articles})
    return context, index


def rehash(context):
    context['sources_sha256'] = digest(context['sources'])
    context['evidence_payload'] = render_evidence(context['sources'])
    keys = ('schema_version', 'dataset_id', 'window_artifact_id', 'article_ranking',
            'scores_sha256', 'config', 'selection_policy', 'tokenizer', 'sources_sha256', 'implementation_sha256')
    context['context_id'] = digest({k: context[k] for k in keys})


class RequestContextTests(unittest.TestCase):
    def test_rehashed_text_and_metadata_cannot_replace_original_sources(self):
        context, index = source_fixture()
        verify_context_sources(context, index)
        for key in ('text', 'title', 'journal', 'publication_date', 'section', 'source_path'):
            changed = copy.deepcopy(context)
            value = changed['sources'][0][key]
            changed['sources'][0][key] = 'x' * len(value) if key == 'text' else 'fabricated'
            rehash(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_context_sources(changed, index)

    def load(self, root, run, *, escape=False):
        context, index = source_fixture()
        request = root/'requests/p01/run.json'
        request.parent.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps(run))
        if escape:
            outside = root/'outside.json'
            outside.write_bytes(request.read_bytes()); request.unlink(); request.symlink_to(outside)
        query = root/'benchmarks/pilot-v1/queries.jsonl'
        query.parent.mkdir(parents=True, exist_ok=True)
        query.write_text('bound questions')
        profile = {'queries': {'path': str(query.relative_to(root)),
                              'sha256': hashlib.sha256(query.read_bytes()).hexdigest()}}
        rows = [{'dataset': 'd', 'windows': 'w', 'embeddings': 'e', 'dataset_id': 'd',
                 'window_artifact_id': 'w', 'embedding_artifact_id': 'e'}]
        with (patch('medlens.generation.request_contexts.load_collection_profile', return_value=(profile, 'profile', rows)),
              patch('medlens.generation.request_contexts.parse_queries', return_value=[SimpleNamespace(query_id='p01', text='question')]),
              patch('medlens.generation.request_contexts.DenseIndex', return_value=index)):
            return load_request_contexts(root, root/'requests', 'profile')

    def run_fixture(self):
        context, _ = source_fixture()
        return {'status': 'complete', 'generate': False, 'rerank': False, 'question': 'question',
                'dataset_id': 'd', 'window_artifact_id': 'w', 'embedding_artifact_id': 'e',
                'context': context, 'retrieval': {'results': [{'pmid': p} for p in context['article_ranking']]}}

    def test_verified_requests_bind_raw_hash_without_claiming_historical_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact, checksum = self.load(Path(directory), self.run_fixture())
        self.assertEqual(artifact['resource_scope'], 'rebuilt-pilot')
        self.assertFalse(artifact['relevance_judgments_bound'])
        self.assertIsNone(artifact['qrels_sha256'])
        self.assertEqual(len(checksum), 64)
        self.assertEqual(len(artifact['source_requests'][0]['sha256']), 64)

    def test_generated_reranked_incomplete_or_mismatched_requests_are_rejected(self):
        changes = ({'generate': True}, {'rerank': True}, {'status': 'failed'}, {'question': 'other'},
                   {'dataset_id': 'other'}, {'embedding_artifact_id': 'other'},
                   {'retrieval': {'results': [{'pmid': '2'}, {'pmid': '1'}]}})
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run = self.run_fixture(); run.update(change)
                with self.assertRaises(ValueError): self.load(Path(directory), run)

    def test_request_symlink_cannot_escape_request_directory(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'escapes'):
            self.load(Path(directory), self.run_fixture(), escape=True)
