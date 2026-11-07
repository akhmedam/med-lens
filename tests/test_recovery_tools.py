"""Resource recovery refuses conflicts and never fetches without permission."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/name
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoveryToolTests(unittest.TestCase):
    def test_asset_conflict_prevents_network_and_preserves_existing_bytes(self):
        tool = script('prepare_models.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'cache').mkdir()
            payload = root/'cache/model.bin'
            payload.write_bytes(b'keep this')
            pins = {'schema_version': 1, 'models': {'bge': {'directory': 'cache',
                'repo_id': 'fixture/model', 'revision': 'pinned', 'files': {'model.bin': {
                    'bytes': 8, 'sha256': hashlib.sha256(b'expected').hexdigest()}}}}}
            (root/'configs/model-assets.json').write_text(json.dumps(pins))
            with patch.object(tool, 'ROOT', root), patch('huggingface_hub.snapshot_download') as download:
                with self.assertRaisesRegex(ValueError, 'checksum conflict'):
                    tool.main(['--download', '--output', str(root/'result.json')])
                download.assert_not_called()
            self.assertEqual(payload.read_bytes(), b'keep this')

    def test_metadata_identity_preserves_original_serialization(self):
        tool = script('prepare_models.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'source.json'
            path.write_text('{"x": 1}')
            tool.save_metadata(path, {'x': 1})
            self.assertEqual(path.read_text(), '{"x": 1}')
            with self.assertRaises(ValueError): tool.save_metadata(path, {'x': 2})

    def test_atomic_publication_never_overwrites_a_file(self):
        tool = script('prepare_models.py')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'asset'
            tool.publish_bytes(path, b'first')
            with self.assertRaises(FileExistsError): tool.publish_bytes(path, b'second')
            self.assertEqual(path.read_bytes(), b'first')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_frozen_or_escaping_corpus_versions_are_rejected(self):
        tool = script('recover_corpus.py')
        for name in ('pilot-v1', 'pilot-v1-002', '../other', '/tmp/other', 'a/b'):
            with self.subTest(name=name), self.assertRaises(argparse.ArgumentTypeError): tool.corpus_version(name)
        self.assertEqual(tool.corpus_version('recovery-20261106'), 'recovery-20261106')

    def test_missing_corpus_is_not_fetched_implicitly(self):
        tool = script('recover_corpus.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'benchmarks/pilot-v1'; folder.mkdir(parents=True)
            (folder/'queries.jsonl').write_text(json.dumps({'query_id': 'p1', 'text': 'Synthetic?',
                'split': 'pilot', 'group_id': 'g1', 'dataset': '../../data/old'})+'\n')
            (folder/'plan.json').write_text(json.dumps({'candidate_limit': 30,
                'queries': [{'query_id': 'p1', 'pubmed_query': 'synthetic'}]}))
            with (patch.object(tool, 'ROOT', root), patch.object(tool, 'load_tokenizer', return_value=(None, {})),
                  patch.object(tool, 'download_snapshot') as download):
                with self.assertRaisesRegex(ValueError, '--fetch'):
                    tool.main(['--version', 'new', '--output', str(root/'profile.json')])
                download.assert_not_called()


    def test_corpus_symlink_cannot_redirect_recovery_into_frozen_data(self):
        tool = script('recover_corpus.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            folder = root/'benchmarks/pilot-v1'; folder.mkdir(parents=True)
            (folder/'queries.jsonl').write_text(json.dumps({'query_id': 'p1', 'text': 'Synthetic?',
                'split': 'pilot', 'group_id': 'g1', 'dataset': '../../data/old'})+'\n')
            (folder/'plan.json').write_text(json.dumps({'candidate_limit': 30,
                'queries': [{'query_id': 'p1', 'pubmed_query': 'synthetic'}]}))
            (root/'data/snapshots').mkdir(parents=True)
            (root/'data/snapshots/new').symlink_to(root/'data/snapshots/pilot-v1')
            with (patch.object(tool, 'ROOT', root), patch.object(tool, 'load_tokenizer', return_value=(None, {})),
                  patch.object(tool, 'download_snapshot') as download):
                with self.assertRaisesRegex(ValueError, 'symlinks'):
                    tool.main(['--version', 'new', '--fetch', '--output', str(root/'profile.json')])
                download.assert_not_called()
