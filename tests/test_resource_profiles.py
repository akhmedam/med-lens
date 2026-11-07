"""Rebuilt resources cannot silently replace a frozen or held-out benchmark."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from medlens.resources import load_collection_profile


class CollectionProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.query_path = self.root/'benchmarks/pilot-v1/queries.jsonl'
        self.query_path.parent.mkdir(parents=True)
        self.query_path.write_text(json.dumps({"query_id": "p1", "text": "Synthetic question?",
            "split": "pilot", "group_id": "g1", "dataset": "../../data/old/p1"})+'\n')
        self.profile = {"schema_version": 1, "kind": "rebuilt-pilot", "version": "new",
            "historical_snapshot_restore": False,
            "queries": {"path": "benchmarks/pilot-v1/queries.jsonl",
                "sha256": hashlib.sha256(self.query_path.read_bytes()).hexdigest()},
            "collections": [{"query_id": "p1", "dataset": "data/new/p1", "windows": "data/windows/p1",
                "embeddings": "data/vectors/p1", "dataset_id": "a"*64,
                "window_artifact_id": "b"*64, "embedding_artifact_id": "c"*64}]}
        self.path = self.root/'profile.json'

    def load(self):
        self.path.write_text(json.dumps(self.profile))
        return load_collection_profile(self.root, self.path)

    def test_new_paths_are_explicit_and_historical_query_paths_stay_unchanged(self):
        _, checksum, rows = self.load()
        self.assertEqual(rows[0]["dataset"], self.root/'data/new/p1')
        self.assertEqual(checksum, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertIn('data/old/p1', self.query_path.read_text())

    def test_query_changes_are_detected(self):
        self.query_path.write_text(self.query_path.read_text().replace('question', 'changed question'))
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.load()

    def test_heldout_path_is_rejected_before_reading_it(self):
        self.profile['queries']['path'] = 'benchmarks/heldout-v1/queries.jsonl'
        with self.assertRaisesRegex(ValueError, 'resource path'):
            self.load()

    def test_external_paths_and_symlink_escapes_are_rejected(self):
        for name in ('../outside', '/tmp/outside', 'data/../../outside'):
            self.profile['collections'][0]['dataset'] = name
            with self.assertRaises(ValueError):
                self.load()
        outside = self.root.parent/(self.root.name+'-outside')
        (self.root/'data').symlink_to(outside)
        self.profile['collections'][0]['dataset'] = 'data/new/p1'
        with self.assertRaisesRegex(ValueError, 'escapes checkout'):
            self.load()

    def test_duplicate_missing_or_reordered_collections_are_rejected(self):
        for rows in ([], self.profile['collections']*2, [{**self.profile['collections'][0], 'query_id': 'other'}]):
            self.profile['collections'] = rows
            with self.assertRaises(ValueError):
                self.load()

    def test_rebuilt_profile_cannot_claim_to_restore_historical_snapshots(self):
        self.profile['historical_snapshot_restore'] = True
        with self.assertRaises(ValueError):
            self.load()
