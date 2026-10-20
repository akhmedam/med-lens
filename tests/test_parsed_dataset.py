"""Parsed datasets retain the source order and reject incomplete or changed files."""

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from medlens.ingestion.snapshots import download_snapshot
from medlens.parsing.dataset import DatasetError, build_dataset, load_dataset
from pubmed_helpers import make_client, search_json, xml_records


class ParsedDatasetTests(unittest.TestCase):
    def snapshot(self, root, empty=False):
        events = [search_json((), 0)] if empty else [search_json(), xml_records(["12", "31"]), xml_records(["25"])]
        client, _, _ = make_client(events)
        directory = root / "snapshot"
        download_snapshot(client, "depression", directory, limit=3, batch_size=2)
        return directory

    def test_build_replay_and_deterministic_identity_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.snapshot(root)
            with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                first = build_dataset(source, root / "first")
                second = build_dataset(source, root / "second")
                loaded = load_dataset(root / "first")
            self.assertEqual(first["dataset_id"], second["dataset_id"])
            self.assertEqual(first["articles_sha256"], second["articles_sha256"])
            self.assertEqual([a["pmid"] for a in loaded.articles], ["31", "12", "25"])
            self.assertEqual(loaded.articles[0]["provenance"]["snapshot_id"], first["source_snapshot_id"])
            self.assertEqual(loaded.articles[0]["provenance"]["source_file"], "fetch-0001.xml")

    def test_empty_snapshot_is_a_valid_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            build_dataset(self.snapshot(root, empty=True), root / "parsed")
            self.assertEqual(load_dataset(root / "parsed").articles, ())

    def test_rejects_corrupted_records_and_source_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.snapshot(root)
            output = root / "parsed"
            build_dataset(source, output)
            article_path = output / "articles.jsonl"
            original = article_path.read_bytes()
            article_path.write_bytes(original + b"{}\n")
            with self.assertRaisesRegex(DatasetError, "checksum"):
                load_dataset(output)
            article_path.write_bytes(original)
            (output / "source-manifest.json").write_text("{}")
            with self.assertRaisesRegex(DatasetError, "checksum"):
                load_dataset(output)

    def test_rejects_incomplete_dataset_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.snapshot(root)
            output = root / "parsed"
            build_dataset(source, output)
            with self.assertRaises(FileExistsError):
                build_dataset(source, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(DatasetError):
                load_dataset(output)

    def test_record_parse_failure_does_not_publish_partial_dataset(self):
        client, _, _ = make_client([search_json(["1"], 1),
            b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID></MedlineCitation></PubmedArticle></PubmedArticleSet>"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "snapshot"
            download_snapshot(client, "depression", source, limit=1)
            with self.assertRaisesRegex(DatasetError, "missing Article"):
                build_dataset(source, root / "parsed")
            self.assertFalse((root / "parsed").exists())

    def test_cli_build_and_inspect_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.snapshot(root)
            output = root / "parsed"
            script = Path(__file__).resolve().parents[1] / "scripts/parse_pubmed_snapshot.py"
            for command in (["build", str(source), "--output", str(output)], ["inspect", str(output)]):
                stdout = io.StringIO()
                with patch.object(sys, "argv", [str(script), *command]), redirect_stdout(stdout):
                    with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                        with self.assertRaises(SystemExit) as result:
                            runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(result.exception.code, 0)
                self.assertEqual(json.loads(stdout.getvalue())["article_count"], 3)


if __name__ == "__main__":
    unittest.main()
