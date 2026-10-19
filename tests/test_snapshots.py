"""Snapshot integrity, incomplete runs, and offline replay."""

import io
import json
import runpy
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from medlens.ingestion.pubmed import PubMedError
from medlens.ingestion.snapshots import SnapshotError, download_snapshot, load_snapshot
from pubmed_helpers import make_client, search_json, xml_records


class SnapshotTests(unittest.TestCase):
    def test_round_trip_batches_preserves_search_order_and_needs_no_http(self):
        client, _, transport = make_client([search_json(), xml_records(["12", "31"]), xml_records(["25"])])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            manifest = download_snapshot(client, "depression", directory, limit=3, batch_size=2)
            with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                loaded = load_snapshot(directory)
            self.assertEqual(loaded.manifest, manifest)
            self.assertEqual(manifest["pmids"], ["31", "12", "25"])
            self.assertEqual(len(transport.calls), 3)
            self.assertEqual(loaded.responses["search.json"], search_json())
            self.assertEqual(len(manifest["responses"]), 3)

    def test_empty_search_does_not_fetch(self):
        client, _, transport = make_client([search_json((), 0)])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            download_snapshot(client, "no matches", directory)
            self.assertEqual(load_snapshot(directory).manifest["pmids"], [])
            self.assertEqual(len(transport.calls), 1)

    def test_keeps_raw_data_and_marks_partial_fetch_as_failed(self):
        client, _, _ = make_client([search_json(), xml_records(["31", "12"]), xml_records([])])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            with self.assertRaises(PubMedError):
                download_snapshot(client, "depression", directory, limit=3, batch_size=2)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertTrue((directory / "fetch-0002.xml").is_file())
            with self.assertRaises(SnapshotError):
                load_snapshot(directory)

    def test_keeps_malformed_search_response(self):
        client, _, _ = make_client([b"<html>temporary problem</html>"])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            with self.assertRaises(PubMedError):
                download_snapshot(client, "depression", directory)
            self.assertEqual((directory / "search.json").read_bytes(), b"<html>temporary problem</html>")
            self.assertEqual(json.loads((directory / "manifest.json").read_text())["status"], "failed")

    def test_rejects_corrupted_response_and_reordered_manifest(self):
        client, _, _ = make_client([search_json(), xml_records(["31", "12", "25"])])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            download_snapshot(client, "depression", directory, limit=3)
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["pmids"].reverse()
            path.write_text(json.dumps(manifest))
            with self.assertRaises(SnapshotError):
                load_snapshot(directory)
            manifest["pmids"].reverse()
            path.write_text(json.dumps(manifest))
            (directory / "fetch-0001.xml").write_bytes(b"changed")
            with self.assertRaisesRegex(SnapshotError, "checksum"):
                load_snapshot(directory)

    def test_refuses_to_overwrite_existing_snapshot(self):
        client, _, transport = make_client([])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                download_snapshot(client, "depression", Path(temp))
            self.assertEqual(transport.calls, [])

    def test_transport_failure_keeps_completed_batch(self):
        client, _, _ = make_client([search_json(), xml_records(["31", "12"]),
                                   URLError("fixture"), URLError("fixture"), URLError("fixture")])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            with self.assertRaises(PubMedError):
                download_snapshot(client, "depression", directory, limit=3, batch_size=2)
            self.assertTrue((directory / "fetch-0001.xml").is_file())
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len(manifest["responses"]), 2)
            with self.assertRaises(SnapshotError):
                load_snapshot(directory)

    def test_inspect_cli_runs_offline(self):
        client, _, _ = make_client([search_json(), xml_records(["31", "12", "25"])])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "snapshot"
            download_snapshot(client, "depression", directory, limit=3)
            stdout = io.StringIO()
            script = Path(__file__).resolve().parents[1] / "scripts/pubmed_snapshot.py"
            with patch.object(sys, "argv", [str(script), "inspect", str(directory)]), redirect_stdout(stdout):
                with patch("socket.socket", side_effect=AssertionError("offline")), patch("socket.getaddrinfo", side_effect=AssertionError("offline")):
                    with self.assertRaises(SystemExit) as exit_info:
                        runpy.run_path(str(script), run_name="__main__")
            self.assertEqual(exit_info.exception.code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["pmids"], ["31", "12", "25"])


if __name__ == "__main__":
    unittest.main()
