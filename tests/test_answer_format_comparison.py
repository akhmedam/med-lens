"""Complete denominators and matching experimental identities are required."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from medlens.evaluation.answer_formats import compare_formats, summarize_formats


def report():
    return {"schema_version": 1, "status": "complete", "context_report_sha256": "context",
        "profile_sha256": "profile", "qrels_sha256": "qrels", "case_ids": ["p1", "p2", "empty"],
        "runtime": {"model_id": "Qwen/Qwen3.5-9B", "revision": "pinned",
            "quantization": {"bits": 4, "group_size": 64}, "packages": {"torch": "pinned"}},
        "per_case": [
            {"case_id": "p1", "kind": "pilot", "status": "partial", "claim_count": 1,
             "stages": [{"stage": "draft", "max_new_tokens": 768}]},
            {"case_id": "p2", "kind": "pilot", "status": "validation_failed", "claim_count": 0,
             "reason": "Invalid JSON output", "stages": [{"stage": "draft", "max_new_tokens": 768}]},
            {"case_id": "empty", "kind": "control", "status": "insufficient_evidence",
             "claim_count": 0, "stages": []}]}


class FormatComparisonTests(unittest.TestCase):
    def test_controls_are_separate_and_validation_failures_stay_in_denominator(self):
        summary = summarize_formats(report())
        self.assertEqual(summary["pilot"]["case_count"], 2)
        self.assertEqual(summary["pilot"]["visible_draft_rate"], .5)
        self.assertEqual(summary["pilot"]["validation_failure_rate"], .5)
        self.assertEqual(summary["control"]["statuses"]["insufficient_evidence"], 1)

    def test_same_contexts_allow_paired_format_comparison(self):
        baseline, candidate = report(), report()
        candidate["per_case"][1].update(status="partial", claim_count=1)
        result = compare_formats(baseline, candidate)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["pilot_visible_draft_rate_delta"], .5)
        self.assertEqual(result["pilot_status_transitions"]["validation_failed -> partial"], 1)

    def test_changed_or_missing_identity_disables_improvement_delta(self):
        for key in ("context_report_sha256", "profile_sha256", "qrels_sha256"):
            for value in ("different", None):
                candidate = report()
                candidate[key] = value
                result = compare_formats(report(), candidate)
                self.assertFalse(result["comparable"])
                self.assertIn(key, result["identity_differences"])
                self.assertIsNone(result["pilot_visible_draft_rate_delta"])
                self.assertIsNone(result["pilot_status_transitions"])

    def test_unjudged_rebuilt_scope_requires_explicit_matching_policy(self):
        a, b = report(), report()
        for row in (a, b):
            row.update(resource_scope="rebuilt-pilot", relevance_judgments_bound=False, qrels_sha256=None)
        self.assertTrue(compare_formats(a, b)["comparable"])
        for change in ({"resource_scope": "original-frozen-pilot"},
                       {"relevance_judgments_bound": True}, {"relevance_judgments_bound": 0},
                       {"qrels_sha256": "unexpected"}):
            candidate = copy.deepcopy(b); candidate.update(change)
            self.assertFalse(compare_formats(a, candidate)["comparable"])
        for row in (a, b):
            row.pop("resource_scope"); row.pop("relevance_judgments_bound")
        self.assertFalse(compare_formats(a, b)["comparable"])

    def test_model_or_budget_changes_are_separate_factors(self):
        for key in ("model_id", "revision", "quantization", "packages"):
            candidate = report()
            candidate["runtime"][key] = "changed"
            self.assertFalse(compare_formats(report(), candidate)["comparable"])
        candidate = report()
        candidate["per_case"][0]["stages"][0]["max_new_tokens"] = 1024
        self.assertFalse(compare_formats(report(), candidate)["comparable"])

    def test_missing_draft_limit_or_changed_review_limit_prevents_pairing(self):
        candidate = report()
        candidate["per_case"][0]["stages"][0].pop("max_new_tokens")
        self.assertFalse(compare_formats(report(), candidate)["comparable"])
        baseline, candidate = report(), report()
        baseline["per_case"][0]["stages"].append({"stage": "support-review", "max_new_tokens": 512})
        candidate["per_case"][0]["stages"].append({"stage": "support-review", "max_new_tokens": 1024})
        self.assertIn("review_output_budgets", compare_formats(baseline, candidate)["identity_differences"])

    def test_incomplete_duplicate_heldout_or_inconsistent_cases_rejected(self):
        for change in (
            lambda r: r.update(status="running"),
            lambda r: r.update(heldout_used=True),
            lambda r: r["per_case"][1].update(case_id="p1"),
            lambda r: r["per_case"][1].update(kind="test"),
            lambda r: r["per_case"][0].update(claim_count=0),
            lambda r: r["per_case"][0].update(claim_count=True),
            lambda r: r["per_case"][0].update(status="failed"),
            lambda r: r["per_case"].pop(),
        ):
            candidate = copy.deepcopy(report())
            change(candidate)
            with self.assertRaises(ValueError):
                compare_formats(report(), candidate)

    def test_mismatched_case_sets_are_not_paired(self):
        candidate = report()
        candidate["per_case"][0]["case_id"] = "other"
        candidate["case_ids"][0] = "other"
        self.assertFalse(compare_formats(report(), candidate)["comparable"])

    def test_cli_records_input_hashes_and_refuses_output_reuse(self):
        path = Path(__file__).resolve().parents[1] / "scripts/compare_answer_formats.py"
        spec = importlib.util.spec_from_file_location("compare_formats_cli", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "summary.json", root / "comparison.json"
            source.write_text(json.dumps(report()))
            args = [str(source), str(source), "--output", str(output)]
            with patch("builtins.print"):
                self.assertEqual(cli.main(args), 0)
            original = output.read_bytes()
            result = json.loads(original)
            self.assertEqual(result["baseline_sha256"], result["candidate_sha256"])
            with self.assertRaises(SystemExit):
                cli.main(args)
            self.assertEqual(output.read_bytes(), original)
