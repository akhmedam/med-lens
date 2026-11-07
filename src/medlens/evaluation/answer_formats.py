"""Compare complete pilot format summaries without loading sources or models."""

from collections import Counter

STATUSES = {"answered", "partial", "insufficient_evidence", "validation_failed"}


def summarize_formats(report):
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("status") != "complete" or report.get("heldout_used", False) is not False):
        raise ValueError("Require a complete version-1 generation summary")
    rows = report.get("per_case")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Require nonempty per-case results")
    seen, groups = set(), {"pilot": [], "control": []}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid case")
        case_id, kind = row.get("case_id"), row.get("kind")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("Missing or duplicate case ID")
        seen.add(case_id)
        status, count = row.get("status"), row.get("claim_count")
        if not isinstance(kind, str) or kind not in groups:
            raise ValueError("Only pilot cases and explicit controls are permitted")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError("Unknown or incomplete answer status")
        if (type(count) is not int or count < 0
                or (status in {"answered", "partial"}) != (count > 0)):
            raise ValueError("Visible claim count disagrees with answer status")
        if status == "validation_failed" and not isinstance(row.get("reason", "unspecified"), str):
            raise ValueError("Failure reason must be text")
        groups[kind].append(row)
    if report.get("case_ids") != [row["case_id"] for row in rows]:
        raise ValueError("Case membership/order differs from the run identity")
    if not groups["pilot"]:
        raise ValueError("Require at least one development pilot case")
    result = {}
    for kind, cases in groups.items():
        counts = Counter(row["status"] for row in cases)
        failures = Counter(row.get("reason", "unspecified") for row in cases
                           if row["status"] == "validation_failed")
        visible = counts["answered"] + counts["partial"]
        result[kind] = {
            "case_count": len(cases), "statuses": {s: counts[s] for s in sorted(STATUSES)},
            "visible_drafts": visible,
            "visible_draft_rate": visible / len(cases) if cases else None,
            "validation_failure_rate": counts["validation_failed"] / len(cases) if cases else None,
            "failure_reasons": dict(sorted(failures.items())),
        }
    return result


def compare_formats(baseline, candidate):
    before, after = summarize_formats(baseline), summarize_formats(candidate)
    differences = []
    def scope(report):
        return (report.get("resource_scope", "original-frozen-pilot"),
                report.get("relevance_judgments_bound", True))
    if scope(baseline) != scope(candidate):
        differences.append("resource_scope_or_judgment_policy")
    unjudged = all(r.get("resource_scope") == "rebuilt-pilot"
                   and r.get("relevance_judgments_bound") is False
                   and r.get("qrels_sha256") is None for r in (baseline, candidate))
    for key in ("context_report_sha256", "profile_sha256", "qrels_sha256"):
        if key == "qrels_sha256" and unjudged:
            continue
        if not baseline.get(key) or baseline.get(key) != candidate.get(key):
            differences.append(key)
    if [(r["case_id"], r["kind"]) for r in baseline["per_case"]] != [
            (r["case_id"], r["kind"]) for r in candidate["per_case"]]:
        differences.append("case_membership_or_order")
    for report in (baseline, candidate):
        if not isinstance(report.get("runtime", {}), dict):
            raise ValueError("Runtime metadata must be an object")
    for key in ("model_id", "revision", "quantization", "packages"):
        old, new = baseline.get("runtime", {}).get(key), candidate.get("runtime", {}).get(key)
        if not old or old != new:
            differences.append("runtime." + key)
    # A changed output budget is a separate experimental factor, not a prompt-only comparison.
    def limits(report):
        return [(row["case_id"], call.get("stage"), call.get("max_new_tokens"))
                for row in report["per_case"] for call in row.get("stages", [])
                if call.get("stage") == "draft"]
    old_limits, new_limits = limits(baseline), limits(candidate)
    if (not old_limits or any(type(limit) is not int or limit <= 0
                             for _, _, limit in old_limits + new_limits)
            or old_limits != new_limits):
        differences.append("draft_output_budgets")
    def review_limits(report):
        return {row["case_id"]: call.get("max_new_tokens") for row in report["per_case"]
                for call in row.get("stages", []) if call.get("stage") == "support-review"}
    old_review, new_review = review_limits(baseline), review_limits(candidate)
    if any(type(value) is not int or value <= 0 for value in [*old_review.values(), *new_review.values()]) or any(
            old_review[key] != new_review[key] for key in old_review.keys() & new_review.keys()):
        differences.append("review_output_budgets")
    comparable = not differences
    transitions = None
    delta = None
    if comparable:
        transitions = dict(sorted(Counter(
            a["status"] + " -> " + b["status"]
            for a, b in zip(baseline["per_case"], candidate["per_case"])
            if a["kind"] == "pilot").items()))
        delta = after["pilot"]["visible_draft_rate"] - before["pilot"]["visible_draft_rate"]
    return {"schema_version": 1, "status": "complete", "inference_run": False,
        "heldout_used": False, "comparable": comparable, "identity_differences": differences,
        "baseline": before, "candidate": after, "pilot_status_transitions": transitions,
        "pilot_visible_draft_rate_delta": delta,
        "interpretation": "Format outcomes only, not semantic accuracy. Different contexts, model, "
            "package versions or output budgets prevent a controlled format comparison."}
