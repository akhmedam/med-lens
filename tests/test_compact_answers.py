"""Compact 4B keeps exact evidence and fails closed on format/consistency/cache errors."""
import copy
from itertools import product
import json
import unittest

from medlens.generation.answers import (AnswerError, accepted_claim_ids, generate_answer,
    parse_draft, render_answer)
from medlens.generation import compact
from medlens.generation.contracts import LEGACY_REVIEW_VERSION, REVIEW_VERSION, draft_contract, review_values
from medlens.generation.reviews import review_messages
import test_answers as fixtures


def supported(**overrides):
    row = {"claim_id": "C1", "population_match": "match", "comparison_match": "not_requested",
        "outcome_match": "match", "verdict": "supported", "reason": "supported_by_excerpt"}
    return row | overrides


class CompactRuntime(fixtures.FakeRuntime):
    answer_policy = "compact-v1"
    def __init__(self, outputs):
        super().__init__(outputs)
        self.cache_identity = {"model_id": "community/pinned4b", "revision": "fixed-revision",
            "artifact_id": "fixed-artifact", "answer_policy": self.answer_policy}
    def generate(self, prompt, *, max_new_tokens):
        return super().generate(prompt, max_new_tokens=max_new_tokens) | {
            "runtime_identity": dict(self.cache_identity)}


def matches(schema, value):
    if "oneOf" in schema:return sum(matches(b, value) for b in schema["oneOf"]) == 1
    if "const" in schema:return type(value) is type(schema["const"]) and value == schema["const"]
    if "enum" in schema:return value in schema["enum"]
    if schema["type"] == "object":
        return isinstance(value, dict) and set(value) == set(schema["required"]) and all(
            matches(schema["properties"][k], v) for k, v in value.items())
    if schema["type"] == "array":
        return isinstance(value, list) and len(value) == len(schema["prefixItems"]) and all(
            matches(s, v) for s, v in zip(schema["prefixItems"], value))
    raise AssertionError("Unexpected finite review schema")


class CompactAnswerTests(unittest.TestCase):
    def test_empty_context_skips_all_generation(self):
        runtime = CompactRuntime([])
        result = generate_answer("Question", fixtures.context(True), runtime)
        self.assertEqual(result["reason"], "empty-context")
        self.assertEqual(result["prompt_version"], "cited-answer-v7")
        self.assertEqual(runtime.calls, [])

    def test_valid_named_draft_review_preserve_exact_evidence_schema_and_caps(self):
        context = fixtures.context();before = copy.deepcopy(context)
        runtime = CompactRuntime([fixtures.draft(), [supported()]])
        result = generate_answer("Question", context, runtime)
        self.assertEqual(context, before)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["review_version"], LEGACY_REVIEW_VERSION)
        self.assertEqual(result["review_prompt_version"], "axes-first-v1")
        self.assertEqual(result["support_check"], "same-model-review-not-independent")
        self.assertEqual([cap for _, cap in runtime.calls], [768, 512])
        first, second = [prompt for prompt, _ in runtime.calls]
        self.assertEqual(first["constraint_kind"], "draft")
        self.assertEqual(first["decoding_schema"], draft_contract(["E1.Q1"]))
        self.assertIn("if", first["decoding_schema"])
        self.assertTrue(first["decoding_schema"]["properties"]["claims"]["items"]["properties"]["citations"]["uniqueItems"])
        prompt, catalog = compact.build_draft_prompt("Question", context, runtime.tokenizer)
        self.assertEqual(first, prompt)
        expected = review_messages("Question", context, result["draft"]["claims"], catalog, protocol=LEGACY_REVIEW_VERSION)[1]
        self.assertEqual(second["messages"][1]["content"], expected)
        self.assertEqual(second["constraint_kind"], "literal")
        for branch in second["decoding_schema"]["prefixItems"][0]["oneOf"]:
            self.assertEqual(branch["type"], "object")
            self.assertEqual(branch["required"], list(compact.FIELD_ORDER))
        citation = result["claims"][0]["citations"][0]
        self.assertEqual(citation["quote"], "alpha beta gamma")
        self.assertEqual(citation["matches"], [{"char_start": 0, "char_end": 16}])
        self.assertIn("requires independent review", render_answer(result))

    def test_contradictory_support_or_reason_withholds_whole_answer_without_repair(self):
        for row in (supported(population_match="mismatch"), supported(outcome_match="unclear"),
                    supported(reason="missing_details"),
                    supported(verdict="not_supported", reason="population_mismatch")):
            runtime = CompactRuntime([fixtures.draft(), [row]])
            result = generate_answer("Question", fixtures.context(), runtime)
            self.assertEqual(result["status"], "validation_failed")
            self.assertEqual(result["validation_error"], {"stage": "support-review", "code": "review_consistency"})
            self.assertEqual(result["claims"], [])
            self.assertNotIn("A scoped observation", render_answer(result))
            self.assertEqual(json.loads(result["model_calls"][1]["text"]), [row])

    def test_passing_axes_do_not_force_factual_support_or_accept_unknown(self):
        for row in (supported(verdict="not_supported", reason="overstatement"),
                    supported(verdict="not_supported", reason="other"),
                    supported(population_match="unclear", verdict="unclear", reason="missing_details"),
                    supported(verdict="unclear", reason="missing_details")):
            result = generate_answer("Question", fixtures.context(), CompactRuntime([fixtures.draft(), [row]]))
            self.assertEqual(result["status"], "insufficient_evidence")
            self.assertEqual(result["review"], [row])
            self.assertEqual(result["claims"], [])

    def test_no_positional_arrays_defaults_or_missing_review_fields(self):
        for rows in ([["C1", "match", "not_requested", "match", "supported", "supported_by_excerpt"]],
                     [{key: value for key, value in supported().items() if key != "verdict"}],
                     [supported(claim_id="C2")]):
            result = generate_answer("Question", fixtures.context(), CompactRuntime([fixtures.draft(), rows]))
            self.assertEqual(result["status"], "validation_failed")
            self.assertEqual(result["claims"], [])

    def test_original_strict_draft_cardinality_and_exact_quote_ids_are_not_trimmed(self):
        for change, code in ((lambda d: d["claims"].__imul__(4), "claim_count"),
                             (lambda d: d["claims"][0].update(citations=["E99.Q9"]), "citation_source"),
                             (lambda d: d["claims"][0].update(citations=["E1.Q1", "E1.Q1"]), "duplicate_citation")):
            value = fixtures.draft();change(value)
            runtime = CompactRuntime([value]);result = generate_answer("Question", fixtures.context(), runtime)
            self.assertEqual(result["validation_error"]["code"], code)
            self.assertEqual(result["claims"], [])
            self.assertEqual(len(runtime.calls), 1)

    def test_cache_requires_exact_runtime_identity_and_decoder_contract(self):
        original = generate_answer("Question", fixtures.context(), CompactRuntime([fixtures.draft(), [supported()]]))
        cache = {call["stage"]: call for call in original["model_calls"]}
        bound = CompactRuntime([])
        reused = generate_answer("Question", fixtures.context(), bound, cached_calls=cache)
        self.assertEqual(bound.calls, [])
        self.assertTrue(all(c["cache_reused"] for c in reused["model_calls"]))
        for mutate in (lambda c: c.pop("runtime_identity"),
                       lambda c: c["runtime_identity"].update(model_id="other9b"),
                       lambda c: c["runtime_identity"].update(artifact_id="other-artifact"),
                       lambda c: c["prompt"].update(decoding_schema={"different": True})):
            changed = copy.deepcopy(cache)
            for call in changed.values():mutate(call)
            runtime = CompactRuntime([fixtures.draft(), [supported()]])
            result = generate_answer("Question", fixtures.context(), runtime, cached_calls=changed)
            self.assertEqual(len(runtime.calls), 2)
            self.assertTrue(all(not c["cache_reused"] for c in result["model_calls"]))

    def test_explicit_v5_is_not_silently_downgraded_for_compact(self):
        runtime = CompactRuntime([])
        with self.assertRaisesRegex(ValueError, "claim-support-v4"):
            generate_answer("Question", fixtures.context(), runtime, review_protocol=REVIEW_VERSION)
        self.assertEqual(runtime.calls, [])

    def test_legacy_fake_keeps_v6_and_legacy_review_behavior(self):
        row = supported(population_match="mismatch")
        runtime = fixtures.FakeRuntime([fixtures.draft(), [row]])
        result = generate_answer("Question", fixtures.context(), runtime)
        self.assertEqual(result["prompt_version"], "cited-answer-v6")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertNotIn("decoding_schema", runtime.calls[0][0])
        self.assertNotIn("review_prompt_version", result)

    def test_changed_catalog_or_over_budget_prompts_fail_without_truncation(self):
        context = fixtures.context();rt = CompactRuntime([])
        _, indexed = compact.build_draft_prompt("Question", context, rt.tokenizer)
        claims = parse_draft(json.dumps(fixtures.draft()), context, catalog=indexed["catalog"], citation_format="strings")["claims"]
        indexed["payload"] += "changed"
        with self.assertRaises(AnswerError):compact.build_review_prompt("Question", context, claims, indexed, rt.tokenizer)
        class Oversized(fixtures.Tokenizer):
            def apply_chat_template(self, *args, **kwargs):return range(6501)
        with self.assertRaises(AnswerError) as caught:compact.build_draft_prompt("Question", context, Oversized())
        self.assertEqual(caught.exception.code, "input_limit")

    def test_consistency_schema_is_disjoint_and_matches_postparse_for_every_finite_row(self):
        schema = compact.review_schema(["C1"])
        values = review_values(LEGACY_REVIEW_VERSION)
        valid = 0
        for combination in product(*(values[key] for key in compact.FIELD_ORDER[1:])):
            row = dict(zip(compact.FIELD_ORDER, ["C1", *combination]))
            hits = sum(matches(branch, row) for branch in schema["prefixItems"][0]["oneOf"])
            self.assertLessEqual(hits, 1)
            if hits:
                valid += 1
                self.assertEqual(compact.parse_compact_review(json.dumps([row]), {"C1"}), [row])
                if row["verdict"] != "supported":
                    self.assertEqual(accepted_claim_ids([row], protocol=LEGACY_REVIEW_VERSION), set())
            else:
                with self.assertRaises(AnswerError):compact.parse_compact_review(json.dumps([row]), {"C1"})
        self.assertEqual(valid, 131)


if __name__ == "__main__":unittest.main()
