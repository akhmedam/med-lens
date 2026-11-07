"""Request-specific output contracts and fail-closed format boundaries."""

import copy
import json
import unittest

import test_answers as helpers
from medlens.generation.answers import (AnswerError, chat_prompt, generate_answer,
    parse_draft, parse_review, strict_json)
from medlens.generation.contracts import draft_contract, review_contract
from medlens.generation.quotes import quote_catalog


def supported(claim_id="C1"):
    return {"claim_id": claim_id, "verdict": "supported", "intervention_match":"match", "qualifiers_preserved":"match", "finding_fidelity":"match", "context_consistency":"match", "population_match": "match",
        "comparison_match": "not_requested", "outcome_match": "match",
        "reason": "supported_by_excerpt"}


class AnswerContractTests(unittest.TestCase):
    def test_protocols_are_explicit_and_resolve_identical_source_spans(self):
        context = helpers.context()
        catalog = quote_catalog(context, helpers.Tokenizer())["catalog"]
        ids = helpers.draft()
        objects = copy.deepcopy(ids)
        objects["claims"][0]["citations"] = [{"quote_id": "E1.Q1"}]
        a = parse_draft(json.dumps(ids), context, catalog=catalog, citation_format="strings")
        b = parse_draft(json.dumps(objects), context, catalog=catalog)
        self.assertEqual(a["claims"], b["claims"])
        for value, protocol in ((ids, "objects"), (objects, "strings")):
            with self.assertRaises(AnswerError):
                parse_draft(json.dumps(value), context, catalog=catalog, citation_format=protocol)

    def test_string_protocol_still_rejects_unknown_ids_and_extra_fields(self):
        context = helpers.context()
        catalog = quote_catalog(context, helpers.Tokenizer())["catalog"]
        for modify in (
            lambda d: d["claims"][0].update(citations=["E99.Q1"]),
            lambda d: d["claims"][0].update(citations=["E1.Q1", "E1.Q1"]),
            lambda d: d["claims"][0].update(limitations=[]),
            lambda d: d.pop("limitations"),
            lambda d: d.update(summary="Unchecked text"),
        ):
            draft = helpers.draft()
            modify(draft)
            with self.assertRaises(AnswerError):
                parse_draft(json.dumps(draft), context, catalog=catalog, citation_format="strings")

    def test_draft_schema_lists_only_current_quote_ids(self):
        schema = draft_contract(["E2.Q3", "E1.Q1"])
        claims = schema["properties"]["claims"]
        self.assertEqual(claims["maxItems"], 3)
        self.assertEqual(claims["items"]["properties"]["citations"]["maxItems"], 3)
        self.assertEqual(claims["items"]["properties"]["citations"]["items"]["enum"], ["E1.Q1", "E2.Q3"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["then"]["properties"]["claims"]["maxItems"], 0)
        self.assertEqual(schema["else"]["properties"]["claims"]["minItems"], 1)

    def test_prompt_contains_live_contracts_without_changing_output_budgets(self):
        draft = helpers.draft()
        draft["claims"].append({"text": "Another finding.", "citations": ["E1.Q1"]})
        runtime = helpers.FakeRuntime([draft, [supported(), supported("C2")]])
        result = generate_answer("Question", helpers.context(), runtime)
        self.assertEqual(result["status"], "partial")
        self.assertEqual([limit for _, limit in runtime.calls], [768, 512])
        schemas = [json.loads(prompt["messages"][0]["content"].split(
            "Output contract (JSON Schema):\n", 1)[1]) for prompt, _ in runtime.calls]
        self.assertEqual(schemas[0], draft_contract(["E1.Q1"]))
        self.assertEqual(schemas[1], review_contract({"C1", "C2"}))

    def test_claim_limit_is_enforced_without_trimming(self):
        draft = helpers.legacy_draft()
        draft["claims"] *= 3
        self.assertEqual(len(parse_draft(json.dumps(draft), helpers.context())["claims"]), 3)
        draft["claims"].append(copy.deepcopy(draft["claims"][0]))
        with self.assertRaises(AnswerError) as caught:
            parse_draft(json.dumps(draft), helpers.context())
        self.assertEqual(caught.exception.code, "claim_count")
        runtime = helpers.FakeRuntime([draft])
        result = generate_answer("Question", helpers.context(), runtime)
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["validation_error"], {"stage": "draft", "code": "claim_count"})
        self.assertEqual(len(runtime.calls), 1)

    def test_citation_count_and_duplicate_quote_ids_remain_errors(self):
        context = helpers.context()
        catalog = quote_catalog(context, helpers.Tokenizer())["catalog"]
        for count, code in ((0, "citation_count"), (2, "duplicate_citation"), (4, "citation_count")):
            draft = helpers.draft()
            draft["claims"][0]["citations"] *= count
            with self.subTest(count=count), self.assertRaises(AnswerError) as caught:
                parse_draft(json.dumps(draft), context, catalog=catalog, citation_format="strings")
            self.assertEqual(caught.exception.code, code)

    def test_invalid_review_enum_reports_review_stage_and_withholds_everything(self):
        review = supported()
        review["comparison_match"] = "not_applicable"
        result = generate_answer("Question", helpers.context(), helpers.FakeRuntime([
            helpers.draft(), [review]]))
        self.assertEqual(result["status"], "validation_failed")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["validation_error"], {"stage": "support-review", "code": "review_enum"})

    def test_json_errors_are_distinct_and_never_repaired(self):
        for text, code in (("not JSON", "invalid_json"), ('{"a":1,"a":2}', "duplicate_json_key"),
                           ('{"a":NaN}', "nonfinite_json"), ("[" * 2000, "invalid_json")):
            with self.subTest(code=code), self.assertRaises(AnswerError) as caught:
                strict_json(text)
            self.assertEqual(caught.exception.code, code)

    def test_review_schema_does_not_replace_coverage_or_enum_validation(self):
        self.assertEqual(review_contract({"C1", "C2"})["items"]["properties"][
            "claim_id"]["enum"], ["C1", "C2"])
        for rows in ([supported()], [supported(), supported()],
                     [supported(), supported("C3")],
                     [supported(), {**supported("C2"), "intervention_match":"match", "qualifiers_preserved":"match", "finding_fidelity":"match", "context_consistency":"match", "population_match": []}]):
            with self.subTest(rows=rows), self.assertRaises(AnswerError):
                parse_review(json.dumps(rows), {"C1", "C2"})

    def test_full_prompt_budget_is_unchanged_and_not_truncated(self):
        class SizedTokenizer:
            size = 6500
            def apply_chat_template(self, messages, **kwargs):
                return range(self.size)
        tokenizer = SizedTokenizer()
        self.assertEqual(chat_prompt("System", "Question", tokenizer)["input_tokens"], 6500)
        tokenizer.size = 6501
        with self.assertRaises(AnswerError) as caught:
            chat_prompt("System", "Question", tokenizer)
        self.assertEqual(caught.exception.code, "input_limit")

    def test_semantic_axis_rejections_still_override_supported_verdict(self):
        for axis in ("population_match", "comparison_match", "outcome_match"):
            for value in ("mismatch", "unclear"):
                result = generate_answer("Question", helpers.context(), helpers.FakeRuntime([
                    helpers.draft(), [{**supported(), axis: value}]]))
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["claims"], [])
