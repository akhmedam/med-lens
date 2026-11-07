"""Prompt contracts; strict parsers remain the output enforcement boundary."""

MAX_CLAIMS = 3
MAX_CITATIONS = 3
MAX_CLAIM_CHARS = 1600
LIMITATIONS = {"limited_excerpts", "population_unclear", "mixed_populations",
               "no_direct_comparison", "conflicting_findings", "methodological_limits"}
LEGACY_REVIEW_VERSION = "claim-support-v4"
REVIEW_VERSION = "claim-support-v5"
REVIEW_REASONS = {"supported_by_excerpt", "population_mismatch", "comparison_mismatch",
                  "outcome_mismatch", "overstatement", "missing_details", "other"}
LEGACY_REVIEW_VALUES = {
    "verdict": ["supported", "not_supported", "unclear"],
    "population_match": ["match", "mismatch", "unclear"],
    "comparison_match": ["match", "mismatch", "unclear", "not_requested"],
    "outcome_match": ["match", "mismatch", "unclear"],
    "reason": sorted(REVIEW_REASONS),
}

REVIEW_VALUES = {**LEGACY_REVIEW_VALUES,
    'reason': sorted(REVIEW_REASONS | {'intervention_mismatch', 'qualifier_loss', 'finding_distortion', 'context_conflict'}),
    **{key: ['match', 'mismatch', 'unclear'] for key in
       ('intervention_match', 'qualifiers_preserved', 'finding_fidelity', 'context_consistency')}}


def review_values(protocol=REVIEW_VERSION):
    if protocol == REVIEW_VERSION: return REVIEW_VALUES
    if protocol == LEGACY_REVIEW_VERSION: return LEGACY_REVIEW_VALUES
    raise ValueError('Unknown support-review protocol')


def draft_contract(quote_ids):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["status", "claims", "limitations"],
        "properties": {
            "status": {"enum": ["answered", "partial", "insufficient_evidence"]},
            "claims": {"type": "array", "maxItems": MAX_CLAIMS, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "citations"], "properties": {
                    "text": {"type": "string", "minLength": 1, "maxLength": MAX_CLAIM_CHARS},
                    "citations": {"type": "array", "minItems": 1, "maxItems": MAX_CITATIONS,
                        "uniqueItems": True, "items": {"type": "string", "enum": sorted(quote_ids)}},
                }}},
            "limitations": {"type": "array", "uniqueItems": True,
                "items": {"enum": sorted(LIMITATIONS)}},
        },
        "if": {"properties": {"status": {"const": "insufficient_evidence"}}},
        "then": {"properties": {"claims": {"maxItems": 0}}},
        "else": {"properties": {"claims": {"minItems": 1}}},
    }


def review_contract(claim_ids, *, protocol=REVIEW_VERSION):
    properties = {"claim_id": {"enum": sorted(claim_ids)}}
    properties.update({key: {"enum": values} for key, values in review_values(protocol).items()})
    return {"type": "array", "minItems": len(claim_ids), "maxItems": len(claim_ids),
        "uniqueItems": True, "items": {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}}
