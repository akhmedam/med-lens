"""Bounded ordered JSON decoding with fresh state and immutable compiled indexes.

Only the declared schema subset is supported. Draft length/uniqueness remain
strict final-parser checks and are explicitly disclosed as decoding relaxations.
Neither numerical scores nor generated JSON are repaired.
"""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path
import re
from time import perf_counter

import numpy as np

MAX_SCHEMA_BYTES = 32768
MAX_SCHEMA_DEPTH = 24
MAX_SCHEMA_NODES = 4096
MAX_ENUM_ITEMS = 512
MAX_COMPILED_INDEXES = 8
OUTLINES_VERSION = "0.2.14"


def check_schema_budget(schema):
    if not isinstance(schema, dict):
        raise ValueError("A schema object is required")
    try:
        payload = json.dumps(schema, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid schema JSON") from exc
    if len(payload) > MAX_SCHEMA_BYTES:
        raise ValueError("Decoding schema exceeds byte budget")
    count = 0
    def visit(value, depth):
        nonlocal count
        count += 1
        if depth > MAX_SCHEMA_DEPTH or count > MAX_SCHEMA_NODES:
            raise ValueError("Decoding schema exceeds structural budget")
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("Schema keys must be strings")
            if "enum" in value and (not isinstance(value["enum"], list)
                    or not 1 <= len(value["enum"]) <= MAX_ENUM_ITEMS):
                raise ValueError("Schema enum exceeds budget")
            for nested in value.values():
                visit(nested, depth + 1)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, depth + 1)
    visit(schema, 0)
    return schema


def _contracts():
    path = Path(__file__).resolve().parents[1] / "generation/contracts.py"
    spec = importlib.util.spec_from_file_location("_medlens_schema_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_schema(schema, kind, *, validate_regex=True):
    check_schema_budget(schema)
    if kind == "draft":
        try:
            quote_ids = schema["properties"]["claims"]["items"]["properties"]["citations"]["items"]["enum"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Unrecognized draft schema") from exc
        if (any(not isinstance(value, str) or not re.fullmatch(r"E[1-9][0-9]*\.Q[1-9][0-9]*", value)
                for value in quote_ids) or len(set(quote_ids)) != len(quote_ids)
                or schema != _contracts().draft_contract(quote_ids)):
            raise ValueError("Draft schema differs from the strict answer contract")
        lowered, metadata = lower_draft_schema(schema)
    elif kind == "review":
        try:
            claim_ids = schema["items"]["properties"]["claim_id"]["enum"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Unrecognized review schema") from exc
        contracts = _contracts()
        if (claim_ids != ["C" + str(index) for index in range(1, len(claim_ids) + 1)]
                or not 1 <= len(claim_ids) <= 3
                or schema != contracts.review_contract(claim_ids, protocol=contracts.LEGACY_REVIEW_VERSION)):
            raise ValueError("Review schema differs from the strict legacy contract")
        lowered, metadata = lower_review_schema(schema)
    elif kind == "literal":
        lowered, metadata = copy.deepcopy(schema), {"decode_relaxations": [], "strict_post_validation_required": True}
    else:
        raise ValueError("Unknown decoding constraint kind")
    check_schema_budget(lowered)
    validate_schema(lowered, allow_large_finite_strings=False)
    if validate_regex:
        required_order_regex(lowered, whitespace_pattern="")
    return lowered, {**metadata, "original_schema_sha256": sha(schema),
        "lowered_schema_sha256": sha(lowered), "kind": kind}


class ConstraintCache:
    """Each worker owns one bounded cache; only immutable indexes are reused."""
    def __init__(self, tokenizer, *, vocab_size=248320, eos=(248044, 248046), maximum=MAX_COMPILED_INDEXES):
        if type(maximum) is not int or not 1 <= maximum <= MAX_COMPILED_INDEXES:
            raise ValueError("Invalid compiled constraint cache bound")
        self.tokenizer, self.vocab_size, self.eos, self.maximum = tokenizer, vocab_size, tuple(eos), maximum
        self._indexes = OrderedDict()

    def get(self, schema, kind):
        lowered, metadata = prepare_schema(schema, kind)
        key = (metadata["original_schema_sha256"], kind)
        hit = key in self._indexes
        started = perf_counter()
        if hit:
            constraint = self._indexes.pop(key)
        else:
            if len(self._indexes) >= self.maximum:
                self._indexes.popitem(last=False)
            constraint = Constraint(lowered, self.tokenizer, vocab_size=self.vocab_size,
                eos=self.eos, preserve_required_order=True, whitespace_pattern="")
        self._indexes[key] = constraint
        return constraint, {**metadata, "cache_hit": hit,
            "compile_seconds": 0.0 if hit else perf_counter() - started,
            "compiled_cache_size": len(self._indexes), "compiled_cache_maximum": self.maximum,
            "regex_sha256": constraint.info["regex_sha256"], "whitespace_pattern": "",
            "object_property_order": "required-list"}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_finite(mx, logits):
    invalid = (mx.any(mx.isnan(logits)) | mx.any(mx.isinf(logits) & (logits > 0))
        | mx.any(mx.all(mx.isinf(logits) & (logits < 0), axis=-1)))
    if invalid.item():
        raise FloatingPointError("Invalid raw or fully masked constrained scores")
    return logits


def byte_decoder():
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {chr(value): value for value in kept}
    for offset, value in enumerate(value for value in range(256) if value not in kept):
        mapping[chr(256 + offset)] = value
    return mapping


def validate_schema(schema, *, allow_large_finite_strings):
    # These keywords are unsupported or are not intersection/cross-field
    # enforcement in outlines-core 0.2.14. Silently dropping them is unsafe.
    unsupported = {"if", "then", "else", "not", "allOf", "contains", "dependentSchemas",
        "dependentRequired", "unevaluatedProperties", "unevaluatedItems", "patternProperties"}
    def visit(value):
        if isinstance(value, dict):
            found = unsupported.intersection(value)
            if found or value.get("uniqueItems") is True:
                raise ValueError("Schema needs explicit lowering before outlines compilation: "
                    + ",".join(sorted(found | ({"uniqueItems"} if value.get("uniqueItems") else set()))))
            maximum = value.get("maxLength")
            if not allow_large_finite_strings and type(maximum) is int and maximum > 64:
                raise ValueError("Large maxLength creates a large eager token DFA; explicit research opt-in required")
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
    visit(schema)


def lower_draft_schema(schema):
    """Explicit research relaxation, retaining maxItems/enums and strict parser.

    Cross-field status/claim counts are made disjoint branches. String length
    and array uniqueness remain fail-closed post-validation, never repair.
    """
    original = copy.deepcopy(schema)
    lowered = copy.deepcopy(schema)
    for key in ("if", "then", "else"):
        lowered.pop(key, None)
    def visit(value):
        if isinstance(value, dict):
            value.pop("uniqueItems", None)
            value.pop("maxLength", None)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
    visit(lowered)
    # Uniqueness implies this original finite-enum upper bound; retaining it
    # prevents an unbounded whitespace/limitation loop after that relaxation.
    limitations = lowered["properties"]["limitations"]
    limitations["maxItems"] = len(limitations["items"]["enum"])
    branches = []
    for statuses, minimum, maximum in ((["answered", "partial"], 1, 3), (["insufficient_evidence"], 0, 0)):
        branch = copy.deepcopy(lowered)
        branch["properties"]["status"] = {"enum": statuses}
        branch["properties"]["claims"]["minItems"] = minimum
        branch["properties"]["claims"]["maxItems"] = maximum
        branches.append(branch)
    return {"oneOf": branches}, {"original_schema_sha256": sha(original),
        "decode_relaxations": ["maxLength", "uniqueItems"],
        "status_condition_lowering": "disjoint-oneOf-branches", "strict_post_validation_required": True}


def lower_review_schema(schema):
    """Require every expected claim once, in a fixed order, with all enum axes."""
    item = schema["items"]
    ids = item["properties"]["claim_id"]["enum"]
    rows = []
    for claim_id in ids:
        row = copy.deepcopy(item)
        row["properties"]["claim_id"] = {"const": claim_id}
        rows.append(row)
    return {"type": "array", "prefixItems": rows, "items": False,
        "minItems": len(ids), "maxItems": len(ids)}, {
            "original_schema_sha256": sha(schema), "decode_relaxations": [],
            "review_claim_order": list(ids), "strict_post_validation_required": True}


def required_order_regex(schema, *, whitespace_pattern):
    """Compile our explicit schema subset while retaining required-list order.

    Outlines still builds the token DFA and handles all token/byte transitions.
    This function only assembles supported JSON structure into a regular
    expression, and rejects every unhandled schema constraint.
    """
    from outlines_core.json_schema import build_regex_from_schema
    if whitespace_pattern not in ("", r"[ \n\r\t]*"):
        raise ValueError("Only compact JSON or explicit JSON whitespace is supported")
    ws = whitespace_pattern

    def literal(value):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        return "".join("\\" + character if character in r"\.+*?()|[]{}^$" else character for character in text)

    def check_keys(value, allowed):
        unknown = set(value) - set(allowed)
        if unknown:
            raise ValueError("Unsupported schema keys in ordered compiler: " + ",".join(sorted(unknown)))

    def integer(value, name, default=None):
        item = value.get(name, default)
        if item is not None and (type(item) is not int or item < 0):
            raise ValueError("Invalid array bound: " + name)
        return item

    def disjoint(left, right):
        def choices(value):
            if "const" in value:
                return {canonical(value["const"])}
            if "enum" in value:
                return {canonical(item) for item in value["enum"]}
            return None
        first, second = choices(left), choices(right)
        if first is not None and second is not None:
            return first.isdisjoint(second)
        if left.get("type") == right.get("type") == "object":
            a, b = left.get("properties", {}), right.get("properties", {})
            required = set(left.get("required", [])) & set(right.get("required", []))
            return any(key in a and key in b and disjoint(a[key], b[key]) for key in required)
        return False

    def compile_value(value):
        if not isinstance(value, dict) or not value:
            raise ValueError("A nonempty explicit schema object is required")
        for union in ("oneOf", "anyOf"):
            if union in value:
                check_keys(value, {union})
                branches = value[union]
                if not isinstance(branches, list) or not branches:
                    raise ValueError("A nonempty union is required")
                if union == "oneOf" and any(not disjoint(left, right)
                        for index, left in enumerate(branches) for right in branches[index + 1:]):
                    raise ValueError("oneOf branches must have provably disjoint enum/const fields")
                return "(?:" + "|".join(compile_value(branch) for branch in branches) + ")"
        if value.get("type") == "object" or "properties" in value:
            check_keys(value, {"type", "properties", "required", "additionalProperties"})
            properties, required = value.get("properties"), value.get("required")
            if (value.get("type", "object") != "object" or value.get("additionalProperties") is not False
                    or not isinstance(properties, dict) or not isinstance(required, list)
                    or any(not isinstance(key, str) for key in required)
                    or len(set(required)) != len(required) or set(required) != set(properties)):
                raise ValueError("Ordered objects require every declared property exactly once")
            pairs = [literal(key) + ws + ":" + ws + compile_value(properties[key]) for key in required]
            return r"\{" + ws + (ws + "," + ws).join(pairs) + ws + r"\}"
        if value.get("type") == "array" or "prefixItems" in value:
            check_keys(value, {"type", "prefixItems", "items", "minItems", "maxItems"})
            if value.get("type", "array") != "array":
                raise ValueError("Invalid array type")
            minimum = integer(value, "minItems", 0)
            maximum = integer(value, "maxItems")
            if maximum is None or maximum > 32:
                raise ValueError("Arrays require an explicit maximum of at most 32 items")
            if maximum is not None and maximum < minimum:
                raise ValueError("Inconsistent array item bounds")
            if "prefixItems" in value:
                rows = value["prefixItems"]
                if (not isinstance(rows, list) or value.get("items") is not False
                        or minimum != len(rows) or maximum != len(rows)):
                    raise ValueError("prefixItems requires an exact bounded tuple with items=false")
                inner = (ws + "," + ws).join(compile_value(row) for row in rows)
            elif maximum == 0:
                # Validate the declared child schema even though no item is legal.
                compile_value(value["items"])
                inner = ""
            else:
                item = "(?:" + compile_value(value.get("items")) + ")"
                upper = "" if maximum is None else str(maximum - 1)
                lower = str(max(0, minimum - 1))
                inner = item + "(?:" + ws + "," + ws + item + "){" + lower + "," + upper + "}"
                if minimum == 0:
                    inner = "(?:" + inner + ")?"
            return r"\[" + ws + inner + ws + r"\]"
        if "enum" in value or "const" in value:
            check_keys(value, {"type", "enum", "const"})
            if ("enum" in value) == ("const" in value):
                raise ValueError("Exactly one enum/const constraint is required")
            choices = value.get("enum") if "enum" in value else [value["const"]]
            if not isinstance(choices, list) or not choices or any(isinstance(choice, (dict, list)) for choice in choices):
                raise ValueError("Only a nonempty finite set of primitive enum values is supported")
            expected = value.get("type")
            for choice in choices:
                actual = ("null" if choice is None else "boolean" if type(choice) is bool else
                    "integer" if type(choice) is int else "number" if type(choice) is float else
                    "string" if isinstance(choice, str) else None)
                if actual is None or (expected is not None and actual != expected
                        and not (expected == "number" and actual == "integer")):
                    raise ValueError("Enum value differs from its declared type")
            return "(?:" + "|".join(literal(choice) for choice in choices) + ")"
        if value.get("type") == "string":
            check_keys(value, {"type", "minLength", "maxLength"})
            minimum = integer(value, "minLength", 0)
            maximum = integer(value, "maxLength")
            if maximum is not None and maximum < minimum:
                raise ValueError("Inconsistent string character bounds")
        elif value.get("type") in {"boolean", "null", "number", "integer"}:
            check_keys(value, {"type"})
        else:
            raise ValueError("Unsupported explicit schema type")
        return build_regex_from_schema(json.dumps(value), whitespace_pattern=ws)

    return compile_value(schema)


class Constraint:
    def __init__(self, schema, hf_tokenizer, vocab_size=248320, eos=(248044, 248046), *,
                 allow_large_finite_strings=False, allow_unicode_escapes=True,
                 preserve_required_order=True, whitespace_pattern=r"[ \n\r\t]*"):
        from outlines_core import Index, Vocabulary
        from outlines_core.json_schema import build_regex_from_schema, STRING_INNER
        if importlib.metadata.version("outlines-core") != OUTLINES_VERSION:
            raise RuntimeError("The pinned outlines-core version is required")
        check_schema_budget(schema)
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("Invalid vocabulary size")
        if (not isinstance(eos, (tuple, list)) or not eos
                or any(type(token) is not int or not 0 <= token < vocab_size for token in eos)
                or len(set(eos)) != len(eos)):
            raise ValueError("Invalid EOS union")
        self.schema = copy.deepcopy(schema)
        validate_schema(self.schema, allow_large_finite_strings=allow_large_finite_strings)
        if type(preserve_required_order) is not bool:
            raise ValueError("preserve_required_order must be boolean")
        ordered = required_order_regex(self.schema, whitespace_pattern=whitespace_pattern)
        started = perf_counter()
        backend = json.loads(hf_tokenizer.backend_tokenizer.to_str())
        if backend.get("decoder", {}).get("type") != "ByteLevel" or backend.get("model", {}).get("type") != "BPE":
            raise ValueError("Only explicit ByteLevel BPE tokenizer mapping is supported")
        decoder = byte_decoder()
        actual_vocab = hf_tokenizer.get_vocab()
        base_vocab = backend["model"]["vocab"]
        token_bytes = {}
        token_count = 0
        for token, token_id in base_vocab.items():
            if actual_vocab.get(token) != token_id or type(token_id) is not int or not 0 <= token_id < vocab_size:
                raise ValueError("Tokenizer vocabulary identity changed")
            if token_id in eos or token_id in hf_tokenizer.all_special_ids:
                continue
            try:
                raw = bytes(decoder[character] for character in token)
            except KeyError as exc:
                raise ValueError("Unexpected ByteLevel vocabulary character") from exc
            if not raw:
                raise ValueError("Empty lexical token is not supported")
            token_bytes.setdefault(raw, []).append(token_id)
            token_count += 1
        self.vocab_size, self.eos = vocab_size, tuple(eos)
        self.canonical_eos = self.eos[-1]
        self.vocabulary = Vocabulary(self.canonical_eos, token_bytes)
        self.regex = ordered if preserve_required_order else build_regex_from_schema(
            json.dumps(self.schema), whitespace_pattern=whitespace_pattern)
        # Upstream STRING_INNER omits \u escapes. Permit scalar Unicode escapes,
        # including paired surrogates; deliberately exclude unpaired surrogates.
        # One surrogate pair is one character for minLength/maxLength counting.
        if allow_unicode_escapes:
            scalar = (r'(?:[^"\\\x00-\x1F]|\\["\\/bfnrt]|'
                r'\\u(?:[0-9a-cA-Ce-fE-F][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2})|'
                r'\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})')
            self.regex = self.regex.replace(STRING_INNER, scalar)
        vocabulary_done = perf_counter()
        self.index = Index(self.regex, self.vocabulary)
        self.info = {"library": "outlines-core", "version": importlib.metadata.version("outlines-core"),
            "schema_sha256": sha(self.schema), "regex_sha256": hashlib.sha256(self.regex.encode()).hexdigest(),
            "tokenizer_backend_sha256": hashlib.sha256(hf_tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "vocab_size": vocab_size, "lexical_token_count": token_count,
            "eos_token_ids": list(eos), "blocked_tokens": "all-added-non-EOS-and-unmapped",
            "unicode_escape_policy": "unicode-scalars-and-paired-surrogates" if allow_unicode_escapes else "upstream",
            "schema_scope": "explicit-supported-subset-not-general-JSON-Schema",
            "object_property_order": "required-list" if preserve_required_order else "upstream-properties-order",
            "preserve_required_order": preserve_required_order, "whitespace_pattern": whitespace_pattern,
            "vocabulary_seconds": vocabulary_done - started,
            "index_seconds": perf_counter() - vocabulary_done,
            "compile_seconds": perf_counter() - started, "regex_bytes": len(self.regex.encode()),
            "semantic_quality_evaluated": False, "posthoc_repair": False}

    def processor(self, mx):
        from outlines_core import Guide
        if mx is np:
            def apply(logits, mask):
                words = mask.view(np.uint32)[0]
                ids = np.arange(self.vocab_size, dtype=np.uint32)
                allowed = ((words[ids >> 5] >> (ids & 31)) & 1).astype(bool)
                return np.where(allowed[None, :], logits, -np.inf)
        else:
            # This is the only native MLX import route; CPU tests never take it.
            from outlines_core.kernels.mlx import apply_token_bitmask as apply
        owner = self
        class Processor:
            def __init__(self):
                self.guide = Guide(owner.index)
                self.mask = np.zeros((1, (owner.vocab_size + 31) // 32), dtype=np.int32)
                self.prompt_length = None
                self.processed = 0
                self.accepted_ids = []
                self.stopped = False
                self.stats = {"processor_calls": 0, "accepted_generated_tokens": 0,
                    "mask_seconds": 0.0, "advance_seconds": 0.0, "raw_validation_seconds": 0.0,
                    "post_mask_validation_seconds": 0.0, "accepting": False, "eos_accepted": False,
                    "prompt_tokens": None, "fresh_guide": True}

            def _set_eos(self):
                for token in owner.eos:
                    word, bit = token >> 5, token & 31
                    view = self.mask.view(np.uint32)
                    view[0, word] |= np.uint32(1 << bit)

            def _advance(self, ids):
                if len(ids.shape) != 1 or int(ids.size) < 1:
                    raise ValueError("A nonempty one-dimensional token prefix is required")
                length = int(ids.size)
                if self.prompt_length is None:
                    self.prompt_length = length
                    self.stats["prompt_tokens"] = length
                    return
                generated = length - self.prompt_length
                if generated != self.processed + 1:
                    raise RuntimeError("Unexpected constrained-generation token sequence")
                token = ids[-1].item()
                if type(token) is not int or not 0 <= token < owner.vocab_size:
                    raise ValueError("Invalid generated token ID")
                if self.stopped:
                    raise RuntimeError("Constrained decoder advanced after terminal EOS")
                if token in owner.eos:
                    if not self.guide.is_finished():
                        raise ValueError("EOS before accepting grammar state")
                    # Index exposes EOS in a final state's mask, but its
                    # next_state explicitly rejects EOS; track terminal state
                    # outside Guide rather than attempt a nonexistent advance.
                    self.stopped = True
                    self.stats["eos_accepted"] = True
                else:
                    self.guide.advance(token, return_tokens=False)
                self.processed = generated
                self.accepted_ids.append(token)
                self.stats["accepted_generated_tokens"] = generated
                self.stats["accepting"] = self.guide.is_finished()

            def __call__(self, ids, logits):
                if len(logits.shape) != 2 or logits.shape != (1, owner.vocab_size):
                    raise ValueError("One full-vocabulary score row is required")
                started = perf_counter()
                require_finite(mx, logits)  # Always inspect raw scores before masking.
                self.stats["raw_validation_seconds"] += perf_counter() - started
                started = perf_counter()
                self._advance(ids)
                self.stats["advance_seconds"] += perf_counter() - started
                started = perf_counter()
                self.guide.write_mask_into(self.mask.ctypes.data, self.mask.size, self.mask.itemsize)
                if self.guide.is_finished():
                    self._set_eos()
                if self.stopped:
                    # MLX performs an extra lookahead forward on the terminal EOS
                    # before yielding it. Keep that unused prediction terminal.
                    self.mask.fill(0)
                    self._set_eos()
                masked = apply(logits, self.mask)
                self.stats["mask_seconds"] += perf_counter() - started
                started = perf_counter()
                require_finite(mx, masked)  # No finite allowed score means failure, never score repair.
                self.stats["post_mask_validation_seconds"] += perf_counter() - started
                self.stats["processor_calls"] += 1
                return masked

            def finish(self, output_ids, *, hit_output_limit=False):
                if not output_ids or hit_output_limit:
                    raise ValueError("Constrained output absent or truncated")
                # The official generate_step lookahead normally advanced the last
                # yielded token already. Explicitly support a sampler without it.
                if len(output_ids) == self.processed + 1:
                    synthetic = np.empty((self.prompt_length + len(output_ids),), dtype=np.int32)
                    synthetic[-1] = output_ids[-1]
                    self._advance(synthetic)
                if (list(output_ids) != self.accepted_ids or len(output_ids) != self.processed
                        or output_ids[-1] not in owner.eos or not self.stopped):
                    raise ValueError("Constrained output did not reach accepting EOS")
                return dict(self.stats)
        return Processor()
