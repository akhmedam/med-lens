"""Transparent lexical question scaffolding for broad PubMed candidate search.

No LLM, biomedical concept dictionary, language/topic classifier or automatic
translation is used. The original question remains unchanged for retrieval
ranking and evidence review; explicit PubMed overrides are submitted verbatim.
"""

import re
import unicodedata

MAX_QUESTION_CHARS = 8000
MAX_OVERRIDE_CHARS = 4000
_DISALLOWED_CATEGORIES = {"Cc", "Cf", "Cs", "Zl", "Zp"}
_CLAUSE = re.compile(
    r'(?:(?<=^)|(?<=[,;]))\s*'
    r'((?:(?:how|what|which|why|when|where)\s+)?'
    r'(?:do|does|did|is|are|was|were|can|could|will|would|should)\b)', re.I)
_PREDICATES = re.compile(
    r'(?<![\w/+\-\u2010-\u2015\u2212])'
    r'(?:compar(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|'
    r'increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|affect(?:s|ed|ing)?|'
    r'influenc(?:e|es|ed|ing)|associat(?:e|es|ed|ing)|contribut(?:e|es|ed|ing)|using)'
    r'(?![\w/+\-\u2010-\u2015\u2212])', re.I)
_REFERENCE_ROLE = re.compile(r'(?<!\w)as\s+(?:(?:the|a)\s+)?(?:reference|gold)\s+standard(?!\w)', re.I)
_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"')
_NEGATED = re.compile(r'\b(?:not|never|no|without)\s+$', re.I)
# Only a simple terminal outcome-assessment phrase is removable. Other time
# restrictions (treatment duration, age, disease onset, or embedded clauses) stay.
_FOLLOWUP_NUMBER = r'(?:[0-9]+(?:\.[0-9]+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)'
_FOLLOWUP = re.compile(
    r'(?<![\w/+\-])(?:after|at)\s+' + _FOLLOWUP_NUMBER
    + r'(?:\s*(?:to|[-–—])\s*' + _FOLLOWUP_NUMBER + r')?'
    + r'\s+(?:minutes?|hours?|days?|weeks?|months?|years?)(?=\s*[?]*\s*$)', re.I)
_OUTCOME_PREDICATE = re.compile(
    r'(?<![\w/+\-\u2010-\u2015\u2212])'
    r'(?:reduc(?:e|es|ed|ing)|improv(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing))'
    r'(?![\w/+\-\u2010-\u2015\u2212])', re.I)
_FOLLOWUP_GUARDS = frozenset(
    'after before following during since when while if unless until with without within '
    'in into on at for from to by as among amongst under over above below between around '
    'who whose that which because not never no aged age onset diagnosis given receiving administered '
    'administration administering treated started starting initiated initiating diagnosed beginning dose doses exposure '
    'treatment therapy course dosing daily weekly'.split())


def _outcome_followup(question, protected):
    """Identify a conservative terminal follow-up; never rewrite the question.

    This is lexical candidate broadening, not PICO extraction. A simple effect
    predicate plus a short unqualified outcome phrase is required. Ambiguous
    time relations and explicit search syntax retain their timing terms.
    """
    if (any(c in question for c in '[]()') or re.search(r'\b(?:AND|OR|NOT)\b', question)
            or re.search(r'\b(?:not|never|no|without)\b', question, re.I)):
        return None
    timing = _FOLLOWUP.search(question)
    if timing is None or any(timing.start() < right and timing.end() > left
                             for left, right in protected):
        return None
    predicates = list(_OUTCOME_PREDICATE.finditer(question, 0, timing.start()))
    if not predicates:
        return None
    predicate = predicates[-1]
    if (predicate.group().isupper() or _NEGATED.search(question[:predicate.start()])
            or any(predicate.start() < right and predicate.end() > left for left, right in protected)):
        return None
    outcome = question[predicate.end():timing.start()].strip()
    if not outcome or any(c in outcome for c in ',;:?!'):
        return None
    words = re.findall(r'\b\w+\b', outcome.casefold())
    if not 1 <= len(words) <= 12 or any(word in _FOLLOWUP_GUARDS for word in words):
        return None
    return timing.span()


def _validate_text(value, *, maximum, label):
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(unicodedata.category(character) in _DISALLOWED_CATEGORIES for character in value)):
        raise ValueError(f"{label} must be nonempty single-line text without control characters, at most {maximum} characters")


def _candidate_query(question):
    protected = [match.span() for match in _QUOTED.finditer(question)]
    removals = []
    for rule, pattern in (("question-clause-scaffold", _CLAUSE),
                          ("generic-predicate-scaffold", _PREDICATES),
                          ("reference-role-scaffold", _REFERENCE_ROLE)):
        for match in pattern.finditer(question):
            start, end = match.span(1) if rule == "question-clause-scaffold" else match.span()
            text = question[start:end]
            if any(start < right and end > left for left, right in protected):
                continue
            if rule != "reference-role-scaffold" and text.isupper():
                continue  # Preserve acronym-like tokens, not a biomedical entity claim.
            if rule == "generic-predicate-scaffold" and _NEGATED.search(question[:start]):
                continue
            if any(start < right and end > left for left, right, _ in removals):
                continue
            removals.append((start, end, rule))
    followup = _outcome_followup(question, protected)
    if followup and not any(followup[0] < right and followup[1] > left for left, right, _ in removals):
        removals.append((*followup, "terminal-outcome-followup"))
    terminal = re.search(r'[?]+\s*$', question)
    if terminal:
        removals.append((*terminal.span(), "terminal-question-punctuation"))
    pieces, transformations, cursor = [], [], 0
    for start, end, rule in sorted(removals):
        pieces.extend((question[cursor:start], " "))
        transformations.append({"rule": rule, "char_start": start, "char_end": end,
            "removed_text": question[start:end]})
        cursor = end
    pieces.append(question[cursor:])
    intermediate = "".join(pieces)
    term = " ".join(intermediate.split())
    if intermediate != term:
        transformations.append({"rule": "normalize-whitespace", "before": intermediate, "after": term})
    if not term:
        raise ValueError("The question has no searchable terms after removing question scaffolding; supply a PubMed query override")
    return term, transformations


def plan_query(question, override=None):
    """Make only recorded lexical changes; save PubMed's translation downstream."""
    _validate_text(question, maximum=MAX_QUESTION_CHARS, label="Research question")
    if override is not None:
        _validate_text(override, maximum=MAX_OVERRIDE_CHARS, label="PubMed query override")
        term, method, transformations = override, "explicit-pubmed-query-v2", []
    else:
        term, transformations = _candidate_query(question)
        method = "lexical-question-scaffold-pubmed-atm-v2"
    return {
        "method": method, "question": question, "submitted_query": term,
        "override": override, "conditions": [], "interventions": [],
        "sort": "PubMed Best Match", "supported_scope": "English biomedical research questions",
        "language_detection": "not-performed", "topic_detection": "not-performed",
        "automatic_translation": False, "query_transformations": transformations,
        "limitations": (
            "Default candidate search removes only recorded generic English question/predicate and "
            "reference-role scaffolding and conservative terminal outcome-follow-up timing, then "
            "normalizes whitespace. Time removal requires a simple effect predicate and an unqualified "
            "outcome phrase; ambiguous durations, age/onset clauses and explicit search syntax retain "
            "their timing. This broadens candidates only: the unchanged original question "
            "is used for Dense ranking and answer review; no biomedical concepts are inferred or "
            "added. Quoted text, acronym-like tokens and identifier punctuation are protected, but "
            "this is a lexical heuristic, not a complete PICO or negation compiler. Remaining terms "
            "may still overconstrain retrieval. Explicit PubMed overrides are submitted exactly, "
            "without scaffold removal or appended filters. PubMed syntax/Automatic Term Mapping may "
            "map or ignore terms; the returned query_translation is saved with the fresh snapshot. "
            "English biomedical questions are the supported scope; language and topic are not "
            "classified or translated. Nonempty results do not establish relevance or exhaustive coverage."
        ),
    }
