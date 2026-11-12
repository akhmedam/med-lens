"""Conservative vetoes for explicit study-language mismatches, never source approval.

No biomedical entity dictionary, network, model, or positive entailment checker.
Absence means absent from the recognized lexical forms, not clinical proof.
"""
import re
import unicodedata
from fractions import Fraction

POLICY_VERSION = 'source-scope-veto-v1'

_STOP = set('the a an of for to from in on with using as and or by versus vs compare compared '
            'does do did is are was were can how what which disease disorder syndrome condition '
            'progression treatment therapy agent drug compound assay test testing measurement'.split())
_NUM = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), range(13)))
_N = r'(?:\d+(?:\.\d+)?|' + '|'.join(_NUM) + r')'
_TIME = re.compile(r'(?<![\w-])(' + _N + r')(?:\s*(?:to|[-–—])\s*(' + _N
                   + r'))?\s*[-‐‑–—]?\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|months?|mos?|years?|yrs?)\b', re.I)
_COMBO = re.compile(r'\b(?:add[ -]on|added to|plus|combination|combined with|(?:half|full|reduced|adjusted|titrated)[ -]dose|dose[ -](?:reduc\w*|titrat\w*|adjust\w*))\b', re.I)
_CONTRAST = re.compile(r'\b(?:compar(?:e|ed|ison)|versus|vs\.?)\b', re.I)
_REFERENCE = re.compile(r'\busing\s+(.+?)\s+as\s+(?:(?:the|a)\s+)?(?:reference|gold)\s+standard\b', re.I)
_CLINICAL = re.compile(r'\b(?:symptoms?|clinical\s+(?:signs?|criteria|assessment))\b', re.I)


def _negated(text, start, end=None):
    prefix = re.split(r'[.;!?]', text[:start])[-1]
    if re.search(r'\b(?:not|never|no|without|exclude|excluded|excluding|neither|nor)\b(?:\W+\w+){0,3}\W*$', prefix, re.I):
        return True
    return end is not None and bool(re.match(r'\s*(?:(?:was|were|are|is|have|been)\s+){0,2}(?:not|never|excluded)\b', text[end:], re.I))


def _validate(question, title, sections, quotes):
    if not isinstance(question, str) or not question.strip() or not isinstance(title, str):
        raise ValueError('Question must be nonempty text; title must be text')
    if not isinstance(sections, (list, tuple)) or not isinstance(quotes, (list, tuple)):
        raise ValueError('Abstract sections and selected quotes must be sequences')
    if any(not isinstance(q, str) or not q.strip() for q in quotes):
        raise ValueError('Selected quotes must contain nonempty strings')
    indices = []
    for i, section in enumerate(sections):
        if (not isinstance(section, dict) or not isinstance(section.get('text'), str)
                or type(section.get('index', i)) is not int or section.get('index', i) < 0):
            raise ValueError('Each abstract section requires text and a nonnegative integer index')
        indices.append(section.get('index', i))
    if len(set(indices)) != len(indices):
        raise ValueError('Abstract section indices must be unique')


def terms(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r"[’']s\b", '', text)
    return {w[:-1] if len(w) > 4 and w.endswith('s') else w
            for w in re.findall(r'\b\w+\b', text) if w not in _STOP}


def _proof(field, text, match):
    return {'field': field, 'char_start': match.start(), 'char_end': match.end(),
            'text': text[match.start():match.end()]}


def _intervals(text):
    result = []
    for m in _TIME.finditer(text):
        number = lambda x: Fraction(_NUM[x.casefold()]) if x.casefold() in _NUM else Fraction(x)
        low, high = number(m[1]), number(m[2] or m[1])
        unit = m[3].casefold().rstrip('s')
        unit = {'sec': 'second', 'min': 'minute', 'hr': 'hour', 'h': 'hour',
                'd': 'day', 'wk': 'week', 'mo': 'month', 'yr': 'year'}.get(unit, unit)
        kind = 'calendar-month' if unit in {'month', 'year'} else 'second'
        factor = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400,
                  'week': 604800, 'month': 1, 'year': 12}[unit]
        if low <= high:
            result.append((m, (kind, low * factor, high * factor)))
    return result


def _terminal_outcome_time(question):
    found = list(re.finditer(r'\b(?:after|at)\s+', question, re.I))
    if not found:
        return None
    opening = found[-1]
    suffix = question[opening.end():].strip().rstrip('?').strip()
    interval = _intervals(suffix)
    if len(interval) != 1 or interval[0][0].span() != (0, len(suffix)):
        return None
    prefix = question[:opening.start()]
    if re.search(r'\b(?:not|never|without)\s*$', prefix, re.I):
        return None
    if not re.search(r'\b(?:reduce|reduces|improve|improves|increase|increases|decrease|decreases)\b', prefix, re.I):
        return None
    match = re.compile(re.escape(question[opening.start():opening.end()] + suffix)).search(question, opening.start())
    return match, interval[0][1]


def check_source_scope(question, title, abstract_sections, selected_quotes=()):
    """Return explicit blocks/cautions and exact source proof spans; never promote.

    A block downgrades a proposed direct relation to related/unclear. Unrecognized
    language is left unvalidated. selected_quotes does not limit full-source vetoes.
    Matching a time anywhere (including treatment duration) or an adult word is
    non-certifying lexical detection; it never establishes endpoint/eligibility support.
    """
    _validate(question, title, abstract_sections, selected_quotes)
    fields = [('title', title)] + [(f'abstract:{s.get("index", i)}', s['text'])
                                     for i, s in enumerate(abstract_sections)]
    result = {'policy_version': POLICY_VERSION, 'blocks': [], 'cautions': [], 'recognized_checks': [],
              'positive_validation_performed': False}
    full = '\n'.join(t for _, t in fields)

    def add(bucket, code, message, proofs, question_match=None):
        row = {'code': code, 'message': message, 'source_proofs': proofs}
        if question_match:
            row['question_proof'] = _proof('question', question, question_match)
        result[bucket].append(row)

    def matches(pattern):
        rx = re.compile(pattern, re.I)
        return [(f, t, m) for f, t in fields for m in rx.finditer(t)]

    if _CONTRAST.search(question) and not _COMBO.search(question):
        addon = re.search(r'\badd[ -]on\b', title, re.I)
        dose = re.search(r'\bdose[ -](?:reduc\w*|titrat\w*|adjust\w*)\b', title, re.I)
        if addon and dose and _CONTRAST.search(title):
            result['recognized_checks'].append('explicit-comparative-add-on-title')
            add('blocks', 'different_addon_dose_regimen',
                'Comparative title explicitly studies add-on plus dose adjustment; question does not request that regimen.',
                [_proof('title', title, addon), _proof('title', title, dose)])

    requested_reference = _REFERENCE.search(question)
    if requested_reference:
        result['recognized_checks'].append('requested-named-reference')
        requested = terms(requested_reference[1])
        named = matches(r'\b(?P<name>consensus(?:[ -]based)?)\s+reference\s+standard\b')
        named += matches(r'\b(?:against|using)\s+(?:the\s+)?(?P<name>[\w-]+(?:\s+[\w-]+){0,5})\s+as\s+(?:the\s+)?(?:reference|gold)\s+standard\b')
        named += matches(r'\b(?P<name>[\w-]+(?:\s+[\w-]+){0,4})\s+(?:served|was used|were used)\s+as\s+(?:the\s+)?(?:reference|gold)\s+standard\b')
        named = [(f, text, m) for f, text, m in named if not _negated(text, m.start(), m.end())
                 and not re.search(r'\b(?:not|never|no|without|excluded)\b', m[0], re.I)]
        requested_analyses = [(f, text, m) for f, text, m in named
                              if requested and requested <= terms(m['name'])]
        for f, text, m in named:
            if requested and not requested & terms(m['name']):
                add('cautions' if requested_analyses else 'blocks',
                    'multiple_reference_analyses' if requested_analyses else 'different_reference_standard',
                    'Source also reports the requested named analysis; separate reference analyses must remain distinct.'
                    if requested_analyses else 'Explicit source reference standard has no requested-name token overlap.',
                    [_proof(f, text, m), *[_proof(f2, text2, m2) for f2, text2, m2 in requested_analyses]], requested_reference)
        combined_question = bool(re.search(r'\b(?:plus|and|combined|composite|both|symptoms?|clinical)\b', requested_reference[1], re.I))
        if not combined_question:
            for f, text, m in matches(r'\b(?:defined|classified|diagnosed)\s+(?:as|by)\s+[^.;]+'):
                if (not _negated(text, m.start(), m.end()) and _CLINICAL.search(m[0]) and re.search(r'\b(?:combined with|plus|and)\b', m[0], re.I)
                        and requested & terms(m[0])):
                    add('cautions' if requested_analyses else 'blocks',
                        'multiple_reference_analyses' if requested_analyses else 'combined_clinical_reference',
                        'Source definition explicitly combines clinical/symptom criteria with requested laboratory terms.',
                        [_proof(f, text, m)], requested_reference)

    target = re.search(r'\bprogression\s+from\s+.+?\s+to\s+([^?]+?)[?.]*\s*$', question, re.I)
    if target:
        requested = terms(target[1])
        source = terms(full)
        acronym = ''.join(w[0] for w in re.findall(r'\b\w+\b', re.sub(r"[’']s\b", '', target[1]).casefold()))
        result['recognized_checks'].append('named-progression-target')
        if requested and not requested <= source and not (len(acronym) > 1 and acronym in source):
            add('blocks', 'progression_target_not_established',
                'Distinctive requested target terms or its explicit acronym are absent from whole frozen source.',
                [{'field': 'title', 'char_start': 0, 'char_end': len(title), 'text': title}], target)

    timing = _terminal_outcome_time(question)
    if timing:
        requested_match, requested_interval = timing
        recognized = [(f, text, m, interval) for f, text in fields for m, interval in _intervals(text)]
        result['recognized_checks'].append('terminal-outcome-time')
        if not any(interval == requested_interval for _, _, _, interval in recognized):
            add('blocks', 'outcome_time_not_established',
                'No exact equivalent interval in recognized source time expressions; a broader interval is not equivalent.',
                [_proof(f, text, m) for f, text, m, _ in recognized], requested_match)

    intervention = re.search(r'\b(?:do|does|did|can|will|would|should)\s+(.+?)\s+(?:compare(?:d)?\s+(?:with|to)|versus|vs\.?)\b', question, re.I)
    if intervention and not _COMBO.search(question):
        requested = terms(intervention[1])
        if requested and not requested & terms(title):
            result['recognized_checks'].append('requested-intervention-not-in-title')
            for f, text, m in matches(r'\b(?:all|both)\s+[^.;]{0,70}?\b(?:received|were given|were treated with|took)\s+[^.;]{1,150}'):
                if (requested <= terms(m[0]) and not _negated(text, m.start(), m.end())
                        and not re.search(r'\b(?:not|never|no|without)\b', m[0], re.I)):
                    add('blocks', 'intervention_given_to_all_groups',
                        'Requested intervention is explicitly given to all/both groups and absent from title topic.',
                        [_proof(f, text, m)], intervention)
            for f, text, m in matches(r'\b(?:received|were given)\s+(?:concomitant|concurrent|background)\s+[^.;]{1,150}|\b(?:received|were given)\s+[^.;]{1,80}?\bconcomitantly'):
                if (requested <= terms(m[0]) and not _negated(text, m.start(), m.end())
                        and not re.search(r'\b(?:not|never|no|without)\b', m[0], re.I)):
                    add('blocks', 'intervention_is_cointervention',
                        'Requested intervention is explicitly concomitant/background care and absent from title topic.',
                        [_proof(f, text, m)], intervention)

    if re.search(r'\badults?\b', question, re.I):
        result['recognized_checks'].append('explicit-adult-eligibility')
        adult = [(f, text, m) for f, text, m in matches(r'\badults?\b')
                 if not _negated(text, m.start(), m.end())]
        ages = matches(r'\b(?:aged|age)\s*(?:≥|>=|>|at least)?\s*(\d+)(?:\s*(?:to|[-–])\s*\d+)?\s*years?\b')
        lower_bound = any(int(m[1]) >= 18 and not _negated(text, m.start(), m.end())
                          and not re.search(r'\b(?:mean|median|average)\s*$', text[:m.start()], re.I)
                          for _, text, m in ages)
        lower_bound |= any(int(m[1]) >= 18 and not _negated(text, m.start(), m.end())
                           and not re.search(r'\b(?:mean|median|average)\s*$', text[:m.start()], re.I)
                           for _, text, m in matches(r'\bage\s*(?:≥|>=)\s*(\d+)\b(?!\s*(?:months?|weeks?|days?|hours?))'))
        if not adult and not lower_bound:
            means = matches(r'\bmean age\b[^.;]{0,70}')
            add('cautions', 'adult_eligibility_not_explicit',
                'No explicit adult label or recognized lower age bound >=18 years; mean age does not establish eligibility.',
                [_proof(f, text, m) for f, text, m in means])
    return result
