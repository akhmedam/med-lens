"""Transparent, bounded English query planning for the current psychiatric scope."""

import re

# Broad candidate retrieval; the original question retains all clinical qualifiers.
CONDITIONS = (
    ('generalized-anxiety', r'\b(?:generalized anxiety|generalised anxiety|gad)\b', '"generalized anxiety disorder"[Title/Abstract]'),
    ('social-anxiety', r'\bsocial anxiety\b', '"social anxiety disorder"[Title/Abstract]'),
    ('panic', r'\bpanic\b', '"panic disorder"[Title/Abstract]'),
    ('depression', r'\bdepress(?:ion|ive|ed)\b', '("Depressive Disorder"[MeSH Terms] OR depress*[Title/Abstract])'),
)
INTERVENTIONS = (
    ('cbt', r'\b(?:cognitive behavio(?:u)?ral therapy|cbt|icbt|internet.delivered cognitive)\b',
        '("Cognitive Behavioral Therapy"[MeSH Terms] OR "cognitive behavioral therapy"[Title/Abstract] OR CBT[Title/Abstract])'),
    ('antidepressants', r'\b(?:antidepressants?|ssris?|selective serotonin reuptake inhibitors?)\b',
        '("Antidepressive Agents"[MeSH Terms] OR antidepressant*[Title/Abstract] OR SSRI*[Title/Abstract])'),
    ('exercise', r'\b(?:exercise|physical activity)\b', '("Exercise"[MeSH Terms] OR exercise[Title/Abstract])'),
    ('mindfulness', r'\b(?:mindfulness|mbct)\b', '("Mindfulness"[MeSH Terms] OR mindfulness[Title/Abstract] OR MBCT[Title/Abstract])'),
    ('exposure', r'\bexposure\b', '(exposure[Title/Abstract])'),
)


def plan_query(question, override=None):
    if not isinstance(question, str) or not question.strip() or len(question) > 8000:
        raise ValueError('Supply a nonempty English research question, at most 8000 characters')
    conditions = [(name, term) for name, pattern, term in CONDITIONS if re.search(pattern, question, re.I)]
    if not conditions and re.search(r'\banxiety\b', question, re.I):
        conditions = [('anxiety', '("Anxiety Disorders"[MeSH Terms] OR anxiety[Title/Abstract])')]
    if not conditions:
        raise ValueError('Live search currently covers depression and anxiety/panic questions in English; broader biomedical topics await manual annotation')
    scope = '(' + ' OR '.join(term for _, term in conditions) + ')'
    interventions = [(name, term) for name, pattern, term in INTERVENTIONS if re.search(pattern, question, re.I)]
    if override is not None:
        if not isinstance(override, str) or not override.strip() or len(override) > 4000 or any(ord(c)<32 for c in override):
            raise ValueError('PubMed query override must be nonempty single-line text, at most 4000 characters')
        term = '(' + override.strip() + ') AND ' + scope
        method = 'explicit-override-with-question-scope-v1'
    else:
        term = scope
        if interventions:
            # OR preserves candidates for comparisons; AND would require both therapies
            # to be mentioned, losing independent comparison evidence.
            term += ' AND (' + ' OR '.join(t for _, t in interventions) + ')'
        method = 'scoped-concepts-v1'
    return {'method': method, 'question': question, 'submitted_query': term,
        'override': override, 'conditions': [n for n, _ in conditions],
        'interventions': [n for n, _ in interventions], 'sort': 'PubMed Best Match',
        'limitations': 'English bounded concept vocabulary, not a complete clinical query compiler. Age, severity, outcome and delivery qualifiers remain in the question for ranking/evidence review; they are not silently added as PubMed filters. Use an explicit PubMed override for unrecognized concepts.'}
