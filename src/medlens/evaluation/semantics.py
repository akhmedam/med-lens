"""Synthetic semantic contrasts, isolated from clinical and held-out evaluation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from medlens.generation.context import ContextConfig, build_context, digest


def load_challenges(path, tokenizer):
    body = Path(path).read_bytes()
    fixture = json.loads(body)
    if (fixture.get('schema_version') != 1 or fixture.get('kind') != 'synthetic-semantic-development'
            or fixture.get('split') != 'development' or fixture.get('heldout_used') is not False):
        raise ValueError('Require explicit synthetic development fixtures')
    groups, seen = [], set()
    for group in fixture['groups']:
        key = group['group_id']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Invalid or duplicate challenge group')
        seen.add(key)
        specs = group['claims']
        if (not 1 <= len(specs) <= 3 or not 1 <= len(group['sources']) <= 6
                or any(type(c['expected_supported']) is not bool or not c['focus'] for c in specs)):
            raise ValueError('Invalid bounded challenge labels/sources')
        articles, windows = [], []
        for number, text in enumerate(group['sources'], 1):
            pmid = str(number)  # Synthetic IDs, never real PubMed evidence.
            provenance = {'source': 'synthetic-development-fixture', 'group': key, 'record': number}
            title, version = 'Synthetic source (not clinical evidence)', digest([key, number, text])
            articles.append({'pmid': pmid, 'title': title, 'journal': None, 'dates': [],
                'abstract': {'sections': [{'index': 0, 'label': 'RESULTS', 'text': text, 'source_path': 'synthetic/0'}]},
                'text_version': {'text_version_id': version}, 'provenance': provenance})
            windows.append({'pmid': pmid, 'window_id': digest([version, text]), 'text_version_id': version,
                'title': title, 'kind': 'abstract', 'section_index': 0, 'section_label': 'RESULTS',
                'source_path': 'synthetic/0', 'char_start': 0, 'char_end': len(text), 'text': text,
                'provenance': provenance})
        manifest = {'dataset_id': digest(articles), 'articles_sha256': digest(articles)}
        dataset = SimpleNamespace(manifest=manifest, articles=articles)
        window_manifest = {**manifest, 'artifact_id': digest(windows)}
        context = build_context(dataset, window_manifest, windows, [a['pmid'] for a in articles],
            [0.5] * len(windows), tokenizer, {'purpose': 'synthetic-development-only'},
            config=ContextConfig(max_per_article=1))
        groups.append({'group_id': key, 'question': group['question'], 'context': context,
            'draft': {'status': 'partial', 'limitations': ['limited_excerpts'],
                'claims': [{k: c[k] for k in ('text', 'citations')} for c in specs]},
            'labels': [{'claim_id': f'C{i}', 'expected_supported': c['expected_supported'], 'focus': c['focus']}
                for i, c in enumerate(specs, 1)]})
    if not groups: raise ValueError('Challenge fixture is empty')
    return groups, hashlib.sha256(body).hexdigest()


def summarize_challenges(rows):
    seen, counts = set(), Counter()
    for row in rows:
        key = (row['group_id'], row['claim_id'])
        if key in seen or type(row['expected_supported']) is not bool or type(row['accepted']) is not bool:
            raise ValueError('Invalid/duplicate semantic result')
        seen.add(key)
        if row['review_status'] not in {'valid', 'validation_failed'}:
            raise ValueError('Incomplete semantic result')
        if row['review_status'] == 'validation_failed' and row['accepted']:
            raise ValueError('Invalid review cannot accept a claim')
        counts['claim_count'] += 1
        expected, accepted = row['expected_supported'], row['accepted']
        counts['expected_supported' if expected else 'expected_withheld'] += 1
        counts['supported_retained' if expected and accepted else 'supported_withheld' if expected
               else 'unsupported_accepted' if accepted else 'unsupported_withheld'] += 1
        if row['review_status'] == 'validation_failed': counts['claims_with_invalid_reviews'] += 1
    if not seen: raise ValueError('No completed semantic results')
    return {key: counts[key] for key in ('claim_count', 'expected_supported', 'expected_withheld',
        'supported_retained', 'supported_withheld', 'unsupported_accepted', 'unsupported_withheld',
        'claims_with_invalid_reviews')}
