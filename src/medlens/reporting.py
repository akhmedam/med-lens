"""Render a local, script-free report with explicit source and draft boundaries."""

from html import escape
import re


def pubmed_link(pmid, label):
    if not isinstance(pmid, str) or re.fullmatch(r'[0-9]+', pmid) is None:
        raise ValueError("Invalid PubMed ID")
    return f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pmid}/">{escape(label)}</a>'


def render_report(report):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>MedLens search report</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#222}h1,h2{line-height:1.25}article{border-top:1px solid #ddd;padding:16px 0}blockquote{border-left:3px solid #bbb;margin-left:0;padding-left:16px;white-space:pre-wrap}aside{background:#fff3cf;padding:16px}.muted{color:#555}a{color:#175e9b}code{overflow-wrap:anywhere}</style>',
        '<h1>MedLens</h1><p>' + escape(report.get('question', '')) + '</p>']
    scope = report.get('scope', {})
    parts.append('<p class="muted">Search scope: provided local collection · '
        + escape(str(scope.get('candidate_count', '?'))) + ' articles. '
        'These results do not cover all of PubMed. Questions work best in English.</p>')
    if report.get('status') in {'failed', 'interrupted'}:
        error = report.get('error', {})
        parts.append('<aside>Run ' + escape(report['status']) + ' at ' + escape(error.get('stage', 'setup'))
            + '. Completed search results remain available. ' + escape(error.get('message', '')) + '</aside>')
    parts.append('<h2>Answer</h2>')
    answer = report.get('answer', {})
    status = answer.get('status', 'not_completed')
    messages = {'not_requested': 'Generation was not requested. Inspect the articles and source excerpts below.',
        'not_completed': 'Generation did not complete. Inspect any available source excerpts below.',
        'validation_failed': 'The generated draft failed validation; no answer is shown.',
        'insufficient_evidence': 'The supplied excerpts did not support an answer in this run.'}
    if status in messages:
        parts.append('<p>' + messages[status] + '</p>')
    elif status in {'answered', 'partial'}:
        parts.append('<aside>' + ('Partial research draft' if status == 'partial' else 'Research draft')
            + ' — requires independent review. Exact quotations do not guarantee that a claim follows from them. '
            'The same model generated and reviewed these claims.</aside>')
        for claim in answer['claims']:
            parts.append('<article><p>' + escape(claim['text']) + '</p>')
            for citation in claim['citations']:
                parts.append(pubmed_link(citation['pmid'], citation['citation_id'])
                    + '<blockquote>' + escape(citation['quote']) + '</blockquote>')
            parts.append('</article>')
        if answer.get('limitations'):
            parts.append('<p>Limitations: ' + escape(', '.join(answer['limitations']).replace('_', ' ')) + '.</p>')
    parts.append('<h2>Articles</h2>')
    hits = report.get('retrieval', {}).get('results', [])
    ranking = report.get('reranking', {}).get('ranking', [h['pmid'] for h in hits])
    by_id = {h['pmid']: h for h in hits}
    if not hits:
        parts.append('<p>No searchable articles were returned.</p>')
    if report.get('reranking', {}).get('status') == 'fallback':
        parts.append('<p>Reranking failed validation; the dense search order was retained.</p>')
    for n, pmid in enumerate(ranking, 1):
        hit = by_id[pmid]
        parts.append('<article>' + pubmed_link(pmid, f'{n}. {hit.get("title") or "Untitled article"}')
            + '<p class="muted">PMID ' + escape(pmid) + ' · cosine similarity '
            + format(hit['score'], '.3f') + '</p></article>')
    parts.append('<h2>Selected source excerpts</h2><p>These are retrieved source passages, '
        'not a judgment of relevance or a verified answer.</p>')
    sources = report.get('context', {}).get('sources', [])
    if not sources:
        parts.append('<p>No source excerpts are available.</p>')
    for source in sources:
        parts.append('<article><p>' + pubmed_link(source['pmid'], source['citation_id'] + ' · ' + (source.get('title') or 'Untitled article'))
            + '</p><p class="muted">' + escape(source.get('section') or 'Unstructured abstract')
            + ' · source characters ' + str(source['char_start']) + '–' + str(source['char_end'])
            + '</p><blockquote>' + escape(source['text']) + '</blockquote></article>')
    parts.append('<p class="muted">Execution status: ' + escape(report.get('status', 'running')) + '</p></html>')
    return '\n'.join(parts)
