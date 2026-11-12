"""Assign quote IDs to exact sentence spans so the model never retypes source text."""

from .context import (COVERAGE_POLICY, canonical, digest, locate_quote,
                      section_sentence_spans, source_quote_spans)


def quote_catalog(context, tokenizer):
    sources, catalog = [], {}
    for source in context['sources']:
        text = source['text']
        spans = source_quote_spans(source)
        if spans is None:
            if context.get('selection_policy') == COVERAGE_POLICY:
                raise ValueError('Complete-source-sentence metadata is required')
            # Historical contexts intentionally preserve the old window-local
            # boundary behavior; they are not relabeled as complete sentences.
            spans = [{'char_start': source['char_start'] + s['char_start'],
                      'char_end': source['char_start'] + s['char_end']}
                     for s in section_sentence_spans(text)]
        elif not spans:
            raise ValueError('Selected source has no complete source sentence')
        sentences = []
        for span in spans:
            start, end = span['char_start'] - source['char_start'], span['char_end'] - source['char_start']
            quote_id = f'{source["citation_id"]}.Q{len(sentences)+1}'
            quote = text[start:end]
            entry = {'quote_id':quote_id, 'citation_id':source['citation_id'], 'quote':quote,
                     'char_start':source['char_start']+start, 'char_end':source['char_start']+end}
            catalog[quote_id] = entry
            sentences.append({'quote_id':quote_id,'text':quote})
        sources.append({**{k:source[k] for k in ('citation_id','pmid','title','journal','publication_date','section')},
                        'sentences':sentences})
    payload = canonical({'sources':sources})
    tokens = len(tokenizer.encode(payload,add_special_tokens=False))
    if tokens > 4000:
        raise ValueError('Quote-indexed evidence exceeds the 4000-token budget')
    return {'catalog':catalog,'payload':payload,'tokens':tokens,'sha256':digest(catalog)}


def resolve_quote(context, catalog, quote_id):
    if not isinstance(quote_id,str) or quote_id not in catalog:
        raise ValueError('Unknown quote ID')
    entry = catalog[quote_id]
    location = locate_quote(context,entry['citation_id'],entry['quote'])
    span = {'char_start':entry['char_start'],'char_end':entry['char_end']}
    if span not in location['matches']:
        raise ValueError('Quote catalog span differs from source')
    source = next(s for s in context['sources'] if s['citation_id'] == entry['citation_id'])
    allowed = source_quote_spans(source)
    if allowed is not None and span not in allowed:
        raise ValueError('Quote is not a complete original source sentence')
    return {**location,'quote_id':quote_id,'matches':[span],'ambiguous':False}
