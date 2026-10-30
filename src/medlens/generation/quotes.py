"""Assign quote IDs to exact sentence spans so the model never retypes source text."""

import re

from .context import canonical, digest, locate_quote


def quote_catalog(context, tokenizer):
    sources, catalog = [], {}
    for source in context['sources']:
        text = source['text']
        # This is a transparent boundary heuristic, not a biomedical sentence parser.
        boundaries = [0, *[m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(])', text)], len(text)]
        sentences = []
        for start, end in zip(boundaries, boundaries[1:]):
            while start < end and text[start].isspace():start += 1
            while end > start and text[end-1].isspace():end -= 1
            if start == end:continue
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
    return {**location,'quote_id':quote_id,'matches':[span],'ambiguous':False}
