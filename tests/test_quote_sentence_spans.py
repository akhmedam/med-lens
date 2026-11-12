"""Window edges must never become new sentence boundaries in coverage contexts."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import unittest

from medlens.generation.answers import check_context
from medlens.generation.context import (COVERAGE_POLICY, ContextConfig, SENTENCE_FIELDS,
    build_context, digest, render_evidence, section_sentence_spans)
from medlens.generation.quotes import quote_catalog, resolve_quote
from medlens.generation.request_contexts import verify_context_sources


class Tokenizer:
    def encode(self, text, **kwargs):return list(text)


def fixture(text, ranges):
    article = {'pmid': '1', 'title': 'Fictional source', 'journal': None, 'dates': [],
        'abstract': {'sections': [{'index': 0, 'label': 'RESULTS', 'text': text, 'source_path': 'abstract/0'}]},
        'text_version': {'text_version_id': 'v1'}, 'provenance': {'source': 'fictional'}}
    windows = [{'pmid': '1', 'window_id': str(n), 'text_version_id': 'v1', 'title': article['title'],
        'kind': 'abstract', 'section_index': 0, 'section_label': 'RESULTS', 'source_path': 'abstract/0',
        'char_start': start, 'char_end': end, 'text': text[start:end], 'provenance': article['provenance']}
        for n, (start, end) in enumerate(ranges)]
    dataset = SimpleNamespace(manifest={'dataset_id': 'd', 'articles_sha256': 'a'}, articles=[article])
    manifest = {'dataset_id': 'd', 'articles_sha256': 'a', 'artifact_id': 'w'}
    index = SimpleNamespace(dataset=dataset, window_manifest=manifest, windows=windows, articles={'1': article})
    return dataset, manifest, windows, index


def build(data, *, coverage=True):
    dataset, manifest, windows, _ = data
    return build_context(dataset, manifest, windows, ['1'], [.9 - n / 10 for n in range(len(windows))],
        Tokenizer(), {}, config=ContextConfig(selection_mode='results-methods' if coverage else 'article-diversity'))


def rehash(context):
    context['sources_sha256'] = digest(context['sources'])
    context['evidence_payload'] = render_evidence(context['sources'])
    keys = ('schema_version', 'dataset_id', 'window_artifact_id', 'article_ranking', 'scores_sha256',
            'config', 'selection_policy', 'tokenizer', 'sources_sha256', 'implementation_sha256')
    context['context_id'] = digest({k: context[k] for k in keys})


class CompleteSourceSentenceTests(unittest.TestCase):
    def truncated(self):
        text = 'Excluded qualifying prefix. Internal complete sentence. Trailing qualifying suffix.'
        start, end = text.index('qualifying') + 3, text.rindex('qualifying') + 5
        data = fixture(text, [(start, end)])
        return text, data, build(data)

    def test_internal_whole_sentence_keeps_exact_original_offsets_and_unicode(self):
        text = 'Discard prefix 🙂 that extends left. Internal café sentence. Discard suffix that extends right.'
        start, end = 5, len(text) - 5
        data = fixture(text, [(start, end)])
        context = build(data)
        verify_context_sources(context, data[3])
        indexed = quote_catalog(context, Tokenizer())
        self.assertEqual([e['quote'] for e in indexed['catalog'].values()], ['Internal café sentence.'])
        entry = indexed['catalog']['E1.Q1']
        self.assertEqual(entry['char_start'], text.index('Internal'))
        self.assertEqual(entry['char_end'], text.index('Internal') + len('Internal café sentence.'))
        self.assertEqual(resolve_quote(context, indexed['catalog'], 'E1.Q1')['matches'],
                         [{'char_start': entry['char_start'], 'char_end': entry['char_end']}])
        self.assertEqual(context['sources'][0]['text'], text[start:end])
        self.assertEqual(context['sources'][0]['source_section_sha256'], hashlib.sha256(text.encode()).hexdigest())

    def test_disjoint_token_windows_do_not_quote_cut_units_at_either_edge(self):
        text = 'First complete. Second qualified complete. Third qualified complete. Fourth complete.'
        data = fixture(text, [(0, text.index('Second') + 5), (text.index('Third') + 5, len(text))])
        context = build(data)
        verify_context_sources(context, data[3])
        indexed = quote_catalog(context, Tokenizer())
        self.assertEqual([s['window_id'] for s in context['sources']], ['0', '1'])
        self.assertEqual([e['quote'] for e in indexed['catalog'].values()], ['First complete.', 'Fourth complete.'])
        self.assertEqual(list(indexed['catalog']), ['E1.Q1', 'E2.Q1'])

    def test_window_with_no_whole_original_unit_is_skipped_without_truncation(self):
        text = 'A single long source sentence has important context before and after the middle.'
        data = fixture(text, [(10, 55)])
        context = build(data)
        self.assertEqual(context['sources'], [])
        self.assertTrue(any(s['reason'] == 'no-complete-source-sentence' for s in context['skipped']))
        self.assertEqual(quote_catalog(context, Tokenizer())['catalog'], {})

    def test_rehashed_forged_metadata_is_checked_against_actual_full_section(self):
        _, data, context = self.truncated()
        changes = [
            lambda s: s.update(source_section_sha256='a' * 64),
            lambda s: s.update(source_section_length=s['source_section_length'] + 1),
            lambda s: s['quote_sentence_spans'][0].update(char_start=s['char_start']),
            lambda s: s['quote_sentence_spans'][0].update(char_start=True),
            lambda s: s.update(quote_sentence_spans=[]),
            lambda s: s.pop('quote_sentence_policy'),
            lambda s: [s.pop(key) for key in SENTENCE_FIELDS],
        ]
        for change in changes:
            with self.subTest(change=change):
                forged = deepcopy(context)
                change(forged['sources'][0]); rehash(forged)
                check_context(forged)  # Hash self-consistency is insufficient.
                with self.assertRaises(ValueError):verify_context_sources(forged, data[3])
        missing = deepcopy(context)
        for key in SENTENCE_FIELDS:missing['sources'][0].pop(key)
        with self.assertRaisesRegex(ValueError, 'metadata is required'):quote_catalog(missing, Tokenizer())

    def test_a_forged_catalog_cannot_restore_a_cut_window_edge_quote(self):
        _, _, context = self.truncated()
        source = context['sources'][0]
        end = source['quote_sentence_spans'][0]['char_start'] - source['char_start']
        quote = source['text'][:end].rstrip()
        forged = {'E1.Q99': {'quote_id': 'E1.Q99', 'citation_id': 'E1', 'quote': quote,
            'char_start': source['char_start'], 'char_end': source['char_start'] + len(quote)}}
        with self.assertRaisesRegex(ValueError, 'not a complete original source sentence'):
            resolve_quote(context, forged, 'E1.Q99')

    def test_legacy_catalog_keeps_its_old_window_local_behavior(self):
        _, data, new = self.truncated()
        legacy = build(data, coverage=False)
        before = quote_catalog(legacy, Tokenizer())
        self.assertGreater(len(before['catalog']), 1)
        self.assertEqual(legacy['sources'][0]['text'], new['sources'][0]['text'])
        self.assertFalse(any(key in legacy['sources'][0] for key in SENTENCE_FIELDS))
        legacy['selection_policy'] = 'article-results-methods-window-cosine-no-overlap-v3'
        rehash(legacy)
        self.assertEqual(quote_catalog(legacy, Tokenizer()), before)
        verify_context_sources(legacy, data[3])

    def test_whole_unpunctuated_source_unit_is_allowed_but_not_a_window_slice(self):
        text = 'A whole source section without terminal punctuation'
        complete = fixture(text, [(0, len(text))])
        context = build(complete)
        self.assertEqual(quote_catalog(context, Tokenizer())['catalog']['E1.Q1']['quote'], text)
        partial = fixture(text, [(0, len(text) - 5)])
        self.assertEqual(build(partial)['sources'], [])
        self.assertEqual(section_sentence_spans('  Text.  Next unit  '),
                         [{'char_start': 2, 'char_end': 7}, {'char_start': 9, 'char_end': 18}])

    def test_fulltext_metadata_is_bound_to_fulltext_section_not_abstract(self):
        text, data, _ = self.truncated()
        dataset, _, windows, _ = data
        article = dataset.articles[0]
        article['full_text'] = {'sections': deepcopy(article['abstract']['sections']),
            'pmcid': 'PMC1', 'license_uri': 'https://example.org/fictional-license'}
        article['abstract']['sections'][0]['text'] = 'Different abstract content.'
        for window in windows:window['kind'] = 'fulltext'
        context = build(data)
        verify_context_sources(context, data[3])
        self.assertEqual(context['sources'][0]['source_section_sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(context['sources'][0]['source_kind'], 'fulltext')
        self.assertEqual([e['quote'] for e in quote_catalog(context, Tokenizer())['catalog'].values()],
                         ['Internal complete sentence.'])


if __name__ == '__main__':unittest.main()
