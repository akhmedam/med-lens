"""Fictional evidence must retain result definitions, scope and exact source spans."""
import copy
from collections import Counter
from types import SimpleNamespace
import unittest

from medlens.generation.context import ContextConfig, build_context
from medlens.generation.quotes import quote_catalog
from medlens.generation.request_contexts import verify_context_sources


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def fixture(count=4):
    articles, windows, scores = [], [], []
    for n in range(1, count + 1):
        pmid = str(n)
        provenance = {'source': 'fictional', 'record_index': n}
        sections = [
            {'index': 0, 'label': 'CONCLUSIONS', 'text': 'The composite changed.', 'source_path': 'abstract/0'},
            {'index': 1, 'label': 'RESULTS', 'text': 'The requested component changed by two units.', 'source_path': 'abstract/1'},
            {'index': 2, 'label': 'METHODS', 'text': 'Only adults were assigned to A or placebo.', 'source_path': 'abstract/2'},
            {'index': 3, 'label': 'SEARCH METHODS', 'text': 'We searched two bibliographic indexes.', 'source_path': 'abstract/3'},
        ]
        article = {'pmid': pmid, 'title': 'Fictional ' + pmid, 'journal': None, 'dates': [],
                   'abstract': {'sections': sections}, 'text_version': {'text_version_id': 'v' + pmid},
                   'provenance': provenance}
        articles.append(article)
        for section, score in zip(sections, (.99, .61, .20, .95)):
            windows.append({'pmid': pmid, 'window_id': pmid + '-' + str(section['index']),
                'text_version_id': 'v' + pmid, 'title': article['title'], 'kind': 'abstract',
                'section_index': section['index'], 'section_label': section['label'],
                'source_path': section['source_path'], 'char_start': 0, 'char_end': len(section['text']),
                'text': section['text'], 'provenance': provenance})
            scores.append(score)
    dataset = SimpleNamespace(manifest={'dataset_id': 'd', 'articles_sha256': 'a'}, articles=articles)
    manifest = {'dataset_id': 'd', 'articles_sha256': 'a', 'artifact_id': 'w'}
    return dataset, manifest, windows, scores


def build(data, *, config=None, ranking=None):
    dataset, manifest, windows, scores = data
    ranking = ranking or [a['pmid'] for a in dataset.articles]
    return build_context(dataset, manifest, windows, ranking, scores, CharacterTokenizer(), {}, config=config)


class CoverageTests(unittest.TestCase):
    def test_methods_and_specific_results_replace_unrelated_article_diversity(self):
        data = fixture()
        legacy = build(data)
        self.assertEqual([s['pmid'] for s in legacy['sources'][:4]], ['1', '2', '3', '4'])
        candidate = build(data, config=ContextConfig(selection_mode='results-methods'))
        self.assertEqual(candidate['article_ranking'], ['1', '2', '3', '4'])
        self.assertEqual([s['pmid'] for s in candidate['sources']], ['1', '1', '2', '2', '3', '3'])
        self.assertEqual([s['section'] for s in candidate['sources']], ['RESULTS', 'METHODS'] * 3)
        self.assertEqual([s['window_cosine'] for s in candidate['sources']], [.61, .20] * 3)
        self.assertEqual(candidate['selection_policy'], 'article-results-methods-complete-source-sentences-v4')
        dataset, manifest, windows, _ = data
        verify_context_sources(candidate, SimpleNamespace(dataset=dataset, window_manifest=manifest,
            windows=windows, articles={a['pmid']: a for a in dataset.articles}))

    def test_explicit_limits_keep_full_ranking_and_real_negative_scores(self):
        data = fixture()
        data[3][:] = [-.99, -.61, -.20, -.95] * 4
        c = build(data, config=ContextConfig(selection_mode='results-methods', max_fragments=3))
        self.assertEqual([s['pmid'] for s in c['sources']], ['1', '1', '2'])
        self.assertEqual([s['window_cosine'] for s in c['sources']], [-.61, -.20, -.61])
        self.assertEqual(c['article_ranking'], ['1', '2', '3', '4'])
        c = build(data, config=ContextConfig(selection_mode='results-methods', max_articles=1))
        self.assertEqual(Counter(s['pmid'] for s in c['sources']), {'1': 2})

    def test_quote_catalog_overhead_is_counted_before_source_is_admitted(self):
        data = fixture(1)
        baseline = build(data, config=ContextConfig(max_fragments=1))
        budget = baseline['evidence_tokens']
        self.assertGreater(quote_catalog(baseline, CharacterTokenizer())['tokens'], budget)
        c = build(data, config=ContextConfig(selection_mode='results-methods', evidence_tokens=budget))
        self.assertEqual(c['sources'], [])
        self.assertTrue(any(s['reason'] == 'quote-token-budget' for s in c['skipped']))

    def test_global_quote_limit_does_not_admit_a_raw_payload_under_4000(self):
        data = fixture(1)
        dataset, _, windows, _ = data
        section = dataset.articles[0]['abstract']['sections'][1]
        section['text'] = 'Sentence. ' * 120
        windows[1].update(text=section['text'], char_end=len(section['text']))
        # Isolate the multi-sentence source so another small excerpt cannot hide
        # the failure of the selected result's indexed budget.
        data = dataset, data[1], [windows[1]], [.5]
        legacy = build(data)
        self.assertLess(legacy['evidence_tokens'], 4000)
        with self.assertRaisesRegex(ValueError, '4000-token'):
            quote_catalog(legacy, CharacterTokenizer())
        c = build(data, config=ContextConfig(selection_mode='results-methods'))
        self.assertEqual(c['sources'], [])
        self.assertTrue(any(s['reason'] == 'quote-token-budget' for s in c['skipped']))

    def test_unstructured_companion_never_overlaps_the_selected_span(self):
        data = fixture(1)
        dataset, manifest, windows, _ = data
        text = 'Alpha beta. Gamma delta. Epsilon final.'
        section = dataset.articles[0]['abstract']['sections'][0]
        section.update(label=None, text=text)
        base = {**windows[0], 'section_label': None}
        windows = [{**base, 'window_id': str(i), 'char_start': start, 'char_end': end,
                    'text': text[start:end]} for i, (start, end) in enumerate(((0, 24), (6, 30), (25, len(text))))]
        c = build((dataset, manifest, windows, [.9, .8, .7]),
                  config=ContextConfig(selection_mode='results-methods'))
        self.assertEqual([s['window_id'] for s in c['sources']], ['0', '2'])
        self.assertTrue(any(s['reason'] == 'overlapping-span' for s in c['skipped']))

    def test_fulltext_citations_keep_their_license_and_original_spans(self):
        data = fixture(1)
        dataset, manifest, windows, scores = data
        article = dataset.articles[0]
        article['full_text'] = {'sections': copy.deepcopy(article['abstract']['sections']),
                               'pmcid': 'PMC1', 'license_uri': 'https://example.org/fictional-license'}
        for w in windows:
            w['kind'] = 'fulltext'
        c = build(data, config=ContextConfig(selection_mode='results-methods'))
        self.assertEqual([s['section'] for s in c['sources']], ['RESULTS', 'METHODS'])
        for source in c['sources']:
            self.assertEqual(source['source_kind'], 'fulltext')
            self.assertEqual(source['pmcid'], 'PMC1')
            self.assertEqual(source['license_uri'], article['full_text']['license_uri'])
            self.assertEqual(source['url'], 'https://pmc.ncbi.nlm.nih.gov/articles/PMC1/')

    def test_candidate_mode_rejects_invalid_limits_or_unknown_modes(self):
        for values in ({'selection_mode': 'unknown'}, {'max_articles': 0}, {'max_articles': True},
                       {'selection_mode': 'results-methods', 'max_fragments': 7},
                       {'selection_mode': 'results-methods', 'max_per_article': 3},
                       {'selection_mode': 'results-methods', 'evidence_tokens': 4001}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ContextConfig(**values)


if __name__ == '__main__':
    unittest.main()
