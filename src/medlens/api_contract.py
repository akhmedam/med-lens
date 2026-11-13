"""Strict, dependency-free API inputs and an allowlisted public report view.

Private run artifacts retain prompts and diagnostic details locally. This module
never imports the ML pipeline and never turns quotation identity or same-model
review into independent scientific verification.
"""

from dataclasses import dataclass
import math
import re
from typing import Optional

from .pubmed_query import plan_query


class RequestValidationError(ValueError):
    """The public request does not match the supported API contract."""


@dataclass(frozen=True)
class SearchRequest:
    question: str
    pubmed_query: Optional[str] = None
    limit: int = 100
    top_k: int = 10
    generate: bool = True
    answer_style: str = 'draft'

    def __post_init__(self):
        if type(self.limit) is not int or not 1 <= self.limit <= 300:
            raise RequestValidationError('limit must be an integer between 1 and 300')
        if type(self.top_k) is not int or not 1 <= self.top_k <= 10:
            raise RequestValidationError('top_k must be an integer between 1 and 10')
        if type(self.generate) is not bool:
            raise RequestValidationError('generate must be a boolean')
        if not isinstance(self.answer_style, str) or self.answer_style not in {'draft', 'source-findings'}:
            raise RequestValidationError('answer_style must be draft or source-findings')
        try:
            # Reuse the real limits and lexical validation. Do not replace the
            # caller's question with the transformed PubMed candidate query.
            plan_query(self.question, self.pubmed_query)
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from exc

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, dict):
            raise RequestValidationError('Request body must be a JSON object')
        allowed = {'question', 'pubmed_query', 'limit', 'top_k', 'generate', 'answer_style'}
        unknown = set(payload) - allowed
        if unknown:
            names = ', '.join(sorted(str(key) for key in unknown))
            raise RequestValidationError('Unknown request fields: ' + names)
        if 'question' not in payload:
            raise RequestValidationError('question is required')
        return cls(**payload)


# XML paths produced by the PubMed/JATS parsers are public source locations,
# unlike filesystem paths contained elsewhere in private provenance objects.
_XML_PATH = re.compile(
    r'^(?:\./|/)(?:PubmedArticleSet|PubmedArticle|PubmedBookArticle|MedlineCitation|'
    r'BookDocument|article|front|body|Abstract|AbstractText)'
    r'(?:\[[0-9]+\])?(?:/[A-Za-z_][A-Za-z0-9_.:-]*(?:\[[0-9]+\])?)*$')
_LOCAL_PATH = re.compile(r'(?:^file:|^[~]|^[A-Za-z]:[\\/]|/(?:Users|private|tmp|home|var)/)')
_LICENSE_URI = re.compile(
    r'https?://creativecommons\.org/(?:licenses/by/(?:1\.0|2\.0|2\.5|3\.0|4\.0)|'
    r'publicdomain/zero/1\.0)/?$', re.I)
_TIMINGS = {
    'query_encoder_constructor_seconds', 'query_encode_seconds', 'dense_search_seconds',
    'retrieval_seconds', 'context_seconds', 'reranking_seconds', 'answer_seconds',
    'pipeline_seconds',
}
# These are fixed program branch codes, never exception messages or model text.
_SAFE_ANSWER_REASONS = frozenset({
    'empty-context', 'model-abstained', 'support-review-complete',
    'no-claims-passed-support-review', 'no-bounded-source-quotes',
    'explicit-source-scope-blocks-all-articles',
    'scoped-source-findings-selected-not-scientifically-verified',
    'no-direct-scoped-source-findings', 'empty-pubmed-result', 'generation-disabled',
})
_PREPARATION_TIMINGS = {
    'bge_tokenizer_load_seconds', 'pubmed_download_seconds', 'parsing_seconds',
    'windowing_seconds', 'passage_embeddings_seconds', 'index_validation_seconds',
    'answer_tokenizer_load_seconds',
}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _dicts(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _texts(source, target, names, *, metadata=False, nullable=False):
    for name in names:
        value = source.get(name)
        if isinstance(value, str) and (not metadata or not _LOCAL_PATH.search(value)):
            target[name] = value
        elif nullable and name in source and value is None:
            target[name] = None


def _booleans(source, target, names):
    for name in names:
        if type(source.get(name)) is bool:
            target[name] = source[name]


def _numbers(source, target, names, *, integer=False, nonnegative=False):
    for name in names:
        value = source.get(name)
        valid = type(value) is int or (not integer and type(value) is float and math.isfinite(value))
        if valid and (not nonnegative or value >= 0):
            target[name] = value


def _string_list(value):
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _span(value):
    value = _mapping(value)
    start, end = value.get('char_start'), value.get('char_end')
    if type(start) is int and type(end) is int and 0 <= start <= end:
        return {'char_start': start, 'char_end': end}
    return {}


def _source_path(source, target):
    value = source.get('source_path')
    if isinstance(value, str) and _XML_PATH.fullmatch(value):
        target['source_path'] = value


def _source_url(source):
    pmcid, pmid = source.get('pmcid'), source.get('pmid')
    if source.get('source_kind') == 'fulltext' and isinstance(pmcid, str) and re.fullmatch(r'PMC[1-9][0-9]*', pmcid):
        return 'https://pmc.ncbi.nlm.nih.gov/articles/' + pmcid + '/'
    if isinstance(pmid, str) and re.fullmatch(r'[1-9][0-9]*', pmid):
        return 'https://pubmed.ncbi.nlm.nih.gov/' + pmid + '/'
    return None


def _source_reference(source):
    public = {}
    _texts(source, public, ('citation_id', 'quote_id', 'pmid', 'pmcid', 'window_id',
                            'text_version_id', 'source_kind'), metadata=True)
    _numbers(source, public, ('section_index',), integer=True, nonnegative=True)
    _source_path(source, public)
    public.update(_span(source))
    uri = source.get('license_uri')
    if isinstance(uri, str) and _LICENSE_URI.fullmatch(uri):
        public['license_uri'] = uri
    url = _source_url(source)
    if url is not None:
        public['url'] = url
    return public


def _citation(source):
    public = _source_reference(source)
    _texts(source, public, ('quote',))  # Preserve quotation bytes as text, no rewrite.
    _booleans(source, public, ('ambiguous',))
    if isinstance(source.get('matches'), list):
        public['matches'] = [span for row in source['matches'] if (span := _span(row))]
    return public


def _proof(source):
    public = {}
    field = source.get('field')
    if isinstance(field, str) and re.fullmatch(r'(?:title|question|(?:abstract|fulltext):[0-9]+)', field):
        public['field'] = field
    _source_path(source, public)
    public.update(_span(source))
    _numbers(source, public, ('section_index',), integer=True, nonnegative=True)
    _texts(source, public, ('pmid',), metadata=True)
    # A proof must identify source/question content, not just carry an opaque
    # provenance object with a local path. Its text remains verbatim.
    if 'field' in public or 'source_path' in public:
        _texts(source, public, ('text',))
    return public


def _caution(source):
    public = {}
    _texts(source, public, ('code', 'message'), metadata=True)
    if isinstance(source.get('source_proofs'), list):
        public['source_proofs'] = [proof for row in _dicts(source['source_proofs']) if (proof := _proof(row))]
    if isinstance(source.get('question_proof'), dict):
        proof = _proof(source['question_proof'])
        if proof:
            public['question_proof'] = proof
    return public


def _claim(source):
    public = {}
    _texts(source, public, ('claim_id', 'source_article_id', 'scope_quote_id'), metadata=True, nullable=True)
    _texts(source, public, ('text', 'source_title'), nullable=True)
    public['citations'] = [_citation(row) for row in _dicts(source.get('citations'))]
    if isinstance(source.get('scope_cautions'), list):
        public['scope_cautions'] = [_caution(row) for row in _dicts(source['scope_cautions'])]
    return public


def _answer(source):
    public = {'status': source.get('status') if isinstance(source.get('status'), str) else None,
              'claims': [_claim(row) for row in _dicts(source.get('claims'))]}
    _texts(source, public, ('answer_style', 'support_check', 'verification_status'), metadata=True)
    _booleans(source, public, ('quote_identity_verified', 'semantic_support_verified',
                              'generated_factual_prose', 'answer_completeness_established'))
    if isinstance(source.get('limitations'), list):
        public['limitations'] = _string_list(source['limitations'])
    reason = source.get('reason')
    if (source.get('status') != 'validation_failed' and isinstance(reason, str)
            and reason in _SAFE_ANSWER_REASONS):
        public['reason'] = reason
    # No draft, model calls, prompt payload, review internals or raw error reason.
    return public


def _evidence(source):
    public = _source_reference(source)
    _texts(source, public, ('title', 'journal', 'publication_date', 'section', 'text'), nullable=True)
    _numbers(source, public, ('article_rank', 'source_section_length'), integer=True, nonnegative=True)
    _numbers(source, public, ('window_cosine',))
    _texts(source, public, ('source_section_sha256', 'quote_sentence_policy'), metadata=True)
    if isinstance(source.get('quote_sentence_spans'), list):
        public['quote_sentence_spans'] = [span for row in source['quote_sentence_spans'] if (span := _span(row))]
    return public


def _articles(source):
    result = []
    for index, row in enumerate(_dicts(source.get('results')), 1):
        item = {}
        _texts(row, item, ('pmid', 'text_version_id'), metadata=True)
        _texts(row, item, ('title',), nullable=True)
        _numbers(row, item, ('rank',), integer=True, nonnegative=True)
        if 'rank' not in item:
            item['rank'] = index
        _numbers(row, item, ('score', 'cosine_distance'))
        url = _source_url(row)
        if url is not None:
            item['url'] = url
        result.append(item)
    return result


def _query_plan(source):
    public = {}
    _texts(source, public, ('method', 'sort', 'supported_scope', 'language_detection', 'topic_detection'), metadata=True)
    _texts(source, public, ('question', 'submitted_query', 'override', 'limitations'), nullable=True)
    _booleans(source, public, ('automatic_translation',))
    for name in ('conditions', 'interventions'):
        if isinstance(source.get(name), list):
            public[name] = _string_list(source[name])
    if isinstance(source.get('query_transformations'), list):
        transformations = []
        for row in _dicts(source['query_transformations']):
            item = {}
            _texts(row, item, ('rule',), metadata=True)
            _texts(row, item, ('removed_text', 'before', 'after'))
            item.update(_span(row))
            transformations.append(item)
        public['query_transformations'] = transformations
    return public


def _search(source):
    public = {}
    _texts(source, public, ('source', 'snapshot_id', 'dataset_id', 'retrieved_started_at',
                            'retrieved_finished_at'), metadata=True)
    _texts(source, public, ('query_translation',))
    _booleans(source, public, ('fresh_search_complete',))
    _numbers(source, public, ('candidate_limit', 'total_pubmed_matches', 'fetched_article_count'), integer=True, nonnegative=True)
    if isinstance(source.get('query_plan'), dict):
        public['query_plan'] = _query_plan(source['query_plan'])
    return public


def _scope(source):
    public = {}
    _texts(source, public, ('source',), metadata=True)
    _numbers(source, public, ('candidate_count', 'searchable_count', 'window_count'), integer=True, nonnegative=True)
    if isinstance(source.get('missing_vector_pmids'), list):
        public['missing_vector_pmids'] = [pmid for pmid in source['missing_vector_pmids']
                                          if isinstance(pmid, str) and re.fullmatch(r'[1-9][0-9]*', pmid)]
    return public


def _timings(source, names):
    public = {}
    _numbers(source, public, names, nonnegative=True)
    return public


def public_report(report):
    """Project completed or partial pipeline artifacts into a JSON-safe API view.

    Pipeline execution status and answer status are separate. Opaque provenance,
    runtime diagnostics and error messages are deliberately not traversed.
    Explicit source quotations remain unchanged. Independent verification is
    false even if an upstream object contains a contrary metadata field.
    """
    if not isinstance(report, dict):
        raise ValueError('A pipeline report must be a mapping')
    answer, context = _mapping(report.get('answer')), _mapping(report.get('context'))
    result = {'schema_version': 1,
              'status': report.get('status') if isinstance(report.get('status'), str) else None,
              'answer': _answer(answer),
              'articles': _articles(_mapping(report.get('retrieval'))),
              'evidence': [_evidence(row) for row in _dicts(context.get('sources'))],
              'search': _search(_mapping(report.get('live'))),
              'scope': _scope(_mapping(report.get('scope'))),
              'timings': _timings(_mapping(report.get('timings')), _TIMINGS),
              'preparation_timings': _timings(_mapping(report.get('preparation_timings')), _PREPARATION_TIMINGS),
              'independent_validation': False}
    _texts(report, result, ('request_id', 'stage', 'answer_style', 'created_at', 'finished_at'), metadata=True)
    _texts(report, result, ('question',))
    _numbers(report, result, ('request_seconds',), nonnegative=True)
    _booleans(report, result, ('generate', 'runtime_closed', 'worker_reaped'))
    return result
