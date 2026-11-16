"""One dependency-free shipping policy; research runners retain their protocols.

The ID names a frozen application configuration, not a claim of scientific
validation. Returned snapshots are fresh JSON-safe values for durable job records.
"""

POLICY_ID = 'medlens-mvp-source-findings-v2'
REQUEST_TIMEOUT_SECONDS = 60.0
REQUEST_CLEANUP_RESERVE_SECONDS = 3.0
REQUEST_DELIVERY_RESERVE_SECONDS = 2.0
DEFAULT_CANDIDATE_LIMIT = 100
MIN_CANDIDATE_LIMIT = 1
MAX_CANDIDATE_LIMIT = 300
DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 10
ANSWER_STYLE = 'source-findings'
ANSWER_MODEL = '4b'
INFERENCE_ENGINE = 'mlx'
EMBEDDING_DEVICE = 'mps'


def policy_snapshot():
    """Describe the application policy without importing models or runtime code."""
    return {
        'id': POLICY_ID,
        'request_deadline': {
            'seconds': REQUEST_TIMEOUT_SECONDS,
            'cleanup_reserve_seconds': REQUEST_CLEANUP_RESERVE_SECONDS,
            'delivery_reserve_seconds': REQUEST_DELIVERY_RESERVE_SECONDS,
            'origin': 'transport-ingress-before-validation-and-admission',
            'enforcement': 'owned-request-process-group-termination-and-reap',
            'timeout_is_abstention': False,
            'client_delivery': 'measured-separately-including-polling-and-output',
        },
        'source': {
            'database': 'PubMed', 'fresh_search': True, 'text': 'title-and-abstract',
            'candidate_limit': {
                'default': DEFAULT_CANDIDATE_LIMIT,
                'min': MIN_CANDIDATE_LIMIT, 'max': MAX_CANDIDATE_LIMIT,
            },
        },
        'retrieval': {
            'method': 'dense-cosine-max-window', 'device': EMBEDDING_DEVICE,
            'top_k': {'default': DEFAULT_TOP_K, 'min': MIN_TOP_K, 'max': MAX_TOP_K},
        },
        'answer': {
            'style': ANSWER_STYLE, 'model_profile': ANSWER_MODEL,
            'engine': INFERENCE_ENGINE, 'independent_validation': False,
        },
    }
