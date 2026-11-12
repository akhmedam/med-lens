"""One fresh PubMed request through verified artifacts and the selected answer pipeline."""

from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

from medlens.generation.context import digest, source_section
from medlens.ingestion.pubmed import PubMedClient
from medlens.ingestion.snapshots import download_snapshot
from medlens.parsing.dataset import build_dataset
from medlens.pipeline import run_question, validate_answer_style
from medlens.pubmed_query import plan_query
from medlens.reporting import render_report
from medlens.retrieval.bge_tokenizer import MODEL_ID, REVISION
from medlens.retrieval.embeddings import QUERY_INSTRUCTION, query_input
from medlens.retrieval.dense import DenseIndex
from medlens.retrieval.vector_cache import VectorCache, build_cached_embeddings
from medlens.retrieval.windows import build_windows


def local_encoder_policy(model_dir, *, device='mps', batch_size=16):
    """Declare the pinned policy without loading weights; the runtime checks it again."""
    source = json.loads((Path(model_dir)/'model-source.json').read_bytes())
    if source['model_id'] != MODEL_ID or source['revision'] != REVISION:
        raise ValueError('Live search requires the pinned BGE assets')
    return {'model': source, 'device': device, 'dtype': 'float32',
        'pooling': 'last_hidden_state[:,0,:]', 'normalization': 'L2', 'dimension': 768,
        'attention': 'eager', 'batch_size': batch_size, 'pad_token_id': 0, 'padding_side': 'right',
        'query_instruction': QUERY_INSTRUCTION, 'passage_instruction': None,
        'torch_version': version('torch'), 'transformers_version': version('transformers'),
        'numpy_version': version('numpy')}


def sample_memory(stage):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row = {'stage': stage, 'process_peak_rss_bytes': rss if sys.platform=='darwin' else rss*1024}
    torch = sys.modules.get('torch')
    if torch is not None and torch.backends.mps.is_available():
        row.update(mps_tensor_bytes=torch.mps.current_allocated_memory(),
                   mps_driver_bytes=torch.mps.driver_allocated_memory())
    return row


def run_live_question(question, output, cache_root, *, tokenizer_loader, encoder_info,
                      encoder_factory, answer_tokenizer_loader, runtime_factory,
                      client=None, limit=100, pubmed_query=None, generate=True, top_k=10, answer_style='draft',
                      progress=None, release_accelerator=None):
    validate_answer_style(answer_style)
    plan = plan_query(question, pubmed_query)
    if type(limit) is not int or not 1 <= limit <= 300:
        raise ValueError('Live candidate limit must be between 1 and 300')
    if type(top_k) is not int or not 1 <= top_k <= 10 or type(generate) is not bool:
        raise ValueError('Invalid top-k or generation flag')
    output, cache_root = Path(output), Path(cache_root)
    if output.resolve().is_relative_to(cache_root.resolve()) or cache_root.resolve().is_relative_to(output.resolve()):
        raise ValueError('Request output and embedding cache must be disjoint')
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    live = {'query_plan': plan, 'candidate_limit': limit, 'fresh_search_complete': False,
            'source': 'PubMed ESearch/EFetch', 'snapshot_directory': str((output/'snapshot').resolve())}
    prep, samples = {}, []
    paths = {key: str((output/key).resolve()) for key in ('dataset', 'windows', 'embeddings')}
    report = {'schema_version': 1, 'question': question, 'status': 'running', 'stage': 'setup',
              'runtime_load_count': 0, 'answer_style': answer_style, 'created_at': datetime.now(timezone.utc).isoformat()}

    def save(value):
        nonlocal report
        report = value
        value.update(live=live, preparation_timings=prep, input_paths=paths,
                     request_seconds=perf_counter()-started, memory_samples=samples)
        if value.get('scope') is not None:
            value['scope']['source'] = 'live-pubmed-candidate-subset'
        for name, payload in [('run.json', json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'),
                              ('report.html', render_report(value))]:
            temporary = output/(name+'.tmp')
            temporary.write_text(payload, encoding='utf-8');temporary.replace(output/name)
        if progress:progress(value['stage'], value['status'])

    def stage(name):
        report['stage'] = name
        samples.append(sample_memory(name));save(report)

    def measure(name, call):
        tick = perf_counter()
        try:return call()
        finally:prep[name+'_seconds'] = perf_counter()-tick

    def release():
        gc.collect()
        if release_accelerator:release_accelerator()

    def measured_runtime():
        instance = runtime_factory()
        samples.append(sample_memory('qwen-loaded'))
        class Runtime:
            tokenizer = instance.tokenizer
            metadata = instance.metadata
            def __getattr__(self, name):
                return getattr(instance, name)
            def close(self):
                close = getattr(instance, "close", None)
                if callable(close):close()
            def generate(self, *args, **kwargs):
                try:return instance.generate(*args, **kwargs)
                finally:samples.append(sample_memory('qwen-call-complete'))
        return Runtime()

    measured_runtime.identity = getattr(runtime_factory, "identity", {"engine": "custom"})
    save(report)
    try:
        stage('query-validation')
        tokenizer, info = measure('bge_tokenizer_load', tokenizer_loader)
        query_input(tokenizer, question)  # Reject oversized query before network I/O.
        stage('pubmed-download')
        sm = measure('pubmed_download', lambda: download_snapshot(client or PubMedClient(),
            plan['submitted_query'], output/'snapshot', limit=limit, batch_size=200))
        live.update(fresh_search_complete=True, snapshot_id=sm['snapshot_id'],
            query_translation=sm['query_translation'], total_pubmed_matches=sm['total_count'],
            fetched_article_count=len(sm['pmids']), retrieved_started_at=sm['started_at'],
            retrieved_finished_at=sm['finished_at'],
            request_attempts=[r['attempts'] for r in sm['responses']])
        stage('parsing')
        dm = measure('parsing', lambda: build_dataset(output/'snapshot', output/'dataset'))
        live['dataset_id'] = dm['dataset_id']
        if not sm['pmids']:
            report.update(status='complete', stage='complete',
                request_id=digest({'plan':plan, 'snapshot_id':sm['snapshot_id'], 'generate':generate, 'answer_style':answer_style}),
                scope={'source':'live-pubmed-candidate-subset','candidate_count':0,'searchable_count':0,
                       'window_count':0,'missing_vector_pmids':[]},
                retrieval={'results':[]}, context={'sources':[],'status':'no-source-excerpts'},
                answer={'status':'insufficient_evidence' if generate else 'not_requested',
                        'reason':'empty-pubmed-result','claims':[],'model_calls':[]})
        else:
            stage('windowing')
            wm = measure('windowing', lambda: build_windows(output/'dataset', tokenizer, info, output/'windows'))
            stage('passage-embeddings')
            cache = VectorCache(cache_root, encoder_info, info)
            try:
                em = measure('passage_embeddings', lambda: build_cached_embeddings(output/'windows',
                    cache, encoder_factory, output/'embeddings'))
                live['passage_cache'] = em['measurements']
                live['cache_policy_sha256'] = cache.policy_sha256
            finally:release()
            stage('index-validation')
            index = measure('index_validation', lambda: DenseIndex(output/'dataset', output/'windows', output/'embeddings'))
            for row in index.windows:source_section(index.articles[row['pmid']], row)
            live.update(window_artifact_id=wm['artifact_id'], embedding_artifact_id=em['artifact_id'])
            answer_tokenizer = measure('answer_tokenizer_load', answer_tokenizer_loader)
            report = run_question(index, question, encoder_factory=encoder_factory,
                tokenizer=answer_tokenizer, runtime_factory=measured_runtime,
                generate=generate, rerank=False, top_k=top_k, answer_style=answer_style, checkpoint=save,
                release_accelerator=release)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error={'type':type(exc).__name__, 'message':str(exc), 'stage':report['stage']})
    finally:
        release()
    samples.append(sample_memory('finished'))
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['memory_scope'] = 'Parent process RSS high-water and stage-boundary MPS samples overlap; do not add. MLX worker RSS/Metal counters, when present in runtime metadata, are separate overlapping observations, not a system-wide or swap peak.'
    save(report)
    return report
