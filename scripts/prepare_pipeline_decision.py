"""Assemble measured pipeline options without selecting a runtime default."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output path')
    paths = {name: ROOT/'docs/experiments'/filename for name, filename in (
        ('quality', 'medcpt-evaluation.json'), ('cost', 'retrieval-cost.json'),
        ('qwen_rerank', 'qwen-reranking.json'), ('generation', 'cited-answers.json'),
        ('requests', 'local-requests.json'), ('qwen_smoke', 'runtime-reports/inference-005.json'))}
    sources = {name: json.loads(path.read_bytes()) for name, path in paths.items()}
    quality, costs = sources['quality'], sources['cost']
    if (quality['heldout_used'] or costs['heldout_used'] or
            quality['input_hashes']['qrels'] != costs['qrels_sha256']):
        raise ValueError('Require the same development judgments and no held-out evaluation')
    mappings = [('dense', 'dense/pool-30'), ('dense_medcpt', 'dense_medcpt/pool-30/depth-10'),
        ('dense_medcpt_depth20', 'dense_medcpt/pool-30/depth-20'), ('medcpt', 'medcpt/pool-30/depth-None')]
    options = {}
    for name, key in mappings:
        group = costs['groups'][key]
        for metric in ('ndcg@10', 'recall@10'):
            if name in quality['macro'] and abs(group['macro'][metric]['mean'] - quality['macro'][name][metric]['mean']) > 1e-12:
                raise ValueError('Quality and cost reports disagree')
        options[name] = {'ndcg_at_10': group['macro']['ndcg@10']['mean'],
            'known_pool_recall_at_10': group['macro']['recall@10']['mean'],
            'judged_top10_coverage': group['macro']['judged_coverage@10']['mean'],
            'resident_cached_seconds': group['cached_seconds']['median'],
            'resident_recompute_vectors_seconds': group['fresh_seconds']['median'],
            'medcpt_pairs_per_ten_queries': group['total_pairs_per_repeat'],
            'answer_integration': 'existing ask.py' if name == 'dense' else 'standalone ranking CLI; requires score-kind and context integration after selection'}
    historical = sources['qwen_rerank']
    options['dense_qwen_historical'] = {'ndcg_at_10': quality['macro']['dense_qwen']['ndcg@10']['mean'],
        'known_pool_recall_at_10': quality['macro']['dense_qwen']['recall@10']['mean'],
        'rerank_only_median_seconds': statistics.median(r['seconds'] for r in historical['per_query'] if r['source'] == 'dense'),
        'scope': 'Historical Qwen permutations rescored on v3, no new timing run; listwise rerank input differs from MedCPT windows.'}
    cases = [r for r in sources['generation']['per_case'] if r['kind'] == 'pilot']
    generation = {'model': 'Qwen/Qwen3.5-9B', 'quantization': 'Metal 4-bit, group 64',
        'pilot_questions': len(cases), 'status_counts': dict(Counter(r['status'] for r in cases)),
        'questions_with_visible_claims': sum(r['claim_count'] > 0 for r in cases),
        'visible_claim_count': sum(r['claim_count'] for r in cases),
        'scope': 'Day 13 Dense+Qwen-reranked contexts; no matched answer-quality comparison of the four retrieval options.',
        'draft_and_review_only_median_seconds': statistics.median(sum(s['seconds'] for s in r['stages']) for r in cases),
        'known_semantic_issue': sources['requests']['known_semantic_issue'],
        'target_air_inference_tested': False,
        'historical_smoke_memory': {k: sources['qwen_smoke'][k] for k in ('memory_samples', 'process_peak_rss_bytes')}}
    result = {'schema_version': 1, 'decision_status': 'pending-user', 'selected_retrieval': None,
        'selected_answer_mode': None, 'runtime_defaults_changed': False, 'heldout_used': False,
        'target': {'machine': 'MacBook Air M3', 'unified_memory_gib': 24, 'inference_validated': False},
        'measurement_machine': {'machine': 'MacBook M3 Pro', 'unified_memory_gib': 36},
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        'retrieval_options': options, 'generation_evidence': generation,
        'answer_modes': ['ranked articles and exact source excerpts', 'experimental cited Qwen draft plus same-model review'],
        'recommendation': 'Dense as retrieval base; keep evidence-only usable. Qwen drafts remain experimental and require an Air memory/inference check before enabling there.',
        'integration_if_medcpt_selected': ['Add an explicit unbounded raw-logit score kind to evidence contexts; do not clamp logits to cosine.',
            'Use the same scored windows and preserve source spans, empty-source behavior and strict generation gates.',
            'Release each retrieval model before Qwen; measure on the Air and add integration checks.'],
        'limitations': ['Ten development queries, AI-assisted nonexpert labels, 208/300 candidate pairs judged.',
            'No primary-pipeline choice has been made by this report.', 'Retrieval nDCG does not establish generated answer correctness.',
            'Cascade costs compose separately measured stages; loads, ingestion and generation excluded.',
            'Qwen same-model review has missed age/subgroup/comparison errors.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
