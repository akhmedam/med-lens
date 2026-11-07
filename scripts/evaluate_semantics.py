"""Paired same-model review diagnostics on explicitly synthetic development contrasts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

from medlens.evaluation.semantics import load_challenges, summarize_challenges
from medlens.generation.answers import AnswerError, accepted_claim_ids, chat_prompt, parse_draft, parse_review
from medlens.generation.contracts import LEGACY_REVIEW_VERSION, REVIEW_VERSION
from medlens.generation.context import digest
from medlens.generation.quotes import quote_catalog
from medlens.generation.reviews import review_messages
from medlens.inference.qwen import MODEL_ID, REVISION, QwenRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT/'benchmarks/semantic-dev-v1/cases.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists() or args.output.resolve() == args.summary.resolve():
        parser.error('Choose two distinct fresh output paths')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True, token=False)
    groups, checksum = load_challenges(args.cases, tokenizer)
    plan = []
    # Construct all prompts before loading weights; labels never enter model messages.
    for protocol in (LEGACY_REVIEW_VERSION, REVIEW_VERSION):
        for group in groups:
            indexed = quote_catalog(group['context'], tokenizer)
            draft = parse_draft(json.dumps(group['draft']), group['context'], catalog=indexed['catalog'], citation_format='strings')
            prompt = chat_prompt(*review_messages(group['question'], group['context'], draft['claims'], indexed, protocol=protocol), tokenizer)
            plan.append((protocol, group, draft, prompt))
    identity = {'fixture_sha256': checksum, 'protocols': [LEGACY_REVIEW_VERSION, REVIEW_VERSION],
        'prompt_sha256': [p[3]['prompt_sha256'] for p in plan], 'max_new_tokens': 512,
        'code_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
            'scripts/evaluate_semantics.py', 'src/medlens/evaluation/semantics.py',
            'src/medlens/generation/answers.py', 'src/medlens/generation/reviews.py',
            'src/medlens/generation/contracts.py', 'src/medlens/inference/qwen.py', 'src/medlens/inference/metal.py')}}
    report = {'schema_version': 1, 'run_id': digest(identity), **identity, 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(), 'heldout_used': False,
        'scope': 'AI-authored synthetic development contrasts; not independent clinical validation',
        'labels_exposed_to_model': False, 'per_group': [], 'per_claim': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:json.dump(report, f, indent=2)
    def save():
        tmp = args.output.with_suffix('.json.tmp');tmp.write_text(json.dumps(report, indent=2)+'\n');tmp.replace(args.output)
    try:
        runtime = QwenRuntime();report['runtime'] = runtime.metadata;save()
        for protocol, group, draft, prompt in plan:
            call = runtime.generate(prompt, max_new_tokens=512)
            review, accepted, error = [], set(), None
            try:
                if call['hit_output_limit']:raise AnswerError('Review output limit reached', code='output_limit')
                review = parse_review(call['text'], {c['claim_id'] for c in draft['claims']}, protocol=protocol)
                accepted = accepted_claim_ids(review, protocol=protocol)
            except AnswerError as exc:
                error = {'code': exc.code, 'message': str(exc)}
            status = 'validation_failed' if error else 'valid'
            report['per_group'].append({'protocol': protocol, 'group_id': group['group_id'],
                'review_status': status, 'error': error, 'review': review, 'model_call': {'prompt': prompt, **call}})
            report['per_claim'].extend({**label, 'protocol': protocol, 'group_id': group['group_id'],
                'accepted': label['claim_id'] in accepted, 'review_status': status} for label in group['labels'])
            save();print(protocol, group['group_id'], status, sorted(accepted), round(call['seconds'],2), flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat();save()
    summary = {k: v for k, v in report.items() if k != 'per_group'}
    summary['local_report_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    summary['per_group'] = [{k: r[k] for k in ('protocol','group_id','review_status','error','review')}
        | {'input_tokens':r['model_call']['prompt']['input_tokens'],
           **{k:r['model_call'][k] for k in ('new_tokens','seconds','hit_output_limit','max_new_tokens')}} for r in report['per_group']]
    summary['by_protocol'] = {p: summarize_challenges([r for r in report['per_claim'] if r['protocol']==p])
        for p in identity['protocols']}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')


if __name__ == '__main__':main()
