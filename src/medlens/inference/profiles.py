"""Pinned answer-model profiles; importing this module needs only the stdlib."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat

DEFAULT_MODEL = '4b'
TOKENIZER_FILES = {
    'chat_template.jinja': {'bytes': 7756, 'sha256': 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'},
    'tokenizer.json': {'bytes': 12807982, 'sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'},
    'tokenizer_config.json': {'bytes': 16710, 'sha256': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'},
}
PROFILES = {
    '4b': {'model_id': 'mlx-community/Qwen3.5-4B-MLX-4bit',
        'revision': '32f3e8ecf65426fc3306969496342d504bfa13f3',
        'directory': '.cache/models/qwen3.5-4b-mlx-4bit-g64', 'dtype': 'bfloat16',
        'quantization': {'method': 'mlx-affine', 'bits': 4, 'group_size': 64, 'coverage': 'all-linear-and-embedding'},
        'tokenizer': {'model_id': 'Qwen/Qwen3.5-4B', 'revision': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                      'directory': '.cache/tokenizers/qwen3.5-4b'},
        'answer_policy': 'compact-v1', 'hidden_size': 2560, 'vocab_size': 248320,
        'tie_word_embeddings': True, 'fp32_exceptions': ['A_log'],
        'upstream_model_id': 'Qwen/Qwen3.5-4B', 'upstream_revision': None},
    '9b': {'model_id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'directory': '.cache/models/qwen3.5-9b-mlx-linear-4bit-g64', 'dtype': 'float16',
        'quantization': {'method': 'mlx-affine', 'bits': 4, 'group_size': 64, 'coverage': 'linear-only-except-lm-head'},
        'tokenizer': {'model_id': 'Qwen/Qwen3.5-9B', 'revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
            'directory': '.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
        'answer_policy': 'legacy-v6', 'hidden_size': 4096, 'vocab_size': 248320,
        'tie_word_embeddings': False, 'fp32_exceptions': ['A_log'],
        'upstream_model_id': 'Qwen/Qwen3.5-9B', 'upstream_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'},
}


def get_profile(name=DEFAULT_MODEL):
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError('Unknown answer model profile; choose 4b or 9b')
    return deepcopy(PROFILES[name])


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def strict_json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate asset metadata key')
            result[key] = value
        return result
    def constant(_): raise ValueError('Nonfinite asset metadata value')
    return json.loads(body, object_pairs_hook=pairs, parse_constant=constant)


def sha_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): result.update(block)
    return result.hexdigest()


def tokenizer_manifest(profile):
    value = {'schema_version': 1, 'kind': 'medlens-answer-tokenizer', 'status': 'complete',
        'model_id': profile['tokenizer']['model_id'], 'revision': profile['tokenizer']['revision'],
        'files': deepcopy(TOKENIZER_FILES)}
    value['artifact_id'] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def verify_tokenizer(root, model=DEFAULT_MODEL):
    """Verify all three pinned tokenizer files, with no model-weight dependency."""
    profile = get_profile(model)
    root = Path(root).resolve(); directory = root/profile['tokenizer']['directory']
    if (not directory.resolve().is_relative_to(root) or not directory.is_dir()
            or (model == '4b' and directory.is_symlink())):
        raise ValueError('Missing or escaping answer tokenizer directory; run prepare_mlx.py')
    expected = tokenizer_manifest(profile)
    if model == '4b':
        manifest_path = directory/'tokenizer-source.json'
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError('Missing regular pinned answer tokenizer-source.json')
        value = strict_json(manifest_path.read_bytes())
        if value != expected: raise ValueError('Answer tokenizer source identity differs from pins')
    for name, identity in TOKENIZER_FILES.items():
        path = directory/name
        # Legacy HF snapshots use in-checkout cache symlinks. New standalone4B
        # assets contain ordinary files; neither route permits checkout escapes.
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or escaping answer tokenizer file: '+name)
        if model == '4b' and not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('Standalone answer tokenizer files must be regular')
        if path.stat().st_size != identity['bytes'] or sha_file(path) != identity['sha256']:
            raise ValueError('Answer tokenizer checksum mismatch: '+name)
    return directory, expected


def load_answer_tokenizer(root, model=DEFAULT_MODEL):
    directory, source = verify_tokenizer(root, model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, token=False, trust_remote_code=False)
    tokenizer._medlens_tokenizer_source = deepcopy(source)
    return tokenizer
