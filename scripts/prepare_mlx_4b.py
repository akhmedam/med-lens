"""Prepare pinned native 4B assets and its official tokenizer without 9B weights or GPU."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import tempfile
import time
from urllib.parse import quote
from urllib.request import urlopen

from medlens.inference.profiles import (canonical, get_profile, sha_file, strict_json,
    tokenizer_manifest, verify_tokenizer)

ROOT = Path(__file__).resolve().parents[1]


def load_pins(root):
    value = strict_json((Path(root)/'configs/mlx-assets.json').read_bytes())
    if type(value.get('schema_version')) is not int or value['schema_version'] != 1:
        raise ValueError('Unsupported MLX asset pins')
    spec = value['models']['4b']; profile = get_profile('4b')
    for key in ('model_id', 'revision', 'directory', 'dtype', 'quantization', 'tie_word_embeddings'):
        if spec.get(key) != profile[key]: raise ValueError('4B pins differ from profile: '+key)
    expected_tokenizer = {**profile['tokenizer'], 'files': tokenizer_manifest(profile)['files']}
    if spec.get('tokenizer') != expected_tokenizer:
        raise ValueError('Official 4B tokenizer pins differ from profile')
    if spec.get('upstream_revision', 'missing') is not None or spec.get('official_source_revision_verified') is not False:
        raise ValueError('Community 4B upstream revision must remain explicitly unverified')
    if set(spec.get('files', {})) != {'config.json', 'model.safetensors'}:
        raise ValueError('Expected exactly pinned config and weights')
    for files in (spec['files'], spec['tokenizer']['files']):
        for name, entry in files.items():
            if (Path(name).name != name or name in ('', '.', '..') or '\\' in name
                    or not isinstance(entry, dict) or type(entry.get('bytes')) is not int or entry['bytes'] < 0
                    or not isinstance(entry.get('sha256'), str) or len(entry['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in entry['sha256'])):
                raise ValueError('Invalid asset file pin')
    source = spec.get('source_directory')
    if not isinstance(source, str) or not (Path(root)/source).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Pinned source directory escapes checkout')
    return deepcopy(spec)


def verify_file(path, identity):
    path = Path(path)
    try: info = path.lstat()
    except FileNotFoundError: raise ValueError('Missing pinned asset: '+str(path)) from None
    if not stat.S_ISREG(info.st_mode): raise ValueError('Pinned asset must be a regular file: '+str(path))
    if info.st_size != identity['bytes'] or sha_file(path) != identity['sha256']:
        raise ValueError('Pinned asset checksum mismatch: '+str(path))


def ensure_source(directory, files, model_id, revision, *, download=False):
    """Verify existing files first; download only absent pinned public files."""
    directory = Path(directory)
    if directory.is_symlink(): raise ValueError('Asset source directory cannot be a symlink')
    missing = []
    for name, identity in files.items():
        path = directory/name
        if path.exists() or path.is_symlink(): verify_file(path, identity)
        else: missing.append(name)
    if missing and not download:
        raise ValueError('Missing pinned files; use --download or provide a verified local source: '+','.join(missing))
    directory.mkdir(parents=True, exist_ok=True)
    for name in missing:
        identity = files[name]
        url = 'https://huggingface.co/'+model_id+'/resolve/'+revision+'/'+quote(name, safe='')
        descriptor, temp_name = tempfile.mkstemp(prefix='.'+name+'.download-', dir=directory)
        temporary = Path(temp_name)
        try:
            print('Downloading '+model_id+'/'+name+' ('+str(identity['bytes'])+' bytes)', flush=True)
            count = 0
            with os.fdopen(descriptor, 'wb') as target, urlopen(url, timeout=60) as response:
                while block := response.read(8*1024*1024):
                    count += len(block)
                    if count > identity['bytes']: raise ValueError('Public asset exceeds pinned size')
                    target.write(block)
                target.flush(); os.fsync(target.fileno())
            verify_file(temporary, identity)
            os.link(temporary, directory/name)  # Atomic no-overwrite file publication.
        finally:
            temporary.unlink(missing_ok=True)
    for name, identity in files.items(): verify_file(directory/name, identity)


def validate_model(directory):
    """Inspect source config and tensor headers without importing native MLX."""
    directory = Path(directory)
    config = strict_json((directory/'config.json').read_bytes()); text = config.get('text_config', {})
    quant = config.get('quantization', {})
    if (config.get('model_type') != 'qwen3_5' or text.get('hidden_size') != 2560
            or text.get('num_hidden_layers') != 32 or text.get('vocab_size') != 248320
            or text.get('eos_token_id') != 248044 or text.get('dtype') != 'bfloat16'
            or text.get('tie_word_embeddings', config.get('tie_word_embeddings')) is not True
            or quant != {'bits': 4, 'group_size': 64, 'mode': 'affine'} or 'model_file' in config):
        raise ValueError('Pinned 4B config differs from model/precision contract')
    path = directory/'model.safetensors'; size = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ValueError('Truncated safetensors prefix')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(32*1024*1024, size-8): raise ValueError('Invalid safetensors header length')
        header = strict_json(stream.read(length))
    if not isinstance(header, dict): raise ValueError('Invalid safetensors header')
    tensors = {}; ranges = []; widths = {'BF16': 2, 'F32': 4, 'U32': 4}
    for name, value in header.items():
        if name == '__metadata__': continue
        if not isinstance(value, dict): raise ValueError('Invalid tensor entry')
        dtype, shape, offsets = value.get('dtype'), value.get('shape'), value.get('data_offsets')
        if (dtype not in widths or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets)):
            raise ValueError('Unexpected 4B tensor layout: '+name)
        start, end = offsets
        if not 0 <= start <= end <= size-8-length or end-start != math.prod(shape)*widths[dtype]:
            raise ValueError('Invalid 4B tensor offsets')
        if dtype == 'F32' and not name.endswith('.A_log'): raise ValueError('Undeclared 4B FP32 parameter')
        tensors[name] = value; ranges.append((start, end))
    ranges.sort()
    if any(b[0] < a[1] for a,b in zip(ranges,ranges[1:])): raise ValueError('Overlapping 4B tensor offsets')
    embeddings = [v for k,v in tensors.items() if k.endswith('embed_tokens.weight')]
    if (len(embeddings) != 1 or embeddings[0]['dtype'] != 'U32' or embeddings[0]['shape'] != [248320,320]
            or any(k.endswith('lm_head.weight') for k in tensors)):
        raise ValueError('Expected tied 4B quantized embedding/head')
    for suffix,dtype in (('.A_log','F32'),('.dt_bias','BF16')):
        selected = [v for k,v in tensors.items() if k.endswith(suffix)]
        if len(selected) != 24 or any(v['dtype'] != dtype or v['shape'] != [32] for v in selected):
            raise ValueError('Unexpected 4B recurrent parameter precision/count')
    return {'tensor_count': len(tensors), 'quantized_embedding_dtype': 'U32', 'tie_word_embeddings': True,
        'floating_dtypes': ['bfloat16','float32'], 'fp32_exceptions': ['A_log'],
        'validation': 'safetensors-headers-layout-dtypes-tied-head'}


def prepare_tokenizer(root, spec, *, source=None, download=False):
    root = Path(root).resolve(); profile = get_profile('4b')
    output = root/profile['tokenizer']['directory']
    if output.exists() or output.is_symlink():
        verify_tokenizer(root, '4b')
        return tokenizer_manifest(profile)
    source = Path(source) if source else root/'.cache/downloads/qwen3.5-4b-tokenizer'
    ensure_source(source, spec['tokenizer']['files'], profile['tokenizer']['model_id'],
        profile['tokenizer']['revision'], download=download)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    try:
        for name, identity in spec['tokenizer']['files'].items():
            shutil.copyfile(source/name, staging/name); verify_file(staging/name, identity)
        manifest = tokenizer_manifest(profile)
        (staging/'tokenizer-source.json').write_bytes(canonical(manifest)+b'\n')
        if output.exists() or output.is_symlink(): raise FileExistsError('Tokenizer output appeared during preparation')
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    verify_tokenizer(root, '4b')
    return manifest


def prepare(root, *, output=None, source=None, tokenizer_source=None, download=False):
    root = Path(root).resolve(); spec = load_pins(root); profile = get_profile('4b')
    output = Path(output).absolute() if output else root/profile['directory']
    if output.exists() or output.is_symlink(): raise FileExistsError('Choose a fresh 4B model output path')
    source = Path(source) if source else root/spec['source_directory']
    if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError('Model output and source must be disjoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    state = {'schema_version':1, 'status':'verifying-source', 'output':str(output), 'model':'4b'}
    def save(): (stage/'preparation-status.json').write_bytes(canonical(state)+b'\n')
    save(); started = time.perf_counter()
    try:
        ensure_source(source, spec['files'], profile['model_id'], profile['revision'], download=download)
        validation = validate_model(source)
        tokenizer = prepare_tokenizer(root, spec, source=tokenizer_source, download=download)
        state['status'] = 'copying-verified-weights'; save()
        prepared = stage/'model'; prepared.mkdir()
        for name, identity in spec['files'].items():
            shutil.copyfile(source/name, prepared/name); verify_file(prepared/name, identity)
        validate_model(prepared)
        manifest = {'schema_version':1, 'kind':'medlens-mlx-model', 'status':'complete',
            'model':{'model_id':profile['model_id'], 'revision':profile['revision']},
            'quantization':profile['quantization'], 'dtype':profile['dtype'], 'fp32_exceptions':['A_log'],
            'tie_word_embeddings':True,
            'source':{'repo_id':profile['model_id'], 'revision':profile['revision'],
                'files':spec['files'], 'upstream_model_id':'Qwen/Qwen3.5-4B', 'upstream_revision':None,
                'official_source_revision_verified':False,
                'converter_claimed_by_card':'mlx-vlm branch pc/fix-qwen35-predicate; exact commit undisclosed',
                'precision_policy':'preserve-community-safetensors-dtypes-without-casting'},
            'tokenizer':{'model_id':tokenizer['model_id'], 'revision':tokenizer['revision'],
                         'artifact_id':tokenizer['artifact_id'], 'directory':profile['tokenizer']['directory']},
            'conversion':{'performed':False, 'dtype_cast_performed':False,
                          'upstream_revision':None, 'exact_converter_revision':None},
            'preparation':{'seconds':time.perf_counter()-started, 'validation':'all-source-and-prepared-file-sizes-and-sha256',
                'implementation_sha256':sha_file(Path(__file__))},
            'tensor_validation':validation, 'files':spec['files']}
        manifest['artifact_id'] = hashlib.sha256(canonical(manifest)).hexdigest()
        (prepared/'medlens-source.json').write_bytes(canonical(manifest)+b'\n')
        if output.exists() or output.is_symlink(): raise FileExistsError('Model output appeared during preparation')
        prepared.rename(output)
        state.update(status='complete',artifact_id=manifest['artifact_id']);save()
        print(str(output),flush=True)
        return manifest
    except BaseException as exc:
        state.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)});save()
        print('Incomplete 4B preparation retained at '+str(stage),flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--source-dir',type=Path,help='Existing pinned community4B raw files')
    parser.add_argument('--tokenizer-source',type=Path,help='Existing official 4B three-file tokenizer payload')
    parser.add_argument('--download',action='store_true',help='Allow absent pinned public model/tokenizer downloads')
    args = parser.parse_args(argv)
    prepare(ROOT,output=args.output,source=args.source_dir,tokenizer_source=args.tokenizer_source,download=args.download)
    return 0


if __name__ == '__main__': raise SystemExit(main())
