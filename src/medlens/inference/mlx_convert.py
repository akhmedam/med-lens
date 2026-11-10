"""Offline, pinned Qwen conversion; importing this module needs only the stdlib."""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import struct
import time

MODEL_ID = 'Qwen/Qwen3.5-9B'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
COVERAGES = ('all-linear-and-embedding', 'linear-only-except-lm-head')
PACKAGES = {'mlx': '0.32.2', 'mlx-metal': '0.32.2', 'mlx-lm': '0.32.2',
            'transformers': '5.20.1', 'numpy': '2.4.6'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def sha_file(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for body in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(body)
    return checksum.hexdigest()


def simple_name(name):
    return isinstance(name, str) and name not in ('', '.', '..') and Path(name).name == name


def verify_source(root, spec):
    """Hash the original pinned files before loading them in the isolated process."""
    root = Path(root).resolve()
    if spec.get('repo_id') != MODEL_ID or spec.get('revision') != REVISION:
        raise ValueError('Conversion requires the pinned Qwen source')
    directory = (root/spec['directory']).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError('Qwen source directory escapes checkout or is missing')
    files = spec.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Missing source file pins')
    for name, identity in files.items():
        if not simple_name(name):
            raise ValueError('Invalid source filename')
        path = directory/name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or escaping source file: '+name)
        if (path.stat().st_size != identity['bytes'] or sha_file(path) != identity['sha256']):
            raise ValueError('Pinned source checksum mismatch: '+name)
    return {'model_id': MODEL_ID, 'revision': REVISION, 'directory': str(directory),
            'files': files, 'validation': 'all-source-file-sizes-and-sha256'}


def tensor_headers(directory):
    """Read bounded safetensors headers without importing or allocating model tensors."""
    directory = Path(directory)
    tensors = {}
    shards = sorted(directory.glob('*.safetensors'))
    if not shards:
        raise ValueError('No converted safetensors')
    dtype_bytes = {'F16': 2, 'F32': 4, 'U32': 4}
    for path in shards:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Converted shards must be ordinary files')
        size = path.stat().st_size
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated safetensors prefix')
            length = struct.unpack('<Q', prefix)[0]
            if not 2 <= length <= min(32*1024*1024, size-8):
                raise ValueError('Invalid safetensors header size')
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError('Invalid safetensors header')
        ranges = []
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key in tensors or not isinstance(value, dict):
                raise ValueError('Duplicate or invalid tensor entry')
            dtype, shape, offsets = value.get('dtype'), value.get('shape'), value.get('data_offsets')
            if (dtype not in dtype_bytes or not isinstance(shape, list)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)):
                raise ValueError('Unexpected converted tensor layout: '+key)
            start, end = offsets
            if (not 0 <= start <= end <= size-8-length
                    or end-start != math.prod(shape)*dtype_bytes[dtype]):
                raise ValueError('Invalid safetensors tensor offsets: '+key)
            ranges.append((start, end))
            tensors[key] = value
        ranges.sort()
        if any(b[0] < a[1] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('Overlapping safetensors tensors')
    return tensors


def validate_converted(directory, coverage):
    if coverage not in COVERAGES:
        raise ValueError('Unsupported quantization coverage')
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Converted model must be an ordinary directory')
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError('Converted model must contain ordinary flat files only')
    config = json.loads((directory/'config.json').read_bytes())
    quant = config.get('quantization', {})
    if (config.get('model_type') != 'qwen3_5' or quant.get('bits') != 4
            or quant.get('group_size') != 64 or quant.get('mode') != 'affine'
            or config.get('text_config', {}).get('dtype') != 'float16'):
        raise ValueError('Converted config does not match the declared policy')
    tensors = tensor_headers(directory)
    expected = 'U32' if coverage == COVERAGES[0] else 'F16'
    for suffix in ('embed_tokens.weight', 'lm_head.weight'):
        selected = [(k, v) for k, v in tensors.items() if k.endswith(suffix)]
        if len(selected) != 1 or selected[0][1]['dtype'] != expected:
            raise ValueError('Unexpected quantization coverage: '+suffix)
        key, value = selected[0]
        shape = [248320, 512 if expected == 'U32' else 4096]
        if value['shape'] != shape:
            raise ValueError('Unexpected embedding/head shape: '+key)
    a_logs = [v for k, v in tensors.items() if k.endswith('.A_log')]
    dt_biases = [v for k, v in tensors.items() if k.endswith('.dt_bias')]
    if (len(a_logs) != 24 or any(v['dtype'] != 'F32' for v in a_logs)
            or len(dt_biases) != 24 or any(v['dtype'] != 'F16' for v in dt_biases)):
        raise ValueError('Unexpected Qwen recurrent parameter dtype/count')
    for key, tensor in tensors.items():
        if tensor['dtype'] == 'F32' and not key.endswith('.A_log'):
            raise ValueError('Undeclared FP32 exception: '+key)
    index = directory/'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_bytes()).get('weight_map', {})
        if set(mapping) != set(tensors) or any(not simple_name(v) or not (directory/v).is_file()
                                              for v in mapping.values()):
            raise ValueError('Converted weight index mismatch')
    return {'tensor_count': len(tensors), 'a_log_count': len(a_logs),
            'dt_bias_count': len(dt_biases), 'embedding_and_head_dtype': expected,
            'validation': 'safetensors-headers-layout-dtypes-and-index'}


def write_manifest(directory, source, coverage, packages, conversion):
    directory = Path(directory)
    validation = validate_converted(directory, coverage)
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': sha_file(p)}
             for p in sorted(directory.iterdir()) if p.name != 'medlens-source.json'}
    value = {'schema_version': 1, 'kind': 'medlens-mlx-model', 'status': 'complete',
        'model': {'model_id': MODEL_ID, 'revision': REVISION},
        'source': {k: v for k, v in source.items() if k != 'directory'},
        'quantization': {'method': 'mlx-affine', 'bits': 4, 'group_size': 64,
                         'coverage': coverage},
        'dtype': 'float16', 'fp32_exceptions': ['A_log'], 'packages': packages,
        'conversion': conversion, 'tensor_validation': validation, 'files': files}
    value['artifact_id'] = hashlib.sha256(canonical(value)).hexdigest()
    path = directory/'medlens-source.json'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coverage', choices=COVERAGES, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Choose a fresh conversion output')
    source = json.loads(args.source_verification.read_bytes())
    if (source.get('model_id') != MODEL_ID or source.get('revision') != REVISION
            or source.get('validation') != 'all-source-file-sizes-and-sha256'):
        raise ValueError('Unverified conversion input')
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    if packages != PACKAGES:
        raise RuntimeError('Isolated MLX package versions differ from pins: '+str(packages))
    # These imports must happen only inside the isolated native environment.
    import mlx.core as mx
    import mlx.nn as nn
    from importlib import import_module
    convert = import_module('mlx_lm.convert').convert
    predicate = None
    if args.coverage == COVERAGES[1]:
        predicate = lambda path, module: isinstance(module, nn.Linear) and not path.endswith('.lm_head')
    started = time.perf_counter()
    convert(hf_path=source['directory'], mlx_path=str(args.output), quantize=True,
            q_bits=4, q_group_size=64, q_mode='affine', dtype='float16',
            quant_predicate=predicate, trust_remote_code=False)
    mx.synchronize()
    conversion = {'seconds': time.perf_counter()-started,
        'mlx_peak_bytes': mx.get_peak_memory(),
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'converter': 'mlx_lm.convert.convert', 'network': 'offline',
        'implementation_sha256': sha_file(Path(__file__))}
    config_path = args.output/'config.json'
    config = json.loads(config_path.read_bytes())
    config['text_config']['dtype'] = 'float16'
    config_path.write_text(json.dumps(config, indent=2)+'\n', encoding='utf-8')
    result = write_manifest(args.output, source, args.coverage, packages, conversion)
    print(json.dumps({'status': 'complete', 'artifact_id': result['artifact_id'],
                      'conversion': conversion}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
