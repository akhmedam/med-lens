"""Prepare native 4B by default; use --model 9b for pinned offline 9B conversion."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from medlens.inference.mlx_convert import COVERAGES, canonical, sha_file, verify_source

ROOT = Path(__file__).resolve().parents[1]


def prepare_9b(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'.cache/models/qwen3.5-9b-mlx-linear-4bit-g64')
    parser.add_argument('--worker-python', type=Path, default=ROOT/'.cache/venvs/mlx/bin/python')
    parser.add_argument('--coverage', choices=COVERAGES, default=COVERAGES[1])
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error('Output already exists; choose a fresh path')
    if not args.worker_python.is_file():
        parser.error('Create the isolated MLX environment first; see docs/native-mlx.md')
    if not 0 < args.timeout <= 3600:
        parser.error('Timeout must be greater than zero and at most 3600 seconds')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    state_path = staging/'preparation-status.json'
    state = {'schema_version': 1, 'status': 'verifying-source', 'output': str(output),
             'staging_directory': str(staging), 'coverage': args.coverage}
    def save():
        state_path.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
    save()
    try:
        pins = json.loads((ROOT/'configs/model-assets.json').read_bytes())
        source = verify_source(ROOT, pins['models']['qwen'])
        verification = staging/'source-verification.json'
        verification.write_bytes(canonical(source)+b'\n')
        state['status'] = 'converting'; save()
        print('Verified original Qwen files; converting offline.', flush=True)
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                   TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        subprocess.run([str(args.worker_python.absolute()), '-I',
            str(ROOT/'src/medlens/inference/mlx_convert.py'),
            '--source-verification', str(verification), '--output', str(staging/'model'),
            '--coverage', args.coverage], check=True, timeout=args.timeout, env=env)
        prepared = staging/'model'
        manifest = json.loads((prepared/'medlens-source.json').read_bytes())
        expected = manifest.pop('artifact_id')
        import hashlib
        if hashlib.sha256(canonical(manifest)).hexdigest() != expected:
            raise ValueError('Prepared manifest checksum mismatch')
        for name, identity in manifest['files'].items():
            path = prepared/name
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != identity['bytes']
                    or sha_file(path) != identity['sha256']):
                raise ValueError('Prepared artifact checksum mismatch: '+name)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Output appeared during conversion')
        prepared.rename(output)
        state.update(status='complete', artifact_id=expected);save()
        print(str(output), flush=True)
    except BaseException as exc:
        state.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        save()
        print('Incomplete conversion retained at '+str(staging), flush=True)
        raise
    return 0


def main(argv=None):
    import sys
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--model', choices=('4b', '9b'), default='4b')
    selected, remainder = selector.parse_known_args(sys.argv[1:] if argv is None else argv)
    if selected.model == '9b':
        return prepare_9b(remainder)
    # Keep4B preparation independent of the native interpreter and9B weights.
    import importlib.util
    path = Path(__file__).with_name('prepare_mlx_4b.py')
    spec = importlib.util.spec_from_file_location('medlens_prepare_4b', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module.main(remainder)


if __name__ == '__main__':
    raise SystemExit(main())
