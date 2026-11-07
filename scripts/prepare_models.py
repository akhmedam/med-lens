"""Verify or download pinned public assets; never overwrite conflicting files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'


def sha_file(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for body in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(body)
    return checksum.hexdigest()


def inspect_files(root, spec):
    root = Path(root).resolve()
    directory = root/spec['directory']
    if not directory.resolve().is_relative_to(root):
        raise ValueError('Asset directory escapes checkout')
    missing = []
    for name, identity in spec['files'].items():
        if Path(name).name != name:
            raise ValueError('Expected simple asset filenames')
        path = directory/name
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError('Asset symlink escapes checkout')
        if not path.is_file():
            if path.exists() or path.is_symlink():
                raise ValueError('Asset path conflict: '+str(path))
            missing.append(name)
        elif path.stat().st_size != identity['bytes'] or sha_file(path) != identity['sha256']:
            raise ValueError('Asset checksum conflict: '+str(path))
    return missing


def publish_bytes(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name+'.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)  # Atomic publication; refuses an existing destination.
    finally:
        Path(name).unlink(missing_ok=True)


def save_metadata(path, value):
    if path.is_symlink(): raise ValueError('Metadata symlink conflict')
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError('Asset metadata conflict: '+str(path))
        return
    publish_bytes(path, (json.dumps(value, indent=2)+'\n').encode())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Allow downloads of missing public assets')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Choose a fresh output path')
    report = {'status': 'running', 'model_inference': False, 'models': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n'); temp.replace(args.output)
    save()
    try:
        pins = json.loads((ROOT/'configs/model-assets.json').read_bytes())
        if pins['schema_version'] != 1:
            raise ValueError('Unsupported asset pins')
        if args.download:
            # The HTTP transport avoids the stalled Xet route seen during recovery.
            # It changes transfer only, not the model or inference runtime.
            os.environ['HF_HUB_DISABLE_XET'] = '1'
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import get_session
            def trim(request):
                value = request.headers.get('user-agent')
                if value is not None: request.headers['user-agent'] = value.rstrip()
            get_session().event_hooks['request'].append(trim)
        # Check every existing payload before starting any download.
        missing = {name: inspect_files(ROOT, spec) for name, spec in pins['models'].items()}
        kernel = pins.get('metal_kernel')
        missing_kernel = inspect_files(ROOT, kernel) if kernel else []
        if (any(missing.values()) or missing_kernel) and not args.download:
            raise ValueError('Missing assets; rerun with --download: '+json.dumps({**missing, 'metal_kernel': missing_kernel}))
        for name, spec in pins['models'].items():
            directory = ROOT/spec['directory']
            if missing[name]:
                if name == 'bge_tokenizer':
                    from medlens.retrieval.bge_tokenizer import download_tokenizer
                    (ROOT/'.cache').mkdir(exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temporary:
                        stage = Path(temporary)/'tokenizer'
                        previous = Path.cwd()
                        try:
                            os.chdir(ROOT)
                            download_tokenizer(stage)
                        finally:
                            os.chdir(previous)
                        staged_spec = {**spec, 'directory': str(stage.relative_to(ROOT))}
                        if inspect_files(ROOT, staged_spec): raise ValueError('Incomplete staged tokenizer')
                        directory.mkdir(parents=True, exist_ok=True)
                        for filename in spec['files']:
                            if filename in missing[name]:
                                publish_bytes(directory/filename, (stage/filename).read_bytes())
                else:
                    snapshot_download(spec['repo_id'], revision=spec['revision'], token=False,
                        allow_patterns=list(spec['files']),
                        local_dir=directory if name != 'qwen' else None, max_workers=4)
            if inspect_files(ROOT, spec): raise ValueError('Downloaded assets remain incomplete')
            if name in {'bge', 'bge_tokenizer', 'medcpt'}:
                value = {'model_id': spec['repo_id'], 'revision': spec['revision'],
                    'files': {file: entry['sha256'] for file, entry in spec['files'].items()}}
                if name == 'medcpt': value = {'schema_version': 1, **value}
                save_metadata(directory/('tokenizer-source.json' if name == 'bge_tokenizer' else 'model-source.json'), value)
            report['models'].append({'name': name, 'revision': spec['revision'],
                'files_checked': len(spec['files']), 'downloaded_files': missing[name]})
            save(); print(name, 'assets_ready', flush=True)
        if kernel:
            if missing_kernel:
                from kernels import get_kernel
                get_kernel(kernel['repo_id'], revision=kernel['revision'])
            if inspect_files(ROOT, kernel): raise ValueError('Pinned Metal kernel remains incomplete')
            report['metal_kernel'] = {'repo_id': kernel['repo_id'], 'revision': kernel['revision'],
                'version': kernel['version'], 'files_checked': len(kernel['files'])}
        report['status'] = 'assets_ready'
    except Exception as exc:
        report.update(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
