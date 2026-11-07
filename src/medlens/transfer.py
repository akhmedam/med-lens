"""Checksummed, split local-artifact bundles; no virtualenv or authentication files."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import tarfile
from datetime import datetime, timezone

ROOTS = ('data', 'results', '.cache/models', '.cache/tokenizers', '.cache/huggingface/hub',
         '.cache/huggingface/models--BAAI--bge-base-en-v1.5')
BLOCKED = {'.DS_Store', '__pycache__', '.locks', 'token', 'stored_tokens', '.env'}
BLOCK_SIZE = 8 * 1024 * 1024


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(BLOCK_SIZE), b''):
            h.update(block)
    return h.hexdigest()


def allowed(name):
    if not isinstance(name, str) or '\\' in name:
        return False
    path = PurePosixPath(name)
    return (not path.is_absolute() and path.as_posix() == name and '..' not in path.parts
        and not any(p in BLOCKED or p.endswith(('.lock', '.incomplete', '.pyc')) for p in path.parts)
        and any(name.startswith(root + '/') for root in ROOTS))


def link_target(name, link):
    if not isinstance(link, str) or not link or link.startswith('/') or '\\' in link:
        raise ValueError('Only relative internal symlinks are permitted')
    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link))
    if not allowed(target):
        raise ValueError('Symlink target outside the artifact roots')
    return target


class PartWriter:
    def __init__(self, output, limit):
        self.output, self.limit = output, limit
        self.parts, self.file, self.size, self.hash = [], None, 0, None

    def write(self, body):
        view = memoryview(body)
        while view:
            if self.file is None:
                name = f'part-{len(self.parts) + 1:04d}.tarpart'
                self.file = (self.output / name).open('xb')
                self.hash, self.size = hashlib.sha256(), 0
            block = view[:self.limit - self.size]
            self.file.write(block)
            self.hash.update(block)
            self.size += len(block)
            view = view[len(block):]
            if self.size == self.limit:
                self.close_part()
        return len(body)

    def close_part(self):
        if self.file is not None:
            name = Path(self.file.name).name
            self.file.close()
            self.parts.append({'name': name, 'bytes': self.size, 'sha256': self.hash.hexdigest()})
            self.file = None


class PartReader:
    def __init__(self, directory, parts):
        self.paths = iter(directory / p['name'] for p in parts)
        self.file = None

    def read(self, size):
        if size < 0:
            raise ValueError('Bounded streaming reads are required')
        blocks, remaining = [], size
        while remaining:
            if self.file is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.file = path.open('rb')
            block = self.file.read(remaining)
            if not block:
                self.file.close()
                self.file = None
                continue
            blocks.append(block)
            remaining -= len(block)
        return b''.join(blocks)

    def close(self):
        if self.file is not None:
            self.file.close()


def inventory(root):
    entries = []
    for prefix in ROOTS:
        base = root / prefix
        if base.is_symlink():
            raise ValueError('Artifact root must not be a symlink')
        if not base.exists():
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            for name in list(dirs):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    files.append(name)
                    dirs.remove(name)
                elif not allowed(relative + '/placeholder'):
                    dirs.remove(name)
            for name in sorted(files):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if not allowed(relative):
                    continue
                if path.is_symlink():
                    link = str(path.readlink())
                    link_target(relative, link)
                    entries.append({'path': relative, 'kind': 'symlink', 'link': link})
                elif path.is_file():
                    entries.append({'path': relative, 'kind': 'file', 'bytes': path.stat().st_size,
                        'sha256': sha_file(path)})
                else:
                    raise ValueError('Only regular files and internal symlinks are permitted')
    entries.sort(key=lambda r: r['path'])
    by_name = {e['path']: e for e in entries}
    for entry in entries:
        if entry['kind'] == 'symlink':
            seen, current = {entry['path']}, entry
            while current['kind'] == 'symlink':
                target = link_target(current['path'], current['link'])
                if target in seen or target not in by_name:
                    raise ValueError('Dangling or cyclic artifact symlink')
                seen.add(target)
                current = by_name[target]
    return entries


def checkout_hashes(root):
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    return {name: sha_file(root / name) for name in names if name and
        (name.startswith(('src/', 'scripts/', 'benchmarks/', 'configs/')) or name in
         ('pyproject.toml', 'requirements-macos-arm64.lock', 'requirements-macos-arm64-day19.lock'))}


def pack(root, output, *, part_bytes=2 * 1024**3):
    root, output = Path(root).resolve(), Path(output).resolve()
    if type(part_bytes) is not int or part_bytes < 1024:
        raise ValueError('Part limit must be at least 1024 bytes')
    if any(output.is_relative_to(root / prefix) for prefix in ROOTS):
        raise ValueError('Bundle output cannot be inside its own artifact inputs')
    output.mkdir(parents=True, exist_ok=False)
    state = {'schema_version': 1, 'status': 'running', 'created_at': datetime.now(timezone.utc).isoformat(),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'roots': list(ROOTS), 'parts': [], 'entries': [], 'checkout_sha256': checkout_hashes(root)}
    manifest = output / 'manifest.json'
    writer = PartWriter(output, part_bytes)
    try:
        manifest.write_text(json.dumps(state, indent=2) + '\n')
        state['entries'] = inventory(root)
        with tarfile.open(mode='w|', fileobj=writer, bufsize=BLOCK_SIZE, format=tarfile.PAX_FORMAT) as archive:
            for entry in state['entries']:
                info = tarfile.TarInfo(entry['path'])
                info.mode = 0o644
                if entry['kind'] == 'symlink':
                    info.type, info.linkname = tarfile.SYMTYPE, entry['link']
                    archive.addfile(info)
                else:
                    info.size = entry['bytes']
                    with (root / entry['path']).open('rb') as stream:
                        archive.addfile(info, stream)
        writer.close_part()
        # Recheck input identities: never publish a successful bundle of moving inputs.
        if state['entries'] != inventory(root) or state['checkout_sha256'] != checkout_hashes(root):
            raise ValueError('Inputs changed during bundling')
        state.update(status='complete', parts=writer.parts,
            file_count=len(state['entries']), artifact_bytes=sum(e.get('bytes', 0) for e in state['entries']),
            archive_bytes=sum(p['bytes'] for p in writer.parts))
    except BaseException as exc:
        writer.close_part()
        state.update(status='failed', parts=writer.parts, error={'type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        manifest.write_text(json.dumps(state, indent=2) + '\n')
    return state


def verify(bundle):
    bundle = Path(bundle)
    if (bundle / 'manifest.json').is_symlink():
        raise ValueError('Manifest must be a regular file')
    manifest = json.loads((bundle / 'manifest.json').read_bytes())
    if manifest['schema_version'] != 1 or manifest['status'] != 'complete' or manifest['roots'] != list(ROOTS):
        raise ValueError('Incomplete or unsupported transfer bundle')
    entries = {}
    for entry in manifest['entries']:
        name = entry['path']
        if not allowed(name) or name in entries:
            raise ValueError('Invalid or duplicate bundle path')
        if entry['kind'] == 'file':
            if (type(entry['bytes']) is not int or entry['bytes'] < 0 or
                    not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
                raise ValueError('Invalid file identity')
        elif entry['kind'] == 'symlink':
            link_target(name, entry['link'])
        else:
            raise ValueError('Unsupported bundle entry type')
        entries[name] = entry
    for entry in entries.values():
        # A symlink can never be an ancestor of another entry.
        for parent in PurePosixPath(entry['path']).parents:
            if parent.as_posix() in entries:
                raise ValueError('Bundle path has a file/symlink ancestor')
        current, seen = entry, {entry['path']}
        while current['kind'] == 'symlink':
            target = link_target(current['path'], current['link'])
            if target not in entries or target in seen:
                raise ValueError('Dangling or cyclic bundle symlink')
            seen.add(target)
            current = entries[target]
    if not manifest['parts']:
        raise ValueError('No archive parts')
    for i, part in enumerate(manifest['parts'], 1):
        if part['name'] != f'part-{i:04d}.tarpart':
            raise ValueError('Invalid part order or name')
        path = bundle / part['name']
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != part['bytes']
                or sha_file(path) != part['sha256']):
            raise ValueError('Transfer part is missing or corrupt: ' + part['name'])
    if (manifest['file_count'] != len(entries) or
            manifest['archive_bytes'] != sum(p['bytes'] for p in manifest['parts']) or
            manifest['artifact_bytes'] != sum(e.get('bytes', 0) for e in entries.values())):
        raise ValueError('Bundle totals differ')
    return manifest


def safe_destination(root, name):
    path = root / name
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError('Destination symlink conflict: ' + name)
    return path


def restore(bundle, root):
    bundle, root = Path(bundle).resolve(), Path(root).resolve()
    manifest = verify(bundle)
    for name, checksum in manifest['checkout_sha256'].items():
        candidate = PurePosixPath(name)
        if candidate.is_absolute() or '..' in candidate.parts:
            raise ValueError('Invalid checkout identity path')
        if sha_file(safe_destination(root, name)) != checksum:
            raise ValueError('Checkout differs from bundle code/config: ' + name)
    by_name = {e['path']: e for e in manifest['entries']}
    # Check every collision before writing anything. Existing regular files must match.
    existing = set()
    for name, entry in by_name.items():
        destination = root / name
        if destination.is_symlink():
            if entry['kind'] != 'symlink' or str(destination.readlink()) != entry['link']:
                raise ValueError('Destination symlink conflict: ' + name)
            safe_destination(root, str(PurePosixPath(name).parent / 'placeholder'))
            existing.add(name)
            continue
        safe_destination(root, name)
        if destination.exists():
            if (entry['kind'] != 'file' or not destination.is_file()
                    or sha_file(destination) != entry['sha256']):
                raise ValueError('Destination file conflict: ' + name)
            existing.add(name)
    reader, seen, delayed_links = PartReader(bundle, manifest['parts']), set(), []
    try:
        with tarfile.open(mode='r|', fileobj=reader, bufsize=BLOCK_SIZE) as archive:
            for member in archive:
                entry = by_name.get(member.name)
                if entry is None or member.name in seen:
                    raise ValueError('Unexpected or duplicate archive member')
                seen.add(member.name)
                if entry['kind'] == 'symlink':
                    if not member.issym() or member.linkname != entry['link']:
                        raise ValueError('Symlink metadata differs')
                    if member.name not in existing:
                        delayed_links.append(entry)
                    continue
                if not member.isfile() or member.size != entry['bytes']:
                    raise ValueError('Regular-file metadata differs')
                destination = safe_destination(root, member.name)
                source, checksum = archive.extractfile(member), hashlib.sha256()
                if member.name in existing:
                    for block in iter(lambda: source.read(BLOCK_SIZE), b''):
                        checksum.update(block)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with destination.open('xb') as stream:
                            for block in iter(lambda: source.read(BLOCK_SIZE), b''):
                                checksum.update(block)
                                stream.write(block)
                        if checksum.hexdigest() != entry['sha256']:
                            raise ValueError('Restored file checksum differs')
                    except BaseException:
                        destination.unlink(missing_ok=True)
                        raise
                if checksum.hexdigest() != entry['sha256']:
                    raise ValueError('Archived file checksum differs')
        if seen != set(by_name):
            raise ValueError('Archive is missing manifest members')
        for entry in delayed_links:
            destination = safe_destination(root, entry['path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(entry['link'])
    finally:
        reader.close()
    return {'status': 'restored', 'file_count': len(seen), 'existing_identical_files': len(existing),
        'source_commit': manifest['source_commit']}
