"""Immutable input-addressed BGE vectors; provenance belongs to window mappings."""

import io
import json
import os
from pathlib import Path
from time import perf_counter
import tempfile

import numpy as np

from . import embeddings
from .embeddings import DIMENSION, EmbeddingError, validate_vectors
from .windows import _json, _sha, load_windows


class CacheError(ValueError):
    """An existing cache entry or reused vector is inconsistent; never repair it."""


class VectorCache:
    def __init__(self, root, encoder_info, tokenizer_info):
        self.root = Path(root)
        if self.root.is_symlink(): raise CacheError('Cache root must not be a symlink')
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = {'schema_version': 1, 'encoder': encoder_info, 'tokenizer': tokenizer_info,
            'encoding_implementation_sha256': _sha(Path(embeddings.__file__).read_bytes()),
            'cache_implementation_sha256': _sha(Path(__file__).read_bytes())}
        self.policy_sha256 = _sha(_json(self.policy))

    def key(self, ids):
        if (not isinstance(ids, list) or not 2 <= len(ids) <= 512 or ids[0] != 101 or ids[-1] != 102
                or any(type(x) is not int or not 0 <= x < 30522 for x in ids)):
            raise CacheError('Invalid unpadded BGE input')
        return _sha(_json({'policy_sha256': self.policy_sha256, 'input_sha256': _sha(_json(ids))}))

    def _path(self, ids):
        key = self.key(ids)
        folder = self.root/key[:2]
        if folder.is_symlink(): raise CacheError('Cache prefix must not be a symlink')
        return key, folder/(key+'.bin')

    def get(self, ids):
        key,path = self._path(ids)
        if path.is_symlink(): raise CacheError('Cache entry must not be a symlink')
        if not path.exists(): return None
        try:
            if not path.is_file() or path.stat().st_size > 8192: raise CacheError('Invalid cache file')
            header,body = path.read_bytes().split(b'\n',1)
            meta = json.loads(header)
            if (len(header) > 4096 or len(body) != DIMENSION*4 or set(meta) != {
                    'schema_version','key','policy_sha256','input_sha256','vector_sha256'}
                    or type(meta['schema_version']) is not int or meta['schema_version'] != 1
                    or meta['key'] != key or meta['policy_sha256'] != self.policy_sha256
                    or meta['input_sha256'] != _sha(_json(ids)) or meta['vector_sha256'] != _sha(body)):
                raise CacheError('Cache identity/checksum mismatch')
            vector = np.frombuffer(body,dtype='<f4').copy()
            validate_vectors(vector[None,:],1)
            return vector
        except (OSError,ValueError,KeyError,TypeError) as exc:
            if isinstance(exc,CacheError): raise
            raise CacheError('Corrupt cache entry') from exc

    def put(self, ids, vector):
        vector = np.asarray(vector)
        validate_vectors(vector[None,:],1)
        existing = self.get(ids)
        if existing is not None:
            difference = float(np.max(np.abs(existing-vector)))
            if difference > 1e-5: raise CacheError('Conflicting vectors for the same input/policy')
            return False,difference
        key,path = self._path(ids);path.parent.mkdir(parents=True,exist_ok=True)
        body = vector.astype('<f4',copy=False).tobytes()
        meta = {'schema_version':1,'key':key,'policy_sha256':self.policy_sha256,
            'input_sha256':_sha(_json(ids)),'vector_sha256':_sha(body)}
        fd,name = tempfile.mkstemp(dir=path.parent,prefix='.'+key,suffix='.tmp')
        try:
            with os.fdopen(fd,'wb') as f:f.write(_json(meta)+b'\n'+body);f.flush();os.fsync(f.fileno())
            try:os.link(name,path)
            except FileExistsError:
                # Another writer won publication: check its complete immutable entry.
                return self.put(ids,vector)
        finally:Path(name).unlink(missing_ok=True)
        return True,0.0


def seed_index(cache, index):
    manifest = index.embedding_manifest
    if manifest['encoder'] != cache.policy['encoder'] or index.window_manifest['tokenizer'] != cache.policy['tokenizer']:
        return {'rows':0,'new_entries':0,'existing_entries':0,'skipped':'encoder-or-tokenizer-policy'}
    native = manifest['implementation_sha256'] == cache.policy['encoding_implementation_sha256']
    cached = (manifest['implementation_sha256'] == cache.policy['cache_implementation_sha256']
        and manifest.get('encoding_implementation_sha256') == cache.policy['encoding_implementation_sha256'])
    if not (native or cached):
        return {'rows':0,'new_entries':0,'existing_entries':0,'skipped':'encoding-implementation'}
    new = existing = 0;delta = 0.0
    for window,vector in zip(index.windows,index.vectors):
        made,difference = cache.put(window['input_ids'],vector)
        new += made;existing += not made;delta = max(delta,difference)
    return {'rows':len(index.windows),'new_entries':new,'existing_entries':existing,
        'max_existing_vector_abs_difference':delta,'skipped':None}


def build_cached_embeddings(windows_dir, cache, encoder_factory, output):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    started = perf_counter()
    source,rows = load_windows(windows_dir)
    if source['tokenizer'] != cache.policy['tokenizer']: raise CacheError('Window tokenizer differs from cache')
    unique = {}
    for row in rows:unique.setdefault(cache.key(row['input_ids']),row['input_ids'])
    values = {key:cache.get(ids) for key,ids in unique.items()}
    missing = [key for key,value in values.items() if value is None]
    measurements = {'window_rows':len(rows),'unique_inputs':len(unique),
        'cache_hit_inputs':len(unique)-len(missing),'encoded_inputs':len(missing),'encoder_loads':0,'cache_lookup_seconds':perf_counter()-started,
        'model_constructor_seconds':0.0,'encode_seconds':0.0,'cache_write_seconds':0.0}
    if missing:
        if encoder_factory is None:raise CacheError('Missing vectors require an explicit encoder factory')
        tick = perf_counter()
        encoder = encoder_factory();measurements['encoder_loads'] = 1
        measurements['model_constructor_seconds'] = perf_counter()-tick
        if encoder.info != cache.policy['encoder'] or encoder.tokenizer_info != cache.policy['tokenizer']:
            raise CacheError('Actual encoder differs from the declared cache policy')
        vectors,timings = encoder.encode([unique[key] for key in missing]);validate_vectors(vectors,len(missing))
        measurements.update(timings)
        tick = perf_counter()
        for key,vector in zip(missing,vectors):
            cache.put(unique[key],vector);values[key] = cache.get(unique[key])
        measurements['cache_write_seconds'] = perf_counter()-tick
    tick = perf_counter()
    vectors = np.stack([values[cache.key(r['input_ids'])] for r in rows]) if rows else np.empty((0,DIMENSION),np.float32)
    validate_vectors(vectors,len(rows))
    buffer = io.BytesIO();np.save(buffer,vectors,allow_pickle=False);body = buffer.getvalue()
    mapping = b''.join(_json({key:row[key] for key in ('window_id','pmid','text_version_id')})+b'\n' for row in rows)
    source_bytes = (Path(windows_dir)/'manifest.json').read_bytes()
    identity = {'schema_version':1,'source_window_artifact_id':source['artifact_id'],
        'source_manifest_sha256':_sha(source_bytes),'vectors_sha256':_sha(body),'rows_sha256':_sha(mapping),
        'encoder':cache.policy['encoder'],'implementation_sha256':cache.policy['cache_implementation_sha256']}
    manifest = {**identity,'artifact_id':_sha(_json(identity)),'status':'complete','dimension':DIMENSION,
        'row_count':len(rows),'encoding_implementation_sha256':cache.policy['encoding_implementation_sha256'],
        'cache_policy_sha256':cache.policy_sha256,'measurements':measurements}
    output.mkdir(parents=True,exist_ok=False)
    for name,payload in [('vectors.npy',body),('rows.jsonl',mapping),('windows-manifest.json',source_bytes),
                         ('manifest.json',_json(manifest)+b'\n')]:
        (output/name).write_bytes(payload)
    measurements['artifact_write_seconds'] = perf_counter()-tick
    # Publish final measurements after payloads; artifact identity excludes timings.
    (output/'manifest.json').write_bytes(_json(manifest)+b'\n')
    return manifest
