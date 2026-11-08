"""Immutable shared PubMed collections with source observations and explicit version selection."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
from time import perf_counter
from types import SimpleNamespace

from medlens.generation.context import source_section
from medlens.parsing.dataset import LoadedDataset, load_dataset
from medlens.parsing.pubmed import _text_version
from medlens.retrieval import windows as window_module
from medlens.retrieval.windows import WindowConfig, article_windows, load_windows, _json, _sha
from medlens.retrieval.embeddings import load_embeddings
from medlens.retrieval.vector_cache import VectorCache, seed_index, build_cached_embeddings

KIND = 'shared-pubmed-collection'
DATASET_KIND = 'pooled-pubmed-dataset'
NATIVE_FILES = ('manifest.json','source-manifest.json','articles.jsonl')
PAYLOADS = ('dataset/manifest.json','windows/manifest.json','embeddings/manifest.json')


class CorpusError(ValueError):
    """A collection cannot be selected or published without a consistent lineage."""


def _write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(_json(value)+b'\n');tmp.replace(path)


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None:raise CorpusError('Retrieval timestamps require a timezone')
    return parsed.astimezone(timezone.utc)


def _semantic(value):
    if isinstance(value,dict):return {k:_semantic(v) for k,v in value.items() if k not in {'provenance','source_path'}}
    if isinstance(value,list):return [_semantic(v) for v in value]
    return value


def _validate_text_version(row):
    def abstract(value):
        return None if value is None else SimpleNamespace(attributes=value['attributes'],
            sections=[SimpleNamespace(**s) for s in value['sections']])
    expected = _text_version(row['pmid'],row.get('title'),row.get('vernacular_title'),row.get('book_title'),
        abstract(row.get('abstract')),[abstract(a) for a in row.get('other_abstracts',[])])
    if asdict(expected) != row['text_version']:raise CorpusError('Article text-version identity mismatch')


def select_records(datasets, resolutions=None):
    resolutions = resolutions or {}
    if not isinstance(resolutions,dict) or any(not isinstance(k,str) or not k.isascii() or not k.isdecimal() or not isinstance(v,str) or not re.fullmatch(r'[0-9a-f]{64}',v) for k,v in resolutions.items()):
        raise CorpusError('Invalid explicit selection resolutions')
    observations = {}
    for dataset_id,dataset in sorted(datasets.items()):
        for article in dataset.articles:
            _validate_text_version(article)
            _time(article['provenance']['retrieved_at'])
            observations.setdefault(article['pmid'],[]).append((dataset_id,article))
    if not set(resolutions) <= set(observations):raise CorpusError('Resolution names an absent PMID')
    articles,catalog = [],[]
    for pmid,items in sorted(observations.items(),key=lambda p:int(p[0])):
        newest = max(_time(a['provenance']['retrieved_at']) for _,a in items)
        candidates = [(sid,a) for sid,a in items if _time(a['provenance']['retrieved_at'])==newest]
        if pmid in resolutions:
            matching = [(sid,a) for sid,a in candidates if sid==resolutions[pmid]]
            if len(matching)!=1:raise CorpusError('Resolution must choose a latest observation: '+pmid)
            selected = matching[0]
        else:
            if len({_sha(_json(_semantic(a))) for _,a in candidates})>1:
                raise CorpusError('Ambiguous latest records for PMID '+pmid+'; choose dataset_id explicitly: '+','.join(sid for sid,_ in candidates))
            selected = min(candidates,key=lambda row:row[0])
        sid,article = selected;articles.append(article)
        catalog.append({'pmid':pmid,'selected_dataset_id':sid,'selected_record_sha256':_sha(_json(article)),
            'selected_text_version_id':article['text_version']['text_version_id'],
            'observations':[{'dataset_id':s,'record_sha256':_sha(_json(a)),
                'text_version_id':a['text_version']['text_version_id'],'retrieved_at':a['provenance']['retrieved_at']}
                for s,a in items]})
    return tuple(articles),catalog


def _child(root, name):
    path = root
    for part in Path(name).parts:
        if part in {'..','/'}:raise CorpusError('Invalid collection path')
        path = path/part
        if path.is_symlink():raise CorpusError('Collection internals must not be symlinks')
    return path


def _read_root(root):
    if root.is_symlink():raise CorpusError('Collection root must not be a symlink')
    m = json.loads(_child(root,'manifest.json').read_bytes())
    if m.get('kind')!=KIND or m.get('schema_version')!=1 or m.get('status')!='complete':
        raise CorpusError('Shared collection is incomplete or unsupported')
    if _sha(_json({k:v for k,v in m.items() if k!='manifest_id'}))!=m['manifest_id']:
        raise CorpusError('Shared manifest identity mismatch')
    if _sha(_json(m['identity']))!=m['collection_id'] or set(m['files'])!=set(PAYLOADS):
        raise CorpusError('Shared collection identity mismatch')
    for name,checksum in m['files'].items():
        if _sha(_child(root,name).read_bytes())!=checksum:raise CorpusError('Shared payload checksum mismatch: '+name)
    return m


def _read_dataset(directory):
    m = json.loads(_child(directory,'manifest.json').read_bytes())
    if m.get('kind')!=DATASET_KIND or m.get('status')!='complete' or m.get('schema_version')!=1:
        raise CorpusError('Incomplete pooled dataset')
    identity = {k:m[k] for k in ('schema_version','kind','articles_sha256','catalog_sha256','sources','resolutions','selection_policy','implementation_sha256')}
    if _sha(_json(identity))!=m['dataset_id']:raise CorpusError('Pooled dataset identity mismatch')
    datasets = {};labels = []
    for source in m['sources']:
        sid = source['dataset_id']
        if (not re.fullmatch(r'[0-9a-f]{64}',sid) or sid in datasets
                or source['collections']!=sorted(set(source['collections'])) or not source['collections']):
            raise CorpusError('Invalid/duplicate pooled source')
        labels.extend(source['collections'])
        folder = _child(directory,'sources/'+sid)
        for filename in NATIVE_FILES:_child(folder,filename)
        if _sha((folder/'manifest.json').read_bytes())!=source['manifest_sha256']:
            raise CorpusError('Copied source manifest differs')
        dataset = load_dataset(folder)
        if dataset.manifest['dataset_id']!=sid or dataset.manifest['article_count']!=source['article_count']:raise CorpusError('Copied source identity differs')
        datasets[sid] = dataset
    if len(set(labels))!=len(labels) or not datasets:raise CorpusError('Duplicate labels or missing sources')
    articles,catalog = select_records(datasets,m['resolutions'])
    body = _child(directory,'articles.jsonl').read_bytes();stored = _child(directory,'catalog.json').read_bytes()
    if (_sha(body)!=m['articles_sha256'] or _sha(stored)!=m['catalog_sha256']
            or json.loads(stored)!=catalog or tuple(json.loads(line) for line in body.splitlines())!=articles
            or len(articles)!=m['article_count'] or [a['pmid'] for a in articles]!=m['pmids']):
        raise CorpusError('Pooled records or selection lineage differ from saved sources')
    return LoadedDataset(m,articles)


def load_pooled_dataset(directory):
    directory = Path(directory)
    try:
        root = _read_root(directory.parent);dataset = _read_dataset(directory)
        if root['identity']['dataset_id']!=dataset.manifest['dataset_id']:
            raise CorpusError('Root and pooled dataset disagree')
        return dataset
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        if isinstance(exc,CorpusError):raise
        raise CorpusError('Invalid shared collection') from exc


def _validate_artifacts(root, dataset):
    wm,rows = load_windows(root/'windows');em,mapping,vectors = load_embeddings(root/'embeddings')
    if (wm['dataset_id']!=dataset.manifest['dataset_id'] or wm['articles_sha256']!=dataset.manifest['articles_sha256']
            or em['source_window_artifact_id']!=wm['artifact_id']
            or list(mapping)!=[{k:r[k] for k in ('window_id','pmid','text_version_id')} for r in rows]):
        raise CorpusError('Shared artifact mapping differs')
    articles = {a['pmid']:a for a in dataset.articles};covered = set()
    for row in rows:
        source_section(articles[row['pmid']],row);covered.add(row['pmid'])
    if [p for p in articles if p not in covered]!=wm['empty_pmids']:raise CorpusError('Shared window coverage differs')
    return wm,em


def inspect_collection(root):
    root = Path(root);m = _read_root(root);dataset = _read_dataset(root/'dataset');wm,em = _validate_artifacts(root,dataset)
    expected = {'dataset_id':dataset.manifest['dataset_id'],'window_artifact_id':wm['artifact_id'],
        'embedding_artifact_id':em['artifact_id'],'implementation_sha256':m['identity']['implementation_sha256']}
    if expected!=m['identity']:raise CorpusError('Root artifact identities differ')
    catalog = json.loads((root/'dataset/catalog.json').read_bytes());sources = dataset.manifest['sources']
    counts = {'input_collections':sum(len(s['collections']) for s in sources),'native_datasets':len(sources),
        'article_memberships':sum(s['article_count']*len(s['collections']) for s in sources),
        'unique_pmids':len(dataset.articles),'duplicate_memberships_removed':sum(s['article_count']*len(s['collections']) for s in sources)-len(dataset.articles),
        'pmids_with_multiple_text_versions':sum(len({o['text_version_id'] for o in c['observations']})>1 for c in catalog),
        'retained_text_versions':sum(len({o['text_version_id'] for o in c['observations']}) for c in catalog),'windows':wm['window_count']}
    if counts!=m['counts']:raise CorpusError('Shared counts differ from lineage')
    return m


def build_collection(sources, tokenizer, tokenizer_info, cache_root, output, *, encoder_factory=None,
                     encoder_info=None, config=None, resolutions=None, seed=True, reuse=None):
    from medlens.retrieval.dense import DenseIndex
    output,cache_root = Path(output),Path(cache_root)
    if output.exists() or output.is_symlink():raise FileExistsError(output)
    if not sources:raise CorpusError('At least one native collection is required')
    roots = [Path(s[k]).resolve() for s in sources for k in ('dataset','windows','embeddings')]
    if reuse:roots.append(Path(reuse).resolve())
    for a,b in [(output.resolve(),p) for p in roots]+[(cache_root.resolve(),p) for p in roots]+[(output.resolve(),cache_root.resolve())]:
        if a.is_relative_to(b) or b.is_relative_to(a):raise CorpusError('Output, cache and input artifact roots must be disjoint')
    started = perf_counter();indices = [];names = set()
    for spec in sources:
        label = spec['query_id']
        if not isinstance(label,str) or not label or label in names:raise CorpusError('Source labels must be distinct')
        names.add(label)
        for filename in NATIVE_FILES:
            if (Path(spec['dataset'])/filename).is_symlink():raise CorpusError('Native source files must not be symlinks')
        load_dataset(spec['dataset'])  # Nested pooled inputs are not a native source list.
        idx = DenseIndex(spec['dataset'],spec['windows'],spec['embeddings'])
        for key,actual in [('dataset_id',idx.dataset.manifest['dataset_id']),('window_artifact_id',idx.window_manifest['artifact_id']),('embedding_artifact_id',idx.embedding_manifest['artifact_id'])]:
            if key in spec and spec[key]!=actual:raise CorpusError('Source profile identity differs: '+label)
        # Reconstruct every source window, binding its encoded IDs to its actual text.
        expected = [w for a in idx.dataset.articles for w in article_windows(a,tokenizer,WindowConfig(**idx.window_manifest['config']))]
        if idx.window_manifest['tokenizer']!=tokenizer_info or expected!=list(idx.windows):
            raise CorpusError('Source windows/tokenizer differ from exact reconstruction')
        indices.append((spec,idx))
    indices.sort(key=lambda x:(x[1].dataset.manifest['dataset_id'],_sha((Path(x[0]['dataset'])/'manifest.json').read_bytes()),x[0]['query_id']))
    datasets = {};descriptors = {};source_paths = {}
    for spec,index in indices:
        sid = index.dataset.manifest['dataset_id']
        if sid not in datasets:
            datasets[sid] = index.dataset;source_paths[sid] = Path(spec['dataset'])
            descriptors[sid] = {'dataset_id':sid,'manifest_sha256':_sha((source_paths[sid]/'manifest.json').read_bytes()),'article_count':len(index.dataset.articles),'collections':[]}
        descriptors[sid]['collections'].append(spec['query_id'])
    for desc in descriptors.values():desc['collections'].sort()
    articles,catalog = select_records(datasets,resolutions)
    config = config or WindowConfig();rows = [w for a in articles for w in article_windows(a,tokenizer,config)]
    counts = {'input_collections':len(sources),'native_datasets':len(datasets),
        'article_memberships':sum(len(i.dataset.articles) for _,i in indices),'unique_pmids':len(articles),
        'duplicate_memberships_removed':sum(len(i.dataset.articles) for _,i in indices)-len(articles),
        'pmids_with_multiple_text_versions':sum(len({o['text_version_id'] for o in c['observations']})>1 for c in catalog),
        'retained_text_versions':sum(len({o['text_version_id'] for o in c['observations']}) for c in catalog),
        'windows':len(rows)}
    output.mkdir(parents=True,exist_ok=False);_write(output/'manifest.json',{'schema_version':1,'kind':KIND,'status':'running'})
    try:
        directory = output/'dataset';directory.mkdir()
        for sid in datasets:
            target = directory/'sources'/sid;target.mkdir(parents=True)
            for name in NATIVE_FILES:shutil.copyfile(source_paths[sid]/name,target/name)
        body = b''.join(_json(a)+b'\n' for a in articles);catalog_body = _json(catalog)+b'\n'
        identity = {'schema_version':1,'kind':DATASET_KIND,'articles_sha256':_sha(body),'catalog_sha256':_sha(catalog_body),
            'sources':list(descriptors.values()),'resolutions':resolutions or {},
            'selection_policy':'latest-retrieval-with-explicit-conflicting-ties-v1','implementation_sha256':_sha(Path(__file__).read_bytes())}
        dm = {**identity,'dataset_id':_sha(_json(identity)),'status':'complete','article_count':len(articles),'pmids':[a['pmid'] for a in articles]}
        (directory/'articles.jsonl').write_bytes(body);(directory/'catalog.json').write_bytes(catalog_body);_write(directory/'manifest.json',dm)
        window_body = b''.join(_json(r)+b'\n' for r in rows)
        wi = {'schema_version':1,'dataset_id':dm['dataset_id'],'articles_sha256':dm['articles_sha256'],'config':asdict(config),
            'tokenizer':tokenizer_info,'windows_sha256':_sha(window_body),'implementation_sha256':_sha(Path(__file__).read_bytes()),
            'input_format':'CLS-title_ids-section_token_slice-SEP-v1'}
        covered = {r['pmid'] for r in rows}
        wm = {**wi,'artifact_id':_sha(_json(wi)),'status':'complete','article_count':len(articles),'window_count':len(rows),
            'empty_pmids':[a['pmid'] for a in articles if a['pmid'] not in covered],
            'title_only_windows':sum(r['kind']=='title' for r in rows),'max_input_token_count':max((r['input_token_count'] for r in rows),default=0)}
        (output/'windows').mkdir();(output/'windows/windows.jsonl').write_bytes(window_body);_write(output/'windows/manifest.json',wm)
        cache = VectorCache(cache_root,encoder_info or indices[0][1].embedding_manifest['encoder'],tokenizer_info)
        seed_reports = []
        if seed:
            for spec,index in indices:seed_reports.append({'source':spec['query_id'],**seed_index(cache,index)})
        if reuse:
            inspect_collection(reuse)
            old = DenseIndex(Path(reuse)/'dataset',Path(reuse)/'windows',Path(reuse)/'embeddings')
            seed_reports.append({'source':'previous-shared-index',**seed_index(cache,old)})
        prepared = perf_counter()
        em = build_cached_embeddings(output/'windows',cache,encoder_factory,output/'embeddings')
        validated = _read_dataset(directory);_validate_artifacts(output,validated)
        ident = {'dataset_id':dm['dataset_id'],'window_artifact_id':wm['artifact_id'],'embedding_artifact_id':em['artifact_id'],
            'implementation_sha256':_sha(Path(__file__).read_bytes())}
        result = {'schema_version':1,'kind':KIND,'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),
            'identity':ident,'collection_id':_sha(_json(ident)),'counts':counts,
            'files':{name:_sha((output/name).read_bytes()) for name in PAYLOADS},
            'measurements':{'source_preparation_seconds':prepared-started,'total_seconds':perf_counter()-started,
                'cache':em['measurements'],'cache_policy_sha256':cache.policy_sha256,'seeds':seed_reports}}
        result['manifest_id'] = _sha(_json(result));_write(output/'manifest.json',result)
        inspect_collection(output)
        return result
    except BaseException as exc:
        _write(output/'manifest.json',{'schema_version':1,'kind':KIND,'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}})
        raise
