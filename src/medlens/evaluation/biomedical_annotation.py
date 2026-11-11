"""Blank, blinded biomedical relevance packets bound to frozen PubMed/Dense runs.

No relevance judgments, model calls, qrels, held-out discovery or quality scores.
Pending decisions and explicit human inability to judge are separate states.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from medlens.generation.context import source_section
from medlens.generation.request_contexts import verify_context_sources
from medlens.ingestion.snapshots import load_snapshot
from medlens.parsing.dataset import load_dataset
from medlens.parsing.pubmed import parse_pubmed_xml
from medlens.retrieval.dense import DenseIndex

VERSION = 1
KIND = 'medlens-biomedical-relevance-package'
EXPORT_KIND = 'medlens-biomedical-relevance-annotations'
CASE_FIELDS = ('query_id', 'question', 'relevance_intent', 'question_type', 'specialty')
STATES = ('pending', 'rated', 'unjudgeable')
ARTIFACT_FILES = {
    'dataset': ('manifest.json', 'articles.jsonl', 'source-manifest.json'),
    'windows': ('manifest.json', 'windows.jsonl'),
    'embeddings': ('manifest.json', 'vectors.npy', 'rows.jsonl', 'windows-manifest.json'),
}


class BiomedicalAnnotationError(ValueError):
    pass


def require(condition, message):
    if not condition: raise BiomedicalAnnotationError(message)


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise BiomedicalAnnotationError('Invalid canonical annotation JSON') from exc


def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def byte_hash(body): return hashlib.sha256(body).hexdigest()


def strict_json(body):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'Duplicate JSON key: '+key)
            value[key] = item
        return value
    def constant(_): raise BiomedicalAnnotationError('Nonfinite JSON value')
    try:
        value = json.loads(body, object_pairs_hook=pairs, parse_constant=constant)
        canonical(value)  # Also reject float overflow and invalid Unicode scalars.
        return value
    except BiomedicalAnnotationError: raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise BiomedicalAnnotationError('Invalid annotation JSON') from exc


def regular_bytes(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Expected a regular source file: '+str(path))
    return path.read_bytes()


def safe_child(directory, relative):
    require(isinstance(relative, str) and relative and '\\' not in relative
        and not Path(relative).is_absolute() and all(p not in ('', '.', '..') for p in Path(relative).parts),
        'Unsafe packet file path')
    result = Path(directory)/relative
    require(result.resolve().is_relative_to(Path(directory).resolve()), 'Packet path escapes directory')
    return result


def parse_plan(body):
    plan = strict_json(body)
    require(isinstance(plan, dict) and type(plan.get('schema_version')) is int and plan['schema_version'] == VERSION
        and plan.get('split') == 'dev' and plan.get('heldout_used') is False
        and isinstance(plan.get('cases'), list) and plan['cases'], 'Require an explicit development-only frozen plan; held-out data is forbidden')
    require(type(plan.get('candidate_limit')) is int and 1 <= plan['candidate_limit'] <= 300
        and type(plan.get('annotation_depth_per_method')) is int
        and 1 <= plan['annotation_depth_per_method'] <= min(100,plan['candidate_limit']), 'Invalid predeclared candidate limit/depth')
    require(isinstance(plan.get('pool_methods'),list) and len(plan['pool_methods']) == 2
        and set(plan['pool_methods']) == {'pubmed-best-match','dense'}, 'Only the frozen PubMed/Dense union policy is supported')
    cases, seen = [], set()
    for raw in plan['cases']:
        require(isinstance(raw, dict) and all(isinstance(raw.get(k), str) and raw[k].strip() for k in CASE_FIELDS),
                'Every planned case needs nonempty question, intent, question_type and specialty')
        case = {key: raw[key] for key in CASE_FIELDS}; qid = case['query_id']
        require(re.fullmatch('[A-Za-z0-9_-]{1,64}', qid) is not None and qid not in seen,
                'Duplicate or unsafe planned query ID')
        if 'split' in raw: require(raw['split'] == 'dev', 'Annotation accepts development cases only')
        seen.add(qid); cases.append(case)
    return plan, cases


def _snapshot_and_index(case, directory, *, original_paths, candidate_limit, depth):
    directory = Path(directory)
    run_bytes = regular_bytes(directory/'run.json'); run = strict_json(run_bytes)
    require(isinstance(run, dict) and run.get('status') == 'complete' and run.get('question') == case['question']
        and run.get('generate') is False and run.get('rerank') is False
        and type(run.get('runtime_load_count')) is int and run['runtime_load_count'] == 0
        and run.get('answer', {}).get('status') == 'not_requested'
        and run.get('answer', {}).get('model_calls', []) == [], 'Require an exact completed search-only source run')
    live = run.get('live', {})
    require(live.get('fresh_search_complete') is True and live.get('query_plan',{}).get('question') == case['question'],
            'Source search is not complete/fresh or its question differs')
    snapshot_dir = directory/'snapshot'; snapshot = load_snapshot(snapshot_dir)
    paths = run.get('input_paths', {})
    if original_paths:
        require(all(isinstance(paths.get(k), str) and Path(paths[k]).is_absolute() for k in ARTIFACT_FILES),
                'Search source must record absolute dataset/windows/embeddings paths')
        dirs = {key: Path(paths[key]) for key in ARTIFACT_FILES}
    else: dirs = {key: directory/key for key in ARTIFACT_FILES}
    dataset = load_dataset(dirs['dataset']); index = DenseIndex(dirs['dataset'], dirs['windows'], dirs['embeddings'])
    sm = snapshot.manifest; dm = dataset.manifest
    require(type(live.get('candidate_limit')) is int and live['candidate_limit'] == sm['limit'] == candidate_limit
        and type(run.get('top_k')) is int and run['top_k'] == depth, 'Actual candidate limit/depth differs from frozen plan')
    require(live.get('snapshot_id') == dm['source_snapshot_id'] == sm['snapshot_id']
        and dm['source_manifest_sha256'] == byte_hash(regular_bytes(snapshot_dir/'manifest.json'))
        and regular_bytes(dirs['dataset']/'source-manifest.json') == regular_bytes(snapshot_dir/'manifest.json')
        and dm['pmids'] == sm['pmids'] and bool(sm['pmids']), 'Source snapshot/dataset is empty or incorrectly bound')
    require(live.get('query_plan', {}).get('submitted_query') == sm['query'], 'Submitted PubMed query differs from snapshot')
    identities = {'dataset_id': dm['dataset_id'], 'window_artifact_id': index.window_manifest['artifact_id'],
                  'embedding_artifact_id': index.embedding_manifest['artifact_id']}
    require(all(run.get(k) == v for k, v in identities.items()) and all(live.get(k) == v for k,v in identities.items()),
            'Run/source artifact identities differ')
    # Parsed display text must agree with the actual original EFetch bytes.
    parsed = {}
    for response in sm['responses'][1:]:
        for article in parse_pubmed_xml(snapshot.responses[response['file']], source_file=response['file'],
                snapshot_id=sm['snapshot_id'], retrieved_at=response['retrieved_at']):
            require(article.pmid not in parsed, 'Duplicate parsed PMID')
            parsed[article.pmid] = asdict(article)
    expected = [strict_json(canonical(parsed[pmid])) for pmid in sm['pmids']]
    require(list(dataset.articles) == expected, 'Parsed article text differs from frozen raw PubMed XML')
    for window in index.windows: source_section(index.articles[window['pmid']], window)
    verify_context_sources(run['context'], index)
    retrieval = run.get('retrieval', {})
    hits = retrieval.get('results')
    require(retrieval.get('method') == 'dense-cosine-max-window' and isinstance(hits, list) and hits
        and len(hits) == min(depth,len(index.articles)-len(index.missing_pmids)),
            'Missing Dense candidates for a planned case')
    seen, previous = set(), float('inf')
    windows = {w['window_id']: w for w in index.windows}
    for rank, hit in enumerate(hits, 1):
        pmid = hit.get('pmid'); score = hit.get('score')
        require(pmid in index.articles and pmid not in seen and hit.get('rank') == rank
            and type(score) in (int, float) and math.isfinite(score) and -1 <= score <= 1 and score <= previous,
            'Invalid saved Dense candidate ranking')
        seen.add(pmid); previous = score
        window = windows[hit['best_window']['window_id']]
        require(window['pmid'] == pmid and hit.get('text_version_id') == index.articles[pmid]['text_version']['text_version_id']
            and hit['best_window'] == {k:window[k] for k in ('window_id','kind','section_index','section_label','source_path','char_start','char_end','text','provenance')},
            'Saved Dense candidate differs from its exact source window')
    require([h['pmid'] for h in hits] == run['context']['article_ranking'], 'Evidence ranking differs from Dense source ranking')
    return run, snapshot, dataset, index, dirs


def _item(case, article):
    value = {**case, **{key: deepcopy(article.get(key)) for key in
        ('pmid', 'title', 'raw_title', 'abstract', 'abstract_status', 'other_abstracts', 'journal', 'dates')},
        'text_version_id': article['text_version']['text_version_id'],
        'article_sha256': digest(article)}
    value['item_id'] = 'B-'+digest([case['query_id'], article['pmid'], value['text_version_id']])[:24]
    value['item_sha256'] = digest(value)
    return value


def _pool(case, run, snapshot, dataset, *, depth, plan_sha):
    pubmed = snapshot.manifest['pmids'][:depth]
    dense = [h['pmid'] for h in run['retrieval']['results'][:depth]]
    selected = set(pubmed)|set(dense)
    require(selected, 'Planned case has no annotation candidates')
    by_id = {a['pmid']: a for a in dataset.articles}
    order = sorted(selected, key=lambda p: byte_hash(('biomedical-blind-v1:'+plan_sha+':'+case['query_id']+':'+p+':'+by_id[p]['text_version']['text_version_id']).encode()))
    return [_item(case, by_id[pmid]) for pmid in order], {'pubmed':pubmed, 'dense':dense}


def _write(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: stream.write(body)


def _implementation():
    root = Path(__file__).resolve().parents[3]
    entries = {'biomedical_annotation.py':Path(__file__), 'biomedical_annotation.html':Path(__file__).with_name('biomedical_annotation.html'),
               'prepare_biomedical_annotation.py':root/'scripts/prepare_biomedical_annotation.py'}
    return {name:regular_bytes(path) for name,path in entries.items()}


def build_packet(plan_path, runs_root, output, *, depth=None):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Choose a fresh annotation output directory')
    plan_bytes = regular_bytes(plan_path); plan, cases = parse_plan(plan_bytes); plan_sha = byte_hash(plan_bytes)
    if depth is None: depth = plan['annotation_depth_per_method']
    require(type(depth) is int and depth == plan['annotation_depth_per_method'], 'Annotation depth must exactly match frozen plan')
    runs_root = Path(runs_root).resolve()
    require(not output.resolve().is_relative_to(runs_root) and not runs_root.is_relative_to(output.resolve()),
            'Annotation output and source runs must be disjoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.preparing-', dir=output.parent))
    inputs = {Path(plan_path).resolve():plan_sha}; sources, items = [], []
    def copy(source, relative):
        body = regular_bytes(source); path = Path(source).resolve(); checksum = byte_hash(body)
        require(path not in inputs or inputs[path] == checksum, 'Source changed between reads')
        inputs[path] = checksum; _write(stage/relative, body)
    try:
        _write(stage/'plan.json', plan_bytes)
        for case in cases:
            qid = case['query_id']; directory = runs_root/qid
            require(directory.is_dir() and not directory.is_symlink(), 'Missing or unsafe planned run directory: '+qid)
            run, snapshot, dataset, index, dirs = _snapshot_and_index(case, directory, original_paths=True, candidate_limit=plan['candidate_limit'], depth=depth)
            selected, methods = _pool(case, run, snapshot, dataset, depth=depth, plan_sha=plan_sha); items.extend(selected)
            copy(directory/'run.json', f'sources/{qid}/run.json')
            for name in ['manifest.json', *[r['file'] for r in snapshot.manifest['responses']]]:
                copy(directory/'snapshot'/name, f'sources/{qid}/snapshot/{name}')
            for kind, names in ARTIFACT_FILES.items():
                for name in names: copy(dirs[kind]/name, f'sources/{qid}/{kind}/{name}')
            sources.append({'query_id':qid, 'run_sha256':inputs[(directory/'run.json').resolve()],
                'snapshot_id':snapshot.manifest['snapshot_id'], 'dataset_id':dataset.manifest['dataset_id'],
                'window_artifact_id':index.window_manifest['artifact_id'], 'embedding_artifact_id':index.embedding_manifest['artifact_id'],
                'candidate_pmids':snapshot.manifest['pmids'], 'methods':methods, 'pool_count':len(selected)})
        require(items, 'No annotation items')
        require(len({(i['query_id'],i['pmid']) for i in items}) == len(items), 'Duplicate annotation pair')
        implementation = _implementation()
        for name, body in implementation.items(): _write(stage/'implementation'/name, body)
        code = {name:byte_hash(body) for name,body in implementation.items()}
        identity = {'schema_version':VERSION, 'kind':KIND, 'plan_sha256':plan_sha,
            'queries_sha256':digest(cases), 'items_sha256':digest(items), 'source_bindings_sha256':digest(sources),
            'depth':depth, 'candidate_limit':plan['candidate_limit'], 'pool_methods':plan['pool_methods'], 'split':'dev', 'heldout_used':False, 'blinding':'sha256-biomedical-blind-v1', 'implementation_sha256':code}
        package = {'schema_version':VERSION, 'kind':KIND, 'package_id':digest(identity), 'split':'dev', 'heldout_used':False,
            **{k:identity[k] for k in ('plan_sha256','items_sha256','source_bindings_sha256','implementation_sha256')},
            'queries':cases, 'items':items,
            'annotation_contract':{'states':list(STATES), 'relevance_values':[0,1,2],
                'unjudgeable_relevance':None, 'pending_is_not_unjudgeable':True, 'manual_human_only':True}}
        package_bytes = canonical(package)+b'\n'; _write(stage/'package.json',package_bytes)
        _write(stage/'items.json',canonical(items)+b'\n')
        embedded = canonical(package).decode().replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        template = implementation['biomedical_annotation.html'].decode('utf-8')
        require(template.count('__PACKAGE_JSON__') == 1, 'Template package placeholder differs')
        _write(stage/'index.html',template.replace('__PACKAGE_JSON__',embedded).encode('utf-8'))
        files = {str(p.relative_to(stage)):{'bytes':p.stat().st_size,'sha256':byte_hash(p.read_bytes())}
                 for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema_version':VERSION, 'kind':'medlens-biomedical-annotation-manifest', 'status':'complete',
            'package_id':package['package_id'], 'identity':identity, 'sources':sources, 'files':files,
            'source_original_sha256':{str(p):v for p,v in inputs.items()}, 'human_labels_generated':False,
            'quality_scores_computed':False, 'question_count':len(cases), 'pair_count':len(items)}
        manifest['manifest_id'] = digest(manifest); _write(stage/'manifest.json',canonical(manifest)+b'\n')
        for path,checksum in inputs.items(): require(byte_hash(regular_bytes(path)) == checksum, 'Original source changed during packet build')
        load_packet(stage)
        require(not output.exists() and not output.is_symlink(), 'Annotation destination appeared during build')
        stage.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage)
        raise


@dataclass(frozen=True)
class LoadedPacket:
    directory: Path
    package: dict
    manifest: dict
    manifest_sha256: str


def load_packet(path):
    directory = Path(path)
    if directory.name == 'package.json': directory = directory.parent
    require(directory.is_dir() and not directory.is_symlink(), 'Missing or unsafe annotation package directory')
    manifest_bytes = regular_bytes(directory/'manifest.json'); manifest = strict_json(manifest_bytes)
    require(isinstance(manifest,dict) and type(manifest.get('schema_version')) is int and manifest['schema_version'] == VERSION
        and manifest.get('kind') == 'medlens-biomedical-annotation-manifest' and manifest.get('status') == 'complete'
        and manifest.get('manifest_id') == digest({k:v for k,v in manifest.items() if k != 'manifest_id'}), 'Invalid annotation manifest')
    for relative, identity in manifest['files'].items():
        body = regular_bytes(safe_child(directory,relative))
        require(type(identity.get('bytes')) is int and len(body) == identity['bytes'] and byte_hash(body) == identity['sha256'],
                'Frozen packet checksum mismatch: '+relative)
    package = strict_json(regular_bytes(directory/'package.json')); plan_bytes = regular_bytes(directory/'plan.json')
    plan, cases = parse_plan(plan_bytes); identity = manifest['identity']
    require(identity.get('split') == 'dev' and identity.get('heldout_used') is False
        and identity.get('depth') == plan['annotation_depth_per_method']
        and identity.get('candidate_limit') == plan['candidate_limit'] and identity.get('pool_methods') == plan['pool_methods'],
        'Frozen pool protocol differs from the development plan')
    require(manifest['package_id'] == package.get('package_id') == digest(identity)
        and identity['plan_sha256'] == package.get('plan_sha256') == byte_hash(plan_bytes)
        and identity['queries_sha256'] == digest(cases) and package.get('queries') == cases,
        'Plan/package identity mismatch')
    require(type(package.get('schema_version')) is int and package['schema_version'] == VERSION
        and package.get('kind') == KIND and package.get('split') == 'dev' and package.get('heldout_used') is False, 'Unknown annotation package kind')
    rebuilt, sources = [], []
    for case in cases:
        qid=case['query_id']; run,snapshot,dataset,index,_ = _snapshot_and_index(case,directory/'sources'/qid,original_paths=False, candidate_limit=plan['candidate_limit'], depth=identity['depth'])
        selected,methods = _pool(case,run,snapshot,dataset,depth=identity['depth'],plan_sha=identity['plan_sha256']); rebuilt.extend(selected)
        sources.append({'query_id':qid,'run_sha256':byte_hash(regular_bytes(directory/'sources'/qid/'run.json')),
            'snapshot_id':snapshot.manifest['snapshot_id'],'dataset_id':dataset.manifest['dataset_id'],
            'window_artifact_id':index.window_manifest['artifact_id'],'embedding_artifact_id':index.embedding_manifest['artifact_id'],
            'candidate_pmids':snapshot.manifest['pmids'],'methods':methods,'pool_count':len(selected)})
    require(manifest['sources'] == sources and identity['source_bindings_sha256'] == package.get('source_bindings_sha256') == digest(sources),
            'Frozen source/pool binding differs')
    require(package.get('items') == rebuilt == strict_json(regular_bytes(directory/'items.json'))
        and identity['items_sha256'] == package.get('items_sha256') == digest(rebuilt), 'Annotation items differ from original sources')
    require(set(identity['implementation_sha256']) == {'biomedical_annotation.py','biomedical_annotation.html','prepare_biomedical_annotation.py'}, 'Unknown frozen implementation files')
    code = {name:byte_hash(regular_bytes(directory/'implementation'/name)) for name in identity['implementation_sha256']}
    require(code == identity['implementation_sha256'] == package.get('implementation_sha256'), 'Frozen builder/template/script hash differs')
    embedded = canonical(package).decode().replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    template = regular_bytes(directory/'implementation/biomedical_annotation.html').decode('utf-8')
    require(template.count('__PACKAGE_JSON__') == 1 and regular_bytes(directory/'index.html') == template.replace('__PACKAGE_JSON__',embedded).encode(),
            'Rendered annotation UI differs from frozen template/items')
    require(manifest.get('question_count') == len(cases) and manifest.get('pair_count') == len(rebuilt)
        and manifest.get('human_labels_generated') is False and manifest.get('quality_scores_computed') is False,
        'Annotation inventory/status differs')
    return LoadedPacket(directory,package,manifest,byte_hash(manifest_bytes))


def validate_annotations(packet, body, *, require_finalized=False):
    if not isinstance(packet,LoadedPacket): packet=load_packet(packet)
    value = strict_json(body); package=packet.package
    keys={'schema_version','kind','annotation_method','judgments_origin','package_id','plan_sha256','items_sha256',
          'source_bindings_sha256','finalized','finalized_at','judgments','reference_sha256'}
    require(isinstance(value,dict) and set(value)==keys and type(value.get('schema_version')) is int
        and value['schema_version']==VERSION and value.get('kind')==EXPORT_KIND
        and value.get('annotation_method')=='manual-human' and value.get('judgments_origin')=='human', 'Unexpected human annotation export fields/origin')
    for key in ('package_id','plan_sha256','items_sha256','source_bindings_sha256'):
        require(value[key]==package[key], 'Human export/source binding differs: '+key)
    require(type(value['finalized']) is bool and value['reference_sha256']==digest({k:v for k,v in value.items()if k!='reference_sha256'}),
            'Human export checksum/finalization type differs')
    if value['finalized']:
        require(isinstance(value['finalized_at'],str) and value['finalized_at'].strip(),'Missing explicit finalization time')
        try: stamp=datetime.fromisoformat(value['finalized_at'].replace('Z','+00:00'))
        except ValueError as exc: raise BiomedicalAnnotationError('Invalid finalization time')from exc
        require(stamp.tzinfo is not None,'Finalization timestamp needs timezone')
    else: require(value['finalized_at'] is None,'Draft export cannot have a finalization timestamp')
    require(not require_finalized or value['finalized'],'Freeze requires an explicitly finalized human export')
    expected={item['item_id']:item for item in package['items']};rows=value['judgments']
    require(isinstance(rows,list) and len(rows)==len(expected),'Human export must preserve every item, including pending rows')
    seen=set();counts={state:0 for state in STATES}
    for row in rows:
        require(isinstance(row,dict)and set(row)=={'item_id','query_id','pmid','text_version_id','item_sha256','state','relevance','rationale'},
                'Unexpected annotation row fields')
        key=row['item_id'];require(isinstance(key,str)and key in expected and key not in seen,'Duplicate or unknown annotation item')
        seen.add(key);item=expected[key]
        require(all(row[k]==item[k] for k in('query_id','pmid','text_version_id','item_sha256')),'Annotation query/PMID/text-version binding differs')
        require(isinstance(row['state'],str)and row['state']in STATES and isinstance(row['rationale'],str),'Invalid human decision state/rationale')
        if row['state']=='rated':require(type(row['relevance'])is int and row['relevance']in(0,1,2),'Rated relevance must be integer0,1or2')
        else: require(row['relevance']is None,'Pending/unjudgeable decisions must retain relevance=null')
        counts[row['state']]+=1
    require(seen==set(expected),'Incomplete annotation item coverage')
    require(not value['finalized']or counts['pending']==0,'Finalization cannot include untouched pending items')
    return value,{'total':len(expected),'rated':counts['rated'],'unjudgeable':counts['unjudgeable'],'pending':counts['pending']}


def validate_export(package, human_export, *, require_finalized=False):
    packet=load_packet(package);body=regular_bytes(human_export)
    value,counts=validate_annotations(packet,body,require_finalized=require_finalized)
    return {'schema_version':VERSION,'kind':'biomedical-human-export-validation','package_id':packet.package['package_id'],
        'annotation_method':'manual-human','judgments_origin':'human','original_export_sha256':byte_hash(body),
        'reference_sha256':value['reference_sha256'],'finalized':value['finalized'],'coverage':counts,
        'manifest_sha256':packet.manifest_sha256,'quality_scores_computed':False}


def freeze_export(package, human_export, output):
    packet=load_packet(package);body=regular_bytes(human_export)
    value,counts=validate_annotations(packet,body,require_finalized=True)
    output=Path(output).absolute()
    require(not output.exists()and not output.is_symlink(),'Choose a fresh frozen-reference directory')
    require(not output.resolve().is_relative_to(packet.directory.resolve())and not packet.directory.resolve().is_relative_to(output.resolve()),
            'Frozen reference and annotation packet must be disjoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.freezing-',dir=output.parent))
    try:
        _write(stage/'original-human-export.json',body);(stage/'original-human-export.json').chmod(0o400)
        receipt={'schema_version':VERSION,'kind':'frozen-biomedical-manual-human-reference',
            'annotation_method':'MANUAL HUMAN ANNOTATION','judgments_origin':'human','assistant_generated_labels':False,
            'human_labels_rewritten':False,'package_id':packet.package['package_id'],
            'plan_sha256':packet.package['plan_sha256'],'items_sha256':packet.package['items_sha256'],
            'source_bindings_sha256':packet.package['source_bindings_sha256'],'manifest_sha256':packet.manifest_sha256,
            'original_export_file':'original-human-export.json','original_export_sha256':byte_hash(body),
            'reference_sha256':value['reference_sha256'],'finalized':True,'finalized_at':value['finalized_at'],
            'frozen_at':datetime.now(timezone.utc).isoformat(),'coverage':counts,'quality_scores_computed':False,
            'unknown_is_not_irrelevant':True,'validator_sha256':byte_hash(Path(__file__).read_bytes())}
        receipt['receipt_id']=digest(receipt);_write(stage/'receipt.json',canonical(receipt)+b'\n')
        require(regular_bytes(human_export)==body,'Original human export changed during freeze')
        check=load_packet(packet.directory);require(check.manifest_sha256==packet.manifest_sha256,'Annotation packet changed during freeze')
        require(not output.exists()and not output.is_symlink(),'Frozen-reference destination appeared during freeze')
        stage.rename(output)
        return receipt
    except BaseException:
        shutil.rmtree(stage)
        raise
