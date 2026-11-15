"""Blank human source references for an explicitly frozen confirmation split.

This package contains source articles and human input controls, never model
answers, suggested judgments or quality metrics. Legacy development packages
retain their separate contracts.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import tempfile

from .biomedical_annotation import (
    ARTIFACT_FILES, CASE_FIELDS, STATES, BiomedicalAnnotationError,
    _item, _pool, _snapshot_and_index, _write, byte_hash, canonical, digest,
    regular_bytes, require, safe_child, strict_json,
)
from . import biomedical_annotation as legacy
from medlens.generation.context import source_section
from medlens.generation.request_contexts import verify_context_sources
from medlens.ingestion.snapshots import load_snapshot
from medlens.parsing.dataset import load_dataset
from medlens.parsing.pubmed import parse_pubmed_xml
from medlens.retrieval.dense import DenseIndex
from medlens import pubmed_query as query_planner

VERSION = 1
KIND = 'medlens-human-confirmation-source-package'
EXPORT_KIND = 'medlens-human-confirmation-source-annotations'
QUERY_STATES = ('pending', 'yes', 'no', 'unknown')
IMPLEMENTATION_FILES = ('confirmation_annotation.py', 'confirmation_annotation.html', 'biomedical_annotation.py', 'pubmed_query.py')
REQUEST_ID_FIELDS = ('question', 'dataset_id', 'window_artifact_id', 'embedding_artifact_id',
    'top_k', 'generate', 'rerank', 'answer_style', 'tokenizer', 'runtime_policy',
    'date_selection', 'implementation_sha256')


def parse_plan(body):
    plan = strict_json(body)
    require(isinstance(plan, dict) and type(plan.get('schema_version')) is int
        and plan['schema_version'] == VERSION and plan.get('kind') == 'medlens-human-confirmation-plan'
        and plan.get('split') == 'test' and plan.get('purpose') == 'prospective independent manual-human confirmation; not model tuning'
        and plan.get('dataset_id') == 'human-confirmation-v1', 'Require an explicit human confirmation test plan')
    require(type(plan.get('candidate_limit')) is int and plan['candidate_limit'] == 100
        and type(plan.get('annotation_depth_per_method')) is int
        and plan['annotation_depth_per_method'] == 10
        and isinstance(plan.get('pool_methods'), list) and len(plan['pool_methods']) == 2
        and set(plan['pool_methods']) == {'pubmed-best-match', 'dense'}, 'Confirmation requires the declared 100/top10 PubMed/Dense pool')
    require(isinstance(plan.get('cases'), list) and plan['cases'], 'Confirmation case list must not be empty')
    seen, groups, cases = set(), set(), []
    for raw in plan['cases']:
        require(isinstance(raw, dict) and all(isinstance(raw.get(k), str) and raw[k].strip()
            for k in CASE_FIELDS), 'Every confirmation case needs the five source-annotation fields')
        case = {key: raw[key] for key in CASE_FIELDS}
        if 'group_id' in raw:
            require(isinstance(raw['group_id'], str) and raw['group_id'].strip(), 'Require a nonempty confirmation group ID')
            require(raw['group_id'] not in groups, 'Duplicate confirmation group ID')
            groups.add(raw['group_id'])
            case['group_id'] = raw['group_id']
        qid = case['query_id']
        require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}', qid) is not None and qid not in seen,
            'Duplicate or unsafe confirmation query ID')
        require('split' not in raw or raw['split'] == 'test', 'Confirmation cases cannot become development cases')
        seen.add(qid)
        cases.append(case)
    require('question_count' not in plan or (type(plan['question_count']) is int
        and plan['question_count'] == len(cases)), 'Declared question count differs from the frozen case list')
    return plan, cases


def _no_hits(case, directory, *, original_paths, candidate_limit, depth):
    """Verify empty searches and title-only candidate sets without inventing labels."""
    run = strict_json(regular_bytes(directory / 'run.json'))
    require(run.get('status') == 'complete' and run.get('question') == case['question']
        and type(run.get('runtime_load_count')) is int and run['runtime_load_count'] == 0
        and run.get('answer', {}).get('status') == 'not_requested'
        and not run.get('answer', {}).get('claims') and not run.get('answer', {}).get('model_calls')
        and run.get('retrieval', {}).get('results') == [] and run.get('context', {}).get('sources') == [],
        'Require a completed empty search-only source run')
    snapshot = load_snapshot(directory / 'snapshot')
    sm = snapshot.manifest
    live, paths = run.get('live', {}), run.get('input_paths', {})
    require(live.get('fresh_search_complete') is True and live.get('candidate_limit') == sm['limit'] == candidate_limit
        and live.get('query_plan', {}).get('question') == case['question']
        and live.get('query_plan', {}).get('submitted_query') == sm['query'], 'Empty-case fresh search binding differs')
    if original_paths:
        require(isinstance(paths.get('dataset'), str) and Path(paths['dataset']).is_absolute(), 'Require absolute source dataset path')
        dirs = {'dataset': Path(paths['dataset'])}
    else:
        dirs = {'dataset': directory / 'dataset'}
    dataset = load_dataset(dirs['dataset'])
    dm = dataset.manifest
    require(live.get('snapshot_id') == dm['source_snapshot_id'] == sm['snapshot_id']
        and live.get('dataset_id') == dm['dataset_id'] and dm['pmids'] == sm['pmids']
        and dm['source_manifest_sha256'] == byte_hash(regular_bytes(directory / 'snapshot/manifest.json'))
        and regular_bytes(dirs['dataset'] / 'source-manifest.json') == regular_bytes(directory / 'snapshot/manifest.json'),
        'Empty-case snapshot/dataset identity differs')
    parsed = {}
    for response in sm['responses'][1:]:
        for article in parse_pubmed_xml(snapshot.responses[response['file']], source_file=response['file'],
                snapshot_id=sm['snapshot_id'], retrieved_at=response['retrieved_at']):
            require(article.pmid not in parsed, 'Duplicate parsed PMID')
            parsed[article.pmid] = asdict(article)
    require(list(dataset.articles) == [strict_json(canonical(parsed[p])) for p in sm['pmids']],
        'Empty-case display text differs from frozen raw XML')
    index = None
    if sm['pmids']:
        require(run.get('generate') is False and run.get('rerank') is False and run.get('top_k') == depth,
            'Title-only sources require exact search-only depth')
        if original_paths:
            require(all(isinstance(paths.get(k), str) and Path(paths[k]).is_absolute() for k in ('windows', 'embeddings')),
                'Require original window/vector paths')
            dirs.update({k: Path(paths[k]) for k in ('windows', 'embeddings')})
        else:
            dirs.update({k: directory / k for k in ('windows', 'embeddings')})
        index = DenseIndex(*(dirs[k] for k in ('dataset', 'windows', 'embeddings')))
        require(len(index.articles) == len(index.missing_pmids), 'Searchable source candidates cannot silently lose their ranking')
        for key, value in {'dataset_id': dm['dataset_id'], 'window_artifact_id': index.window_manifest['artifact_id'],
                           'embedding_artifact_id': index.embedding_manifest['artifact_id']}.items():
            require(run.get(key) == live.get(key) == value, 'Title-only source artifact binding differs')
        for window in index.windows:
            source_section(index.articles[window['pmid']], window)
        verify_context_sources(run['context'], index)
    else:
        require(not dataset.articles and run.get('generate', False) is False
            and run.get('rerank', False) is False, 'An empty PubMed run cannot generate or rerank')
    return run, snapshot, dataset, index, dirs


def _sources(case, directory, *, original_paths, plan, expected_query_plan):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'Missing or unsafe confirmation source directory')
    run = strict_json(regular_bytes(directory / 'run.json'))
    require(isinstance(expected_query_plan, dict) and expected_query_plan.get('question') == case['question']
        and expected_query_plan.get('override') is None
        and canonical(run.get('live', {}).get('query_plan')) == canonical(expected_query_plan),
        'Saved candidate query plan differs from the frozen expected question plan')
    args = dict(original_paths=original_paths, candidate_limit=plan['candidate_limit'], depth=plan['annotation_depth_per_method'])
    if run.get('retrieval', {}).get('results'):
        result = _snapshot_and_index(case, directory, **args)
    else:
        result = _no_hits(case, directory, **args)
    snapshot = result[1]
    if snapshot.manifest['pmids']:
        require(all(key in run for key in REQUEST_ID_FIELDS), 'Missing source request identity fields')
        expected_id = digest({key: run[key] for key in REQUEST_ID_FIELDS})
    else:
        expected_id = digest({'plan': expected_query_plan, 'snapshot_id': snapshot.manifest['snapshot_id'],
            'generate': False, 'answer_style': run.get('answer_style')})
    require(run.get('request_id') == expected_id, 'Source request identity differs from the frozen question/encoding inputs')
    return result


def _selection(case, run, snapshot, dataset, plan, plan_sha):
    if snapshot.manifest['pmids']:
        items, methods = _pool(case, run, snapshot, dataset, depth=plan['annotation_depth_per_method'], plan_sha=plan_sha)
    else:
        items, methods = [], {'pubmed': [], 'dense': []}
    state = 'empty-pubmed' if not snapshot.manifest['pmids'] else 'searchable' if methods['dense'] else 'no-searchable-fragments'
    source = {'query_id': case['query_id'], 'snapshot_id': snapshot.manifest['snapshot_id'],
        'dataset_id': dataset.manifest['dataset_id'], 'candidate_pmids': snapshot.manifest['pmids'],
        'methods': methods, 'pool_count': len(items), 'source_state': state}
    if 'group_id' in case:
        source['group_id'] = case['group_id']
    binding = {'query_id': case['query_id'], 'question_sha256': digest(case['question']),
        'source_pool_sha256': digest([{k: i[k] for k in ('item_id', 'pmid', 'text_version_id', 'item_sha256')} for i in items])}
    return items, source, binding


def _inventory(sources):
    return [{key: source[key] for key in ('query_id', 'group_id', 'source_state', 'pool_count') if key in source}
        for source in sources]


def _implementation():
    return {name: regular_bytes(Path(query_planner.__file__) if name == 'pubmed_query.py'
        else Path(legacy.__file__) if name == 'biomedical_annotation.py'
        else Path(__file__).with_name(name)) for name in IMPLEMENTATION_FILES}


def build_packet(plan_path, runs_root, output):
    plan_bytes = regular_bytes(plan_path)
    plan, cases = parse_plan(plan_bytes)
    # Capture the complete question-to-candidate policy before opening any run.
    # Later verification reads this frozen data, never executes archived Python.
    planner_body = regular_bytes(query_planner.__file__)
    expected_plans = {case['query_id']: query_planner.plan_query(case['question']) for case in cases}
    require(regular_bytes(query_planner.__file__) == planner_body, 'Planner changed while freezing candidate query plans')
    plan_sha, runs_root, output = byte_hash(plan_bytes), Path(runs_root).resolve(), Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Choose a fresh confirmation packet directory')
    require(not output.resolve().is_relative_to(runs_root) and not runs_root.is_relative_to(output.resolve()), 'Packet and source directories must be disjoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.preparing-', dir=output.parent))
    inputs, items, sources, bindings = {Path(plan_path).resolve(): plan_sha}, [], [], []
    def copy(path, relative):
        body = regular_bytes(path)
        path, checksum = Path(path).resolve(), byte_hash(body)
        require(path not in inputs or inputs[path] == checksum, 'Source changed between reads')
        inputs[path] = checksum
        _write(stage / relative, body)
    try:
        _write(stage / 'plan.json', plan_bytes)
        for case in cases:
            qid = case['query_id']
            directory = runs_root / qid
            run, snapshot, dataset, index, dirs = _sources(case, directory, original_paths=True,
                plan=plan, expected_query_plan=expected_plans[qid])
            selected, source, binding = _selection(case, run, snapshot, dataset, plan, plan_sha)
            items.extend(selected)
            copy(directory / 'run.json', f'sources/{qid}/run.json')
            for name in ['manifest.json', *[r['file'] for r in snapshot.manifest['responses']]]:
                copy(directory / 'snapshot' / name, f'sources/{qid}/snapshot/{name}')
            for kind, directory_path in dirs.items():
                for name in ARTIFACT_FILES[kind]:
                    copy(directory_path / name, f'sources/{qid}/{kind}/{name}')
            source['run_sha256'] = inputs[(directory / 'run.json').resolve()]
            sources.append(source)
            bindings.append(binding)
        require(len({i['item_id'] for i in items}) == len(items), 'Duplicate source annotation items')
        implementation = _implementation()
        require(implementation['pubmed_query.py'] == planner_body, 'Planner source changed during packet build')
        for name, body in implementation.items():
            _write(stage / 'implementation' / name, body)
        code = {name: byte_hash(body) for name, body in implementation.items()}
        identity = {'schema_version': VERSION, 'kind': KIND, 'split': 'test', 'dataset_id': plan['dataset_id'],
            'plan_sha256': plan_sha, 'queries_sha256': digest(cases), 'items_sha256': digest(items),
            'source_bindings_sha256': digest(sources), 'query_bindings_sha256': digest(bindings),
            'expected_query_plans_sha256': digest(expected_plans),
            'implementation_sha256': code}
        package = {**identity, 'package_id': digest(identity), 'queries': cases, 'items': items, 'query_bindings': bindings,
            'query_inventory': _inventory(sources),
            'expected_query_plans': expected_plans,
            'annotation_contract': {'article_states': list(STATES), 'article_relevance_values': [0, 1, 2],
                'query_states': list(QUERY_STATES), 'answerability_scope': 'frozen-visible-annotation-pool',
                'manual_human_only': True, 'no_model_answers_or_suggested_labels': True,
                'pending_is_not_unknown': True}}
        _write(stage / 'package.json', canonical(package) + b'\n')
        _write(stage / 'items.json', canonical(items) + b'\n')
        embedded = canonical(package).decode().replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        template = implementation['confirmation_annotation.html'].decode()
        require(template.count('__PACKAGE_JSON__') == 1, 'Template placeholder differs')
        _write(stage / 'index.html', template.replace('__PACKAGE_JSON__', embedded).encode())
        files = {str(p.relative_to(stage)): {'bytes': p.stat().st_size, 'sha256': byte_hash(p.read_bytes())}
            for p in sorted(stage.rglob('*')) if p.is_file()}
        manifest = {'schema_version': VERSION, 'kind': 'medlens-human-confirmation-source-manifest', 'status': 'complete',
            'package_id': package['package_id'], 'identity': identity, 'sources': sources, 'files': files,
            'question_count': len(cases), 'pair_count': len(items), 'human_labels_generated': False,
            'answer_generation_performed': False, 'quality_scores_computed': False,
            'source_original_sha256': {str(p): checksum for p, checksum in inputs.items()}}
        manifest['manifest_id'] = digest(manifest)
        _write(stage / 'manifest.json', canonical(manifest) + b'\n')
        for path, checksum in inputs.items():
            require(byte_hash(regular_bytes(path)) == checksum, 'Original source changed during packet build')
        load_packet(stage)
        require(not output.exists() and not output.is_symlink(), 'Packet destination appeared during build')
        stage.rename(output)
        return package
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
    require(directory.is_dir() and not directory.is_symlink(), 'Missing or unsafe confirmation packet')
    body = regular_bytes(directory / 'manifest.json')
    manifest = strict_json(body)
    require(isinstance(manifest, dict) and type(manifest.get('schema_version')) is int and manifest['schema_version'] == VERSION
        and manifest.get('kind') == 'medlens-human-confirmation-source-manifest'
        and manifest.get('status') == 'complete' and manifest.get('manifest_id') == digest({k: v for k, v in manifest.items() if k != 'manifest_id'}),
        'Invalid confirmation manifest')
    for relative, expected in manifest['files'].items():
        data = regular_bytes(safe_child(directory, relative))
        require(type(expected['bytes']) is int and expected['bytes'] == len(data) and expected['sha256'] == byte_hash(data), 'Frozen packet checksum mismatch: ' + relative)
    plan_bytes = regular_bytes(directory / 'plan.json')
    plan, cases = parse_plan(plan_bytes)
    package = strict_json(regular_bytes(directory / 'package.json'))
    identity = manifest['identity']
    require(isinstance(package, dict) and isinstance(identity, dict)
        and package.get('kind') == KIND and package.get('split') == 'test' and package.get('dataset_id') == plan['dataset_id']
        and type(package.get('schema_version')) is int and package['schema_version'] == VERSION
        and package.get('package_id') == manifest['package_id'] == digest(identity), 'Confirmation package identity differs')
    require(identity.get('kind') == KIND and identity.get('split') == 'test' and identity.get('dataset_id') == plan['dataset_id']
        and all(package.get(k) == v for k, v in identity.items())
        and identity['plan_sha256'] == byte_hash(plan_bytes) and package.get('queries') == cases
        and identity['queries_sha256'] == digest(cases), 'Frozen confirmation plan differs')
    expected_plans = package.get('expected_query_plans')
    require(isinstance(expected_plans, dict) and set(expected_plans) == {c['query_id'] for c in cases}
        and digest(expected_plans) == identity.get('expected_query_plans_sha256'), 'Frozen expected candidate query plans differ')
    items, sources, bindings = [], [], []
    for case in cases:
        source_dir = directory / 'sources' / case['query_id']
        run, snapshot, dataset, _, _ = _sources(case, source_dir, original_paths=False,
            plan=plan, expected_query_plan=expected_plans[case['query_id']])
        selected, source, binding = _selection(case, run, snapshot, dataset, plan, identity['plan_sha256'])
        source['run_sha256'] = byte_hash(regular_bytes(source_dir / 'run.json'))
        items.extend(selected); sources.append(source); bindings.append(binding)
    require(items == package.get('items') == strict_json(regular_bytes(directory / 'items.json'))
        and digest(items) == identity['items_sha256'] and sources == manifest['sources']
        and digest(sources) == identity['source_bindings_sha256']
        and bindings == package.get('query_bindings') and digest(bindings) == identity['query_bindings_sha256'], 'Confirmation source/items/query bindings differ')
    inventory = _inventory(sources)
    require(package.get('query_inventory') == inventory, 'Confirmation question inventory differs')
    require(set(identity['implementation_sha256']) == set(IMPLEMENTATION_FILES), 'Unknown frozen builder files')
    code = {name: byte_hash(regular_bytes(directory / 'implementation' / name)) for name in IMPLEMENTATION_FILES}
    require(code == identity['implementation_sha256'], 'Frozen confirmation implementation differs')
    template = regular_bytes(directory / 'implementation/confirmation_annotation.html').decode()
    embedded = canonical(package).decode().replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    require(template.count('__PACKAGE_JSON__') == 1 and regular_bytes(directory / 'index.html') == template.replace('__PACKAGE_JSON__', embedded).encode(), 'Frozen rendered confirmation UI differs')
    require(manifest.get('question_count') == len(cases) and manifest.get('pair_count') == len(items)
        and manifest.get('human_labels_generated') is False and manifest.get('answer_generation_performed') is False
        and manifest.get('quality_scores_computed') is False, 'Confirmation status/counts differ')
    return LoadedPacket(directory, package, manifest, byte_hash(body))


def validate_annotations(packet, body, *, require_finalized=False):
    if not isinstance(packet, LoadedPacket): packet = load_packet(packet)
    value = strict_json(body)
    package = packet.package
    binding_keys = ('package_id', 'plan_sha256', 'items_sha256', 'source_bindings_sha256', 'query_bindings_sha256')
    keys = {'schema_version', 'kind', 'annotation_method', 'judgments_origin', *binding_keys,
            'finalized', 'finalized_at', 'judgments', 'query_judgments', 'reference_sha256'}
    require(isinstance(value, dict) and set(value) == keys and type(value.get('schema_version')) is int
        and value['schema_version'] == VERSION and value.get('kind') == EXPORT_KIND
        and value.get('annotation_method') == 'manual-human' and value.get('judgments_origin') == 'human', 'Unexpected confirmation human export fields/origin')
    require(all(value[k] == package[k] for k in binding_keys), 'Human export confirmation/source binding differs')
    require(type(value['finalized']) is bool and value['reference_sha256'] == digest({k: v for k, v in value.items() if k != 'reference_sha256'}), 'Human export checksum/finalization differs')
    if value['finalized']:
        try:
            require(isinstance(value['finalized_at'], str) and re.fullmatch(
                r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})',
                value['finalized_at']) is not None, 'Require an explicit ISO timestamp with timezone')
            stamp = datetime.fromisoformat(value['finalized_at'].replace('Z', '+00:00'))
            require(stamp.tzinfo is not None, 'Finalization timestamp needs timezone')
        except (TypeError, ValueError, AttributeError) as exc:
            raise BiomedicalAnnotationError('Invalid finalization timestamp') from exc
    else: require(value['finalized_at'] is None, 'Draft export cannot have a finalization timestamp')
    require(not require_finalized or value['finalized'], 'Require an explicitly finalized human export')
    expected = {i['item_id']: i for i in package['items']}
    require(isinstance(value['judgments'], list) and len(value['judgments']) == len(expected), 'Human export must retain every article row')
    seen, counts = set(), {s: 0 for s in STATES}
    for row in value['judgments']:
        require(isinstance(row, dict) and set(row) == {'item_id','query_id','pmid','text_version_id','item_sha256','state','relevance','rationale'}, 'Unexpected article judgment fields')
        key = row['item_id']
        require(isinstance(key, str) and key in expected and key not in seen, 'Unknown or duplicate article row')
        item = expected[key]; seen.add(key)
        require(all(row[k] == item[k] for k in ('query_id','pmid','text_version_id','item_sha256')), 'Article judgment source binding differs')
        require(isinstance(row['state'], str) and row['state'] in STATES and isinstance(row['rationale'], str), 'Invalid article judgment state/rationale')
        if row['state'] == 'rated': require(type(row['relevance']) is int and row['relevance'] in (0,1,2), 'Relevance must be integer 0, 1 or 2')
        else: require(row['relevance'] is None, 'Pending/unjudgeable relevance must remain null')
        counts[row['state']] += 1
    bindings = {b['query_id']: b for b in package['query_bindings']}
    by_query = {q['query_id']: {i['pmid'] for i in package['items'] if i['query_id'] == q['query_id']} for q in package['queries']}
    rows = value['query_judgments']
    require(isinstance(rows, list) and len(rows) == len(bindings), 'Human export must retain every question row, including empty searches')
    seen, query_counts = set(), {'total':len(bindings), 'decided':0, 'unknown':0, 'pending':0, 'incomplete_support':0}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'query_id','question_sha256','source_pool_sha256','state','supporting_pmids','rationale'}, 'Unexpected query judgment fields')
        qid = row['query_id']
        require(isinstance(qid, str) and qid in bindings and qid not in seen, 'Unknown or duplicate question row')
        seen.add(qid)
        require(all(row[k] == bindings[qid][k] for k in ('question_sha256','source_pool_sha256')), 'Question/source-pool binding differs')
        require(isinstance(row['state'], str) and row['state'] in QUERY_STATES and isinstance(row['rationale'], str), 'Invalid answerability state/rationale')
        refs = row['supporting_pmids']
        require(isinstance(refs, list) and all(isinstance(p, str) and p in by_query[qid] for p in refs)
            and len(set(refs)) == len(refs), 'Supporting PMIDs must be unique members of this question source pool')
        require(row['state'] == 'yes' or not refs, 'Only an explicit yes can retain supporting PMIDs')
        if row['state'] == 'yes' and not refs: query_counts['incomplete_support'] += 1
        if row['state'] in ('yes','no'): query_counts['decided'] += 1
        else: query_counts[row['state']] += 1
    require(not value['finalized'] or (counts['pending'] == 0 and query_counts['pending'] == 0
        and query_counts['incomplete_support'] == 0), 'Finalization cannot include pending decisions or yes without source support')
    return value, {'articles': {'total':len(expected), **counts}, 'queries':query_counts}


def _structural_coverage(counts):
    """Expose completion only; semantic human values stay quarantined."""
    articles, queries = counts['articles'], counts['queries']
    pending_queries = queries['pending'] + queries['incomplete_support']
    return {'articles': {'total': articles['total'],
                'completed': articles['total'] - articles['pending'], 'pending': articles['pending']},
            'queries': {'total': queries['total'],
                'completed': queries['total'] - pending_queries, 'pending': pending_queries}}


def validate_export(package, human_export, *, require_finalized=False):
    packet = package if isinstance(package, LoadedPacket) else load_packet(package)
    body = regular_bytes(human_export)
    value, counts = validate_annotations(packet, body, require_finalized=require_finalized)
    return {'schema_version':VERSION, 'kind':'confirmation-human-export-validation',
        'package_id':packet.package['package_id'], 'annotation_method':'manual-human', 'judgments_origin':'human',
        'original_export_sha256':byte_hash(body), 'reference_sha256':value['reference_sha256'],
        'manifest_sha256':packet.manifest_sha256, 'finalized':value['finalized'], 'coverage':_structural_coverage(counts),
        'quality_scores_computed':False, 'labels_or_rationales_returned':False}


def freeze_export(package, human_export, output):
    packet = package if isinstance(package, LoadedPacket) else load_packet(package)
    body = regular_bytes(human_export)
    value, counts = validate_annotations(packet, body, require_finalized=True)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Choose a fresh frozen human-reference directory')
    require(not output.resolve().is_relative_to(packet.directory.resolve()) and not packet.directory.resolve().is_relative_to(output.resolve()), 'Frozen reference and packet must be disjoint')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.freezing-', dir=output.parent))
    try:
        _write(stage / 'original-human-export.json', body)
        (stage / 'original-human-export.json').chmod(0o400)
        receipt = {'schema_version':VERSION, 'kind':'frozen-confirmation-manual-human-source-reference',
            'annotation_method':'MANUAL HUMAN ANNOTATION', 'judgments_origin':'human',
            'assistant_generated_labels':False, 'human_labels_rewritten':False,
            **{k:packet.package[k] for k in ('package_id','plan_sha256','items_sha256','source_bindings_sha256','query_bindings_sha256')},
            'manifest_sha256':packet.manifest_sha256, 'original_export_file':'original-human-export.json',
            'original_export_sha256':byte_hash(body), 'reference_sha256':value['reference_sha256'],
            'finalized':True, 'finalized_at':value['finalized_at'], 'frozen_at':datetime.now(timezone.utc).isoformat(),
            'coverage':_structural_coverage(counts), 'quality_scores_computed':False, 'validator_sha256':byte_hash(Path(__file__).read_bytes())}
        receipt['receipt_id'] = digest(receipt)
        _write(stage / 'receipt.json', canonical(receipt) + b'\n')
        require(regular_bytes(human_export) == body and load_packet(packet.directory).manifest_sha256 == packet.manifest_sha256,
            'Human original or packet changed during freeze')
        require(not output.exists() and not output.is_symlink(), 'Reference destination appeared during freeze')
        stage.rename(output)
        return receipt
    except BaseException:
        shutil.rmtree(stage)
        raise
