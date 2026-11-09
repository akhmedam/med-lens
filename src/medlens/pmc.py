"""Bounded PMC OAI snapshots and paired JATS abstract/narrative collections."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zlib

from medlens.corpus import inspect_collection, load_pooled_dataset
from medlens.parsing.dataset import LoadedDataset
from medlens.retrieval.embeddings import load_embeddings
from medlens.retrieval.vector_cache import VectorCache, build_cached_embeddings
from medlens.retrieval.windows import WindowConfig, article_windows, load_windows, _json, _sha

BASE = 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/'
MAX_BYTES = 8*1024*1024
PARSER = 'jats-narrative-v1'
KIND = 'pmc-pilot-dataset'
EXCLUDED = {'fig','table-wrap','table-wrap-group','supplementary-material','ref-list','fn-group',
    'app-group','boxed-text','media','statement','disp-formula','inline-formula','tex-math','math'}
LICENSE = re.compile(r'https?://creativecommons\.org/(licenses/by/(?:1\.0|2\.0|2\.5|3\.0|4\.0)|publicdomain/zero/1\.0)/?$', re.I)


class PMCError(ValueError):
    """An artifact or protocol response is inconsistent."""


class NotEligible(PMCError):
    """The record does not meet the frozen pilot admission policy."""


def code_sha():
    return _sha(Path(__file__).read_bytes())


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(_json(value)+b'\n');temporary.replace(path)


def local(tag):
    return tag.rsplit('}',1)[-1]


def text(node):
    return ' '.join(''.join(node.itertext()).split()) if node is not None else ''


def parse_jats(body, pmid, pmcid):
    if len(body)>MAX_BYTES or b'<!ENTITY' in body.upper() or b'<!DOCTYPE' in body.upper():
        raise PMCError('Oversized XML or prohibited DTD/entity declaration')
    try:root = ET.fromstring(body)
    except ET.ParseError as exc:raise PMCError('Invalid OAI XML') from exc
    if root.tag != '{http://www.openarchives.org/OAI/2.0/}OAI-PMH':raise PMCError('Expected OAI-PMH envelope')
    errors = [e for e in root if local(e.tag)=='error']
    if errors:
        code = errors[0].get('code')
        if code in {'cannotDisseminateFormat','idDoesNotExist','noRecordsMatch'}:raise NotEligible('oai:'+code)
        raise PMCError('Unexpected OAI error: '+str(code))
    # Strip namespaces only after validating the protocol envelope. Paths below
    # are relative to the namespace-stripped JATS article element, not XML bytes.
    nodes = list(root.iter())
    if len(nodes)>200000:raise PMCError('Too many XML nodes')
    for node in nodes:node.tag = local(node.tag)
    records = root.findall('./GetRecord/record')
    if len(records)!=1:raise PMCError('Expected one OAI record')
    record = records[0];header = record.find('header')
    if header is None or header.get('status')=='deleted':raise NotEligible('deleted-record')
    if header.findtext('identifier')!='oai:pubmedcentral.nih.gov:'+pmcid[3:]:raise PMCError('OAI identifier mismatch')
    articles = record.findall('./metadata/article')
    if len(articles)!=1:raise PMCError('Expected one JATS article')
    article = articles[0]
    if article.get('{http://www.w3.org/XML/1998/namespace}lang','en').lower() not in {'en','eng'}:
        raise NotEligible('non-english')
    if article.findall('.//sub-article'):raise NotEligible('sub-article-not-supported')
    meta = article.find('./front/article-meta')
    if meta is None:raise PMCError('Missing JATS article metadata')
    identifiers = [(e.get('pub-id-type'),text(e)) for e in meta.findall('article-id')]
    actual_pmids = {v for k,v in identifiers if k=='pmid'}
    actual_pmc = {v if v.startswith('PMC') else 'PMC'+v for k,v in identifiers if k in {'pmc','pmcid','pmcaid'}}
    if actual_pmids!={pmid} or actual_pmc!={pmcid}:raise PMCError('JATS PMID/PMCID mismatch')
    licenses = meta.findall('./permissions/license')
    uris = set()
    for license in licenses:
        values = [v for k,v in license.attrib.items() if local(k)=='href']
        values += [v for e in license.iter() if local(e.tag)=='ext-link' for k,v in e.attrib.items() if local(k)=='href']
        matches = [v for v in values if LICENSE.fullmatch(v)]
        if not matches:raise NotEligible('license-not-explicit-cc-by-or-cc0')
        uris.update(matches)
    if len(uris)!=1:raise NotEligible('missing-or-ambiguous-license')
    title = text(meta.find('./title-group/article-title'))
    if not title:raise NotEligible('missing-title')
    paths = {id(article):'.'}
    def locate(node):
        counts = Counter()
        for child in node:
            counts[child.tag]+=1
            paths[id(child)] = paths[id(node)]+'/'+child.tag+'['+str(counts[child.tag])+']'
            locate(child)
    locate(article)
    excluded = Counter();abstract=[];sections=[]
    def paragraphs(node, headings, category, target):
        tag = node.tag
        if tag in EXCLUDED:
            excluded[tag]+=1;return
        if tag=='sec':
            heading = text(node.find('title'))
            headings = [*headings,heading] if heading else headings
            hint = (node.get('sec-type') or heading).casefold().strip()
            if hint in {'results','result','conclusions','conclusion'}:category = hint
        if tag=='p':
            if any(local(e.tag) in EXCLUDED or (local(e.tag)=='p' and e is not node) for e in node.iter()):
                excluded['paragraph-with-unsupported-content']+=1;return
            value = text(node)
            if value:
                target.append({'index':len(target)+1,'label':' / '.join(headings) or None,
                    'section_hierarchy':headings,'nlm_category':category,'text':value,
                    'raw_text':''.join(node.itertext()),'source_path':paths[id(node)],
                    'jats_id':node.get('id')})
            return
        for child in node:
            if child.tag!='title':paragraphs(child,headings,category,target)
    for node in meta.findall('abstract'):
        if node.get('abstract-type') in {'graphical','toc'}:excluded['non-narrative-abstract']+=1;continue
        paragraphs(node,[],None,abstract)
    body_node = article.find('body')
    if body_node is None:raise NotEligible('no-body')
    paragraphs(body_node,[],None,sections)
    if not abstract or not sections:raise NotEligible('empty-abstract-or-narrative-body')
    if len(abstract)+len(sections)>6000:raise NotEligible('paragraph-budget')
    return {'pmid':pmid,'pmcid':pmcid,'title':title,'license_uri':next(iter(uris)),
        'license_text':[text(v) for v in licenses], 'oai_datestamp':header.findtext('datestamp'),
        'abstract':abstract,'body':sections,'excluded':dict(excluded),
        'text_normalization':'Unicode XML itertext joined then whitespace collapsed; offsets index normalized paragraph characters'}


class PMCClient:
    """One sequential, bounded OAI client; at most two requests per second."""
    def __init__(self, *, opener=urlopen, sleep=time.sleep, clock=time.monotonic):
        self.opener,self.sleep,self.clock = opener,sleep,clock
        self.next_request = 0.0

    def fetch(self, pmcid):
        if not isinstance(pmcid,str) or not re.fullmatch(r'PMC[1-9][0-9]*',pmcid):raise PMCError('Invalid PMCID')
        url = BASE+'?'+urlencode({'verb':'GetRecord','identifier':'oai:pubmedcentral.nih.gov:'+pmcid[3:],'metadataPrefix':'pmc'})
        for attempt in range(1,4):
            delay = self.next_request-self.clock()
            if delay>0:self.sleep(delay)
            self.next_request = self.clock()+0.5
            request = Request(url,headers={'User-Agent':'medlens/0.0.1','Accept-Encoding':'gzip, deflate'})
            try:
                try:response = self.opener(request,timeout=25)
                except HTTPError as exc:
                    if exc.code==400:response = exc  # OAI protocol errors carry an XML envelope.
                    else:raise
                with response:
                    raw = response.read(MAX_BYTES+1)
                    if len(raw)>MAX_BYTES:raise PMCError('Oversized HTTP response')
                    encoding = response.headers.get('Content-Encoding','').lower()
                    if encoding in {'gzip','deflate'}:
                        decoder = zlib.decompressobj(16+zlib.MAX_WBITS if encoding=='gzip' else zlib.MAX_WBITS)
                        body = decoder.decompress(raw,MAX_BYTES+1)
                        if len(body)>MAX_BYTES or not decoder.eof or decoder.unused_data:raise PMCError('Oversized or incomplete compressed response')
                    elif not encoding:body = raw
                    else:raise PMCError('Unsupported HTTP compression')
                    if not body:raise PMCError('Empty HTTP response')
                    return body,{'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat(),'attempts':attempt,'http_status':response.code}
            except HTTPError as exc:
                status = exc.code;retry = exc.headers.get('Retry-After') if exc.headers else None;exc.close()
                if status not in {429,500,502,503,504} or attempt==3:raise PMCError('OAI HTTP '+str(status)) from exc
                try:wait = max(2**(attempt-1),float(retry or 0))
                except ValueError:raise PMCError('Unsupported Retry-After; retry later')
                if not 0<=wait<=30:raise PMCError('Retry wait exceeds budget; retry later')
                self.sleep(wait)
            except (URLError,TimeoutError,ConnectionError) as exc:
                if attempt==3:raise PMCError('OAI transport failed') from exc
                self.sleep(2**(attempt-1))
        raise AssertionError('unreachable')


def download_pilot(base, output, *, limit=6, max_candidates=36, client=None):
    if type(limit) is not int or not 1<=limit<=10 or type(max_candidates) is not int or not limit<=max_candidates<=40:
        raise PMCError('Pilot requires 1–10 articles and at most 40 sequential candidates')
    base,output = Path(base),Path(output)
    if output.exists():raise FileExistsError(output)
    original = inspect_collection(base);dataset = load_pooled_dataset(base/'dataset')
    if output.resolve().is_relative_to(base.resolve()) or base.resolve().is_relative_to(output.resolve()):raise PMCError('Snapshot and base must be disjoint')
    candidates=[]
    for row in sorted(dataset.articles,key=lambda a:int(a['pmid'])):
        ids = {i['value'] for i in row.get('identifiers',[]) if i['kind']=='pmc'}
        if len(ids)==1 and re.fullmatch(r'PMC[1-9][0-9]*',next(iter(ids))):candidates.append((row,next(iter(ids))))
    output.mkdir(parents=True);client = client or PMCClient()
    manifest = {'schema_version':1,'kind':'pmc-oai-pilot-snapshot','status':'running',
        'base_collection_id':original['collection_id'],'base_dataset_id':dataset.manifest['dataset_id'],
        'base_articles_sha256':dataset.manifest['articles_sha256'],'candidate_count':len(candidates),
        'policy':{'limit':limit,'max_candidates':max_candidates,'order':'ascending-numeric-PMID',
            'licenses':'explicit article CC BY 1.0/2.0/2.5/3.0/4.0 or CC0 1.0 URI only','content':'English JATS abstract and narrative body; no sub-articles'},
        'responses':[],'selected_pmids':[]}
    write(output/'manifest.json',manifest);selected=[]
    try:
        for row,pmcid in candidates[:max_candidates]:
            body,response = client.fetch(pmcid)
            item = {**response,'pmid':row['pmid'],'pmcid':pmcid,'sha256':_sha(body)}
            try:parsed = parse_jats(body,row['pmid'],pmcid)
            except NotEligible as exc:item.update(selected=False,reason=str(exc))
            else:
                # Persist source XML only after explicit license and format acceptance.
                filename=pmcid+'.xml';(output/filename).write_bytes(body)
                item.update(selected=True,license_uri=parsed['license_uri'],file=filename);selected.append(row)
                manifest['selected_pmids'].append(row['pmid'])
            manifest['responses'].append(item);write(output/'manifest.json',manifest)
            print('PMC',pmcid,item.get('reason','accepted'),flush=True)
            if len(selected)==limit:break
        if len(selected)!=limit:raise PMCError('Insufficient eligible articles within the frozen candidate budget')
        body = b''.join(_json(a)+b'\n' for a in selected);(output/'pubmed-articles.jsonl').write_bytes(body)
        manifest.update(pubmed_articles_sha256=_sha(body),status='complete')
        manifest['snapshot_id'] = _sha(_json({k:v for k,v in manifest.items() if k!='snapshot_id'}))
        write(output/'manifest.json',manifest);load_snapshot(output)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        write(output/'manifest.json',manifest);raise


def load_snapshot(root):
    root = Path(root)
    if root.is_symlink():raise PMCError('Snapshot must not be a symlink')
    try:
        m = json.loads((root/'manifest.json').read_bytes())
        if m['kind']!='pmc-oai-pilot-snapshot' or m['schema_version']!=1 or m['status']!='complete':raise PMCError('Incomplete PMC snapshot')
        if _sha(_json({k:v for k,v in m.items() if k!='snapshot_id'}))!=m['snapshot_id']:raise PMCError('Snapshot identity mismatch')
        p=root/'pubmed-articles.jsonl'
        if p.is_symlink() or _sha(p.read_bytes())!=m['pubmed_articles_sha256']:raise PMCError('PubMed observation checksum mismatch')
        rows=tuple(json.loads(v) for v in p.read_bytes().splitlines())
        if [r['pmid'] for r in rows]!=m['selected_pmids'] or len(set(m['selected_pmids']))!=len(rows):raise PMCError('PMC snapshot membership mismatch')
        parsed=[];selected=[r for r in m['responses'] if r['selected']]
        if [r['pmid'] for r in selected]!=m['selected_pmids']:raise PMCError('Selected OAI membership mismatch')
        for r in m['responses']:
            if (not re.fullmatch(r'PMC[1-9][0-9]*',r['pmcid']) or type(r.get('selected')) is not bool
                    or not re.fullmatch(r'[0-9a-f]{64}',r['sha256'])):raise PMCError('Invalid OAI response record')
            if r['selected']:
                if r.get('file')!=r['pmcid']+'.xml' or not r.get('license_uri'):
                    raise PMCError('Selected OAI record lacks licensed source')
                path=root/r['file']
                if path.is_symlink() or _sha(path.read_bytes())!=r['sha256']:raise PMCError('OAI response checksum mismatch')
                value=parse_jats(path.read_bytes(),r['pmid'],r['pmcid'])
                if value['license_uri']!=r['license_uri']:raise PMCError('Saved license differs')
                parsed.append(value)
            elif 'file' in r or 'license_uri' in r:
                raise PMCError('Rejected OAI response must not retain article content/license')
        return m,rows,tuple(parsed)
    except (OSError,KeyError,TypeError,ValueError,AttributeError) as exc:
        if isinstance(exc,PMCError):raise
        raise PMCError('Invalid PMC snapshot') from exc


def derived_articles(manifest, native, parsed):
    result=[];responses={r['pmid']:r for r in manifest['responses'] if r['selected']}
    for old,jats in zip(native,parsed):
        if old['pmid']!=jats['pmid']:raise PMCError('Native/JATS order differs')
        row=deepcopy(old);r=responses[row['pmid']]
        row['metadata_provenance']=row['provenance'];row['metadata_text_version']=row['text_version']
        row['title']=jats['title'];row['abstract']={'sections':jats['abstract'],'attributes':{},'copyright':None,'source_path':'./front/article-meta/abstract'}
        row['other_abstracts']=[];row['abstract_status']='present'
        row['provenance']={'source':'pmc-oai','snapshot_id':manifest['snapshot_id'],'source_file':r['file'],
            'retrieved_at':r['retrieved_at'],'raw_response_sha256':r['sha256'],'pmcid':r['pmcid'],
            'license_uri':r['license_uri'],'oai_datestamp':jats['oai_datestamp'],'path_basis':'namespace-stripped JATS article element'}
        row['full_text']={k:jats[k] for k in ('pmcid','license_uri','license_text','excluded','text_normalization')}
        row['full_text']['sections']=jats['body']
        value={'title':row['title'],'abstract':[(s['label'],s['text']) for s in jats['abstract']],
            'body':[(s['section_hierarchy'],s['text']) for s in jats['body']]}
        digest=_sha(_json(value));row['text_version']={'parser_version':PARSER,'text_sha256':digest,'text_version_id':'pmc:'+row['pmid']+':'+digest}
        result.append(row)
    return tuple(result)


def pmc_windows(article, tokenizer, mode, config):
    rows=article_windows(article,tokenizer,config)
    if mode=='fulltext':
        proxy={**article,'abstract':{'sections':article['full_text']['sections']}}
        for row in article_windows(proxy,tokenizer,config):
            if row['kind']=='title':continue
            row['kind']='fulltext'
            row['window_id']=_sha(_json({'row':{k:v for k,v in row.items() if k!='window_id'},'config':asdict(config)}))
            rows.append(row)
    return rows


def load_pmc_dataset(directory):
    directory=Path(directory)
    try:
        root=json.loads((directory.parent/'manifest.json').read_bytes())
        if root['status']!='complete' or root['kind']!='pmc-pilot-collection':raise PMCError('Incomplete PMC collection')
        if _sha(_json({k:v for k,v in root.items() if k!='manifest_id'}))!=root['manifest_id']:raise PMCError('Collection manifest identity mismatch')
        for name in ('dataset/manifest.json','windows/manifest.json','embeddings/manifest.json'):
            path=directory.parent/name
            if path.is_symlink() or _sha(path.read_bytes())!=root['files'][name]:raise PMCError('Collection payload checksum mismatch')
        m=json.loads((directory/'manifest.json').read_bytes())
        if m['kind']!=KIND or m['status']!='complete' or m['parser_sha256']!=code_sha():raise PMCError('Unsupported PMC dataset/parser')
        identity={k:m[k] for k in ('schema_version','kind','snapshot_id','articles_sha256','parser_sha256')}
        if _sha(_json(identity))!=m['dataset_id'] or root['dataset_id']!=m['dataset_id']:raise PMCError('PMC dataset identity mismatch')
        sm,native,parsed=load_snapshot(directory/'snapshot')
        rows=derived_articles(sm,native,parsed);body=(directory/'articles.jsonl').read_bytes()
        if sm['snapshot_id']!=m['snapshot_id'] or _sha(body)!=m['articles_sha256'] or tuple(json.loads(v) for v in body.splitlines())!=rows:
            raise PMCError('PMC records differ from saved observations')
        if [a['pmid'] for a in rows]!=m['pmids'] or len(rows)!=m['article_count']:raise PMCError('PMC article membership mismatch')
        return LoadedDataset(m,rows)
    except (OSError,KeyError,TypeError,ValueError,AttributeError) as exc:
        if isinstance(exc,PMCError):raise
        raise PMCError('Invalid PMC collection') from exc


def build_collection(snapshot, tokenizer, tokenizer_info, cache_root, output, encoder_info, encoder_factory, *, mode):
    if mode not in {'abstract','fulltext'}:raise PMCError('Expected abstract or fulltext mode')
    snapshot,cache_root,output=map(Path,(snapshot,cache_root,output))
    if output.exists():raise FileExistsError(output)
    roots=[p.resolve() for p in (snapshot,cache_root,output)]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):raise PMCError('Snapshot, output and cache must be disjoint')
    started=time.perf_counter();sm,native,parsed=load_snapshot(snapshot);articles=derived_articles(sm,native,parsed)
    output.mkdir(parents=True);write(output/'manifest.json',{'kind':'pmc-pilot-collection','status':'running'})
    try:
        ds=output/'dataset';ds.mkdir();shutil.copytree(snapshot,ds/'snapshot')
        body=b''.join(_json(a)+b'\n' for a in articles);(ds/'articles.jsonl').write_bytes(body)
        identity={'schema_version':1,'kind':KIND,'snapshot_id':sm['snapshot_id'],'articles_sha256':_sha(body),'parser_sha256':code_sha()}
        dm={**identity,'dataset_id':_sha(_json(identity)),'status':'complete','article_count':len(articles),'pmids':[a['pmid'] for a in articles]}
        write(ds/'manifest.json',dm)
        config=WindowConfig();rows=[w for a in articles for w in pmc_windows(a,tokenizer,mode,config)]
        payload=b''.join(_json(r)+b'\n' for r in rows)
        wi={'schema_version':1,'dataset_id':dm['dataset_id'],'articles_sha256':dm['articles_sha256'],'config':asdict(config),
            'tokenizer':tokenizer_info,'windows_sha256':_sha(payload),'implementation_sha256':code_sha(),'input_format':'CLS-title_ids-section_token_slice-SEP-v1'}
        wm={**wi,'artifact_id':_sha(_json(wi)),'status':'complete','article_count':len(articles),'window_count':len(rows),
            'empty_pmids':[],'title_only_windows':0,'max_input_token_count':max(r['input_token_count'] for r in rows)}
        (output/'windows').mkdir();(output/'windows/windows.jsonl').write_bytes(payload);write(output/'windows/manifest.json',wm)
        cache=VectorCache(cache_root,encoder_info,tokenizer_info)
        em=build_cached_embeddings(output/'windows',cache,encoder_factory,output/'embeddings')
        result={'schema_version':1,'kind':'pmc-pilot-collection','status':'complete','mode':mode,'dataset_id':dm['dataset_id'],
            'window_artifact_id':wm['artifact_id'],'embedding_artifact_id':em['artifact_id'],
            'counts':{'articles':len(articles),'windows':len(rows),'abstract_windows':sum(r['kind']=='abstract' for r in rows),
                'body_windows':sum(r['kind']=='fulltext' for r in rows),'narrative_paragraphs':sum(len(a['full_text']['sections']) for a in articles)},
            'measurements':{'cache':em['measurements'],'prepublication_seconds':time.perf_counter()-started},
            'files':{name:_sha((output/name).read_bytes()) for name in ('dataset/manifest.json','windows/manifest.json','embeddings/manifest.json')}}
        result['manifest_id']=_sha(_json(result));write(output/'manifest.json',result)
        # Reopen through the ordinary reader before considering publication successful.
        from medlens.retrieval.dense import DenseIndex
        index=DenseIndex(ds,output/'windows',output/'embeddings')
        from medlens.generation.context import source_section
        for row in index.windows:source_section(index.articles[row['pmid']],row)
        if index.window_manifest['artifact_id']!=result['window_artifact_id'] or index.embedding_manifest['artifact_id']!=result['embedding_artifact_id']:
            raise PMCError('Published artifact identity mismatch')
        return result
    except BaseException as exc:
        write(output/'manifest.json',{'kind':'pmc-pilot-collection','status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}});raise
