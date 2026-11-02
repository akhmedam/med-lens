"""Rank verified local articles by their maximum MedCPT window logit."""

import hashlib
import math
from pathlib import Path
from time import perf_counter

from medlens.generation.context import digest, source_section
from medlens.parsing.dataset import load_dataset
from medlens.retrieval.windows import load_windows
from .medcpt import MAX_TOKENS, MODEL_ID, REVISION


def aggregate_windows(windows, scores, article_order):
    """Exact ties retain input article order and first source window, including negatives."""
    if len(windows) != len(scores) or len(set(article_order)) != len(article_order):
        raise ValueError('Window scores or article order are inconsistent')
    best = {}
    allowed = set(article_order)
    for window, score in zip(windows, scores):
        if (window['pmid'] not in allowed or isinstance(score,bool)
                or not isinstance(score,(int,float)) or not math.isfinite(score)):
            raise ValueError('Expected one finite raw score per eligible source window')
        pmid = window['pmid']
        if pmid not in best or score > best[pmid][0]:
            best[pmid] = (float(score),window)
    rows = [(pmid,*best[pmid]) for pmid in article_order if pmid in best]
    rows.sort(key=lambda row:-row[1])
    return rows


class MedCPTRanker:
    def __init__(self, dataset_dir, windows_dir):
        self.dataset = load_dataset(Path(dataset_dir))
        self.window_manifest,self.windows = load_windows(Path(windows_dir))
        if (self.window_manifest['dataset_id'] != self.dataset.manifest['dataset_id']
                or self.window_manifest['articles_sha256'] != self.dataset.manifest['articles_sha256']):
            raise ValueError('MedCPT dataset/window identities differ')
        self.articles = {article['pmid']:article for article in self.dataset.articles}
        covered = set()
        for window in self.windows:
            if window['pmid'] not in self.articles:
                raise ValueError('Window PMID outside candidate set')
            source_section(self.articles[window['pmid']],window)
            covered.add(window['pmid'])
        self.missing_pmids = [p for p in self.articles if p not in covered]
        if self.missing_pmids != self.window_manifest['empty_pmids']:
            raise ValueError('Candidate window coverage differs')

    def eligible_windows(self, eligible_pmids):
        if not isinstance(eligible_pmids,(list,tuple)):
            raise ValueError('Eligible PMIDs must be an ordered list or tuple')
        if (not all(isinstance(p,str) for p in eligible_pmids)
                or len(set(eligible_pmids)) != len(eligible_pmids)
                or not set(eligible_pmids) <= set(self.articles)):
            raise ValueError('Eligible PMIDs must be unique candidate IDs')
        allowed = set(eligible_pmids)
        return [window for window in self.windows if window['pmid'] in allowed]

    def run(self, question, scorer, *, eligible_pmids=None, top_k=10):
        if not isinstance(question,str) or not question.strip():
            raise ValueError('A nonempty question is required')
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError('top_k must be between one and ten')
        all_candidates = eligible_pmids is None
        order = list(self.articles) if all_candidates else eligible_pmids
        windows = self.eligible_windows(order)
        tick = perf_counter()
        if windows:
            scored = scorer.score([(question,w['title'],w['text']) for w in windows])
            lengths = scored['pair_lengths']
            if (scored['pair_count'] != len(windows) or len(lengths) != len(windows)
                    or any(type(n) is not int or not 3 <= n <= MAX_TOKENS for n in lengths)):
                raise ValueError('MedCPT score/input mapping differs from selected windows')
            scores = scored['scores']
        else:
            scored = {'pair_count':0,'pair_lengths':[],'scores':[],'score_seconds':0.0}
            scores,lengths = [],[]
        aggregated = aggregate_windows(windows,scores,order)
        hits = []
        for rank,(pmid,score,window) in enumerate(aggregated[:top_k],1):
            hits.append({'rank':rank,'pmid':pmid,'score':score,'title':self.articles[pmid].get('title'),
                'url':f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/','text_version_id':window['text_version_id'],
                'best_window':{key:window[key] for key in ('window_id','kind','section_index',
                    'section_label','source_path','char_start','char_end','text','provenance')}})
        spec = {'dataset_id':self.dataset.manifest['dataset_id'],
            'window_artifact_id':self.window_manifest['artifact_id'],'query':question,
            'candidate_policy':'all-articles' if all_candidates else 'explicit-article-subset',
            'eligible_pmids':list(order),'top_k':top_k,'article_score':'max-window-raw-logit',
            'article_ties':'eligible-input-order','window_ties':'source-window-order',
            'negative_scores':'retain','model_id':MODEL_ID,'revision':REVISION,
            'ranker_config':{k:v for k,v in scorer.metadata.items() if k!='load_seconds'} if windows else None,
            'code_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ('articles.py','medcpt.py')}}
        return {'schema_version':1,'run_id':digest(spec),**spec,
            'runtime':scorer.metadata if windows else None,
            'candidate_count':len(self.articles),'eligible_count':len(order),
            'ranked_article_count':len(aggregated),'missing_window_pmids':[p for p in order if p in self.missing_pmids],
            'scoring':{key:value for key,value in scored.items() if key not in {'scores','pair_lengths'}},
            'window_scores':[{'window_id':w['window_id'],'pmid':w['pmid'],
                'text_version_id':w['text_version_id'],'input_tokens':length,'score':float(score)}
                for w,score,length in zip(windows,scores,lengths)],
            'ranking_seconds':perf_counter()-tick,'results':hits}
