"""Inclusive publication-date filters with explicit partial-date intervals."""

import calendar
from datetime import date
import re


def bound(value):
    if value is None:return None
    if not isinstance(value,str) or re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',value) is None:
        raise ValueError('Publication bounds require YYYY-MM-DD')
    return date.fromisoformat(value)


def publication_interval(article):
    # Preserve the native PubMed journal-issue date policy; never use retrieval,
    # OAI release/revision, Entrez indexing or record-completion dates.
    values = {d.get('iso_date') for d in article.get('dates',[]) if d['kind']=='journal_issue'}
    if len(values)!=1:return None
    value = next(iter(values))
    if not isinstance(value,str):return None
    try:
        if re.fullmatch(r'[0-9]{4}',value):return date(int(value),1,1),date(int(value),12,31)
        if re.fullmatch(r'[0-9]{4}-[0-9]{2}',value):
            year,month=map(int,value.split('-'))
            return date(year,month,1),date(year,month,calendar.monthrange(year,month)[1])
        if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}',value):return date.fromisoformat(value),date.fromisoformat(value)
    except ValueError:return None
    return None


def select_publication_dates(articles, published_from=None, published_until=None):
    start,end=bound(published_from),bound(published_until)
    if start and end and start>end:raise ValueError('Publication start must not follow end')
    selected=[];unknown=[];outside=[]
    for article in articles:
        interval=publication_interval(article)
        if interval is None:unknown.append(article['pmid'])
        if not (start or end):selected.append(article['pmid'])
        elif interval is None:continue
        elif (start and interval[1]<start) or (end and interval[0]>end):outside.append(article['pmid'])
        else:selected.append(article['pmid'])
    return selected,{'published_from':published_from,'published_until':published_until,
        'policy':'inclusive-overlap-with-journal-issue-date-interval-v1',
        'unknown_date_pmids':unknown,'outside_date_pmids':outside,
        'input_article_count':len(articles),'eligible_article_count':len(selected),
        'unknown_dates_included':not (start or end)}
