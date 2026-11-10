"""Verify the immutable MANUAL HUMAN snapshot and recompute its numeric audit.

Standard library only. Reads this package and its compact assessment; never loads
models, source articles, generated answers, private local reports, or other labels.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA256 = {
    'human-manual-claims-original.json': '4b7844755ffd4d7cdc13183b4ac5f5d37fa11c4dc4836c15501b3bfe4e8822bc',
    'human-manual-answers-original.json': 'e5263b3ba8411cf9cad6f3f79c57577eb2eec5ff1112eb751cd4940ffee56374',
}
FIELDS = {'factual_support': {'supported','not_supported','unclear'},
          'population': {'match','mismatch','unclear'},
          'comparison': {'match','mismatch','unclear','not_requested'},
          'outcome': {'match','mismatch','unclear'}}


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        value = {}
        for key,item in items:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def invalid(_):raise ValueError('Nonfinite JSON value')
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,parse_constant=invalid)


def require(condition,message):
    if not condition:raise ValueError(message)


def verify_artifact(value):
    require(value.get('artifact_id')==digest({k:v for k,v in value.items()if k!='artifact_id'}),'Artifact identity differs')


def classify(row):
    require(all(row.get(k)in allowed for k,allowed in FIELDS.items()),'Missing/invalid MANUAL HUMAN label')
    if row['factual_support']=='not_supported' or any(row[k]=='mismatch'for k in('population','comparison','outcome')):return 'bad'
    if row['factual_support']=='supported'and row['population']==row['outcome']=='match'and row['comparison']in('match','not_requested'):return 'eligible'
    return 'uncertain'


def interval(low,high,total):
    return {'count_lower':low,'count_upper':high,'denominator':total,
            'rate_lower':low/total if total else None,'rate_upper':high/total if total else None}


def exact(count,total):return {'count':count,'denominator':total,'rate':count/total if total else None}


def verify():
    provenance=read(HERE/'provenance.json');verify_artifact(provenance)
    require(provenance['annotation_method']=='MANUAL HUMAN ANNOTATION'
        and provenance['assistant_generated_labels']is False and provenance['human_rationales_rewritten']is False,
        'Manual origin declaration differs')
    for name,identity in provenance['files'].items():
        path=HERE/name
        require(Path(name).name==name and path.is_file()and not path.is_symlink(),'Invalid package file')
        require(path.stat().st_size==identity['bytes']and file_hash(path)==identity['sha256'],'Package checksum differs: '+name)
    for name,expected in ORIGINAL_SHA256.items():
        require(file_hash(HERE/name)==expected,'Original MANUAL HUMAN export bytes changed')
    manual=read(HERE/'human-manual-claims-original.json')
    require(manual['finalized']is True and manual['judgments_origin']=='human'
        and manual['reference_sha256']==digest({k:v for k,v in manual.items()if k!='reference_sha256'}),
        'Require the finalized original MANUAL HUMAN claim reference')
    labels={row['claim_id']:row for row in manual['labels']}
    require(len(labels)==len(manual['labels'])==30,'Human claim coverage differs')
    audit=read(HERE/'claim-audit.json');verify_artifact(audit)
    require(audit['reference_sha256']==manual['reference_sha256']and audit['package_id']==manual['package_id'],
        'Human/audit reference binding differs')
    require(len(audit['rows'])==30 and {r['human_claim_id']for r in audit['rows']}==set(labels),'Audit claim coverage differs')
    report=read(HERE.parents[2]/'docs/experiments/qwen4b-manual-human-assessment.json');verify_artifact(report)
    require(report['evidence_provenance_sha256']==file_hash(HERE/'provenance.json'),'Assessment/package binding differs')
    classes=Counter();unique_votes={m:defaultdict(list)for m in('4b','9b')}
    occurrences=set();question_hashes=set();context_hashes=set()
    for row in audit['rows']:
        key=row['human_claim_id'];human=labels[key]
        for name in('claim_sha256','question_sha256','context_sha256'):
            require(row[name]==human[name],'Human source/claim binding differs')
        require(row['manual_labels']=={k:human[k]for k in FIELDS},'Manual label value changed')
        group=classify(human);require(row['derived_protocol_class']==group,'Protocol grouping differs')
        classes[group]+=1;question_hashes.add(row['question_sha256']);context_hashes.add(row['context_sha256'])
        require(row['occurrences'],'Human claim has no frozen occurrence')
        for origin in row['occurrences']:
            occurrence=(origin['case_id'],origin['original_claim_id'])
            require(occurrence not in occurrences,'Duplicate frozen occurrence');occurrences.add(occurrence)
            require(set(origin['reviewer_accepted'])=={'4b','9b'}and all(type(x)is bool for x in origin['reviewer_accepted'].values()),
                'Incomplete reviewer prediction')
            for model in('4b','9b'):unique_votes[model][key].append(origin['reviewer_accepted'][model])
    require(len(occurrences)==32 and len(question_hashes)==2 and len(context_hashes)==6,'Scope differs')
    require(dict(classes)==report['claims']['human_classes'],'Human class counts differ')
    require(classes==Counter(eligible=24,bad=5,uncertain=1),'Snapshot human class inventory differs')
    for model in('4b','9b'):
        low=Counter();high=Counter();occ=Counter();accept=Counter();discordance=0
        for key,votes in unique_votes[model].items():
            group=classify(labels[key]);low[group]+=all(votes);high[group]+=any(votes)
            occ[group]+=len(votes);accept[group]+=sum(votes);discordance+=all(votes)!=any(votes)
        expected_unique={'false_accept_bad':interval(low['bad'],high['bad'],classes['bad']),
            'false_reject_eligible':interval(classes['eligible']-high['eligible'],classes['eligible']-low['eligible'],classes['eligible']),
            'recall_eligible':interval(low['eligible'],high['eligible'],classes['eligible']),
            'accepted_uncertain':interval(low['uncertain'],high['uncertain'],classes['uncertain'])}
        expected_occ={'false_accept_bad':exact(accept['bad'],occ['bad']),
            'false_reject_eligible':exact(occ['eligible']-accept['eligible'],occ['eligible']),
            'recall_eligible':exact(accept['eligible'],occ['eligible']),
            'accepted_uncertain':exact(accept['uncertain'],occ['uncertain'])}
        saved=report['claims']['models'][model]
        require(saved['unique']==expected_unique and saved['occurrences']==expected_occ,'Reviewer numeric audit differs: '+model)
        require(saved['duplicate_discordance_count']==discordance,'Duplicate bounds were silently collapsed')
        require(saved['zero_observed_bad_acceptances']==(accept['bad']==0),'Safety count differs')
    candidate=report['claims']['models']['4b']['unique']['recall_eligible']
    baseline=report['claims']['models']['9b']['unique']['recall_eligible']
    require(10*candidate['count_lower']>=9*baseline['count_upper'],'Relative recall threshold differs')
    answers=read(HERE/'human-manual-answers-original.json');answer_audit=read(HERE/'answer-audit.json');verify_artifact(answer_audit)
    require(answers['judgments_origin']=='human'and answers['package_id']==answer_audit['package_id'],'Answer reference origin/binding differs')
    original={(c['case_id'],v['blind_id']):(c,v)for c in answers['cases']for v in c['variants']}
    require(len(original)==len(answer_audit['rows'])==12,'Answer coverage differs')
    seen=set();by_model=defaultdict(list)
    for row in answer_audit['rows']:
        key=(row['case_id'],row['blind_id']);require(key in original and key not in seen,'Duplicate/unknown blind answer');seen.add(key)
        case,human=original[key]
        require(row['ratings']==human['judgments']and row['output_sha256']==human['output_sha256']
            and row['context_sha256']==case['context_sha256']and row['source_run_sha256']==case['source_run_sha256']
            and row['rationale_sha256']==hashlib.sha256(human['rationale'].encode()).hexdigest(),'Original answer judgment/binding differs')
        require(row['model']in('4b-named-axes-first','9b-named-axes-first'),'Unknown unblinded model')
        by_model[row['model']].append(row)
    for model,rows in by_model.items():
        require(len(rows)==6,'Paired answer model coverage differs')
        for field in('source_support','question_scope','qualifiers_preserved','completeness','overall_usable'):
            raw=Counter(r['ratings'][field]for r in rows);applicable=Counter();na=unknown=0
            for row in rows:
                value=row['ratings'][field]
                if value=='unknown'and row['empty_answer']and field in('source_support','question_scope','qualifiers_preserved'):na+=1
                elif value=='unknown':unknown+=1
                else:applicable[value]+=1
            saved=report['answers']['models'][model]['criteria'][field]
            require(saved['raw_human_counts']==dict(raw)and saved['applicable_counts']==dict(applicable)
                and saved['not_applicable_empty_count']==na and saved['unknown_count']==unknown
                and saved['judgeable_denominator']==sum(applicable.values()),'Answer numeric audit differs')
    require(report['decision']['4b_zero_known_bad_acceptance_gate']=='fail'
        and report['decision']['9b_zero_known_bad_acceptance_gate']=='fail'
        and report['decision']['candidate_overall_usability_gate']=='fail','Failed gates hidden')
    require(report['reviewer_native_speed']['scope'].startswith('native reviewer'),'Speed scope differs')
    return {'status':'passed','annotation_method':'MANUAL HUMAN ANNOTATION',
        'unique_claims':len(labels),'claim_occurrences':len(occurrences),'paired_contexts':6,'answers':12,
        'source_articles_or_generated_answers_read':False,'models_loaded':False}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
