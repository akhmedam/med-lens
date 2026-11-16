"""Conservative vetoes for explicit study-language mismatches, never source approval.

No biomedical entity dictionary, network, model, or positive entailment checker.
Absence means absent from the recognized lexical forms, not clinical proof.
"""
import re
import unicodedata
from fractions import Fraction

POLICY_VERSION = 'source-scope-veto-v4'

_STOP = set('the a an of for to from in on with using as and or by versus vs compare compared '
            'does do did is are was were can how what which disease disorder syndrome condition '
            'progression treatment therapy agent drug compound assay test testing measurement'.split())
_NUM = dict(zip('zero one two three four five six seven eight nine ten eleven twelve'.split(), range(13)))
_N = r'(?:\d+(?:\.\d+)?|' + '|'.join(_NUM) + r')'
_TIME = re.compile(r'(?<![\w-])(' + _N + r')(?:\s*(?:to|[-–—])\s*(' + _N
                   + r'))?\s*[-‐‑–—]?\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|h|days?|d|weeks?|wks?|months?|mos?|years?|yrs?)\b', re.I)
_COMBO = re.compile(r'\b(?:add[ -]on|added to|plus|combination|combined with|(?:half|full|reduced|adjusted|titrated)[ -]dose|dose[ -](?:reduc\w*|titrat\w*|adjust\w*))\b', re.I)
_CONTRAST = re.compile(r'\b(?:compar(?:e|ed|ison)|versus|vs\.?)\b', re.I)
_REFERENCE = re.compile(r'\busing\s+(.+?)\s+as\s+(?:(?:the|a)\s+)?(?:reference|gold)\s+standard\b', re.I)
_CLINICAL = re.compile(r'\b(?:symptoms?|clinical\s+(?:signs?|criteria|assessment))\b', re.I)


def _negated(text, start, end=None):
    prefix = re.split(r'[.;!?]', text[:start])[-1]
    if re.search(r'\b(?:not|never|no|without|exclude|excluded|excluding|neither|nor)\b(?:\W+\w+){0,3}\W*$', prefix, re.I):
        return True
    return end is not None and bool(re.match(r'\s*(?:(?:was|were|are|is|have|been)\s+){0,2}(?:not|never|excluded)\b', text[end:], re.I))


def _validate(question, title, sections, quotes):
    if not isinstance(question, str) or not question.strip() or not isinstance(title, str):
        raise ValueError('Question must be nonempty text; title must be text')
    if not isinstance(sections, (list, tuple)) or not isinstance(quotes, (list, tuple)):
        raise ValueError('Abstract sections and selected quotes must be sequences')
    if any(not isinstance(q, str) or not q.strip() for q in quotes):
        raise ValueError('Selected quotes must contain nonempty strings')
    indices = []
    for i, section in enumerate(sections):
        if (not isinstance(section, dict) or not isinstance(section.get('text'), str)
                or type(section.get('index', i)) is not int or section.get('index', i) < 0):
            raise ValueError('Each abstract section requires text and a nonnegative integer index')
        indices.append(section.get('index', i))
    if len(set(indices)) != len(indices):
        raise ValueError('Abstract section indices must be unique')


def terms(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    text = re.sub(r"[’']s\b", '', text)
    return {w[:-1] if len(w) > 4 and w.endswith('s') else w
            for w in re.findall(r'\b\w+\b', text) if w not in _STOP}


def _proof(field, text, match):
    return {'field': field, 'char_start': match.start(), 'char_end': match.end(),
            'text': text[match.start():match.end()]}


def _intervals(text):
    result = []
    for m in _TIME.finditer(text):
        number = lambda x: Fraction(_NUM[x.casefold()]) if x.casefold() in _NUM else Fraction(x)
        low, high = number(m[1]), number(m[2] or m[1])
        unit = m[3].casefold().rstrip('s')
        unit = {'sec': 'second', 'min': 'minute', 'hr': 'hour', 'h': 'hour',
                'd': 'day', 'wk': 'week', 'mo': 'month', 'yr': 'year'}.get(unit, unit)
        kind = 'calendar-month' if unit in {'month', 'year'} else 'second'
        factor = {'second': 1, 'minute': 60, 'hour': 3600, 'day': 86400,
                  'week': 604800, 'month': 1, 'year': 12}[unit]
        if low <= high:
            result.append((m, (kind, low * factor, high * factor)))
    return result


def _terminal_outcome_time(question):
    found = list(re.finditer(r'\b(?:after|at)\s+', question, re.I))
    if not found:
        return None
    opening = found[-1]
    suffix = question[opening.end():].strip().rstrip('?').strip()
    interval = _intervals(suffix)
    if len(interval) != 1 or interval[0][0].span() != (0, len(suffix)):
        return None
    prefix = question[:opening.start()]
    if re.search(r'\b(?:not|never|without)\s*$', prefix, re.I):
        return None
    if not re.search(r'\b(?:reduce|reduces|improve|improves|increase|increases|decrease|decreases)\b', prefix, re.I):
        return None
    match = re.compile(re.escape(question[opening.start():opening.end()] + suffix)).search(question, opening.start())
    return match, interval[0][1]



def question_task_kind(question):
    """A narrow lexical prompt hint; it neither checks nor approves sources.

    Unknown and mixed forms stay unknown. Proofs contain only literal question
    cues; no entity, reference standard, population or causal fact is inferred.
    Existing question_contract anchors and every source gate remain separate.
    """
    if not isinstance(question,str)or not question.strip():
        raise ValueError('Question must be nonempty text')
    result={'policy_version':'question-task-kind-v2-draft','kind':'unknown','recognition_rule':None,
            'question_proofs':[],'advisory_only':True,'positive_validation_performed':False}
    if re.search(r'[?;]\s*\S|\b(?:and|or)\s+(?:what|which|how|does|do|did|can|is|are)\b|'
                 r'(?:[.!]\s+|\n\s*)(?:what|which|how|does|do|did|can|is|are)\b|'
                 r'\b(?:and|or)\s+(?:(?:the|a|an)\s+)?(?:diagnostic\s+(?:accuracy|performance)|'
                 r'(?:association|correlation)\s+between|mechanisms?\s+(?:of|by)|effects?\s+of)\b',question,re.I):
        return result
    opening=r'^\s*(?:(?:in|among)\s+[^,;?.]{1,160},\s*)?'
    # Explanatory action grammar is not a how-to or comparison grammar. Opaque
    # subjects/targets are literal question text, not recognized entity facts.
    if re.search(opening+r'how\s+(?:does|do)\s+(?:one|you|we|i|they|someone|anyone)\b',question,re.I):
        return result
    if re.search(opening+r'how\s+(?:does|do)\s+[^?;]{1,160}?\s+(?:compare|compares|perform|performs|conduct|conducts|carry\s+out)\b',question,re.I):
        return result
    grammars={
        'mechanism':[
            r'(?:what|which)\s+(?:(?:is|are)\s+)?(?:the\s+)?(?:(?:molecular|cellular|biological)\s+)?mechanisms?\b',
            r'(?:by|through)\s+(?:what|which)\s+(?:(?:molecular|cellular|biological)\s+)?(?:mechanisms?|pathways?)\b',
            r'how\s+(?:does|do)\s+[^?;]{1,160}?\s+(?:affect|promote|alter|contribute\s+to)\b(?=\s+\S)'],
        'diagnostic':[r'what\s+(?:is|are)\s+(?:the\s+)?diagnostic\s+(?:accuracy|performance)\b'],
        'association':[
            r'what\s+(?:is|are)\s+(?:the\s+)?(?:association|correlation)\b',
            r'(?:is|are|was|were)\s+[^?;]{1,160}?\s+(?:associated|correlated)\s+with\b',
            r'(?:does|do)\s+[^?;]{1,160}?\s+correlate\s+with\b'],
        'effect':[
            r'(?:does|do|did|can)\s+[^?;]{1,160}?\s+(?:reduce|improve|increase|decrease|prevent|affect)\b',
            r'what\s+(?:is|are)\s+(?:the\s+)?effects?\s+of\b']}
    candidates=[]
    for kind,patterns in grammars.items():
        for pattern in patterns:
            match=re.search(opening+pattern,question,re.I)
            if match:
                candidates.append((kind,match));break
    if len(candidates)==1:
        kind,match=candidates[0]
        rule=('how-agent-action-explanation'if kind=='mechanism'and re.search(
            opening+r'how\s+(?:does|do)\b',question,re.I)else'explicit-kind-question')
        result.update(kind=kind,recognition_rule=rule,question_proofs=[_proof('question',question,match)])
    return result


def question_contract(question):
    """Expose only recognized, exact question spans, without entity dictionaries.

    Missing fields stay missing: this lexical profile is not a PICO parser or a
    positive semantic check. The selector still reads the complete question.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question must be nonempty text')
    result = {}
    outcome = re.search(
        r'\b(?:reduce|reduces|improve|improves|increase|increases|decrease|decreases|prevent|prevents)\s+'
        r'(?P<target>.+?)(?=\s+(?:in|among|for|compared\s+(?:with|to)|versus|after|at)\b|[?.]|$)',
        question, re.I)
    if outcome:
        start, end = outcome.span('target')
        result['outcome'] = {'text': question[start:end], 'char_start': start, 'char_end': end}
    comparison = re.search(
        r'\b(?:compar(?:e|ed)\s+(?:with|to)|versus|vs\.?)\s+(?P<target>.+?)'
        r'(?=\s+(?:in|among|for|after|at|reduce|improve|increase|decrease|prevent)\b|[?.]|$)', question, re.I)
    if comparison:
        start, end = comparison.span('target')
        result['comparison'] = {'text': question[start:end], 'char_start': start, 'char_end': end}
    intervention = re.search(r'\b(?:does|do|did|can|will|would|should)\s+(?P<target>.+?)\s+(?:reduce|improve|increase|decrease|prevent|compar(?:e|ed)\s+(?:with|to)|versus)\b', question, re.I)
    if intervention:
        start, end = intervention.span('target')
        result['intervention'] = {'text': question[start:end], 'char_start': start, 'char_end': end}
    population = re.search(r'\b(?:in|among)\s+(?P<target>.+?)(?=,\s*(?:how|what|does|do|is|are|can|will|should)\b|\s+using\b|[?.]|$)', question, re.I)
    if population:
        start, end = population.span('target')
        while end > start and question[end - 1] in '?.': end -= 1
        result['population'] = {'text': question[start:end], 'char_start': start, 'char_end': end}
    return result


def _outcome_terms(text):
    """Small morphology normalization; no biomedical target or scale whitelist."""
    ignored = {'symptom', 'score', 'level', 'rate', 'risk', 'severity', 'clinical',
               'change', 'outcome', 'treatment', 'reported'}
    result = set()
    for word in terms(text):
        if word in ignored: continue
        for suffix in ('ation', 'ment', 'ness', 'ing', 'ion', 'ive', 'ed'):
            if word.endswith(suffix) and len(word) - len(suffix) >= 5:
                word = word[:-len(suffix)]; break
        result.add(word)
    return result


def procedural_only_sentence(text):
    """Recognize explicit objective/procedure-only language, never prove a result.

    No number is compulsory. Observed mechanism, null and qualitative results
    stay eligible; measurement/analysis being performed is not itself a finding.
    """
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Sentence must be nonempty text')
    observed = re.search(r'\b(?:we (?:found|show|showed|observed|demonstrated)|'
        r'(?:results?|findings?) (?:show|showed|indicate|indicated|demonstrate|demonstrated)|'
        r'observed that|revealed that)\b',text,re.I)
    purpose = re.search(r'\b(?:aims?|aimed|objectives?|purpose|hypothes(?:is|ized|ised))\b|'
        r'^\s*To (?:assess|evaluate|determine|examine|investigate|compare|estimate)\b',text,re.I)
    if purpose and not observed:return True
    procedure = re.search(r'\b(?:we|authors?) (?:calculated|assessed|measured|evaluated|tested|compared|'
        r'examined|analy[sz]ed|determined|conducted|performed|recorded|collected)\b|'
        r'\b(?:was|were|are|is) (?:also )?(?:calculated|assessed|measured|evaluated|tested|compared|'
        r'examined|analy[sz]ed|determined|conducted|performed|recorded|collected|randomi[sz]ed|assigned)\b',text,re.I)
    if not procedure or observed:return False
    if re.search(r'\b(?:whether|to (?:assess|evaluate|determine|examine|investigate|compare|estimate))\b',text,re.I):
        return True
    result_predicate = re.search(r'\b(?:improv\w*|reduc\w*|increas\w*|decreas\w*|higher|lower|'
        r'greater|less|difference|similar|significant|correlat\w*|associat\w*|inhibit\w*|'
        r'induc\w*|protect\w*|prevent\w*|no detectable|could not detect)\b',text,re.I)
    return not bool(result_predicate)


def _selected_proofs(abstract_sections, finding_quotes):
    proofs=[]
    for quote in finding_quotes:
        for i,section in enumerate(abstract_sections):
            start=section['text'].find(quote)
            if start>=0:
                proofs.append({'field':f'abstract:{section.get("index",i)}','char_start':start,
                               'char_end':start+len(quote),'text':quote})
                break
    return proofs


def _selected_activity_instead_of_diagnosis(question,title,sections,finding_quotes):
    """Recognize an explicit disease-activity target substitution, not approve.

    Established-patient populations and actual activity goals are required. The
    word active alone is insufficient: active infection may be a presence target.
    The selected result, not another analysis in the article, controls this veto.
    """
    result={"blocks":[],"cautions":[],"recognized_checks":[]}
    diagnostic=re.search(r'\b(?:diagnostic\s+accuracy|diagnos\w*|sensitivity|specificity)\b',question,re.I)
    if not diagnostic or re.search(r'\b(?:activity|severity|remission|relapse|flare|monitor\w*)\b',question,re.I):return result
    condition=re.search(r'\bfor\s+(.+?)(?=\s+(?:in|among|using|against)\b|[?.]|$)',question,re.I)
    if not condition:
        condition=re.search(r'\bdiagnos(?:e|ing|is of)\s+(.+?)(?=\s+(?:in|among|using|against)\b|[?.]|$)',question,re.I)
    if not condition:return result
    name=condition[1].strip();words=re.findall(r'\b\w+\b',name)
    acronym=''.join(word[0]for word in words)
    names=[re.escape(name)]
    if len(words)>1 and len(acronym)>1:names.append(re.escape(acronym))
    disease=r'(?:'+'|'.join(names)+r')'
    activity=re.compile(r'\b(?:(?:disease|clinical|endoscopic|histologic(?:al)?|mucosal)\s+activity|'
        r'severity|remission)\b|\b'+disease+r'\s+activity\b',re.I)
    selected_activity=re.compile(activity.pattern+r'|\b(?:diagnos\w*|detect\w*|classif\w*)\s+active\s+disease\b',re.I)
    known=re.compile(r'\bpatients?\s+(?:(?:diagnosed\s+)?with|of|having|suffering\s+from)\s+[^.;!?]{0,100}?\b'+disease+
        r'\b|\b'+disease+r'\s+patients?\b',re.I)
    uncertain=re.compile(r'\b(?:suspected|possible|potential|unconfirmed|undiagnosed|at risk|no|not|without|excluded|negative for|free of|rule out)\b',re.I)
    fields=[('title',title,None)]+[(f'abstract:{section.get("index",i)}',section['text'],section)
        for i,section in enumerate(sections)]
    population=[];goals=[]
    for field,text,section in fields:
        label=' '.join(str((section or{}).get('label')or'').casefold().replace('-',' ').split())
        category=str((section or{}).get('nlm_category')or'').casefold()
        background=category=='background'or bool(re.match(r'^(?:background|introduction|rationale|context)\b',label))
        if not background:
            for match in known.finditer(text):
                prefix=text[max(0,match.start()-45):match.start()]
                if not uncertain.search(match[0])and not _negated(text,match.start(),match.end())and not re.search(r'\b(?:suspected|possible|unconfirmed)\s*$',prefix,re.I):
                    population.append(_proof(field,text,match))
        goal_field=field=='title'or category in {'objective','methods'}or bool(re.search(
            r'\b(?:objectives?|aims?|purpose|methods?|design|participants?|patients?)\b',label))
        if goal_field:
            goals.extend(_proof(field,text,match)for match in activity.finditer(text)
                         if not _negated(text,match.start(),match.end()))
    if not population or not goals:return result
    # A different genuine diagnostic result in the article does not rescue an
    # activity-only selection. Conversely, do not blanket-veto a selected
    # presence result simply because the same article also studies activity.
    selected=[quote for quote in finding_quotes if selected_activity.search(quote)]
    if not selected:return result
    real_presence=[quote for quote in finding_quotes if re.search(
        r'\b(?:sensitivity|specificity|accuracy|auc|diagnos\w*|detect\w*|distinguish\w*)\b',quote,re.I)
        and re.search(r'\b(?:healthy\s+controls?|non[ -]?cases?|non[ -]?diseased|disease[ -]?free|'
                      r'presence\s+(?:or|and)\s+absence|with\s+and\s+without)\b',quote,re.I)
        and not selected_activity.search(quote)]
    result["recognized_checks"].append("diagnostic-presence-versus-disease-activity")
    proof=_selected_proofs(sections,selected)
    bucket="cautions"if real_presence else"blocks"
    result[bucket].append({"code":"selected_mixed_diagnostic_and_activity_targets"if real_presence else"selected_activity_instead_of_diagnostic_presence",
        "message":"Selected findings mix a disease-presence result with a separate activity endpoint; keep targets distinct."if real_presence else
            "Actual study goal and population concern established disease activity/severity/remission; selected activity findings do not establish diagnostic disease presence.",
        "source_proofs":[*goals,*population,*proof],"question_proof":_proof("question",question,diagnostic)})
    return result


def _endpoint_literal_name(text):
    """Only a complete literal name; unknown abbreviations remain inconclusive."""
    text=text.strip()
    words=text.split()
    if not text or len(text)>120 or not re.fullmatch(r'\w+(?:[ -]\w+){0,7}',text):return None
    if any(word.casefold()in {'and','or','versus','without','not'}for word in words):return None
    if re.search(r'\b(?:[a-z]{1,3})?[A-Z]{2,}[A-Za-z0-9]*\b',text):return None
    return r'(?<!\w)'+r'\s+'.join(re.escape(word)for word in words)+r'(?!\w)'


def _endpoint_section_role(section):
    category=str(section.get('nlm_category')or'').casefold()
    label=' '.join(str(section.get('label')or'').casefold().replace('-',' ').split())
    if category=='methods'or re.match(r'^(?:methods?|materials and methods|patients and methods|design|participants?)\b',label):
        return 'methods'
    if category=='results'or re.match(r'^(?:results?|findings|main results)\b',label):return 'results'
    if category=='background'or re.match(r'^(?:background|introduction|rationale|context)\b',label):return 'background'
    if category=='objective'or re.match(r'^(?:objectives?|aims?|purpose)\b',label):return 'objective'
    if category=='conclusions'or re.match(r'^conclusions?\b',label):return 'conclusions'
    if category=='unassigned'and not label:return 'unknown'
    return 'other'if category or label else'unknown'


def _endpoint_observed(text):
    """Recognize a reported clause, preserving nulls but excluding future work."""
    if procedural_only_sentence(text):return False
    if re.search(r'\b(?:may|might|would|will|should|future|planned|aims?|aimed)\b',text,re.I):return False
    if re.search(r'\bcould\b(?!\s+not\b)',text,re.I):return False
    if re.search(r'\b(?:no|without)\s+(?:data|information|results?)\b|'
                 r'\bnot\s+(?:assessed|measured|evaluated|reported)\b',text,re.I):return False
    return bool(re.search(r'\b(?:was|were|is|are|had|showed|found|observed|yielded|'
        r'demonstrated|revealed|remained|developed|reached|occurred|occur)\b|\d',text,re.I))


def _endpoint_alias_uncertain(text):
    # Statistical abbreviations do not name a clinical endpoint. Every other
    # acronym is left unresolved; no medical expansion or grade list is used.
    statistical={'AUC','ROC','PPV','NPV','CI','HR','OR','RR'}
    return any(match[0]not in statistical for match in re.finditer(
        r'\b(?:[a-z]{1,3})?[A-Z]{2,}[A-Za-z0-9]*\b',text))


def _selected_endpoint_contracts(question, sections, finding_quotes):
    """Bounded postselection-only vetoes for two explicit English contracts.

    A veto means the requested qualified endpoint is not established by these
    selected findings, never that clinical truth or source relevance was proved.
    Unstructured roles, aliases and definitions stay inconclusive.
    """
    result={'blocks':[],'cautions':[],'recognized_checks':[]}
    if re.search(r'[?;]\s*\S|\b(?:and|or)\s+(?:what|how|does|is|are)\b',question,re.I):
        return result
    by_index={section.get('index',i):section for i,section in enumerate(sections)}
    selected=[];primary_selected=[]
    for proof in _selected_proofs(sections,finding_quotes):
        section=by_index[int(proof['field'].split(':')[1])]
        role=_endpoint_section_role(section)
        primary_selected.append((proof['text'],role))
        if role=='results'and _endpoint_observed(proof['text']):
            selected.append(proof['text'])
    if not selected:return result
    diagnostic=re.search(r'\bdiagnostic\s+accuracy\b',question,re.I)
    qualified=re.search(r'\bfor\s+(?P<qualified>clinically\s+significant\s+'
        r'(?P<name>.+?))(?=\s+(?:in|among|using|against)\b|[?.;,]|$)',question,re.I)
    if diagnostic and qualified:
        name=_endpoint_literal_name(qualified['name'])
        if name and not _negated(question,qualified.start('qualified'),qualified.end('qualified')):
            qualified_endpoint=re.compile(r'\bclinically\s+significant\s+'+name,re.I)
            measurement=r'\b(?:sensitivity|specificity|accuracy|predictive\s+value|auc|area\s+under)\b'
            unqualified=re.compile(r'\b(?:for\s+(?:(?:the\s+)?(?:detection|diagnosis|'
                r'identification|classification)\s+of\s+|detecting\s+)?|in\s+detecting\s+)'+name,re.I)
            candidates=[quote for quote in selected if re.search(measurement,quote,re.I)
                and unqualified.search(quote)]
            if candidates and not any(_endpoint_alias_uncertain(quote)for quote in selected):
                # A real Methods definition or qualified analysis can make an
                # unqualified result shorthand. Title/background or a future
                # conclusion cannot establish that analysis.
                from .context import section_sentence_spans
                established=False
                for section in sections:
                    role=_endpoint_section_role(section)
                    if role not in {'methods','results','unknown'}:continue
                    for span in section_sentence_spans(section['text']):
                        sentence=section['text'][span['char_start']:span['char_end']]
                        has_qualifier=bool(qualified_endpoint.search(sentence))
                        inverted_definition=(role=='methods'and bool(re.search(
                            name+r'\s+(?:was|were|is|are)\s+(?:defined|classified|considered)\s+'
                            r'(?:as|by|to\s+mean|based\s+on)\s+[^.;]{0,80}\bclinically\s+significant\b'
                            r'|\bdefined\s+(?:the\s+)?'+name+r'\s+as\s+[^.;]{0,80}\bclinically\s+significant\b',
                            sentence,re.I)))
                        future=bool(re.search(r'\b(?:may|might|could|would|will|should|future|aims?|aimed)\b',sentence,re.I))
                        if (has_qualifier or inverted_definition) and not future and (role=='methods'or role=='unknown'
                                or _endpoint_observed(sentence)):
                            established=True
                        if role=='methods'and _endpoint_alias_uncertain(sentence)and re.search(
                                r'\b(?:defined|definition|endpoint|target)\b',sentence,re.I):
                            established=True
                selected_qualified=any(role in {'results','conclusions','unknown'}and qualified_endpoint.search(quote)and _endpoint_observed(quote)
                    for quote,role in primary_selected)
                if not established and not selected_qualified:
                    result['recognized_checks'].append('selected-qualified-diagnostic-endpoint')
                    result['blocks'].append({
                        'code':'selected_requested_diagnostic_qualifier_unestablished',
                        'message':'Selected observed accuracy names the unqualified literal target; no actual primary Methods/Results qualified definition or analysis was recognized.',
                        'question_proof':{'field':'question','char_start':qualified.start('qualified'),
                            'char_end':qualified.end('qualified'),'text':qualified['qualified']},
                        'source_proofs':_selected_proofs(sections,candidates)})
    progression=re.search(r'\b(?P<request>progression\s+to\s+(?P<name>.+?))'
        r'(?=\s+(?:in|among|with|using|after|at)\b|[?.;,]|$)',question,re.I)
    clinical_context=any(_endpoint_section_role(section)in {'methods','results'}and re.search(
        r'\b(?:patients?|participants?|people|adults?|cohort|clinical)\b',section['text'],re.I)
        for section in sections)
    if progression and clinical_context and not _negated(question,progression.start(),progression.end()):
        name=_endpoint_literal_name(progression['name'])
        if name:
            clinical=[]
            pattern=re.compile(r'\b(?:hazard|risk|rate|probability|likelihood|incidence|odds)\s+of\s+'
                r'(?:[\w-]+\s+){0,5}regression\b|\bregression\s+'
                r'(?:hazard|risk|rate|probability|likelihood|incidence|odds)\b',re.I)
            for quote in selected:
                for match in pattern.finditer(quote):
                    occurrence=re.search(r'\bregression\b',match[0],re.I)
                    start,end=match.start()+occurrence.start(),match.start()+occurrence.end()
                    before,after=quote[max(0,start-50):start],quote[end:end+50]
                    statistical=(re.search(r'\b(?:cox|logistic|linear|poisson|probit|polynomial|'
                        r'statistical|multivariable|multivariate|univariate|ridge|lasso|quantile|'
                        r'ordinal|nonlinear|multiple|simple|robust|bayesian|elastic[ -]net)\s*$',before,re.I)
                        or re.match(r'\s+(?:models?|analys(?:is|es)|coefficients?|equations?|'
                            r'parameters?|to\s+(?:the\s+)?mean)\b',after,re.I))
                    if not statistical:clinical.append(quote);break
            actual_target=re.compile(r'\b(?:progression\s+(?:from\s+[^.;]{1,120}?\s+)?to|progressed\s+to|developed|reached)\s+(?:the\s+)?'+name+
                r'|\b(?:risk|hazard|rate|incidence|probability|likelihood|odds)\s+of\s+'
                r'(?:progression\s+to\s+)?(?:the\s+)?'+name,re.I)
            genuine=any(actual_target.search(quote)for quote in selected)
            unresolved_alias=any(_endpoint_alias_uncertain(quote)and re.search(
                r'\b(?:progression|progressed|developed|reached|risk|hazard|rate|incidence|'
                r'probability|likelihood|odds)\b',quote,re.I)for quote in selected if quote not in clinical)
            if clinical and not genuine and not unresolved_alias:
                result['recognized_checks'].append('selected-progression-versus-clinical-regression')
                result['blocks'].append({'code':'selected_clinical_regression_instead_of_requested_progression',
                    'message':'Selected observed clinical regression endpoint does not establish progression to the requested literal target; statistical regression is not this check.',
                    'question_proof':_proof('question',question,progression),
                    'source_proofs':_selected_proofs(sections,list(dict.fromkeys(clinical)))})
    return result


def check_selected_findings(question, title, abstract_sections, finding_quotes):
    """Demote recognized selected-outcome substitutions, never approve a card.

    Full-source scope and selected-result applicability are separate checks. A
    source may measure the requested endpoint yet the selected sentences report
    only a different endpoint or correlation between two changes.
    """
    _validate(question, title, abstract_sections, finding_quotes)
    result = {'policy_version': POLICY_VERSION, 'blocks': [], 'cautions': [],
              'recognized_checks': [], 'positive_validation_performed': False}
    if not finding_quotes:return result
    proofs=_selected_proofs(abstract_sections,finding_quotes)
    if all(procedural_only_sentence(quote)for quote in finding_quotes):
        result['recognized_checks'].append('selected-result-versus-procedure')
        result['blocks'].append({'code':'selected_findings_are_purpose_or_procedure',
            'message':'Selected sentences state objectives or procedures without an observed scientific result.',
            'source_proofs':proofs})
        return result
    activity=_selected_activity_instead_of_diagnosis(question,title,abstract_sections,finding_quotes)
    for bucket in ('blocks','cautions','recognized_checks'):
        result[bucket].extend(activity[bucket])
    if activity['blocks']:return result
    endpoint=_selected_endpoint_contracts(question,abstract_sections,finding_quotes)
    for bucket in ('blocks','cautions','recognized_checks'):
        result[bucket].extend(endpoint[bucket])
    if endpoint['recognized_checks']:
        result['postselection_endpoint_contract_version']='selected-endpoint-contract-v1'
    if endpoint['blocks']:return result
    target = question_contract(question).get('outcome')
    if not target: return result
    wanted = _outcome_terms(target['text'])
    result['recognized_checks'].append('selected-requested-outcome')
    # Association between endpoint changes is not the intervention's effect on
    # either endpoint. This also preserves null *treatment* results as eligible.
    association = re.compile(
        r'\b(?:changes?|improvements?|increases?|decreases?)\s+in\b.+?'
        r'\b(?:correlat\w*|associat\w*|relationship)\b.+?'
        r'\b(?:changes?|improvements?|increases?|decreases?)\s+in\b', re.I)
    accounting = re.compile(r'(?=.*\b(?:studies|trials|participants|subjects|patients)\b)'
        r'(?=.*\b(?:includ\w*|randomi[sz]\w*|found|identified|met|enrolled)\b)', re.I)
    reported_result = re.compile(r'\b(?:improv\w*|reduc\w*|increas\w*|decreas\w*|higher|lower|'
        r'difference|similar|significant|correlat\w*|associat\w*|effect|sensitivity|specificity|ratio|versus|vs)\b', re.I)
    if all(accounting.search(quote) and not reported_result.search(quote) for quote in finding_quotes):
        result['blocks'].append({'code':'selected_findings_are_study_accounting',
            'message':'Selected sentences only count enrolled participants or included studies; they report no requested outcome result.',
            'source_proofs':proofs,'question_proof':{'field':'question',**target}})
        return result
    if not wanted: return result
    relevant = [quote for quote in finding_quotes if wanted & _outcome_terms(quote)
                and not association.search(quote)]
    if relevant: return result
    # Unknown acronyms and assay/scale results cannot be judged by lexical
    # absence alone. Veto only when a selected endpoint-change association
    # explicitly mentions the requested target; otherwise leave it to facets.
    if any(wanted & _outcome_terms(quote) and association.search(quote) for quote in finding_quotes):
        result['blocks'].append({'code': 'selected_outcome_is_change_association',
            'message': 'Selected target evidence reports association between changes, not intervention effect on the requested outcome.',
            'source_proofs': proofs, 'question_proof': {'field': 'question', **target}})
    return result


def _actual_placebo_comparison(question, title, sections):
    """Veto an explicitly different actual control, never infer an absent arm.

    Historical placebo trials cannot establish this study's allocation. A
    genuinely reported placebo arm/analysis keeps mixed-control sources eligible.
    No treatment/disease dictionary or positive scientific validation is used.
    """
    result={"blocks":[],"cautions":[],"recognized_checks":[]}
    request=re.search(r'\b(?:compar(?:e|ed)\s+(?:with|to)|versus|vs\.?)\s+'
        r'(?:(?:a|the|matched|matching)\s+)*placebo\b|\bplacebo[ -]controlled\b',question,re.I)
    if not request or _negated(question,request.start(),request.end()):return result
    from .context import section_sentence_spans
    result["recognized_checks"].append("requested-placebo-versus-actual-control")
    allocation=re.compile(r'\b(?:randomly\s+assigned|randomi[sz]ed|assigned|allocated)\b',re.I)
    alternative=re.compile(
        r'\bimmediat(?:e|ely)\b.{0,240}?\b(?:or|and|versus|vs)\b.{0,120}?'
        r'\b(?:defer(?:red|ral)?|delayed?|wait(?:ing)?[ -]?list)\b'
        r'|\b(?:or|versus|vs|compared\s+(?:with|to))\s+(?:(?:the|a|an)\s+)?'
        r'(?:defer(?:red|ral)?|delayed?|wait(?:ing)?[ -]?list|usual\s+care|standard\s+care|no\s+(?:treatment|intervention))\b'
        r'|\bcontrol(?:\s+(?:group|arm))?\b.{0,70}?\b(?:received|given)\b.{0,30}?'
        r'\b(?:usual\s+care|standard\s+care|no\s+(?:treatment|intervention))\b',re.I)
    previous=re.compile(r'\b(?:previous\w*|prior|earlier|other)\s+(?:[\w-]+\s+){0,3}'
        r'(?:trials?|studies|research|reports?)\b|\b(?:than|in contrast to)\s+(?:in\s+)?placebo[ -]controlled\s+trials?\b',re.I)
    review=bool(re.search(r'\b(?:review|meta[ -]analysis)\b',title,re.I))
    different=[];placebo=[]
    if re.search(r'\bplacebo[ -]controlled\b',title,re.I):
        placebo.append({"field":"title","char_start":0,"char_end":len(title),"text":title})
    for i,section in enumerate(sections):
        label=" ".join(str(section.get("label")or"").casefold().replace("-"," ").split())
        category=str(section.get("nlm_category")or"").casefold()
        if category in {"background","objective"}or re.match(
                r'^(?:background|objectives?|rationale|context|funding|registration)\b',label):continue
        text=section["text"];field=f'abstract:{section.get("index",i)}'
        method=category=="methods"or bool(re.search(r'\b(?:methods?|design|interventions?|participants?|patients?)\b',label))
        for span in section_sentence_spans(text):
            sentence=text[span["char_start"]:span["char_end"]]
            if previous.search(sentence):continue
            proof={"field":field,"char_start":span["char_start"],"char_end":span["char_end"],"text":sentence}
            assigned=allocation.search(sentence)
            if assigned and _negated(sentence,assigned.start(),assigned.end()):assigned=None
            arms=sentence[assigned.end():]if assigned else sentence
            control=re.search(r'\bplacebo\b',arms,re.I)
            active_placebo=bool(control and not _negated(arms,control.start(),control.end())and(
                assigned or re.search(r'\b(?:placebo\s+(?:group|arm)|(?:group|arm)\s+received\s+placebo)\b',arms,re.I)
                or (method and re.search(r'\bplacebo[ -]controlled\b',arms,re.I))
                or (review and re.search(r'\b(?:pooled|included|compar\w*|randomi[sz]\w*|meta[ -]analysis)\b',arms,re.I))))
            if active_placebo:placebo.append(proof)
            if assigned and alternative.search(arms)and not active_placebo:different.append(proof)
    if different:
        bucket="cautions"if placebo else"blocks"
        result[bucket].append({"code":"multiple_actual_control_analyses"if placebo else"different_actual_control_than_requested_placebo",
            "message":"Source reports an actual placebo arm/analysis as well as other controls; keep them distinct."if placebo else
                "Actual allocation explicitly uses deferred/waitlist/usual-care or no-intervention control; background placebo trials do not establish a placebo arm.",
            "source_proofs":[*different,*placebo],"question_proof":_proof("question",question,request)})
    return result


def check_source_scope(question, title, abstract_sections, selected_quotes=()):
    """Return explicit blocks/cautions and exact source proof spans; never promote.

    A block downgrades a proposed direct relation to related/unclear. Unrecognized
    language is left unvalidated. selected_quotes does not limit full-source vetoes.
    Matching a time anywhere (including treatment duration) or an adult word is
    non-certifying lexical detection; it never establishes endpoint/eligibility support.
    """
    _validate(question, title, abstract_sections, selected_quotes)
    fields = [('title', title)] + [(f'abstract:{s.get("index", i)}', s['text'])
                                     for i, s in enumerate(abstract_sections)]
    result = {'policy_version': POLICY_VERSION, 'blocks': [], 'cautions': [], 'recognized_checks': [],
              'positive_validation_performed': False}
    full = '\n'.join(t for _, t in fields)
    actual_control=_actual_placebo_comparison(question,title,abstract_sections)
    for bucket in ('blocks','cautions','recognized_checks'):
        result[bucket].extend(actual_control[bucket])

    def add(bucket, code, message, proofs, question_match=None):
        row = {'code': code, 'message': message, 'source_proofs': proofs}
        if question_match:
            row['question_proof'] = _proof('question', question, question_match)
        result[bucket].append(row)

    def matches(pattern):
        rx = re.compile(pattern, re.I)
        return [(f, t, m) for f, t in fields for m in rx.finditer(t)]

    # General study-purpose and treatment-strategy distinctions. These checks
    # never contain disease/intervention names and never infer positive support.
    request_stopping = re.search(r'\b(?:discontinu(?:e|ed|ing|ation)|withdraw(?:al|ing)?|stopping|stop)\b', question, re.I)
    request_continuing = re.search(r'\b(?:continu(?:e|ed|ing|ation)|maintenance|maintain(?:ed|ing)?)\b', question, re.I)
    if request_stopping and request_continuing:
        result['recognized_checks'].append('continuation-versus-stopping-strategy')
        study_fields={'title'} | {f'abstract:{section.get("index",i)}'for i,section in enumerate(abstract_sections)
            if not (str(section.get('label')or '').casefold()in {'background','objective','objectives','context','rationale'}
                or str(section.get('nlm_category')or '').casefold()in {'background','objective'})}
        attrition=re.compile(r'\b(?:protocol|prematurely|dropout|attrition|lost to follow|withdrew from|withdrawn from)\b',re.I)
        design=re.compile(r'\b(?:randomi[sz]\w*|assign\w*|compar\w*|versus|group|arm|continu\w*|maintain\w*)\b',re.I)
        contrasts=[(f,t,m)for f,t,m in matches(
            r'[^.;!?]*\b(?:discontinu(?:e|ed|ing|ation)|withdraw(?:al|ing|n)?|stopping|stop)\b[^.;!?]*')
            if f in study_fields and not attrition.search(m[0])and design.search(m[0])]
        placebo=[(f,t,m)for f,t,m in matches(r'[^.;!?]*\bplacebo\b[^.;!?]*')
            if f in study_fields and not attrition.search(m[0])and design.search(m[0])]
        # A placebo arm alone can represent *starting* another treatment after
        # remission with a different modality. Require an explicit withdrawal
        # comparison or evidence that the same named drug was already used for
        # stabilization/open-label response before continuation vs placebo.
        prior=[]
        patterns=(r'\bopen[ -]label\s+(?:(?:treatment|therapy|phase)\s+(?:with|using)\s+)?(?P<drug>[\w-]+)',
                  r'\b(?:stabili[sz]ed|responded|remitted)\s+(?:on|with|to)\s+(?P<drug>[\w-]+)',
                  r'\bstable\s+(?:on|treatment\s+with)\s+(?P<drug>[\w-]+)')
        structural={'treatment','therapy','phase','trial','period','medication','drug','study'}
        for pattern in patterns:
            prior.extend((f,t,m)for f,t,m in matches(pattern)
                if f in study_fields and m['drug'].casefold()not in structural)
        linked=[]
        for f,t,m in placebo:
            allocation=re.search(r'\b(?:randomi[sz]ed|assigned|allocated)\b',m[0],re.I)
            arms=m[0][allocation.end():]if allocation else m[0]
            if any(re.search(r'(?<![\w-])'+re.escape(before['drug'])+r'(?![\w-])',arms,re.I)
                   or re.search(r'\b(?:same (?:drug|medication|treatment|therapy)|'
                       r'continu(?:e|ed|ing) (?:the |their |study )?(?:drug|medication|treatment|therapy))\b',arms,re.I)
                   for _,_,before in prior):
                linked.append((f,t,m))
        if not contrasts and not linked:
            add('blocks', 'requested_stopping_comparison_not_established',
                'No recognized withdrawal/stopping comparison or same-drug stabilization to continuation/placebo transition; placebo alone and protocol attrition do not establish stopping ongoing medication.',
                [{'field': 'title', 'char_start': 0, 'char_end': len(title), 'text': title},
                 *[_proof(f,t,m)for f,t,m in placebo]], request_stopping)

    # A question about reducing an outcome in people *with* a condition asks
    # about established disease. Prevention-only eligibility is a different
    # clinical setting. Presence of a prevention word alone never vetoes a trial.
    established = re.search(r'\b(?:in|among)\s+[^?;.]+?\bwith\s+[^?;.]+', question, re.I)
    prevention_question = re.search(r'\b(?:prevent\w*|incidence|risk of|onset)\b', question, re.I)
    if established and not prevention_question:
        prevention = matches(r'[^.;!?]*\b(?:prevent\w*|incidence|onset)\b[^.;!?]*')
        condition_terms = _outcome_terms(re.split(r'\bwith\s+', established[0], flags=re.I)[-1])
        no_condition = [(f, t, m) for f, t, m in matches(
            r'[^.;!?]*\b(?:without|free of|no history of|at risk (?:of|for))\b[^.;!?]*')
            if condition_terms & _outcome_terms(re.split(
                r'\b(?:without|free of|no history of|at risk (?:of|for))\b', m[0], flags=re.I)[-1])]
        prenatal = matches(r'[^.;!?]*\b(?:antenatal|prenatal|during pregnancy|pregnant women)\b[^.;!?]*')
        postpartum = re.search(r'\bpost(?:partum|natal)\b', established[0], re.I)
        if prevention and (no_condition or (postpartum and prenatal)):
            result['recognized_checks'].append('prevention-versus-established-condition')
            # A mixed treatment/prevention review can still supply an eligible
            # treatment subset; do not blanket-veto it when explicitly reported.
            treatment_subset = matches(r'[^.;!?]*\b(?:diagnosed with|existing|established|met (?:the )?(?:diagnostic )?criteria for)\b[^.;!?]*')
            mixed_purpose = matches(r'[^.;!?]*\b(?:treatment\s+(?:or|and)\s+prevention|prevention\s+(?:or|and)\s+treatment)\b[^.;!?]*')
            if mixed_purpose:
                add('cautions','mixed_treatment_prevention_review',
                    'Source includes treatment and prevention; only a distinct established-condition treatment result can answer the question.',
                    [_proof(f,t,m)for f,t,m in mixed_purpose],established)
            if not treatment_subset and not mixed_purpose:
                add('blocks', 'prevention_population_for_established_condition',
                    'Reported prevention/at-risk eligibility differs from treating participants with an existing condition.',
                    [_proof(f, t, m) for f, t, m in [*prevention, *no_condition, *prenatal]], established)

    if _CONTRAST.search(question) and not _COMBO.search(question):
        addon = re.search(r'\badd[ -]on\b', title, re.I)
        dose = re.search(r'\bdose[ -](?:reduc\w*|titrat\w*|adjust\w*)\b', title, re.I)
        if addon and dose and _CONTRAST.search(title):
            result['recognized_checks'].append('explicit-comparative-add-on-title')
            add('blocks', 'different_addon_dose_regimen',
                'Comparative title explicitly studies add-on plus dose adjustment; question does not request that regimen.',
                [_proof('title', title, addon), _proof('title', title, dose)])

    requested_reference = _REFERENCE.search(question)
    if requested_reference:
        result['recognized_checks'].append('requested-named-reference')
        requested = terms(requested_reference[1])
        named = matches(r'\b(?P<name>consensus(?:[ -]based)?)\s+reference\s+standard\b')
        named += matches(r'\b(?:against|using)\s+(?:the\s+)?(?P<name>[\w-]+(?:\s+[\w-]+){0,5})\s+as\s+(?:the\s+)?(?:reference|gold)\s+standard\b')
        named += matches(r'\b(?P<name>[\w-]+(?:\s+[\w-]+){0,4})\s+(?:served|was used|were used)\s+as\s+(?:the\s+)?(?:reference|gold)\s+standard\b')
        named = [(f, text, m) for f, text, m in named if not _negated(text, m.start(), m.end())
                 and not re.search(r'\b(?:not|never|no|without|excluded)\b', m[0], re.I)]
        requested_analyses = [(f, text, m) for f, text, m in named
                              if requested and requested <= terms(m['name'])]
        for f, text, m in named:
            if requested and not requested & terms(m['name']):
                add('cautions' if requested_analyses else 'blocks',
                    'multiple_reference_analyses' if requested_analyses else 'different_reference_standard',
                    'Source also reports the requested named analysis; separate reference analyses must remain distinct.'
                    if requested_analyses else 'Explicit source reference standard has no requested-name token overlap.',
                    [_proof(f, text, m), *[_proof(f2, text2, m2) for f2, text2, m2 in requested_analyses]], requested_reference)
        combined_question = bool(re.search(r'\b(?:plus|and|combined|composite|both|symptoms?|clinical)\b', requested_reference[1], re.I))
        if not combined_question:
            for f, text, m in matches(r'\b(?:defined|classified|diagnosed)\s+(?:as|by)\s+[^.;]+'):
                if (not _negated(text, m.start(), m.end()) and _CLINICAL.search(m[0]) and re.search(r'\b(?:combined with|plus|and)\b', m[0], re.I)
                        and requested & terms(m[0])):
                    add('cautions' if requested_analyses else 'blocks',
                        'multiple_reference_analyses' if requested_analyses else 'combined_clinical_reference',
                        'Source definition explicitly combines clinical/symptom criteria with requested laboratory terms.',
                        [_proof(f, text, m)], requested_reference)

    target = re.search(r'\bprogression\s+from\s+.+?\s+to\s+([^?]+?)[?.]*\s*$', question, re.I)
    if target:
        requested = terms(target[1])
        source = terms(full)
        acronym = ''.join(w[0] for w in re.findall(r'\b\w+\b', re.sub(r"[’']s\b", '', target[1]).casefold()))
        result['recognized_checks'].append('named-progression-target')
        if requested and not requested <= source and not (len(acronym) > 1 and acronym in source):
            add('blocks', 'progression_target_not_established',
                'Distinctive requested target terms or its explicit acronym are absent from whole frozen source.',
                [{'field': 'title', 'char_start': 0, 'char_end': len(title), 'text': title}], target)

    timing = _terminal_outcome_time(question)
    if timing:
        requested_match, requested_interval = timing
        recognized = [(f, text, m, interval) for f, text in fields for m, interval in _intervals(text)]
        result['recognized_checks'].append('terminal-outcome-time')
        if not any(interval == requested_interval for _, _, _, interval in recognized):
            add('blocks', 'outcome_time_not_established',
                'No exact equivalent interval in recognized source time expressions; a broader interval is not equivalent.',
                [_proof(f, text, m) for f, text, m, _ in recognized], requested_match)

    intervention = re.search(r'\b(?:do|does|did|can|will|would|should)\s+(.+?)\s+(?:compare(?:d)?\s+(?:with|to)|versus|vs\.?)\b', question, re.I)
    if intervention and not _COMBO.search(question):
        requested = terms(intervention[1])
        if requested and not requested & terms(title):
            result['recognized_checks'].append('requested-intervention-not-in-title')
            for f, text, m in matches(r'\b(?:all|both)\s+[^.;]{0,70}?\b(?:received|were given|were treated with|took)\s+[^.;]{1,150}'):
                if (requested <= terms(m[0]) and not _negated(text, m.start(), m.end())
                        and not re.search(r'\b(?:not|never|no|without)\b', m[0], re.I)):
                    add('blocks', 'intervention_given_to_all_groups',
                        'Requested intervention is explicitly given to all/both groups and absent from title topic.',
                        [_proof(f, text, m)], intervention)
            for f, text, m in matches(r'\b(?:received|were given)\s+(?:concomitant|concurrent|background)\s+[^.;]{1,150}|\b(?:received|were given)\s+[^.;]{1,80}?\bconcomitantly'):
                if (requested <= terms(m[0]) and not _negated(text, m.start(), m.end())
                        and not re.search(r'\b(?:not|never|no|without)\b', m[0], re.I)):
                    add('blocks', 'intervention_is_cointervention',
                        'Requested intervention is explicitly concomitant/background care and absent from title topic.',
                        [_proof(f, text, m)], intervention)

    if re.search(r'\badults?\b', question, re.I):
        result['recognized_checks'].append('explicit-adult-eligibility')
        adult = [(f, text, m) for f, text, m in matches(r'\badults?\b')
                 if not _negated(text, m.start(), m.end())]
        ages = matches(r'\b(?:aged|age)\s*(?:≥|>=|>|at least)?\s*(\d+)(?:\s*(?:to|[-–])\s*\d+)?\s*years?\b')
        lower_bound = any(int(m[1]) >= 18 and not _negated(text, m.start(), m.end())
                          and not re.search(r'\b(?:mean|median|average)\s*$', text[:m.start()], re.I)
                          for _, text, m in ages)
        lower_bound |= any(int(m[1]) >= 18 and not _negated(text, m.start(), m.end())
                           and not re.search(r'\b(?:mean|median|average)\s*$', text[:m.start()], re.I)
                           for _, text, m in matches(r'\bage\s*(?:≥|>=)\s*(\d+)\b(?!\s*(?:months?|weeks?|days?|hours?))'))
        if not adult and not lower_bound:
            means = matches(r'\bmean age\b[^.;]{0,70}')
            add('cautions', 'adult_eligibility_not_explicit',
                'No explicit adult label or recognized lower age bound >=18 years; mean age does not establish eligibility.',
                [_proof(f, text, m) for f, text, m in means])
    return result
