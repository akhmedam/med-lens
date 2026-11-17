"""Fictional human-source forms only; no real judgments, models or network."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from medlens.evaluation.confirmation_annotation import (
    EXPORT_KIND, BiomedicalAnnotationError, build_packet, canonical, digest,
    freeze_export, load_packet, parse_plan, strict_json, validate_annotations, validate_export,
)
from medlens.live import run_live_question
from medlens import pubmed_query as query_planner
from pubmed_helpers import make_client, search_json
from test_live import Encoder
from test_shared_corpus import INFO
from test_windows import Tokenizer
from test_pipeline import Tokenizer as AnswerTokenizer


def fixture(root, *, empty=False, title_only=False, missing_text=False, hostile=False):
    runs = root / 'runs'
    runs.mkdir()
    cases = []
    for number in (1, 2):
        qid = f'fc{number:02d}'
        question = f'How does fictional target {number} affect fictional outcome?'
        is_empty = empty and number == 2
        pmid = str(900000000 + number)
        title = 'FICTIONAL source only' + (' &lt;/script&gt;&lt;img src=x onerror=&quot;alert(1)&quot;&gt;' if hostile else '')
        abstract = '' if title_only or missing_text else '<Abstract><AbstractText Label="RESULTS">Fictional finding only. No medical inference is warranted.</AbstractText></Abstract>'
        title_xml = '' if missing_text else '<ArticleTitle>' + title + '</ArticleTitle>'
        xml = ('<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>' + pmid +
            '</PMID><Article>' + title_xml + abstract +
            '</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>').encode()
        client, _, _ = make_client([search_json([], 0)] if is_empty else [search_json([pmid], 1), xml])
        no_runtime = Mock(side_effect=AssertionError('No model permitted'))
        no_runtime.identity = {'engine': 'fictional-only'}
        with patch('medlens.live.sample_memory', side_effect=lambda stage: {'stage': stage}):
            report = run_live_question(question, runs / qid, root / 'cache',
                tokenizer_loader=lambda: (Tokenizer(), INFO), encoder_info=Encoder().info,
                encoder_factory=Encoder, answer_tokenizer_loader=AnswerTokenizer,
                runtime_factory=no_runtime, client=client, limit=100, top_k=10,
                generate=False, answer_style='source-findings')
        if report['status'] != 'complete':
            raise AssertionError(report)
        no_runtime.assert_not_called()
        cases.append({'query_id': qid, 'question': question, 'relevance_intent': 'Fictional source-only test intent.',
            'question_type': 'fictional', 'specialty': 'fictional'})
    plan = {'schema_version': 1, 'kind': 'medlens-human-confirmation-plan',
        'split': 'test', 'purpose': 'prospective independent manual-human confirmation; not model tuning', 'dataset_id': 'human-confirmation-v1',
        'candidate_limit': 100, 'annotation_depth_per_method': 10,
        'pool_methods': ['pubmed-best-match', 'dense'], 'cases': cases}
    path = root / 'plan.json'
    path.write_bytes(canonical(plan) + b'\n')
    return path, runs


def export_value(packet, *, finalized=False, state='pending'):
    package = packet.package
    value = {'schema_version': 1, 'kind': EXPORT_KIND, 'annotation_method': 'manual-human',
        'judgments_origin': 'human', **{k: package[k] for k in
            ('package_id','plan_sha256','items_sha256','source_bindings_sha256','query_bindings_sha256')},
        'finalized': finalized, 'finalized_at': '2000-01-01T00:00:00Z' if finalized else None,
        'judgments': [{**{k: i[k] for k in ('item_id','query_id','pmid','text_version_id','item_sha256')},
            'state': state, 'relevance': None, 'rationale': ''} for i in package['items']],
        'query_judgments': [{**binding, 'state': 'unknown' if state == 'unjudgeable' else 'pending',
            'supporting_pmids': [], 'rationale': ''} for binding in package['query_bindings']]}
    return rehash(value)


def rehash(value):
    value['reference_sha256'] = digest({k: v for k, v in value.items() if k != 'reference_sha256'})
    return value


class ConfirmationAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def packet(self, **kwargs):
        plan, runs = fixture(self.root, **kwargs)
        package = build_packet(plan, runs, self.root / 'packet')
        packet = load_packet(self.root / 'packet')
        self.assertEqual(package, packet.package)
        return plan, runs, packet

    def test_confirmation_plan_is_distinct_from_legacy_development_contract(self):
        plan, runs = fixture(self.root)
        original = strict_json(plan.read_bytes())
        for key, value in [('split','dev'), ('dataset_id','old'), ('purpose','development'),
                           ('candidate_limit', True), ('annotation_depth_per_method', 9)]:
            bad = deepcopy(original)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(BiomedicalAnnotationError):
                parse_plan(canonical(bad))
        bad = deepcopy(original)
        bad['cases'][0]['query_id'] = '../outside'
        with self.assertRaises(BiomedicalAnnotationError): parse_plan(canonical(bad))
        for declared in (1, True):
            bad = deepcopy(original); bad['question_count'] = declared
            with self.subTest(question_count=declared), self.assertRaises(BiomedicalAnnotationError):
                parse_plan(canonical(bad))
        bad = deepcopy(original)
        for row in bad['cases']: row['group_id'] = 'same-group'
        with self.assertRaisesRegex(BiomedicalAnnotationError, 'group'):
            parse_plan(canonical(bad))

    def test_empty_cases_remain_in_inventory_and_every_decision_starts_pending(self):
        _, runs, packet = self.packet(empty=True)
        self.assertEqual(len(packet.package['queries']), 2)
        self.assertEqual(len(packet.package['items']), 1)
        inventory = packet.package['query_inventory']
        self.assertEqual(inventory[1], {'query_id':'fc02','source_state':'empty-pubmed','pool_count':0})
        value, counts = validate_annotations(packet, canonical(export_value(packet)))
        self.assertEqual(counts['queries']['pending'], 2)
        self.assertEqual(counts['articles']['pending'], 1)
        self.assertTrue(all(row['relevance'] is None for row in value['judgments']))
        self.assertTrue(all(row['state'] == 'pending' and not row['supporting_pmids'] for row in value['query_judgments']))
        self.assertFalse(any({'rank','score','methods','relevance','rationale'} & set(i) for i in packet.package['items']))
        shutil.rmtree(runs)
        self.assertEqual(load_packet(packet.directory).package['package_id'], packet.package['package_id'])

    def test_title_only_records_remain_as_ungraded_annotation_cards(self):
        _, _, packet = self.packet(title_only=True)
        self.assertEqual(len(packet.package['items']), 2)
        self.assertTrue(all(not (i['abstract'] or {}).get('sections') for i in packet.package['items']))

    def test_declared_confirmation_groups_remain_bound_in_query_inventory(self):
        plan, runs = fixture(self.root)
        value = strict_json(plan.read_bytes())
        value['cases'][0]['group_id'] = 'fictional-independent-group'
        plan.write_bytes(canonical(value))
        package = build_packet(plan, runs, self.root / 'packet')
        self.assertEqual(package['queries'][0]['group_id'], 'fictional-independent-group')
        self.assertEqual(package['query_inventory'][0]['group_id'], 'fictional-independent-group')
        self.assertEqual(load_packet(self.root / 'packet').package['package_id'], package['package_id'])

    def test_candidate_query_and_dense_input_identity_cannot_be_relabelled(self):
        plan, runs = fixture(self.root)
        original_plan = strict_json(plan.read_bytes())
        path = runs / 'fc01/run.json'
        original_run = strict_json(path.read_bytes())
        for question in ['How does another fictional exposure affect a different fictional outcome?',
                         'What does fictional target 1 affect fictional outcome?']:
            value, raw = deepcopy(original_plan), deepcopy(original_run)
            value['cases'][0]['question'] = question
            raw['question'] = question
            if question.startswith('What'):
                raw['live']['query_plan'] = query_planner.plan_query(question)
                self.assertEqual(raw['live']['query_plan']['submitted_query'], original_run['live']['query_plan']['submitted_query'])
            else:
                raw['live']['query_plan']['question'] = question
            plan.write_bytes(canonical(value)); path.write_bytes(canonical(raw))
            with self.subTest(question=question), self.assertRaises(BiomedicalAnnotationError):
                build_packet(plan, runs, self.root / 'bad-packet')
            self.assertFalse((self.root / 'bad-packet').exists())

    def test_frozen_query_map_is_verified_without_executing_a_future_planner(self):
        _, _, packet = self.packet(empty=True)
        expected = packet.package['expected_query_plans']
        self.assertEqual(packet.manifest['identity']['expected_query_plans_sha256'], digest(expected))
        self.assertEqual((packet.directory / 'implementation/pubmed_query.py').read_bytes(), Path(query_planner.__file__).read_bytes())
        with patch.object(query_planner, 'plan_query', side_effect=AssertionError('Future planner execution forbidden')):
            self.assertEqual(load_packet(packet.directory).package['package_id'], packet.package['package_id'])

    def test_empty_case_request_identity_cannot_be_replaced(self):
        plan, runs = fixture(self.root, empty=True)
        path = runs / 'fc02/run.json'
        run = strict_json(path.read_bytes()); run['request_id'] = '0' * 64
        path.write_bytes(canonical(run))
        with self.assertRaisesRegex(BiomedicalAnnotationError, 'request identity'):
            build_packet(plan, runs, self.root / 'bad-packet')

    def test_records_without_searchable_text_keep_pubmed_pool_without_dense_hits(self):
        _, _, packet = self.packet(missing_text=True)
        self.assertEqual(len(packet.package['items']), 2)
        self.assertTrue(all(i['source_state'] == 'no-searchable-fragments' for i in packet.package['query_inventory']))

    def test_yes_requires_unique_own_source_refs_for_finalization_and_never_infers_labels(self):
        _, _, packet = self.packet()
        value = export_value(packet, state='unjudgeable')
        first = value['query_judgments'][0]
        first['state'] = 'yes'
        _, counts = validate_annotations(packet, canonical(rehash(value)))
        self.assertEqual(counts['queries']['incomplete_support'], 1)
        value.update(finalized=True, finalized_at='2000-01-01T00:00:00Z')
        with self.assertRaisesRegex(BiomedicalAnnotationError, 'yes without'):
            validate_annotations(packet, canonical(rehash(value)))
        own = [i['pmid'] for i in packet.package['items'] if i['query_id'] == first['query_id']][0]
        other = [i['pmid'] for i in packet.package['items'] if i['query_id'] != first['query_id']][0]
        for refs in ([other], [own,own], [True]):
            first['supporting_pmids'] = refs
            with self.subTest(refs=refs), self.assertRaises(BiomedicalAnnotationError):
                validate_annotations(packet, canonical(rehash(value)))
        first['supporting_pmids'] = [own]
        result, counts = validate_annotations(packet, canonical(rehash(value)), require_finalized=True)
        self.assertEqual(result['judgments'][0]['state'], 'unjudgeable')
        self.assertEqual(counts['queries']['incomplete_support'], 0)
        first['state'] = 'no'
        with self.assertRaisesRegex(BiomedicalAnnotationError, 'Only an explicit yes'):
            validate_annotations(packet, canonical(rehash(value)))

    def test_pending_unknown_and_empty_query_finalization_are_distinct(self):
        _, _, packet = self.packet(empty=True)
        pending = export_value(packet, finalized=True)
        with self.assertRaisesRegex(BiomedicalAnnotationError, 'pending decisions'):
            validate_annotations(packet, canonical(pending))
        unknown = export_value(packet, state='unjudgeable', finalized=True)
        _, coverage = validate_annotations(packet, canonical(unknown), require_finalized=True)
        self.assertEqual(coverage['queries']['unknown'], 2)
        self.assertEqual(coverage['articles']['unjudgeable'], 1)
        unknown['query_judgments'][1]['state'] = 'no'
        self.assertTrue(validate_annotations(packet, canonical(rehash(unknown)), require_finalized=True)[0]['finalized'])

    def test_finalization_requires_a_real_iso_calendar_date(self):
        _, _, packet = self.packet()
        value = export_value(packet, state='unjudgeable', finalized=True)
        for stamp in ['2000-02-30T00:00:00Z', '1900-02-29T00:00:00Z',
                      '2000-01-01T24:00:00Z', '2000-01-01T00:00:00+24:00', '0000-01-01T00:00:00Z']:
            value['finalized_at'] = stamp
            with self.subTest(stamp=stamp), self.assertRaises(BiomedicalAnnotationError):
                validate_annotations(packet, canonical(rehash(value)))
        for stamp in ['2000-02-29T00:00:00Z', '2000-02-29T00:00:00.123456+03:00']:
            value['finalized_at'] = stamp
            self.assertTrue(validate_annotations(packet, canonical(rehash(value)), require_finalized=True)[0]['finalized'])

    def test_foreign_origins_bindings_duplicate_rows_and_nonnumeric_grades_fail_closed(self):
        _, _, packet = self.packet()
        good = export_value(packet)
        changes = [lambda v:v.update(kind='medlens-biomedical-relevance-annotations'),
            lambda v:v.update(judgments_origin='model'), lambda v:v.update(annotation_method='assistant'),
            lambda v:v['query_judgments'].pop(),
            lambda v:v['query_judgments'].__setitem__(1, deepcopy(v['query_judgments'][0])),
            lambda v:v['query_judgments'][0].update(source_pool_sha256='changed'),
            lambda v:v['judgments'][0].update(text_version_id='changed'),
            lambda v:v['judgments'][0].update(state='rated', relevance=True),
            lambda v:v['judgments'][0].update(state='rated', relevance=1.0)]
        for change in changes:
            bad = deepcopy(good); change(bad); rehash(bad)
            with self.subTest(change=change), self.assertRaises(BiomedicalAnnotationError):
                validate_annotations(packet, canonical(bad))
        for body in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(BiomedicalAnnotationError): strict_json(body)

    def test_source_markup_is_not_executable_and_frozen_files_are_bound(self):
        _, _, packet = self.packet(hostile=True)
        html = (packet.directory / 'index.html').read_text()
        self.assertNotIn('<img src=x', html)
        self.assertIn('\\u003c/script\\u003e', html)
        self.assertNotIn('innerHTML', html)
        self.assertIn('out.textContent=text', html)
        self.assertIn('query_judgments', html)
        for name in ['index.html','sources/fc01/snapshot/fetch-0001.xml','implementation/confirmation_annotation.py']:
            path = packet.directory / name
            before = path.read_bytes(); path.write_bytes(before+b' ')
            with self.subTest(name=name), self.assertRaisesRegex(BiomedicalAnnotationError, 'checksum'):
                load_packet(packet.directory)
            path.write_bytes(before)

    def test_stable_packet_identity_no_overwrite_and_byte_exact_human_freeze(self):
        plan, runs, packet = self.packet()
        other = build_packet(plan, runs, self.root / 'other')
        self.assertEqual(other['package_id'], packet.package['package_id'])
        with self.assertRaises(BiomedicalAnnotationError): build_packet(plan, runs, packet.directory)
        value = export_value(packet, state='unjudgeable', finalized=True)
        value['query_judgments'][0]['rationale'] = '  FICTIONAL HUMAN NOTE\nkeep whitespace  '
        rehash(value)
        body = ('\n '+json.dumps(value, ensure_ascii=False, indent=7)+'\n\n').encode()
        original = self.root / 'human.json'; original.write_bytes(body)
        checked = validate_export(packet, original, require_finalized=True)
        self.assertFalse(checked['labels_or_rationales_returned'])
        self.assertNotIn('FICTIONAL HUMAN NOTE', json.dumps(checked))
        self.assertEqual(set(checked['coverage']['queries']), {'total','completed','pending'})
        self.assertEqual(set(checked['coverage']['articles']), {'total','completed','pending'})
        receipt = freeze_export(packet, original, self.root / 'reference')
        self.assertEqual((self.root / 'reference/original-human-export.json').read_bytes(), body)
        self.assertEqual(original.read_bytes(), body)
        self.assertEqual((self.root / 'reference/original-human-export.json').stat().st_mode & 0o777, 0o400)
        self.assertFalse(receipt['quality_scores_computed'])
        self.assertFalse(receipt['human_labels_rewritten'])
        self.assertEqual(set(receipt['coverage']['queries']), {'total','completed','pending'})
        self.assertEqual(set(receipt['coverage']['articles']), {'total','completed','pending'})
        with self.assertRaises(BiomedicalAnnotationError): freeze_export(packet, original, self.root / 'reference')

    def test_browser_logic_blank_export_hash_and_import_validation_match_python(self):
        node = shutil.which('node')
        self.assertIsNotNone(node, 'Install Node.js 22 or newer to run the offline browser logic tests')
        _, _, packet = self.packet()
        template = (packet.directory / 'implementation/confirmation_annotation.html')
        runner = self.root / 'probe.js'
        runner.write_text("""const fs=require('fs'),vm=require('vm'),{webcrypto}=require('crypto');
const pkg=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
let code=fs.readFileSync(process.argv[3],'utf8').split("'use strict';")[1].split('blank();try{')[0];
const ctx={document:{getElementById:()=>({textContent:JSON.stringify(pkg)})},crypto:webcrypto,TextEncoder};vm.createContext(ctx);vm.runInContext(code,ctx);
function mustReject(code,label){let rejected=false;try{vm.runInContext(code,ctx);}catch(_){rejected=true;}if(!rejected)throw Error(label+' accepted');}
async function signed(value){return {...value,reference_sha256:await vm.runInContext('hash('+JSON.stringify(value)+')',ctx)};}
(async()=>{
const value=vm.runInContext('blank();validate(payload(),false);payload()',ctx);
mustReject(`strictParse('{"a":1,"a":2}')`,'duplicate key');
const rated=vm.runInContext('(()=>{const v=payload();v.judgments[0].state="rated";v.judgments[0].relevance=1;return v})()',ctx);
const signedRated=await signed(rated);
for(const replacement of ['1.0','1e0','1E+0']){
 const raw=JSON.stringify(signedRated).replace('"relevance":1','"relevance":'+replacement);
 mustReject('validate(strictParse('+JSON.stringify(raw)+'),true)','noninteger lexical relevance '+replacement);
}
const rawSchema=JSON.stringify(await signed(value)).replace('"schema_version":1','"schema_version":1e0');
mustReject('validate(strictParse('+JSON.stringify(rawSchema)+'),true)','exponent schema version');
mustReject('(()=>{const v=payload(true,"2000-01-01T00:00:00Z");validate(v,false)})()','pending finalization');
vm.runInContext('for(const row of labels.values()){row.state="unjudgeable";row.relevance=null;}for(const row of queries.values())row.state="unknown"',ctx);
for(const stamp of ['2000-02-30T00:00:00Z','1900-02-29T00:00:00Z','2000-01-01T24:00:00Z','2000-01-01T00:00:00+24:00','0000-01-01T00:00:00Z']){
 const complete=vm.runInContext('payload(true,'+JSON.stringify(stamp)+')',ctx);
 mustReject('validate(strictParse('+JSON.stringify(JSON.stringify(await signed(complete)))+'),true)','invalid ISO calendar date '+stamp);
}
for(const stamp of ['2000-02-29T00:00:00Z','2000-02-29T00:00:00.123456+03:00'])vm.runInContext('validate(payload(true,'+JSON.stringify(stamp)+'),false)',ctx);
vm.runInContext('queries.get(pkg.query_bindings[0].query_id).state="yes";validate(payload(),false)',ctx);
mustReject('(()=>{const v=payload(true,"2000-01-01T00:00:00Z");validate(v,false)})()','yes without source support');
vm.runInContext('blank()',ctx);const hash=await vm.runInContext('hash(payload())',ctx);
console.log(JSON.stringify({value,hash,strict_integer_regressions:4,invalid_calendar_regressions:5}));
})().catch(error=>{console.error(error);process.exitCode=1;});
""")
        run = subprocess.run([node, str(runner), str(packet.directory / 'package.json'), str(template)], capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stderr)
        output = json.loads(run.stdout)
        self.assertEqual(output['hash'], digest(output['value']))
        output['value']['reference_sha256'] = output['hash']
        validated, coverage = validate_annotations(packet, canonical(output['value']))
        self.assertEqual(coverage['queries']['pending'], 2)


if __name__ == '__main__':
    unittest.main()
