"""Fictional-only prospective harness checks; no model, network or held-out reads."""

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import validate_confirmation as tool


def source(qid='fictional', pmid='999', text='Fictional evidence.', version='fictional-v1'):
    return {'query_id': qid, 'question': 'What does the fictional experiment show?',
        'item_id': qid + '-' + pmid + '-' + version, 'pmid': pmid, 'title': 'Fictional article',
        'raw_title': 'Fictional article', 'abstract': {'sections': [{'text': text}]},
        'abstract_status': 'available', 'other_abstracts': [], 'text_version_id': version,
        'item_sha256': tool.legacy.digest([qid, pmid, text, version])}


def prep(items):
    queries = [{'query_id': qid, 'question': next(i['question'] for i in items if i['query_id'] == qid)}
        for qid in sorted({i['query_id'] for i in items})]
    value = {'judgments_origin': 'ai', 'human_annotation': False,
        'status': 'complete-ai-assessment', 'queries': queries,
        'article_judgments': [{**{k: i[k] for k in ('item_id', 'query_id', 'pmid', 'text_version_id', 'item_sha256')},
            'state': 'rated', 'relevance': 2, 'rationale': 'Fictional fixture only.'} for i in items],
        'query_judgments': [{'query_id': q['query_id'], 'state': 'yes',
            'supporting_item_ids': [i['item_id'] for i in items if i['query_id'] == q['query_id']]} for q in queries]}
    return queries, value


def binding(items, qid='fictional', state='available'):
    return {'query_id': qid, 'question_sha256': tool.legacy.digest(items[0]['question']),
        'source_keys': sorted(tool.source_key(i) for i in items), 'pool_state': state}


def quality_rows(count=20):
    return [{'query_id': 'fictional-' + str(n), 'useful_selection': 'yes' if n < count // 2 else 'no',
        'answerable': 'yes', 'unnecessary_abstention': 'no', 'semantic_abstention': False,
        'critical_scope_mismatch': 'no'} for n in range(count)]


class ProspectiveConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exact_question_text_and_version_required_but_timestamp_ignored(self):
        old = source()
        queries, labels = prep([old])
        unchanged = {**old, 'retrieved_at': 'different timestamp', 'item_sha256': 'new-provenance-only'}
        value = tool.reconcile_values([old], queries, labels, [unchanged], [binding([unchanged])])
        self.assertEqual(len(value['mapped']), 1)
        self.assertEqual(value['delta_items'], [])
        self.assertFalse(value['queries'][0]['pool_changed'])
        for changed in (source(text='Changed fictional text.'), source(version='fictional-v2')):
            result = tool.reconcile_values([old], queries, labels, [changed], [binding([changed])])
            self.assertEqual(result['mapped'], [])
            self.assertEqual(result['delta_items'][0]['state'], 'pending')
            self.assertIsNone(result['delta_items'][0]['relevance'])
            self.assertEqual(result['queries'][0]['state'], 'pending')
        changed = {**unchanged, 'question': 'A different fictional question?'}
        with self.assertRaisesRegex(ValueError, 'question identity'):
            tool.reconcile_values([old], queries, labels, [changed], [binding([changed])])

    def test_membership_change_reconfirms_answerability_and_preserves_ai_origin(self):
        old, added = source(), source(pmid='998')
        queries, labels = prep([old])
        result = tool.reconcile_values([old], queries, labels, [old, added], [binding([old, added])])
        self.assertTrue(result['queries'][0]['pool_changed'])
        self.assertEqual(result['queries'][0]['supporting_pmids'], [])
        self.assertEqual(result['mapped'][0]['judgments_origin'], 'ai')
        self.assertIsNone(result['delta_items'][0]['judgments_origin'])
        self.assertFalse(result['source_labels_establish_output_usefulness'])
        human = {**labels, 'judgments_origin': 'human', 'human_annotation': True}
        with self.assertRaisesRegex(ValueError, 'origin'):
            tool.reconcile_values([old], queries, human, [old], [binding([old])])

    def test_source_position_and_copyright_do_not_change_exact_text_identity(self):
        old = source()
        old['abstract'] = {'attributes': {}, 'copyright': 'Old publisher metadata',
            'source_path': './PubmedArticle[1]/Abstract', 'sections': [
                {'index': 0, 'label': 'RESULTS', 'nlm_category': 'RESULTS',
                 'text': 'Fictional finding.', 'raw_text': 'Fictional finding.',
                 'source_path': './PubmedArticle[1]/Abstract/AbstractText[1]'}]}
        old['other_abstracts'] = [{'attributes': {'Language': 'fre', 'Type': 'Publisher'},
            'copyright': 'Old translated copyright', 'source_path': './PubmedArticle[1]/OtherAbstract[1]',
            'sections': [{'index': 0, 'label': 'CONCLUSION', 'nlm_category': 'CONCLUSIONS',
                'text': 'Texte fictif.', 'raw_text': 'Texte fictif.',
                'source_path': './PubmedArticle[1]/OtherAbstract[1]/AbstractText[1]'}]}]
        moved = deepcopy(old)
        for abstract in [moved['abstract'], *moved['other_abstracts']]:
            abstract['source_path'] = abstract['source_path'].replace('[1]', '[7]', 1)
            abstract['copyright'] = 'New copyright metadata'
            for section in abstract['sections']:
                section['source_path'] = section['source_path'].replace('[1]', '[7]', 1)
        moved.update(retrieved_at='New acquisition', item_sha256='New provenance hash')
        queries, labels = prep([old])
        result = tool.reconcile_values([old], queries, labels, [moved], [binding([moved])])
        self.assertEqual(len(result['mapped']), 1)
        self.assertFalse(result['queries'][0]['pool_changed'])
        self.assertEqual(result['delta_items'], [])
        changes = [
            lambda i: i['abstract']['sections'][0].update(text='Changed finding.'),
            lambda i: i['abstract']['sections'][0].update(raw_text='Different raw spacing. '),
            lambda i: i['abstract']['sections'][0].update(label='METHODS'),
            lambda i: i['abstract']['sections'][0].update(nlm_category='METHODS'),
            lambda i: i['other_abstracts'][0]['sections'][0].update(text='Autre texte.'),
            lambda i: i['other_abstracts'][0]['attributes'].update(Language='eng'),
            lambda i: i['other_abstracts'][0]['attributes'].update(Type='Other'),
            lambda i: i.update(text_version_id='fictional-new-version'),
            lambda i: i.update(raw_title='Changed raw title'),
        ]
        for change in changes:
            changed = deepcopy(moved)
            change(changed)
            with self.subTest(change=change):
                result = tool.reconcile_values([old], queries, labels, [changed], [binding([changed])])
                self.assertEqual(result['mapped'], [])
                self.assertEqual(result['delta_items'][0]['state'], 'pending')
                self.assertTrue(result['queries'][0]['pool_changed'])

    def test_invalid_partial_run_preserves_original_bytes_and_supervisor_cleanup(self):
        directory = self.root / 'job'
        (directory / 'run').mkdir(parents=True)
        raw = b'{"status":"running","unfinished":'
        (directory / 'run/run.json').write_bytes(raw)
        supervisor = {'status': 'failed', 'answer': {'status': 'not_completed'},
            'native_worker_cleanup': {'confirmed': True, 'worker_reaped': True, 'group_absent': True}}
        tool.legacy.save_new(directory / 'worker-outcome.json', supervisor)
        run, outcome, hashes, errors = tool.read_run_artifacts(directory / 'job.json')
        self.assertIsNone(run)
        self.assertEqual(outcome, supervisor)
        self.assertEqual(errors, {'run': 'JSONDecodeError'})
        self.assertEqual(hashes['run'], tool.legacy.file_sha(directory / 'run/run.json'))
        self.assertTrue(tool.cleanup_confirmed(outcome, {'observed_native_workers_all_reaped': True}))
        self.assertEqual((directory / 'run/run.json').read_bytes(), raw)

    def test_failed_source_pools_are_unavailable_and_integrity_failures_stay_distinct(self):
        output = self.root / 'live'
        output.mkdir()
        cases, hashes = [], {}
        variants = [('timeout-malformed', 'failed', 'passed', b'{partial'),
            ('timeout-before-ranking', 'failed', 'passed', b'{"live":{"fresh_search_complete":true}}'),
            ('failed-source-audit', 'failed', 'failed', b'{}'),
            ('altered-partial', 'failed', 'passed', b'{}'),
            ('successful-incomplete-pool', 'succeeded', 'passed', b'{"live":{"fresh_search_complete":true}}')]
        for qid, status, audit, raw in variants:
            cases.append({'query_id': qid, 'question': 'Fictional question only.'})
            directory = output / 'cases' / qid / 'jobs' / 'fixture'
            (directory / 'run').mkdir(parents=True)
            tool.legacy.save_new(directory / 'job.json', {'status': status})
            (directory / 'run/run.json').write_bytes(raw)
            record = {'job_path': 'jobs/fixture/job.json',
                'job_sha256': tool.legacy.file_sha(directory / 'job.json'),
                'run_sha256': tool.legacy.file_sha(directory / 'run/run.json'),
                'job_status': status, 'operational_pass': status == 'succeeded',
                'source_audit': {'status': audit, 'error_type': 'FictionalSourceMismatch'}}
            if qid == 'altered-partial':
                (directory / 'run/run.json').write_bytes(b'{"changed":true}')
            path = output / ('case-' + qid + '.json')
            tool.legacy.save_new(path, record)
            hashes[qid] = tool.legacy.file_sha(path)
        tool.legacy.save_new(output / 'live-summary.json', {'records': hashes})
        forbidden = unittest.mock.Mock(side_effect=AssertionError('Incomplete source pool must not be loaded or ranked'))
        modules = {'medlens.evaluation.biomedical_annotation': SimpleNamespace(_pool=forbidden),
            'medlens.ingestion.snapshots': SimpleNamespace(load_snapshot=forbidden),
            'medlens.parsing.dataset': SimpleNamespace(load_dataset=forbidden),
            'medlens.parsing.pubmed': SimpleNamespace(parse_pubmed_xml=forbidden)}
        with patch.dict(sys.modules, modules), patch.object(tool, 'parse_plan', return_value=({}, cases)):
            items, bindings = tool.actual_sources(self.root, output)
        self.assertEqual(items, [])
        self.assertEqual([b['pool_state'] for b in bindings],
            ['unavailable', 'unavailable', 'integrity_failed', 'integrity_failed', 'integrity_failed'])
        self.assertTrue(all(b['pool_error'] for b in bindings))
        forbidden.assert_not_called()

    def finalize_fixture(self, name):
        output, reconciliation = self.root / name, self.root / (name + '-sources')
        output.mkdir()
        reconciliation.mkdir()
        items, bindings, review_queries, records = [], [], [], {}
        for n in range(20):
            qid = 'fictional-' + str(n)
            item = source(qid=qid, pmid=str(9000+n))
            items.append(item)
            pool = {**binding([item], qid=qid), 'pool_changed': False, 'state': 'yes'}
            bindings.append(pool)
            claim = {'text': 'Fictional finding only.', 'source_article_id': item['pmid'], 'citations': []}
            job_directory = output / 'cases' / qid / 'jobs' / 'fictional-job'
            job_directory.mkdir(parents=True)
            job = {'result': {'answer': {'status': 'partial', 'claims': [claim]}}}
            tool.legacy.save_new(job_directory / 'job.json', job)
            record = {'job_path': 'jobs/fictional-job/job.json',
                'job_sha256': tool.legacy.file_sha(job_directory / 'job.json'),
                'operational_pass': True, 'source_audit': {'status': 'passed'},
                'native_cleanup_confirmed': True, 'wall_pass': True, 'full_wall_seconds': 1.0}
            record_path = output / ('case-' + qid + '.json')
            tool.legacy.save_new(record_path, record)
            records[qid] = tool.legacy.file_sha(record_path)
            review_queries.append({'query_id': qid, 'question_sha256': pool['question_sha256'],
                'case_record_sha256': records[qid], 'source_pool_sha256': tool.legacy.digest(pool['source_keys']),
                'useful_selection': 'yes' if n < 10 else 'no', 'critical_scope_mismatch': 'no',
                'answerable': 'yes', 'unnecessary_abstention': 'no', 'cards': [
                    {'claim_sha256': tool.card_binding(claim), **{field: 'yes' for field in tool.CARD_FACETS}}]})
        tool.legacy.save_new(output / 'live-summary.json', {'records': records, 'operational_pass': True})
        rec = {'manifest_id': 'fictional-freeze', 'live_summary_sha256': tool.legacy.file_sha(output / 'live-summary.json'),
            'actual_items': items, 'queries': bindings}
        rec['reconciliation_id'] = tool.legacy.digest(rec)
        tool.legacy.save_new(reconciliation / 'reconciliation.json', rec)
        reference = {'reconciliation_id': rec['reconciliation_id'], 'queries': bindings}
        reference['source_reference_id'] = tool.legacy.digest(reference)
        tool.legacy.save_new(reconciliation / 'source-reference.json', reference)
        review = {'kind': 'fixed20-assistant-actual-output-review', 'judgments_origin': 'ai',
            'human_authored': False, 'finalized': True, 'finalized_at': '2000-01-01T00:00:00Z',
            'independent_human_validation': False, 'clinical_validation': False,
            'manifest_id': rec['manifest_id'], 'reconciliation_id': rec['reconciliation_id'],
            'source_reference_id': reference['source_reference_id'], 'queries': review_queries}
        return output, reconciliation, review, items, bindings

    def finalize_fictional(self, fixture):
        output, reconciliation, review, items, bindings = fixture
        review_path = output / 'review.json'
        tool.legacy.save_new(review_path, review)
        with patch.object(tool, 'verify_manifest'), patch.object(tool, 'actual_sources', return_value=(items, bindings)):
            return tool.finalize(self.root, output, reconciliation, review_path)

    def test_finalize_rejects_contradictory_core_scope_even_for_nonuseful_case(self):
        for facet in tool.CORE_FACETS:
            for decision in ('no', 'unknown'):
                with self.subTest(facet=facet, decision=decision):
                    fixture = self.finalize_fixture(facet + '-' + decision)
                    fixture[2]['queries'][-1]['cards'][0][facet] = decision
                    # Ten other useful queries must not override this logical
                    # contradiction on the displayed nonuseful twentieth card.
                    with self.assertRaisesRegex(ValueError, 'core scope'):
                        self.finalize_fictional(fixture)

    def test_known_core_mismatch_fails_gate_after_consistent_review(self):
        fixture = self.finalize_fixture('known-core-mismatch')
        query = fixture[2]['queries'][-1]
        query['cards'][0]['population_system'] = 'no'
        query['critical_scope_mismatch'] = 'yes'
        report = self.finalize_fictional(fixture)
        self.assertEqual(report['gates']['useful'], 'pass')
        self.assertEqual(report['gates']['critical_scope'], 'fail')
        self.assertEqual(report['overall_ai_gate'], 'fail')

    def test_unknown_core_scope_keeps_confirmation_not_established(self):
        fixture = self.finalize_fixture('unknown-core-scope')
        query = fixture[2]['queries'][-1]
        query['cards'][0]['outcome_target'] = 'unknown'
        query['critical_scope_mismatch'] = 'unknown'
        report = self.finalize_fictional(fixture)
        self.assertEqual(report['gates']['useful'], 'pass')
        self.assertEqual(report['gates']['critical_scope'], 'not_established')
        self.assertEqual(report['overall_ai_gate'], 'not_established')

    def test_support_and_qualifier_failure_or_unknown_cannot_pass_release(self):
        for facet in tool.SUPPORT_FACETS:
            for decision, expected in (('no', 'fail'), ('unknown', 'not_established')):
                with self.subTest(facet=facet, decision=decision):
                    fixture = self.finalize_fixture(facet + '-' + decision)
                    fixture[2]['queries'][-1]['cards'][0][facet] = decision
                    report = self.finalize_fictional(fixture)
                    self.assertEqual(report['gates']['useful'], 'pass')
                    self.assertEqual(report['gates']['displayed_card_support'], expected)
                    self.assertEqual(report['overall_ai_gate'], expected)

    def test_malformed_or_partial_source_labels_fail_closed(self):
        old = source()
        queries, labels = prep([old])
        for alter in (lambda v: v['article_judgments'][0].update(relevance=True),
                      lambda v: v['article_judgments'][0].update(item_sha256='different'),
                      lambda v: v.update(article_judgments=[]),
                      lambda v: v['query_judgments'][0].update(state='pending')):
            changed = deepcopy(labels)
            alter(changed)
            with self.assertRaises(ValueError):
                tool.reconcile_values([old], queries, changed, [old], [binding([old])])

    def test_half_useful_threshold_fixed_twenty_and_unknown_bounds(self):
        rows = quality_rows()
        score = tool.score_rows(rows)
        self.assertEqual(score['useful']['denominator'], 20)
        self.assertEqual(score['useful']['rate'], .5)
        self.assertEqual(score['useful']['gate'], 'pass')
        rows[0]['useful_selection'] = 'unknown'
        score = tool.score_rows(rows)
        self.assertEqual(score['useful']['bounds'], [.45, .5])
        self.assertEqual(score['useful']['gate'], 'not_established')
        rows[0]['useful_selection'] = 'no'
        self.assertEqual(tool.score_rows(rows)['useful']['gate'], 'fail')
        with self.assertRaisesRegex(ValueError, 'denominator'):
            tool.score_rows(rows[:-1])

    def test_critical_mismatch_is_a_separate_gate(self):
        rows = quality_rows()
        rows[-1]['critical_scope_mismatch'] = 'yes'
        score = tool.score_rows(rows)
        self.assertEqual(score['useful']['gate'], 'pass')
        self.assertEqual(score['critical_scope']['gate'], 'fail')
        rows[-1]['critical_scope_mismatch'] = 'unknown'
        self.assertEqual(tool.score_rows(rows)['critical_scope']['gate'], 'not_established')

    def test_runtime_closed_alone_never_establishes_native_cleanup(self):
        run = {'runtime_load_count': 1, 'runtime_closed': True}
        observations = {'observed_native_workers_all_reaped': True}
        self.assertFalse(tool.cleanup_confirmed(run, observations))
        run['native_worker_cleanup'] = {'confirmed': True, 'worker_reaped': True, 'group_absent': True}
        self.assertTrue(tool.cleanup_confirmed(run, observations))
        observations['observed_native_workers_all_reaped'] = False
        self.assertFalse(tool.cleanup_confirmed(run, observations))

    def test_freeze_never_parses_plan_or_any_reference_values(self):
        output = self.root / 'candidate'
        selector = {'SELECTOR_SYSTEM': 'Fictional prompt.'}
        def constants(path, names):
            return {'PROFILES': {'4b': {'fictional': True}}} if 'PROFILES' in names else dict(selector)
        with patch.object(tool.legacy, 'file_sha', return_value=tool.PLAN_SHA256), \
                patch.object(tool.legacy, 'constants', side_effect=constants), \
                patch.object(tool.legacy, 'asset_identity', return_value={}), \
                patch.object(tool.legacy, 'packages', return_value={}), \
                patch.object(tool.legacy, 'worker_environment', return_value={}), \
                patch.object(tool, 'source_hashes', return_value={}), \
                patch.object(tool, 'application_policy', return_value={}), \
                patch.object(tool.subprocess, 'check_output', return_value='fictional-head'), \
                patch.object(tool, 'load', side_effect=AssertionError('No JSON/reference parsing permitted')), \
                patch.object(tool, 'parse_plan', side_effect=AssertionError('No plan parsing permitted')):
            result = tool.freeze(self.root, output)
        self.assertFalse(result['semantic_cases_parsed'])
        self.assertTrue((output / 'manifest.json').is_file())

    def test_false_abstention_denominator_and_unknown_answerability_bounds(self):
        rows = quality_rows()
        for n, row in enumerate(rows):
            row['answerable'] = 'yes' if n < 4 else 'no'
            row['semantic_abstention'] = n < 2
            row['unnecessary_abstention'] = 'yes' if n < 2 else 'no'
        score = tool.score_rows(rows)['false_abstention']
        self.assertEqual(score['rate'], .5)
        self.assertEqual(score['answerable_yes'], 4)
        self.assertEqual(score['gate'], 'pass')
        rows[4].update(answerable='unknown', semantic_abstention=True, unnecessary_abstention='unknown')
        score = tool.score_rows(rows)['false_abstention']
        self.assertEqual(score['bounds'], [.5, .6])
        self.assertEqual(score['gate'], 'not_established')

    def test_zero_answerability_never_automatically_passes_and_errors_not_abstention(self):
        rows = quality_rows()
        for row in rows:
            row.update(answerable='no', unnecessary_abstention='no', semantic_abstention=False)
        score = tool.score_rows(rows)['false_abstention']
        self.assertIsNone(score['rate'])
        self.assertEqual(score['gate'], 'not_established')
        rows[0].update(answerable='yes', unnecessary_abstention='yes')
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            tool.score_rows(rows)

    def test_durable_one_shot_marker_precedes_parse_failure_and_refuses_restart(self):
        output = self.root / 'confirmation'
        output.mkdir()
        with patch.object(tool, 'verify_manifest', return_value={'manifest_id': 'fictional-freeze'}), \
                patch.object(tool, 'parse_plan', side_effect=ValueError('fictional parse failure')) as parse:
            with self.assertRaisesRegex(ValueError, 'parse failure'):
                tool.execute(self.root, output)
            self.assertEqual(parse.call_count, 1)
            self.assertTrue((output / 'execution-started.json').is_file())
            self.assertFalse(tool.load(output / 'live-summary.json')['complete'])
            with self.assertRaises(FileExistsError):
                tool.execute(self.root, output)
            self.assertEqual(parse.call_count, 1)

    def test_every_case_order_once_and_hash_checks_even_when_request_fails(self):
        output = self.root / 'confirmation'
        output.mkdir()
        cases = [{'query_id': 'fictional-' + str(n), 'question': 'Fictional question ' + str(n)} for n in range(20)]
        def execute(root, directory, case):
            return {'query_id': case['query_id'], 'native_cleanup_confirmed': True,
                'operational_pass': case['query_id'] != 'fictional-3', 'job_status': 'failed',
                'full_wall_seconds': 1., 'wall_pass': True}
        with patch.object(tool, 'verify_manifest', return_value={'manifest_id': 'fictional-freeze'}) as verify, \
                patch.object(tool, 'parse_plan', return_value=({}, cases)), \
                patch.object(tool, 'execute_cli', side_effect=execute) as runner:
            summary = tool.execute(self.root, output)
        self.assertEqual([call.args[2]['query_id'] for call in runner.call_args_list], [c['query_id'] for c in cases])
        self.assertEqual(verify.call_count, 41)
        self.assertEqual(summary['recorded_count'], 20)
        self.assertTrue(summary['complete'])
        self.assertFalse(summary['operational_pass'])

    def test_output_review_cannot_open_before_separate_source_reference_finalized(self):
        reconciliation = self.root / 'reconciliation'
        reconciliation.mkdir()
        tool.legacy.save_new(reconciliation / 'reconciliation.json', {})
        with self.assertRaises((FileNotFoundError, ValueError)):
            tool.review_template(self.root, reconciliation, self.root / 'review.json')

    def test_source_delta_freeze_keeps_new_decisions_explicit(self):
        directory = self.root / 'reconciliation'
        directory.mkdir()
        item = {**source(), 'source_key': tool.source_key(source()), 'state': 'pending',
            'relevance': None, 'rationale': '', 'judgments_origin': None}
        query = {**binding([item]), 'pool_changed': True, 'state': 'pending',
            'supporting_pmids': [], 'rationale': '', 'judgments_origin': None}
        rec = {'delta_items': [item],
            'queries': [query], 'actual_items': [item], 'mapped': []}
        rec['reconciliation_id'] = tool.legacy.digest(rec)
        tool.legacy.save_new(directory / 'reconciliation.json', rec)
        review = {'kind': 'assistant-source-delta-review', 'reconciliation_id': rec['reconciliation_id'],
            'judgments_origin': 'ai', 'human_authored': False, 'finalized': True,
            'items': [item], 'queries': [query]}
        path = self.root / 'delta.json'
        tool.legacy.save_new(path, review)
        with self.assertRaisesRegex(ValueError, 'pending'):
            tool.finalize_sources(directory, path)
        review['items'][0].update(state='rated', relevance=2, judgments_origin='ai')
        review['queries'][0].update(state='yes', supporting_pmids=['999'], judgments_origin='ai')
        path.write_bytes(tool.legacy.canonical(review))
        result = tool.finalize_sources(directory, path)
        self.assertEqual(result['status'], 'ai-source-reference-frozen')
        self.assertEqual((directory / 'original-ai-delta-source-review.json').read_bytes(), path.read_bytes())
        self.assertFalse(tool.load(directory / 'source-reference.json')['human_authored'])

    def require_process_inventory(self):
        import psutil
        try:
            psutil.Process().children(recursive=True)
        except (psutil.Error, OSError):
            self.skipTest('Targeted process inventory unavailable in sandbox; run this fixture with process-observation permission')

    def process_fixture(self, *, ignore_term=False):
        """CLI -> separate-session request -> native child; no models/network."""
        self.require_process_inventory()
        request_script = self.root / ('request-' + str(ignore_term) + '.py')
        request_script.write_text(
            'import json,os,pathlib,signal,subprocess,sys,time\n'
            'child=None\n'
            + ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if ignore_term else
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
                "def stop(*args):\n"
                "    child.wait(timeout=1)\n"
                "    raise SystemExit(0)\n"
                "signal.signal(signal.SIGTERM,stop)\n")
            + "pathlib.Path(sys.argv[1]).write_text(json.dumps({'request':os.getpid(),'native':child.pid if child else None,'pgid':os.getpgrp(),'sid':os.getsid(0)}))\n"
              'time.sleep(30)\n')
        cli_script = self.root / ('cli-' + str(ignore_term) + '.py')
        identity_path = self.root / ('identity-' + str(ignore_term) + '.json')
        cli_script.write_text(
            'import signal,subprocess,sys,time\n'
            'child=subprocess.Popen([sys.executable,' + repr(str(request_script)) + ',' + repr(str(identity_path)) +
            '],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n'
            'def stop(*args):\n'
            '    child.wait(timeout=2)\n'
            '    raise SystemExit(0)\n'
            'signal.signal(signal.SIGTERM,stop)\n'
            'time.sleep(30)\n')
        return cli_script, identity_path

    def test_watchdog_cleans_separate_session_and_native_descendant(self):
        script, identity_path = self.process_fixture()
        def launch(command, **options):
            return subprocess.Popen([sys.executable, str(script)], **options)
        policy = {**tool.POLICY, 'outer_watchdog_seconds': .3, 'outer_cleanup_seconds': 2}
        with patch.object(tool, 'POLICY', policy), patch.object(tool, 'application_policy', return_value={}):
            record = tool.execute_cli(self.root, self.root / 'case',
                {'query_id': 'fictional-watchdog', 'question': 'Fictional question only.'}, runner=launch)
        identities = json.loads(identity_path.read_text())
        self.assertNotEqual(identities['pgid'], os.getpgrp())
        self.assertEqual(identities['pgid'], identities['sid'])
        self.assertTrue(record['watchdog_forced_kill'])
        self.assertTrue(record['outer_forced_cleanup'])
        self.assertFalse(record['wall_pass'])
        self.assertFalse(record['operational_pass'])
        cleanup = record['outer_cleanup']
        self.assertTrue(cleanup['confirmed_observed_tree_absent'], cleanup)
        self.assertTrue(cleanup['cli_reaped'])
        self.assertEqual(cleanup['remaining_process_groups'], [])
        self.assertEqual(cleanup['survivors'], [])
        signalled = {row.get('process_group') for row in cleanup['signals']}
        self.assertIn(identities['pgid'], signalled)
        self.assertNotIn(os.getpgrp(), signalled)
        self.assertFalse(cleanup['unobserved_descendant_absence_established'])
        for pid in (identities['request'], identities['native']):
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_uncooperative_owned_session_gets_kill_then_cli_is_reaped(self):
        script, identity_path = self.process_fixture(ignore_term=True)
        process = subprocess.Popen([sys.executable, str(script)], start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tool.OwnedProcessTree(process) as tree:
            try:
                deadline = time.monotonic() + 3
                while not identity_path.is_file() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(identity_path.is_file())
                tree.observe()
                identity = json.loads(identity_path.read_text())
                cleanup = tree.cleanup(seconds=2)
                process.communicate(timeout=1)
                self.assertTrue(cleanup['confirmed_observed_tree_absent'], cleanup)
                self.assertTrue(any(row.get('process_group') == identity['pgid'] and
                    row['signal'] == signal.SIGKILL for row in cleanup['signals']))
                self.assertTrue(cleanup['cli_reaped'])
            finally:
                if process.poll() is None:
                    tree.cleanup(seconds=1)
                    process.communicate(timeout=1)

    def test_stale_birth_identity_and_parent_shell_group_never_authorize_signal(self):
        self.require_process_inventory()
        process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
            start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with tool.OwnedProcessTree(process) as tree:
            try:
                original = tree.members[process.pid]['created_at']
                with tree.lock:
                    tree.members[process.pid]['created_at'] = original + 1
                with patch.object(tool.os, 'killpg') as killpg:
                    tree.signal_owned(signal.SIGTERM)
                    killpg.assert_not_called()
                self.assertEqual(tree._group_members(os.getpgrp()), [])
                self.assertIsNone(process.poll())
            finally:
                with tree.lock:
                    tree.members[process.pid]['created_at'] = original
                cleanup = tree.cleanup(seconds=2)
                process.communicate(timeout=1)
                self.assertTrue(cleanup['confirmed_observed_tree_absent'], cleanup)

    def test_harness_exception_cleans_observed_separate_session_before_unwinding(self):
        script, identity_path = self.process_fixture()
        process = subprocess.Popen([sys.executable, str(script)], start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tree = tool.OwnedProcessTree(process)
        with self.assertRaisesRegex(RuntimeError, 'fictional harness failure'):
            with tree:
                deadline = time.monotonic() + 3
                while not identity_path.is_file() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(identity_path.is_file())
                tree.observe()
                raise RuntimeError('fictional harness failure')
        process.communicate(timeout=1)
        cleanup = tree.result()
        self.assertTrue(cleanup['confirmed_observed_tree_absent'], cleanup)
        self.assertTrue(cleanup['cli_reaped'])


if __name__ == '__main__':
    unittest.main()
