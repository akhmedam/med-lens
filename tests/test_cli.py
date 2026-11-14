"""CLI behavior with real job threads and fake search runners, without ML/network."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from medlens import cli
from medlens.api_service import JobService


QUESTION = 'Does fictional intervention A reduce outcome B?'
OPTIONS = {'limit': 10, 'top_k': 3, 'generate': True,
           'answer_style': 'draft', 'pubmed_query': None}


def complete(request, *, answer_status=None):
    return {'status': 'complete', 'stage': 'complete', 'question': request.question,
            'runtime_load_count': 0, 'answer_style': request.answer_style,
            'answer': {'status': answer_status or ('partial' if request.generate else 'not_requested'),
                       'claims': [], 'limitations': []},
            'retrieval': {'results': [{'pmid': '123', 'title': 'Retrieved study', 'score': .8}]},
            'context': {'sources': []}}


class CLISessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = io.StringIO()
        self.count = 0

    def service(self, runner, cls=JobService):
        self.count += 1
        service = cls(self.root / str(self.count), runner)
        self.addCleanup(service.close)
        return service

    def session(self, service, lines, options=None, **kwargs):
        lines = iter(lines)
        def read(prompt):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError
        return cli.run_session(service, options or OPTIONS, read=read,
                               output=self.output, poll_interval=.001, **kwargs)

    def test_invalid_question_then_valid_search_and_commands(self):
        calls = []
        def runner(request, output, progress):
            calls.append(request)
            return complete(request)
        service = self.service(runner)
        code = self.session(service, ['Does?', '/evidence', QUESTION, '/evidence',
                                      '/help', '/unknown', '/quit'])
        self.assertEqual(code, 0)
        self.assertEqual([r.question for r in calls], [QUESTION])
        text = self.output.getvalue()
        self.assertIn('Некорректный запрос:', text)
        self.assertIn('Сначала выполните запрос.', text)
        self.assertIn(cli.HELP, text)
        self.assertIn('Неизвестная команда.', text)
        self.assertEqual(service.health()['status'], 'closed')
        jobs = list(service.storage_root.glob('*/job.json'))
        self.assertEqual(len(jobs), 1)
        self.assertEqual(json.loads(jobs[0].read_text())['status'], 'succeeded')

    def test_search_only_and_source_findings_flags_reach_runner(self):
        calls = []
        def runner(request, output, progress):
            calls.append(request)
            return complete(request)
        service = self.service(runner)
        options = {**OPTIONS, 'limit': 7, 'top_k': 2, 'generate': False,
                   'answer_style': 'source-findings', 'pubmed_query': 'fictional[Title]'}
        self.assertEqual(self.session(service, [QUESTION, '/quit'], options), 0)
        request, = calls
        self.assertEqual((request.limit, request.top_k, request.generate,
                          request.answer_style, request.pubmed_query),
                         (7, 2, False, 'source-findings', 'fictional[Title]'))
        path, = service.storage_root.glob('*/job.json')
        self.assertEqual(json.loads(path.read_text())['result']['answer']['status'], 'not_requested')

    def test_expected_upstream_failure_recovers_for_next_question(self):
        calls = []
        def runner(request, output, progress):
            calls.append(request)
            report = complete(request)
            if len(calls) == 1:
                report.update(status='failed', stage='pubmed-download',
                    error={'stage': 'pubmed-download', 'message': '/private/secret-diagnostic'})
            return report
        service = self.service(runner)
        self.assertEqual(self.session(service, [QUESTION, QUESTION, '/quit']), 0)
        self.assertEqual(len(calls), 2)
        jobs = [json.loads(path.read_text()) for path in service.storage_root.glob('*/job.json')]
        self.assertEqual(sorted(j['status'] for j in jobs), ['failed', 'succeeded'])
        self.assertNotIn('/private/secret-diagnostic', self.output.getvalue())

    def test_uncertain_cleanup_stops_session_before_another_prompt(self):
        def runner(request, output, progress):
            report = complete(request)
            report.update(runtime_load_count=1, runtime_closed=False)
            return report
        service = self.service(runner)
        prompts = []
        def read(prompt):
            prompts.append(prompt)
            return QUESTION
        self.assertEqual(cli.run_session(service, OPTIONS, read=read, output=self.output,
                                         poll_interval=.001), 1)
        self.assertEqual(len(prompts), 1)
        self.assertIn('Требуется перезапуск.', self.output.getvalue())
        self.assertEqual(service.health()['status'], 'closed')

    def test_complete_checkpoint_does_not_advance_prompt_before_cleanup(self):
        at_cleanup, release, cleaned = Event(), Event(), Event()
        self.addCleanup(release.set)
        def runner(request, output, progress):
            progress('complete', 'complete')
            at_cleanup.set()
            if not release.wait(5):
                raise TimeoutError('test cleanup was not released')
            cleaned.set()
            return complete(request)
        class CheckpointService(JobService):
            def submit(inner, request):
                submitted = super().submit(request)
                self.assertTrue(at_cleanup.wait(2))
                return submitted
        service = self.service(runner, CheckpointService)
        prompts = []
        def read(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return QUESTION
            self.assertTrue(cleaned.is_set(), 'CLI prompted while cleanup still owned the lane')
            return '/quit'
        def tick(_):
            self.assertEqual(len(prompts), 1)
            self.assertEqual(service.health()['status'], 'busy')
            self.assertFalse(cleaned.is_set())
            release.set()
            self.assertTrue(cleaned.wait(2))
            service._future.result(timeout=2)
        with patch.object(cli.time, 'sleep', side_effect=tick):
            code = cli.run_session(service, OPTIONS, read=read, output=self.output)
        self.assertEqual(code, 0)
        self.assertIn(cli.STAGES['complete'], self.output.getvalue())
        self.assertEqual(len(prompts), 2)

    def test_ctrl_c_drains_real_worker_and_protects_shutdown_from_sigint(self):
        running, release, cleaned = Event(), Event(), Event()
        self.addCleanup(release.set)
        def runner(request, output, progress):
            running.set()
            if not release.wait(5):
                raise TimeoutError('test work was not released')
            cleaned.set()
            return complete(request)
        observed = []
        class DrainingService(JobService):
            def close(inner):
                if inner.health()['status'] != 'closed':
                    observed.append(signal.getsignal(signal.SIGINT))
                    release.set()
                    # A real repeated signal must be ignored while closing.
                    signal.raise_signal(signal.SIGINT)
                super().close()
        service = self.service(runner, DrainingService)
        original = signal.getsignal(signal.SIGINT)
        def interrupt(_):
            self.assertTrue(running.wait(2))
            raise KeyboardInterrupt
        with patch.object(cli.time, 'sleep', side_effect=interrupt):
            code = self.session(service, [], question=QUESTION)
        self.assertEqual(code, 130)
        self.assertTrue(cleaned.is_set())
        self.assertEqual(observed, [signal.SIG_IGN])
        self.assertIs(signal.getsignal(signal.SIGINT), original)
        self.assertIn('задание не отменено', self.output.getvalue())
        path, = service.storage_root.glob('*/job.json')
        self.assertEqual(json.loads(path.read_text())['status'], 'succeeded')
        self.assertEqual(service.health()['status'], 'closed')

    def test_idle_ctrl_c_and_eof_close_without_running_search(self):
        for interrupt, expected in [(KeyboardInterrupt, 130), (EOFError, 0)]:
            with self.subTest(interrupt=interrupt):
                calls = []
                service = self.service(lambda *args: calls.append(args))
                def read(prompt):
                    raise interrupt
                self.assertEqual(cli.run_session(service, OPTIONS, read=read, output=self.output), expected)
                self.assertFalse(calls)
                self.assertEqual(service.health()['status'], 'closed')

    def test_one_shot_exit_codes_preserve_answer_and_execution_outcomes(self):
        for status, expected in [('partial', 0), ('insufficient_evidence', 0),
                                 ('not_requested', 0), ('validation_failed', 2)]:
            with self.subTest(status=status):
                service = self.service(lambda request, *args: complete(request, answer_status=status))
                self.assertEqual(self.session(service, [], question=QUESTION), expected)
        for pipeline_status, expected in [('failed', 1), ('interrupted', 130)]:
            with self.subTest(pipeline_status=pipeline_status):
                def runner(request, *args):
                    report = complete(request)
                    report.update(status=pipeline_status, stage='answer')
                    return report
                self.assertEqual(self.session(self.service(runner), [], question=QUESTION), expected)

    def test_export_commands_keep_public_artifacts_and_do_not_overwrite(self):
        def runner(request, output, progress):
            output.mkdir()
            (output / 'report.html').write_text('PRIVATE-DIAGNOSTIC')
            report = complete(request)
            report['retrieval']['results'][0]['title'] = '<script>alert(1)</script>'
            report['error'] = {'message': 'PRIVATE-DIAGNOSTIC'}
            return report
        service = self.service(runner)
        json_path = self.root / 'public report.json'
        html_path = self.root / 'public report.html'
        code = self.session(service, [QUESTION, f'/export "{json_path}"',
                                      f'/export "{html_path}"', '/export', '/export',
                                      '/export missing.txt', '/export "broken', '/quit'])
        self.assertEqual(code, 0)
        exported = json.loads(json_path.read_text())
        self.assertEqual(exported['result']['articles'][0]['title'], '<script>alert(1)</script>')
        self.assertNotIn('PRIVATE-DIAGNOSTIC', json_path.read_text())
        html = html_path.read_text()
        self.assertNotIn('<script>', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('PRIVATE-DIAGNOSTIC', html)
        default, = service.storage_root.glob('*/report-public.html')
        self.assertEqual(default.read_text(), html)
        self.assertIn('Файл уже существует.', self.output.getvalue())
        self.assertIn('Не удалось сохранить отчёт.', self.output.getvalue())
        self.assertIn('Ошибка команды:', self.output.getvalue())
        previous = json_path.read_bytes()
        with self.assertRaises(FileExistsError):
            cli.export_job(exported, json_path)
        self.assertEqual(json_path.read_bytes(), previous)
        link = self.root / 'link.html'
        link.symlink_to(html_path)
        with self.assertRaises(FileExistsError):
            cli.export_job(exported, link)


class CLIStartupTests(unittest.TestCase):
    def test_invalid_flags_and_overlapping_roots_do_not_create_service(self):
        for args in [['--limit', '0'], ['--top-k', '11'], ['--pubmed-query', ''],
                     ['--question', 'Does?'], ['--storage-dir', '/tmp/cache', '--cache', '/tmp/cache']]:
            with self.subTest(args=args), patch.object(cli, 'JobService') as create, \
                 redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    cli.main(args)
                self.assertEqual(caught.exception.code, 2)
                create.assert_not_called()

    def test_main_preserves_options_and_constructs_only_one_service(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(cli, 'make_local_runner') as runner, \
             patch.object(cli, 'JobService') as create, patch.object(cli, 'run_session', return_value=0) as session:
            self.assertEqual(cli.main(['--root', temporary, '--limit', '8', '--top-k', '2',
                                       '--search-only', '--answer-style', 'source-findings',
                                       '--pubmed-query', 'fictional[Title]']), 0)
            root = Path(temporary).resolve()
            runner.assert_called_once_with(root, root / '.cache/embeddings/bge-input-v1')
            create.assert_called_once_with(root / 'results/cli-requests', runner.return_value)
            self.assertEqual(session.call_args.args[1],
                {'limit': 8, 'top_k': 2, 'generate': False, 'answer_style': 'source-findings',
                 'pubmed_query': 'fictional[Title]'})
            self.assertIsNone(session.call_args.kwargs['question'])

    def test_help_and_idle_exit_are_available_without_api_or_ml_packages(self):
        source = str(Path(__file__).resolve().parents[1] / 'src')
        code = """import contextlib, io, sys, tempfile
sys.path.insert(0, SOURCE)
from medlens import cli
with contextlib.redirect_stdout(io.StringIO()):
    try: cli.main(['--help'])
    except SystemExit as exc: assert exc.code == 0
    with tempfile.TemporaryDirectory() as root:
        old = sys.stdin
        try:
            sys.stdin = io.StringIO('/quit\\n')
            assert cli.main(['--root', root]) == 0
        finally: sys.stdin = old
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules
    for prefix in ('fastapi', 'pydantic', 'uvicorn', 'httpx', 'numpy', 'torch',
                   'transformers', 'mlx', 'mlx_lm', 'medlens.api', 'medlens.live',
                   'medlens.inference', 'medlens.retrieval'))
print('lazy')
""".replace('SOURCE', repr(source))
        process = subprocess.run([sys.executable, '-I', '-S', '-c', code],
                                 capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), 'lazy')


if __name__ == '__main__':
    unittest.main()
