"""Plain-text local research CLI over the same service used by HTTP."""

import argparse
from html import escape
import json
import logging
import os
from pathlib import Path
import shlex
import signal
import sys
import time

from .api_backend import make_local_runner
from .api_contract import RequestValidationError, SearchRequest
from .api_service import JobService, ServiceBusy, ServiceUnavailable, UnknownRequest
from .cli_output import render_evidence, render_result, safe_text


HELP = """Введите биомедицинский вопрос на английском и нажмите Enter.
/help                 Справка
/evidence             Все отобранные фрагменты последнего запроса
/export [PATH]        Сохранить последний результат: .html или .json
/quit                 Выход
Настройки поиска задаются флагами запуска; medlens --help.
Ctrl+C во время поиска ждёт завершения задания и освобождения модели."""

STAGES = {
    'queued': 'Запрос принят', 'setup': 'Подготовка',
    'query-validation': 'Проверка запроса', 'pubmed-download': 'Поиск и загрузка PubMed',
    'parsing': 'Подготовка статей', 'windowing': 'Разбиение на фрагменты',
    'passage-embeddings': 'Подготовка векторов фрагментов',
    'index-validation': 'Проверка поискового индекса', 'retrieval': 'Ранжирование статей',
    'retrieval-complete': 'Статьи ранжированы', 'context': 'Отбор доказательств',
    'context-complete': 'Доказательства отобраны', 'reranking': 'Переранжирование',
    'reranking-complete': 'Переранжирование завершено',
    'answer': 'Подготовка ответа и проверка цитат',
    'complete': 'Завершение задания и освобождение ресурсов',
    'runtime-cleanup': 'Освобождение модели', 'unknown': 'Выполнение запроса',
}
TERMINAL = {'succeeded', 'failed', 'interrupted'}


def export_job(job, destination):
    """Export only the public envelope, never the private diagnostic HTML."""
    path = Path(destination).expanduser()
    if path.suffix.lower() == '.json':
        body = json.dumps(job, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    elif path.suffix.lower() == '.html':
        text = render_result(job) + '\n\n' + render_evidence(job)
        body = ('<!doctype html><html lang="ru"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>MedLens</title><style>body{max-width:70rem;margin:2rem auto;'
                'padding:0 1rem;font:16px/1.5 system-ui}pre{white-space:pre-wrap;'
                'overflow-wrap:anywhere}</style><body><pre>' + escape(text) + '</pre></body></html>\n')
    else:
        raise ValueError('Укажите расширение .html или .json.')
    # Serialize before touching the destination; exclusive creation also rejects links.
    payload = body.encode('utf-8')
    with path.open('xb') as stream:
        try:
            stream.write(payload)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path


def _close(service):
    """A second SIGINT must not abandon the native worker during shutdown."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        service.close()
    finally:
        signal.signal(signal.SIGINT, previous)


def _exit_code(job):
    if job['status'] == 'interrupted':
        return 130
    if (job.get('error') or {}).get('code') == 'answer_validation_failed':
        return 2
    return 0 if job['status'] == 'succeeded' else 1


def run_session(service, options, *, question=None, read=None, output=None,
                poll_interval=0.25):
    """Own a single service until idle exit or graceful drain after Ctrl+C.

    The injectable input/output boundary keeps tests independent of models/network.
    A failed search can be followed by another question if service health is ready.
    """
    output = output if output is not None else sys.stdout
    read = read if read is not None else input

    def say(text):
        print(text, file=output, flush=True)

    last_job = None
    one_shot = question is not None
    try:
        if not one_shot:
            say('MedLens — поиск биомедицинской литературы. /help — справка; /quit — выход.')
            say('Вопросы — на английском. Ответы исследовательские; независимая проверка не выполнена.')
            say(f"До {options['limit']} статей; top-{options['top_k']}; "
                + ('только поиск.' if not options['generate'] else f"режим {options['answer_style']}."))
        while True:
            if not one_shot:
                try:
                    line = read('Вопрос> ')
                except EOFError:
                    return 0
                if not line.strip():
                    continue
                if line.startswith('/'):
                    try:
                        parts = shlex.split(line)
                    except ValueError:
                        say('Ошибка команды: проверьте кавычки. /help — справка.')
                        continue
                    if parts in (['/quit'], ['/exit']):
                        return 0
                    if parts == ['/help']:
                        say(HELP)
                    elif parts == ['/evidence']:
                        say(render_evidence(last_job) if last_job else 'Сначала выполните запрос.')
                    elif parts and parts[0] == '/export' and len(parts) <= 2:
                        if last_job is None:
                            say('Сначала выполните запрос.')
                            continue
                        target = (parts[1] if len(parts) == 2 else
                                  service.storage_root / last_job['request_id'] / 'report-public.html')
                        try:
                            saved = export_job(last_job, target)
                        except FileExistsError:
                            say('Файл уже существует. Выберите другое имя; перезапись отключена.')
                        except (OSError, ValueError, UnicodeError):
                            say('Не удалось сохранить отчёт. Проверьте путь и расширение .html/.json.')
                        else:
                            say('Сохранено: ' + safe_text(str(saved)))
                    else:
                        say('Неизвестная команда. /help — справка.')
                    continue
                question = line
            try:
                request = SearchRequest.from_payload({'question': question, **options})
            except RequestValidationError as exc:
                say('Некорректный запрос: ' + safe_text(str(exc)))
                if one_shot:
                    return 2
                continue
            try:
                job = service.submit(request)
            except ServiceBusy:
                say('Сервис занят другим запросом. Завершите его перед новым запуском.')
                return 1
            except ServiceUnavailable:
                say('Сервис недоступен. Проверьте локальные артефакты и перезапустите CLI.')
                return 1
            say('Запрос: ' + safe_text(job['request_id']))
            started = time.monotonic()
            previous = None
            while True:
                try:
                    job = service.get(job['request_id'])
                except UnknownRequest:
                    say('Не удалось прочитать состояние запроса. Ожидаю завершения перед выходом.')
                    return 1
                if job['status'] in TERMINAL:
                    break
                if job['stage'] != previous:
                    say(f'[{time.monotonic() - started:.1f} с] '
                        + STAGES.get(job['stage'], STAGES['unknown']))
                    previous = job['stage']
                time.sleep(poll_interval)
            last_job = job
            say(render_result(job))
            if one_shot:
                return _exit_code(job)
            if not service.health()['accepting_requests']:
                say('Сервис недоступен для новых запросов. Требуется перезапуск.')
                return 1
    except KeyboardInterrupt:
        if service.health().get('active_request_id'):
            say('\nВыход после текущего запроса: задание не отменено. Жду завершения и освобождения модели…')
        else:
            say('\nВыход.')
        return 130
    finally:
        _close(service)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Простой интерактивный поиск MedLens на macOS/Apple Silicon.')
    parser.add_argument('--question', help='Один вопрос на английском; затем выход вместо диалога')
    parser.add_argument('--limit', type=int, default=100, help='Кандидатов PubMed: 1–300 (100)')
    parser.add_argument('--top-k', type=int, default=10, help='Статей в выдаче: 1–10 (10)')
    parser.add_argument('--search-only', action='store_true', help='Поиск и фрагменты без генерации ответа')
    parser.add_argument('--answer-style', choices=('draft', 'source-findings'), default='draft',
                        help='Режим ответа; source-findings — экспериментальный')
    parser.add_argument('--pubmed-query', help='Точное поисковое выражение PubMed для этой сессии')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2], help='Корень проекта с ресурсами')
    parser.add_argument('--storage-dir', type=Path, help='Хранилище запросов; ROOT/results/cli-requests')
    parser.add_argument('--cache', type=Path, help='Кеш векторов; ROOT/.cache/embeddings/bge-input-v1')
    args = parser.parse_args(argv)
    # Keep the terminal focused on our stage messages, not dependency loading bars.
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    logging.getLogger('torch.distributed.elastic.multiprocessing.redirects').addFilter(
        lambda record: record.levelno >= logging.ERROR)
    options = {'limit': args.limit, 'top_k': args.top_k, 'generate': not args.search_only,
               'answer_style': args.answer_style, 'pubmed_query': args.pubmed_query}
    # Validate flags before starting the REPL, even when no question is supplied.
    try:
        SearchRequest.from_payload({'question': args.question if args.question is not None else
                                    'What is the role of inflammation in diabetes?', **options})
    except RequestValidationError as exc:
        parser.error(safe_text(str(exc)))
    try:
        root = args.root.expanduser().resolve()
        store = (args.storage_dir.expanduser() if args.storage_dir else root / 'results/cli-requests').resolve()
        cache = (args.cache.expanduser() if args.cache else root / '.cache/embeddings/bge-input-v1').resolve()
        if store.is_relative_to(cache) or cache.is_relative_to(store):
            parser.error('Хранилище запросов и кеш векторов должны быть раздельными каталогами.')
        service = JobService(store, make_local_runner(root, cache))
    except OSError:
        parser.error('Не удалось создать хранилище запросов. Проверьте --root и --storage-dir.')
    try:
        return run_session(service, options, question=args.question)
    except BrokenPipeError:
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
