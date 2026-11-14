"""Plain terminal views of the public research-job envelope."""

import math


_JOB_STATUSES = {
    "queued": "ожидает запуска", "running": "выполняется", "succeeded": "завершено",
    "failed": "ошибка", "interrupted": "прервано",
}
_ANSWER_STATUSES = {
    "answered": "исследовательский черновик", "partial": "частичный исследовательский черновик",
    "insufficient_evidence": "недостаточно доказательств; ответ не сформирован",
    "validation_failed": "проверка ответа не пройдена; ответ скрыт",
    "not_requested": "генерация не запрашивалась (только поиск)",
    "not_completed": "не сформирован",
}
_ABSTENTION_REASONS = {
    "empty-pubmed-result": "PubMed не вернул статей.",
    "empty-context": "Фрагменты для ответа отсутствуют.",
    "model-abstained": "Модель отказалась формировать ответ.",
    "no-claims-passed-support-review": "Утверждения отклонены проверкой поддержки источниками.",
    "no-bounded-source-quotes": "Подходящие цитаты не найдены.",
    "explicit-source-scope-blocks-all-articles": "Найденные статьи не соответствуют ограничениям вопроса.",
    "no-direct-scoped-source-findings": "Прямые находки по вопросу не найдены.",
}


def safe_text(value):
    """Escape terminal controls without dropping or truncating source text."""
    if value is None:
        return ""
    result = []
    for char in str(value):
        code = ord(char)
        if code < 32 or 127 <= code <= 159:
            result.append("\\x" + format(code, "02x"))
        elif 0xD800 <= code <= 0xDFFF or 0x202A <= code <= 0x202E or 0x2066 <= code <= 0x2069:
            result.append("\\u" + format(code, "04x"))
        else:
            result.append(char)
    return "".join(result)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _label(table, value, fallback):
    return table.get(value, safe_text(value) or fallback) if isinstance(value, str) else fallback


def _reference(source):
    parts = []
    for key, label in (("citation_id", ""), ("pmid", "PMID "), ("pmcid", "PMCID ")):
        if source.get(key):
            parts.append(label + safe_text(source[key]))
    return " · ".join(parts) or "источник"


def _source_lines(source):
    lines = ["  [" + _reference(source) + "]"]
    if source.get("url"):
        lines.append("  " + safe_text(source["url"]))
    return lines


def _answer_lines(answer):
    status = answer.get("status")
    findings = answer.get("answer_style") == "source-findings"
    label = _label(_ANSWER_STATUSES, status, "не сформирован")
    if findings and status in {"answered", "partial"}:
        label = "частичная подборка цитат" if status == "partial" else "подборка цитат"
    lines = ["Ответ: " + label]
    if status in {"answered", "partial"}:
        lines.append("Научная корректность и полнота ответа независимо не проверены.")
        if answer.get("quote_identity_verified") is True:
            lines.append("Программа подтвердила точное совпадение цитат с источниками.")
        elif answer.get("quote_identity_verified") is False:
            lines.append("Точное совпадение цитат не подтверждено.")
        if answer.get("semantic_support_verified") is False:
            lines.append("Семантическая поддержка утверждений не подтверждена.")
        if answer.get("answer_completeness_established") is False:
            lines.append("Полнота ответа не установлена.")
        claims = _rows(answer.get("claims"))
        for index, claim in enumerate(claims, 1):
            title = claim.get("source_title") if findings else claim.get("text")
            lines.extend(["", str(index) + ". " + safe_text(title or "Находка из источника")])
            for caution in _rows(claim.get("scope_cautions")):
                message = caution.get("message") or caution.get("code")
                if message:
                    lines.append("Ограничение применимости: " + safe_text(message))
            for citation in _rows(claim.get("citations")):
                lines.extend(_source_lines(citation))
                if "quote" in citation:
                    lines.append("  Цитата: " + safe_text(citation["quote"]))
        if not claims:
            lines.append("Утверждения отсутствуют.")
    elif status == "insufficient_evidence":
        lines.append("Доступные фрагменты не позволили сформировать ответ на этот вопрос.")
        reason = answer.get("reason")
        if isinstance(reason, str) and reason in _ABSTENTION_REASONS:
            lines.append(_ABSTENTION_REASONS[reason])
    if isinstance(answer.get("limitations"), list) and answer["limitations"]:
        lines.append("Ограничения: " + "; ".join(safe_text(item) for item in answer["limitations"]))
    return lines


def _article_lines(articles):
    lines = ["Статьи (порядок Dense)"]
    if not articles:
        return lines + ["Статьи отсутствуют."]
    for index, article in enumerate(articles, 1):
        rank = article.get("rank", index)
        lines.append(safe_text(rank) + ". " + safe_text(article.get("title") or "Без названия"))
        details = []
        if article.get("pmid"):
            details.append("PMID " + safe_text(article["pmid"]))
        if details:
            lines.append("  " + " · ".join(details))
        if article.get("url"):
            lines.append("  " + safe_text(article["url"]))
    return lines


def render_result(job):
    """Render execution, answer/citations and articles; never private artifacts."""
    job = _mapping(job)
    result, request = _mapping(job.get("result")), _mapping(job.get("request"))
    lines = ["Задание: " + safe_text(job.get("request_id") or "неизвестно"),
             "Выполнение: " + _label(_JOB_STATUSES, job.get("status"), "неизвестно")]
    question = result.get("question", request.get("question"))
    if question:
        lines.append("Вопрос: " + safe_text(question))
    error = _mapping(job.get("error"))
    if error:
        lines.append("Ошибка [" + safe_text(error.get("code") or "search_failed")
                     + "] на этапе " + safe_text(error.get("stage") or job.get("stage") or "неизвестно")
                     + ": " + safe_text(error.get("message") or "Не удалось выполнить запрос."))
    if not result:
        lines.append("Результат пока недоступен.")
        return "\n".join(lines)
    seconds = result.get("request_seconds")
    if type(seconds) in {int, float} and math.isfinite(seconds) and seconds >= 0:
        lines.append("Время обработки: " + format(seconds, ".2f") + " с")
    search = _mapping(result.get("search"))
    counts = []
    for key, label in (("fetched_article_count", "загружено "), ("total_pubmed_matches", "совпадений ")):
        count = search.get(key)
        if type(count) is int and count >= 0:
            counts.append(label + str(count))
    if counts:
        lines.append("PubMed: " + ", ".join(counts))
    lines.extend(["", *_answer_lines(_mapping(result.get("answer"))), "",
                  *_article_lines(_rows(result.get("articles")))])
    evidence = _rows(result.get("evidence"))
    lines.append("\nОтобранных фрагментов: " + str(len(evidence)) + ". Показать: /evidence")
    return "\n".join(lines)


def render_evidence(job):
    """Render every public evidence passage with a simple source reference."""
    result = _mapping(_mapping(job).get("result"))
    evidence = _rows(result.get("evidence"))
    lines = ["Отобранные фрагменты источников",
             "Это результат поиска; релевантность и поддержка ответа независимо не проверены."]
    if not evidence:
        return "\n".join(lines + ["Фрагменты отсутствуют."])
    for index, source in enumerate(evidence, 1):
        lines.extend(["", str(index) + ". " + safe_text(source.get("title") or "Без названия")])
        lines.extend(_source_lines(source))
        if source.get("section"):
            lines.append("  Раздел: " + safe_text(source["section"]))
        if "text" in source:
            lines.append("  Текст: " + safe_text(source["text"]))
    return "\n".join(lines)
