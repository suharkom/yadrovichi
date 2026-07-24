"""Батч-постобработка реплик: чистит текст пачками (в UI — каждые ~10 реплик).

Работает ПО СЛОВАМ, сохраняя уверенность ASR у каждого слова, — чтобы не
ломать подсветку низкоуверенных слов. Сейчас движок — правила (безопасно,
идемпотентно, без моделей и нагрузки на GPU). Архитектура пригодна под замену:
`refine_utterances(..., refiner=<своя функция по словам>)` — сюда позже можно
подставить LLM-достройку форм слов.

Правила (консервативные, чтобы не переисказить):
- убрать одиночные слова-филлеры (э, мм, эм, кхм…);
- схлопнуть немедленный повтор длинного слова (>=4 букв): «хотим хотим» -> «хотим»;
- заглавная в начале реплики и после конца предложения.
"""

from __future__ import annotations

from typing import Any, Callable

from app.services.alignment import join_word_tokens

FILLERS = {
    "э", "ээ", "эээ", "мм", "ммм", "эм", "эмм", "мэ", "кхм", "кх", "гм",
}

WordList = list[dict[str, Any]]


def _norm(token: str) -> str:
    return token.lower().strip(".,!?…-—:;\"'()")


def _cap_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def _rule_refine_words(words: WordList) -> WordList:
    """Правило-based достройка по словам с сохранением probability."""
    cleaned: WordList = []
    for word in words:
        token = str(word.get("text", "")).strip()
        if not token:
            continue
        key = _norm(token)
        if key in FILLERS:
            continue
        if (
            cleaned
            and key
            and key == _norm(str(cleaned[-1]["text"]))
            and len(key) >= 4
        ):
            continue
        cleaned.append(dict(word))

    capitalize_next = True
    for word in cleaned:
        token = str(word["text"])
        if capitalize_next:
            word["text"] = _cap_first(token)
        stripped = token.rstrip()
        capitalize_next = bool(stripped) and stripped[-1] in ".!?…"

    return cleaned


def refine_utterances(
    utterances: list[dict[str, Any]],
    refiner: Callable[[WordList], WordList] = _rule_refine_words,
) -> list[dict[str, Any]]:
    """Пройтись по репликам, дочистить каждую. Мутирует на месте и возвращает.

    Идемпотентно: повторный прогон уже вычищенных реплик ничего не портит,
    поэтому в UI можно звать хоть каждые 10 реплик на всём накопленном списке.
    """
    for utterance in utterances:
        words = utterance.get("words")
        if words:
            refined = refiner(words)
            utterance["words"] = refined
            utterance["text"] = join_word_tokens(
                [str(w["text"]) for w in refined]
            )
        else:
            # Нет по-словных данных (батч/старый прогон) — чистим текст целиком.
            utterance["text"] = _refine_plain_text(utterance.get("text", ""))
    return utterances


def _refine_plain_text(text: str) -> str:
    tokens = [
        {"text": token}
        for token in str(text).split()
        if token.strip()
    ]
    refined = _rule_refine_words(tokens)
    return join_word_tokens([str(w["text"]) for w in refined])
