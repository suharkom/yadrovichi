"""Пользовательский словарь замен ошибок ASR, редактируемый из интерфейса.

Базовый словарь (`configs.text_replacements`) в коде, а правки пользователя
лежат поверх него в `data/replacements.json` — их можно добавлять на лету, не
трогая код. Итоговый словарь = база + оверрайды пользователя.
"""

from __future__ import annotations

import json
from pathlib import Path

from configs.text_replacements import TEXT_REPLACEMENTS

OVERRIDES_PATH = Path("data/replacements.json")

_cache: dict[str, str] | None = None


def _load_overrides() -> dict[str, str]:
    if OVERRIDES_PATH.exists():
        try:
            data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def get_replacements() -> dict[str, str]:
    """Итоговый словарь (база + оверрайды), с кешем. Быстрый — зовётся часто."""
    global _cache
    if _cache is None:
        _cache = {**TEXT_REPLACEMENTS, **_load_overrides()}
    return _cache


def user_overrides() -> dict[str, str]:
    """Только пользовательские правки (для отображения в интерфейсе)."""
    return _load_overrides()


def add_replacement(wrong: str, correct: str) -> dict[str, str]:
    """Добавить/обновить замену, сохранить и сбросить кеш. Вернуть оверрайды."""
    wrong = wrong.strip()
    correct = correct.strip()
    if not wrong:
        return _load_overrides()
    overrides = _load_overrides()
    overrides[wrong] = correct
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    global _cache
    _cache = None
    return overrides


def remove_replacement(wrong: str) -> dict[str, str]:
    """Убрать пользовательскую замену (базовую не трогает)."""
    overrides = _load_overrides()
    overrides.pop(wrong.strip(), None)
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    global _cache
    _cache = None
    return overrides
