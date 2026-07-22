from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


def interval_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(
        0.0,
        min(end_a, end_b) - max(start_a, start_b),
    )


def interval_distance(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    if end_a < start_b:
        return start_b - end_a

    if end_b < start_a:
        return start_a - end_b

    return 0.0


def assign_speaker_to_word(
    word: dict[str, Any],
    speaker_turns: list[dict[str, Any]],
    max_nearest_distance: float = 0.5,
) -> str:
    if not speaker_turns:
        return "UNKNOWN"

    word_start = float(word["start"])
    word_end = float(word["end"])
    word_duration = max(word_end - word_start, 1e-6)

    overlap_by_speaker: dict[str, float] = defaultdict(float)

    for turn in speaker_turns:
        overlap = interval_overlap(
            word_start,
            word_end,
            float(turn["start"]),
            float(turn["end"]),
        )

        if overlap > 0:
            overlap_by_speaker[str(turn["speaker"])] += overlap

    if overlap_by_speaker:
        best_speaker, best_overlap = max(
            overlap_by_speaker.items(),
            key=lambda item: item[1],
        )

        if best_overlap / word_duration > 0:
            return best_speaker

    nearest_speaker = "UNKNOWN"
    nearest_distance = float("inf")

    for turn in speaker_turns:
        distance = interval_distance(
            word_start,
            word_end,
            float(turn["start"]),
            float(turn["end"]),
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest_speaker = str(turn["speaker"])

    if nearest_distance <= max_nearest_distance:
        return nearest_speaker

    return "UNKNOWN"


def assign_speakers_to_words(
    words: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assigned_words: list[dict[str, Any]] = []

    for word in words:
        assigned_word = dict(word)
        assigned_word["speaker"] = assign_speaker_to_word(
            word=word,
            speaker_turns=speaker_turns,
        )
        assigned_words.append(assigned_word)

    return assigned_words


def _finish_utterance(
    current: dict[str, Any],
) -> dict[str, Any]:
    result = dict(current)
    result["text"] = join_word_tokens(
        result.pop("_words")
    )
    return result


def join_word_tokens(words: list[str]) -> str:
    """Собирает word-level ASR токены без пробелов перед пунктуацией."""
    text = " ".join(word.strip() for word in words if word.strip())
    text = re.sub(r"\s+([,.;:!?…%)\]}])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def build_utterances(
    words: list[dict[str, Any]],
    max_gap_seconds: float = 1.0,
    max_utterance_duration: float = 20.0,
) -> list[dict[str, Any]]:
    """
    Объединяет последовательные слова одного спикера в реплики.

    Новая реплика начинается, если:
    - сменился спикер;
    - пауза превысила max_gap_seconds;
    - текущая реплика достигла max_utterance_duration.
    """
    if not words:
        return []

    ordered_words = sorted(
        words,
        key=lambda item: (
            float(item["start"]),
            float(item["end"]),
        ),
    )

    utterances: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for word in ordered_words:
        speaker = str(word["speaker"])
        start = float(word["start"])
        end = float(word["end"])
        text = str(word["text"]).strip()

        if not text:
            continue

        current_duration = (
            start - float(current["start"])
            if current is not None
            else 0.0
        )

        starts_new_utterance = (
            current is None
            or speaker != current["speaker"]
            or start - float(current["end"]) > max_gap_seconds
            or current_duration >= max_utterance_duration
        )

        if starts_new_utterance:
            if current is not None:
                utterances.append(
                    _finish_utterance(current)
                )

            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "_words": [text],
            }
        else:
            current["end"] = end
            current["_words"].append(text)

    if current is not None:
        utterances.append(
            _finish_utterance(current)
        )

    return utterances


def smooth_short_speaker_turns(
    utterances: list[dict[str, Any]],
    max_duration: float = 1.5,
    max_words: int = 3,
    max_neighbor_gap: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Исправляет короткое ложное переключение вида A -> B -> A.

    Средняя реплика переприсваивается спикеру A, если:
    - предыдущая и следующая реплики принадлежат одному спикеру;
    - средняя реплика принадлежит другому спикеру;
    - её длительность не больше max_duration;
    - в ней не больше max_words слов.

    Настоящие короткие ответы вида A -> B -> A также возможны,
    поэтому правило намеренно применяется только к очень коротким
    фрагментам.
    """
    if len(utterances) < 3:
        return [dict(item) for item in utterances]

    smoothed = [
        dict(item)
        for item in utterances
    ]

    for index in range(1, len(smoothed) - 1):
        previous = smoothed[index - 1]
        current = smoothed[index]
        following = smoothed[index + 1]

        previous_speaker = str(previous["speaker"])
        current_speaker = str(current["speaker"])
        following_speaker = str(following["speaker"])

        if previous_speaker != following_speaker:
            continue

        if current_speaker == previous_speaker:
            continue

        if current_speaker == "UNKNOWN":
            continue

        duration = max(
            0.0,
            float(current["end"])
            - float(current["start"]),
        )

        word_count = len(
            str(current["text"]).split()
        )

        gap_before = max(
            0.0,
            float(current["start"]) - float(previous["end"]),
        )
        gap_after = max(
            0.0,
            float(following["start"]) - float(current["end"]),
        )

        normalized_text = re.sub(
            r"[^\w\s-]",
            " ",
            str(current["text"]).lower().replace("ё", "е"),
        ).strip()
        first_word = normalized_text.split(maxsplit=1)[0] if normalized_text else ""
        likely_short_response = first_word in {
            "да",
            "нет",
            "ага",
            "угу",
            "ладно",
            "хорошо",
        }

        is_short = (
            duration <= max_duration
            and word_count <= max_words
            and gap_before <= max_neighbor_gap
            and gap_after <= max_neighbor_gap
            and not likely_short_response
        )

        if is_short:
            current["speaker"] = previous_speaker
            current["smoothed_from_speaker"] = (
                current_speaker
            )

    return merge_adjacent_utterances(smoothed)


def merge_adjacent_utterances(
    utterances: list[dict[str, Any]],
    max_gap_seconds: float = 1.0,
    max_utterance_duration: float = 20.0,
) -> list[dict[str, Any]]:
    """
    После сглаживания снова объединяет соседние фрагменты
    одного спикера, но не создаёт реплики длиннее лимита.
    """
    if not utterances:
        return []

    merged: list[dict[str, Any]] = []

    for utterance in utterances:
        item = dict(utterance)

        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]

        same_speaker = (
            previous["speaker"] == item["speaker"]
        )

        gap = (
            float(item["start"])
            - float(previous["end"])
        )

        combined_duration = (
            float(item["end"])
            - float(previous["start"])
        )

        can_merge = (
            same_speaker
            and gap <= max_gap_seconds
            and combined_duration <= max_utterance_duration
        )

        if can_merge:
            previous["end"] = item["end"]
            previous["text"] = (
                f"{previous['text']} {item['text']}"
            ).strip()

            if "smoothed_from_speaker" in item:
                previous.setdefault(
                    "smoothed_fragments",
                    [],
                ).append(
                    {
                        "original_speaker": item[
                            "smoothed_from_speaker"
                        ],
                        "start": item["start"],
                        "end": item["end"],
                        "text": item["text"],
                    }
                )
        else:
            merged.append(item)

    return merged
