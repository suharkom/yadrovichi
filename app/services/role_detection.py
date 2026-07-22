from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from configs.teacher_markers import (
    MARKER_WEIGHTS,
    TEACHER_MARKERS,
)


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def count_marker(
    normalized_text: str,
    marker: str,
) -> int:
    normalized_marker = normalize_text(marker)

    if not normalized_marker:
        return 0

    pattern = (
        r"(?<!\w)"
        + re.escape(normalized_marker)
        + r"(?!\w)"
    )

    return len(
        re.findall(
            pattern,
            normalized_text,
        )
    )


def collect_speaker_texts(
    utterances: list[dict[str, Any]],
) -> dict[str, str]:
    pieces: dict[str, list[str]] = defaultdict(list)

    for utterance in utterances:
        speaker = str(utterance["speaker"])
        text = str(utterance["text"]).strip()

        if speaker == "UNKNOWN" or not text:
            continue

        pieces[speaker].append(text)

    return {
        speaker: " ".join(texts)
        for speaker, texts in pieces.items()
    }


def calculate_marker_score(
    text: str,
) -> dict[str, Any]:
    normalized = normalize_text(text)
    word_count = len(normalized.split())

    raw_score = 0.0
    matches: list[dict[str, Any]] = []

    for category, markers in TEACHER_MARKERS.items():
        category_weight = MARKER_WEIGHTS[category]

        for marker in markers:
            count = count_marker(
                normalized_text=normalized,
                marker=marker,
            )

            if count == 0:
                continue

            weighted_score = count * category_weight
            raw_score += weighted_score

            matches.append(
                {
                    "marker": marker,
                    "category": category,
                    "count": count,
                    "weight": category_weight,
                    "weighted_score": weighted_score,
                }
            )

    markers_per_1000_words = (
        raw_score
        / max(word_count, 300)
        * 1000
    )

    return {
        "word_count": word_count,
        "raw_marker_score": raw_score,
        "markers_per_1000_words": markers_per_1000_words,
        "matches": matches,
    }


def calculate_speech_statistics(
    utterances: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    durations: dict[str, float] = defaultdict(float)
    total_duration = 0.0

    for utterance in utterances:
        speaker = str(utterance["speaker"])

        if speaker == "UNKNOWN":
            continue

        duration = max(
            0.0,
            float(utterance["end"])
            - float(utterance["start"]),
        )

        durations[speaker] += duration
        total_duration += duration

    return {
        speaker: {
            "speech_seconds": duration,
            "speech_share": (
                duration / total_duration
                if total_duration > 0
                else 0.0
            ),
        }
        for speaker, duration in durations.items()
    }


def calculate_neighbors(
    utterances: list[dict[str, Any]],
) -> dict[str, int]:
    neighbors: dict[str, set[str]] = defaultdict(set)

    valid_utterances = [
        utterance
        for utterance in utterances
        if utterance["speaker"] != "UNKNOWN"
    ]

    for previous, current in zip(
        valid_utterances,
        valid_utterances[1:],
    ):
        first = str(previous["speaker"])
        second = str(current["speaker"])

        if first == second:
            continue

        neighbors[first].add(second)
        neighbors[second].add(first)

    return {
        speaker: len(speaker_neighbors)
        for speaker, speaker_neighbors in neighbors.items()
    }


def detect_teacher(
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    speaker_texts = collect_speaker_texts(utterances)
    speech_stats = calculate_speech_statistics(utterances)
    neighbor_counts = calculate_neighbors(utterances)

    if not speaker_texts:
        raise ValueError(
            "Невозможно определить преподавателя: "
            "нет текста, привязанного к спикерам."
        )

    maximum_neighbor_count = max(
        neighbor_counts.values(),
        default=1,
    )

    scores: dict[str, dict[str, Any]] = {}

    for speaker, text in speaker_texts.items():
        marker_result = calculate_marker_score(text)

        speech_share = speech_stats.get(
            speaker,
            {},
        ).get("speech_share", 0.0)

        speech_seconds = speech_stats.get(
            speaker,
            {},
        ).get("speech_seconds", 0.0)

        neighbor_count = neighbor_counts.get(
            speaker,
            0,
        )

        normalized_neighbor_score = (
            neighbor_count / maximum_neighbor_count
            if maximum_neighbor_count > 0
            else 0.0
        )

        final_score = (
            marker_result["markers_per_1000_words"]
            + 1.5 * normalized_neighbor_score
            + 0.5 * speech_share
        )

        scores[speaker] = {
            **marker_result,
            "speech_seconds": speech_seconds,
            "speech_share": speech_share,
            "unique_neighbors": neighbor_count,
            "neighbor_score": normalized_neighbor_score,
            "final_score": final_score,
        }

    ranking = sorted(
        scores,
        key=lambda speaker: scores[speaker]["final_score"],
        reverse=True,
    )

    teacher_speaker = ranking[0]
    best_score = scores[teacher_speaker]["final_score"]

    second_score = (
        scores[ranking[1]]["final_score"]
        if len(ranking) > 1
        else 0.0
    )

    margin = best_score - second_score

    score_confidence = (
        margin / max(abs(best_score), 1.0)
    )
    score_confidence = max(
        0.0,
        min(score_confidence, 1.0),
    )

    marker_count = sum(
        match["count"]
        for match in scores[teacher_speaker]["matches"]
    )

    evidence_factor = min(
        marker_count / 3.0,
        1.0,
    )

    heuristic_confidence = (
        score_confidence
        * (0.5 + 0.5 * evidence_factor)
    )

    low_confidence = (
        heuristic_confidence < 0.2
        or marker_count == 0
    )

    return {
        "teacher_speaker": teacher_speaker,
        "heuristic_confidence": heuristic_confidence,
        "low_confidence": low_confidence,
        "score_margin": margin,
        "ranking": ranking,
        "scores": scores,
    }


def create_speaker_mapping(
    utterances: list[dict[str, Any]],
    teacher_speaker: str,
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {
        teacher_speaker: {
            "speaker_id": 0,
            "role": "teacher",
            "display_name": "Преподаватель",
        }
    }

    next_student_id = 1

    for utterance in utterances:
        source_speaker = str(utterance["speaker"])

        if (
            source_speaker == "UNKNOWN"
            or source_speaker in mapping
        ):
            continue

        mapping[source_speaker] = {
            "speaker_id": next_student_id,
            "role": "student",
            "display_name": f"Ученик {next_student_id}",
        }

        next_student_id += 1

    return mapping


def apply_roles(
    utterances: list[dict[str, Any]],
    speaker_mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for utterance in utterances:
        item = dict(utterance)
        source_speaker = str(item["speaker"])

        role_info = speaker_mapping.get(
            source_speaker,
            {
                "speaker_id": None,
                "role": "unknown",
                "display_name": "Неизвестный спикер",
            },
        )

        item["source_speaker"] = source_speaker
        item.update(role_info)
        item.pop("speaker", None)

        result.append(item)

    return result
