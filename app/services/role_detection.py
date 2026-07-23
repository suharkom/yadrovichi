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


def calculate_neighbor_scores(
    utterances: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    speakers = {
        str(item["speaker"])
        for item in utterances
        if str(item["speaker"]) != "UNKNOWN"
    }
    neighbors: dict[str, set[str]] = {
        speaker: set()
        for speaker in speakers
    }

    previous_speaker: str | None = None
    transition_count = 0

    for utterance in utterances:
        speaker = str(utterance["speaker"])
        if speaker == "UNKNOWN":
            continue

        if previous_speaker is not None and speaker != previous_speaker:
            neighbors[previous_speaker].add(speaker)
            neighbors[speaker].add(previous_speaker)
            transition_count += 1

        previous_speaker = speaker

    maximum_neighbors = max(len(speakers) - 1, 1)
    return {
        speaker: {
            "unique_neighbor_count": len(speaker_neighbors),
            "neighbor_score": len(speaker_neighbors) / maximum_neighbors,
            "neighbors": sorted(speaker_neighbors),
            "transition_count": transition_count,
        }
        for speaker, speaker_neighbors in neighbors.items()
    }


def detect_teacher(
    utterances: list[dict[str, Any]],
) -> dict[str, Any]:
    speaker_texts = collect_speaker_texts(utterances)
    speech_stats = calculate_speech_statistics(utterances)
    neighbor_stats = calculate_neighbor_scores(utterances)

    if not speaker_texts:
        raise ValueError(
            "Невозможно определить преподавателя: "
            "нет текста, привязанного к спикерам."
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

        neighbor_result = neighbor_stats.get(speaker, {})

        scores[speaker] = {
            **marker_result,
            **neighbor_result,
            "speech_seconds": speech_seconds,
            "speech_share": speech_share,
        }

    graph_ranking = sorted(
        scores,
        key=lambda speaker: scores[speaker]["unique_neighbor_count"],
        reverse=True,
    )
    best_graph_score = scores[graph_ranking[0]]["unique_neighbor_count"]
    graph_candidates = [
        speaker
        for speaker in graph_ranking
        if scores[speaker]["unique_neighbor_count"] == best_graph_score
    ]
    graph_is_decisive = (
        len(scores) > 2
        and len(graph_candidates) == 1
        and best_graph_score > 0
    )

    if graph_is_decisive:
        scoring_basis = "speaker_graph"
        ranking = graph_ranking
        score_key = "unique_neighbor_count"
        used_speech_share_tiebreaker = False
    else:
        scoring_basis = "teacher_markers"
        candidates = (
            list(scores)
            if len(scores) <= 2
            else graph_candidates
        )
        candidate_ranking = sorted(
            candidates,
            key=lambda speaker: (
                scores[speaker]["markers_per_1000_words"],
                scores[speaker]["speech_share"],
            ),
            reverse=True,
        )
        ranking = candidate_ranking + [
            speaker
            for speaker in graph_ranking
            if speaker not in candidate_ranking
        ]
        score_key = "markers_per_1000_words"
        best_marker_score = scores[candidate_ranking[0]][score_key]
        used_speech_share_tiebreaker = sum(
            scores[speaker][score_key] == best_marker_score
            for speaker in candidate_ranking
        ) > 1

    teacher_speaker = ranking[0]
    best_score = scores[teacher_speaker][score_key]

    second_score = (
        scores[ranking[1]][score_key]
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

    if graph_is_decisive:
        graph_evidence = min(
            scores[teacher_speaker]["transition_count"] / 10.0,
            1.0,
        )
        heuristic_confidence = score_confidence * graph_evidence
    else:
        heuristic_confidence = (
            score_confidence
            * (0.5 + 0.5 * evidence_factor)
        )

    low_confidence = (
        heuristic_confidence < 0.2
        or (not graph_is_decisive and marker_count == 0)
        or used_speech_share_tiebreaker
    )

    return {
        "teacher_speaker": teacher_speaker,
        "heuristic_confidence": heuristic_confidence,
        "low_confidence": low_confidence,
        "score_margin": margin,
        "scoring_basis": scoring_basis,
        "used_speech_share_tiebreaker": used_speech_share_tiebreaker,
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
