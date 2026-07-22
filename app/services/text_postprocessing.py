from __future__ import annotations

import re
from typing import Any

from configs.text_replacements import TEXT_REPLACEMENTS


def preserve_case(
    original: str,
    replacement: str,
) -> str:
    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]

    return replacement


def apply_text_replacements(
    text: str,
) -> str:
    result = text

    for wrong, correct in TEXT_REPLACEMENTS.items():
        pattern = re.compile(
            rf"(?<!\w){re.escape(wrong)}(?!\w)",
            flags=re.IGNORECASE,
        )

        result = pattern.sub(
            lambda match: preserve_case(
                match.group(0),
                correct,
            ),
            result,
        )

    return result


def postprocess_asr_result(
    asr_result: dict[str, Any],
) -> dict[str, Any]:
    result = dict(asr_result)

    result["segments"] = [
        {
            **segment,
            "text": apply_text_replacements(
                str(segment["text"])
            ),
        }
        for segment in asr_result["segments"]
    ]

    result["words"] = [
        {
            **word,
            "text": apply_text_replacements(
                str(word["text"])
            ),
        }
        for word in asr_result["words"]
    ]

    return result
