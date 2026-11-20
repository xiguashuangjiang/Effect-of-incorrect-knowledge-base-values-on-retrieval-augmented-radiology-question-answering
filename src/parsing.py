"""
src/parsing.py
Created on October 13, 2026

@author: Yangtingting Wang
https://github.com/xiguashuangjiang
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParsedAnswer:
    answer_letter: Optional[str]
    answer_status: str
    confidence: Optional[float]


def _strip_reasoning_prefix(text: str) -> str:
    if not text:
        return ""

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]

    return text.strip()


def _remove_markdown_emphasis(text: str) -> str:
    return text.replace("**", "").replace("__", "")


def parse_answer_letter(text: str) -> Optional[str]:

    text = _strip_reasoning_prefix(text)
    text = _remove_markdown_emphasis(text)

    matches = re.findall(
        r"FINAL[_ ]ANSWER\s*[:：]\s*\(?([A-E])\b",
        text,
        flags=re.IGNORECASE,
    )

    if matches:
        return matches[-1].upper()

    matches = re.findall(
        r"(?m)^\s*(?i:FINAL[_ ]ANSWER)\s+\(?([A-E])\)?\.?\s*$",
        text,
    )

    if matches:
        return matches[-1].upper()

    fallback = re.match(
        r"^\s*(?:OPTION\s+)?([A-E])(?:[.):]|(?=\s*(?:\n|$)))",
        text,
        flags=re.IGNORECASE,
    )

    if fallback:
        return fallback.group(1).upper()

    return None


def parse_confidence(text: str) -> Optional[float]:

    text = _strip_reasoning_prefix(text)
    text = _remove_markdown_emphasis(text)

    matches = re.findall(
        r"CONFIDENCE\s*[:：]\s*(\d{1,3}(?:\.\d+)?)(?![\w]|[.,]\d)",
        text,
        flags=re.IGNORECASE,
    )

    if matches:
        value = float(matches[-1])
        return value if 0 <= value <= 100 else None

    matches = re.findall(
        r"(?m)^\s*(?i:CONFIDENCE)\s+"
        r"(\d{1,3}(?:\.\d+)?)\s*%?\s*$",
        text,
    )

    if not matches:
        return None

    value = float(matches[-1])
    return value if 0 <= value <= 100 else None


def parse_model_response(text: str) -> ParsedAnswer:
    answer_letter = parse_answer_letter(text)
    confidence = parse_confidence(text)

    return ParsedAnswer(
        answer_letter=answer_letter,
        answer_status="parsed" if answer_letter is not None else "unparsed",
        confidence=confidence,
    )
