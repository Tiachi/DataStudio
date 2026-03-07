"""Language detection utilities."""

import re
from typing import Literal


def detect_language(text: str) -> Literal["chinese", "english", "unknown"]:
    """
    Detect if text is primarily Chinese or English.

    Args:
        text: Input text to analyze.

    Returns:
        "chinese", "english", or "unknown".
    """
    # Remove punctuation and digits, keep only letters and Chinese characters
    clean_text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbfa-zA-Z]", "", text)

    if not clean_text:
        return "unknown"

    # Count Chinese characters
    chinese_chars = re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", clean_text)

    if len(chinese_chars) > 0:
        return "chinese"
    else:
        return "english"


def is_mixed_language(question: str, answer: str) -> bool:
    """
    Check if question and answer are in different languages.

    Useful for detecting cases where question is English but answer is Chinese.

    Args:
        question: Question text.
        answer: Answer text.

    Returns:
        True if languages differ (specifically English question, Chinese answer).
    """
    question_lang = detect_language(question)
    answer_lang = detect_language(answer)
    return question_lang == "english" and answer_lang == "chinese"
