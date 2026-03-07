"""Text repetition detection utilities."""

from typing import Optional, Tuple


def has_meaningful_content(text: str, threshold: float = 0.5) -> bool:
    """
    Check if string contains sufficient proportion of alphanumeric characters.

    Args:
        text: Text to check.
        threshold: Minimum ratio of alphanumeric characters (0-1).

    Returns:
        True if ratio meets threshold.
    """
    if not text:
        return False

    alphanumeric_chars = sum(1 for char in text if char.isalnum())
    total_chars = len(text)

    return (alphanumeric_chars / total_chars) >= threshold


def detect_consecutive_repetition(
    text: str, min_times: int = 10
) -> Tuple[bool, Optional[str]]:
    """
    Detect consecutive repetitive patterns at the end of text.

    Looks for patterns like "ABCABCABC..." where the same substring
    is repeated multiple times consecutively.

    Args:
        text: Text to analyze.
        min_times: Minimum number of consecutive repetitions to detect.

    Returns:
        Tuple of (is_repetitive, repeated_pattern).
    """
    n = len(text)
    if n < 1000:
        return False, None

    # Excluded patterns (markdown/formatting characters)
    excluded_chars = ["|", "__", "--", "——", "=", "*", "#", "xx"]

    for length in range(2, n // 2 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length

        if count >= min_times:
            stripped = candidate.strip()
            # Skip patterns that are just formatting characters
            if stripped and not any(exc in stripped for exc in excluded_chars):
                return True, candidate

    return False, None
