"""Normalize <think> tags rewriter."""

import re
from typing import Any, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Rewriter


@OPERATORS.register_module()
class NormThinkRewriter(Rewriter):
    """
    Normalize <think> tag format in answers.

    Ensures consistent format:
    - Lowercase tags
    - Newline after opening tag
    - Newline before closing tag

    Example:
        Input:  "<THINK>reasoning</THINK>answer"
        Output: "<think>\\nreasoning\\n</think>\\nanswer"
    """

    # Patterns for different think tag formats
    THINK_VARIANTS = [
        (re.compile(r"<THINK>", re.IGNORECASE), "<think>"),
        (re.compile(r"</THINK>", re.IGNORECASE), "</think>"),
        (re.compile(r"<think>(?!\n)"), "<think>\n"),
        (re.compile(r"(?<!\n)</think>"), "\n</think>"),
        (re.compile(r"</think>(?!\n)(?=\S)"), "</think>\n"),
    ]

    def __init__(
        self,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the rewriter.

        Args:
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Normalize think tags in an answer."""
        qa = item.get_qa(qa_idx)
        new_answer = self._normalize(qa.answer)

        if new_answer != qa.answer:
            return new_answer
        return None

    def _normalize(self, text: str) -> str:
        """Apply all normalization patterns."""
        result = text
        for pattern, replacement in self.THINK_VARIANTS:
            result = pattern.sub(replacement, result)
        return result
