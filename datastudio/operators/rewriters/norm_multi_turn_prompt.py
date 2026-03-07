"""Normalize multi-turn prompt rewriter."""

import re
from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Result, Rewriter

# Instruction patterns to detect and remove
INSTRUCTION_PATTERNS = [
    r"Answer concisely\.?\s*",
    r"Answer briefly\.?\s*",
    r"Be concise\.?\s*",
    r"Provide a short answer\.?\s*",
    r"Keep your answer brief\.?\s*",
]


@OPERATORS.register_module()
class NormMultiTurnPromptRewriter(Rewriter):
    """
    Normalize instruction patterns in multi-turn conversations.

    If the first turn contains instruction patterns (like "Answer concisely"),
    removes similar patterns from subsequent turns to avoid redundancy.
    """

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
        self._patterns = [re.compile(p, re.IGNORECASE) for p in INSTRUCTION_PATTERNS]

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def _has_patterns(self, text: str) -> bool:
        """Check if text contains instruction patterns."""
        for pattern in self._patterns:
            if pattern.search(text):
                return True
        return False

    def _remove_patterns(self, text: str) -> Tuple[str, bool]:
        """Remove instruction patterns from text."""
        result = text
        changed = False
        for pattern in self._patterns:
            new_result = pattern.sub("", result)
            if new_result != result:
                changed = True
                result = new_result
        return result.strip(), changed

    def process(self, item: DataItem) -> Result:
        """Remove redundant instruction patterns from multi-turn (item-level)."""
        result = Result(item_idx=item.idx)

        # Only process multi-turn
        if item.qa_count <= 1:
            return result

        # Check if first turn has instruction patterns
        first_question = item.get_question(0)
        if not self._has_patterns(first_question):
            return result

        # Process subsequent turns (skip first)
        for qa in item.qa_pairs[1:]:
            cleaned, changed = self._remove_patterns(qa.question)
            if changed:
                result.add_rewrite(
                    qa_idx=qa.idx,
                    new_question=cleaned,
                    message="removed redundant instruction patterns",
                )

        return result

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Not used - item-level rewriter."""
        return None
