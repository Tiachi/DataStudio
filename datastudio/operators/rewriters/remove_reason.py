"""Remove Reasoning: prefix rewriter."""

import re
from typing import Any, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Rewriter


@OPERATORS.register_module()
class RemoveReasonRewriter(Rewriter):
    """
    Remove "Reasoning:" prefix from responses.

    Some data may have "Reasoning:" prefixes in responses that
    should be removed for clean output.
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

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """Remove Reasoning: prefix from an answer."""
        qa = item.get_qa(qa_idx)
        if qa.answer.startswith("Reasoning:"):
            return re.sub(r"^Reasoning:\s*", "", qa.answer).strip()
        return None
