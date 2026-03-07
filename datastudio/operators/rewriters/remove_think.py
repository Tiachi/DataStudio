"""Remove <think> tags rewriter."""

import re
from typing import Any, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Rewriter

# Pattern to match <think>...</think> tags
THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@OPERATORS.register_module()
class RemoveThinkRewriter(Rewriter):
    """
    Remove <think>...</think> tags from answers.

    These tags often contain model reasoning that shouldn't be
    in the final output.

    Example:
        rewriter = RemoveThinkRewriter()
        pipeline = Pipeline([rewriter])
        result, _ = pipeline(data_list)
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
        """Remove think tags from an answer."""
        qa = item.get_qa(qa_idx)
        original_answer = qa.answer
        new_answer = THINK_PATTERN.sub("", original_answer).strip()

        # Compare stripped versions to handle whitespace differences
        if new_answer and new_answer != original_answer.strip():
            return new_answer
        return None
