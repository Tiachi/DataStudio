"""Add empty think tags rewriter."""

from typing import Any, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Rewriter


@OPERATORS.register_module()
class AddNoThinkRewriter(Rewriter):
    """
    Add empty <think> tags to responses that don't have them.

    This ensures consistent format for responses that should have
    think tags but are missing them.
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
        """Add empty think tags to a response missing them."""
        qa = item.get_qa(qa_idx)
        has_think = "<think>" in qa.answer and "</think>" in qa.answer

        if not has_think:
            return "<think>\n\n</think>\n\n" + qa.answer
        return None
