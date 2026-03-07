"""Response tag filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter


@OPERATORS.register_module()
class ResponseTagFilter(Filter):
    """
    Filter data items that have image/video tags in responses.

    Responses should not contain <image> or <video> tags as these
    indicate potential data quality issues.
    """

    def __init__(
        self,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)

        if logger:
            logger.info(f"[{self.name}] Initialized")

    def _has_tags_in_response(self, answer: str) -> bool:
        """Check if answer contains image/video/audio tags."""
        return "<image>" in answer or "<video>" in answer or "<audio>" in answer

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Check if response contains image/video tags."""
        # Only check if item has images
        if not item.has_image:
            return False, ""

        qa = item.get_qa(qa_idx)
        if self._has_tags_in_response(qa.answer):
            return True, "Response contains image/video/audio tags (should not exist)"

        return False, ""
