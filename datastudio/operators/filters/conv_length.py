"""Conversation length filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter, Result


@OPERATORS.register_module()
class ConvLengthFilter(Filter):
    """
    Filter data items based on conversation length (number of QA pairs).

    Example:
        filter = ConvLengthFilter(min_length=1, max_length=10)

        # Check a single item
        result = filter.process(item)

        # Use in pipeline
        pipeline = Pipeline([filter])
        kept, rejected = pipeline(data_list)
    """

    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 36,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            min_length: Minimum number of QA pairs (inclusive).
            max_length: Maximum number of QA pairs (inclusive).
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

        if logger:
            logger.info(
                f"[{self.name}] Initialized: range=[{min_length}, {max_length}]"
            )

    def process(self, item: DataItem) -> Result:
        """
        Check if conversation length is within bounds.

        This is an item-level filter, so we override process() directly.
        """
        qa_count = item.qa_count

        if qa_count < self.min_length:
            result = Result(item_idx=item.idx)
            result.add_filter(
                qa_idx=-1,
                rejected=True,
                reason=f"QA count {qa_count} < min {self.min_length}",
            )
            return result

        if qa_count > self.max_length:
            result = Result(item_idx=item.idx)
            result.add_filter(
                qa_idx=-1,
                rejected=True,
                reason=f"QA count {qa_count} > max {self.max_length}",
            )
            return result

        return Result(item_idx=item.idx)

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Not used - this filter operates at item level."""
        return False, ""
