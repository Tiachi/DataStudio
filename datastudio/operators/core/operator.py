"""Base classes for all operators.

Operators return :class:`Result` objects; the pipeline applies them
uniformly via :meth:`Result.apply_to`.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from .data_item import DataItem
from .result import Result


class Operator(ABC):
    """Base class for all operators.

    Subclasses implement :meth:`process` for single-item logic, or
    override :meth:`process_batch` for batch-level optimization
    (e.g., MLLM operators).
    """

    def __init__(
        self, name: Optional[str] = None, logger: Optional[Any] = None, **kwargs
    ):
        """
        Initialize the operator.

        Args:
            name: Operator name (defaults to class name).
            logger: Logger instance.
        """
        self._name = name
        self.logger = logger

    @property
    def name(self) -> str:
        """Operator name for records."""
        return self._name or self.__class__.__name__

    @abstractmethod
    def process(self, item: DataItem) -> Result:
        """
        Process a single data item.

        Args:
            item: DataItem to process.

        Returns:
            Result with filter and/or rewrite decisions.
        """
        pass

    def process_batch(self, items: List[DataItem]) -> List[Result]:
        """
        Process a batch of items.

        Default implementation calls process() for each item.
        Override for batch optimization (e.g., MLLM operators).

        Args:
            items: List of DataItems.

        Returns:
            List of Results (same order as input).
        """
        return [self.process(item) for item in items]


class Filter(Operator):
    """
    Base class for filter operators.

    Subclasses implement check(item, qa_idx) for per-QA filtering.

    Example:
        class MyFilter(Filter):
            def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
                if len(item.get_answer(qa_idx)) < 10:
                    return True, "answer too short"
                return False, ""
    """

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """
        Check if a QA pair should be rejected.

        Args:
            item: DataItem to check.
            qa_idx: QA pair index.

        Returns:
            Tuple of (rejected, reason). rejected=True means filter out.
        """
        return False, ""

    def process(self, item: DataItem) -> Result:
        """Process by calling check() for each QA pair."""
        result = Result(item_idx=item.idx)
        for qa in item.qa_pairs:
            rejected, reason = self.check(item, qa.idx)
            if rejected:
                result.add_filter(qa.idx, rejected=True, reason=reason)
        return result


class Rewriter(Operator):
    """
    Base class for rewrite operators.

    Subclasses implement rewrite(item, qa_idx) for per-QA rewriting.

    Example:
        class MyRewriter(Rewriter):
            def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
                answer = item.get_answer(qa_idx)
                stripped = answer.strip()
                return stripped if stripped != answer else None
    """

    def rewrite(self, item: DataItem, qa_idx: int) -> Optional[str]:
        """
        Rewrite a QA pair's answer.

        Args:
            item: DataItem to rewrite.
            qa_idx: QA pair index.

        Returns:
            New answer text, or None if no change.
        """
        return None  # Default: no change

    def process(self, item: DataItem) -> Result:
        """Process by calling rewrite() for each QA pair."""
        result = Result(item_idx=item.idx)
        for qa in item.qa_pairs:
            new_content = self.rewrite(item, qa.idx)
            if new_content is not None:
                result.add_rewrite(qa.idx, new_answer=new_content, message="rewritten")
        return result
