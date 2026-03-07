"""Sub-pipeline for sequential operator execution."""

from typing import Any, Dict, List, Optional

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.core.operator import Operator
from datastudio.operators.core.result import Result
from datastudio.utils.registry import PIPELINES


def wrap_items(datas: List[Dict]) -> List[DataItem]:
    """Wrap raw dicts into DataItems."""
    return [DataItem(d, i) for i, d in enumerate(datas)]


def unwrap_items(items: List[DataItem]) -> List[Dict]:
    """Extract raw dicts from DataItems."""
    return [item.data for item in items]


@PIPELINES.register_module()
class SubPipeline:
    """Execute a sequence of operators on data items.

    Runs each operator in order, splitting items into kept and rejected
    after each step.  Rejected items do not continue to subsequent operators.

    Args:
        operators: List of operators to execute in order.
        name: Optional name for the sub-pipeline.
        priority: Execution priority (lower = earlier).
        logger: Logger instance.

    Example::

        sub_pipeline = SubPipeline([
            ConvLengthFilter(min_length=1, max_length=10),
            RemoveThinkRewriter(),
        ])
        result = sub_pipeline(data_list)
    """

    def __init__(
        self,
        operators: List[Operator],
        name: Optional[str] = None,
        priority: int = 0,
        logger: Optional[Any] = None,
    ):
        self.operators = operators
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.logger = logger

    def __call__(self, datas: List[Dict]) -> List[Dict]:
        """
        Execute the sub-pipeline.

        Args:
            datas: List of raw data dicts to process.

        Returns:
            List of all processed data dicts (both kept and rejected).
            Rejected items have 'rejected=True' in their dict.
        """
        if not datas:
            return datas

        # Wrap raw dicts
        items = wrap_items(datas)
        all_rejected: List[DataItem] = []

        # Execute each operator
        for op in self.operators:
            if not items:
                break

            result = self._execute_operator(op, items)
            # Split for next operator: only non-rejected items continue
            items = [i for i in result if not i.is_rejected]
            all_rejected.extend([i for i in result if i.is_rejected])

        # Unwrap and merge kept + rejected into single list
        return unwrap_items(items) + unwrap_items(all_rejected)

    def _execute_operator(
        self,
        op: Operator,
        items: List[DataItem],
    ) -> List[DataItem]:
        """
        Execute a single operator and apply results.

        Returns:
            List of DataItems (both kept and rejected).
        """
        # Get results (all operators use process_batch, which defaults to calling process())
        results: List[Result] = op.process_batch(items)

        # Validate results length matches items
        if len(results) != len(items):
            if self.logger:
                self.logger.warning(
                    f"Operator {op.name} returned {len(results)} results for {len(items)} items"
                )
            # Pad with empty results if needed
            while len(results) < len(items):
                results.append(Result(item_idx=len(results)))

        # Apply results
        output: List[DataItem] = []
        for item, result in zip(items, results):
            kept_item, rejected_item = result.apply_to(item, op.name)
            if kept_item:
                output.append(kept_item)
            if rejected_item:
                output.append(rejected_item)

        return output
