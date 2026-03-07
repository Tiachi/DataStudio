"""Split multi-turn conversations rewriter."""

import copy
from typing import Any, Dict, List, Optional

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Operator, Result


@OPERATORS.register_module()
class SplitRewriter(Operator):
    """
    Split multi-turn conversations into single-turn conversations.

    Each QA pair becomes a separate data item, preserving
    all metadata and adjusting indices accordingly.

    Note: This is a special Operator that returns multiple items,
    handled specially by Pipeline.
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

    def process(self, item: DataItem) -> Result:
        """
        Split is handled specially - we mark items for expansion.

        The actual splitting is done in expand_items().
        """
        return Result(item_idx=item.idx)

    def process_batch(self, items: List[DataItem]) -> List[Result]:
        """Process batch - returns keep result for each item."""
        return [Result(item_idx=item.idx) for item in items]

    def expand_items(self, items: List[DataItem]) -> List[DataItem]:
        """
        Expand multi-turn items into single-turn items.

        This should be called directly instead of through Pipeline
        for splitting operations.
        """
        expanded = []

        for item in items:
            if item.qa_count <= 1:
                expanded.append(item)
                continue

            # Split each QA pair into separate item
            for qa_idx in range(item.qa_count):
                new_data = self._extract_single_turn(item.data, qa_idx, item.qa_count)
                expanded.append(DataItem(new_data, idx=len(expanded)))

        return expanded

    def _extract_single_turn(
        self,
        data: Dict,
        qa_idx: int,
        total_qa_count: int,
    ) -> Dict:
        """Extract a single turn from multi-turn data."""
        new_data = copy.deepcopy(data)

        # Extract single conversation pair
        new_data["conversations"] = data["conversations"][qa_idx * 2 : (qa_idx + 1) * 2]

        # Re-index metadata fields
        self._reindex_metadata(new_data, qa_idx)

        # Add split record
        if "rewrite_ops" not in new_data:
            new_data["rewrite_ops"] = {}
        new_data["rewrite_ops"][self.name] = {
            0: f"split from {total_qa_count}-turn conversation (was qa_{qa_idx})"
        }

        return new_data

    def _reindex_metadata(self, data: Dict, target_idx: int):
        """Re-index metadata for extracted single turn."""
        metadata_fields = ["ori_answer", "ori_question", "score"]

        for field in metadata_fields:
            if field not in data or not isinstance(data[field], dict):
                continue

            original = data[field]
            value = original.get(str(target_idx))
            if value is not None:
                data[field] = {"0": value}
            else:
                data[field] = {}

        # Handle nested ops records
        for ops_field in ["filter_ops", "rewrite_ops"]:
            if ops_field not in data or not isinstance(data[ops_field], dict):
                continue

            new_ops = {}
            for op_name, records in data[ops_field].items():
                if isinstance(records, dict):
                    value = records.get(target_idx) or records.get(str(target_idx))
                    if value is not None:
                        new_ops[op_name] = {0: value}
                else:
                    # Non-dict record (e.g., full-item filter reason)
                    new_ops[op_name] = records
            data[ops_field] = new_ops
