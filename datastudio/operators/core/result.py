"""Operator result types.

Immutable decision objects representing filter and rewrite outcomes.
:class:`Result` bundles decisions and applies them to a :class:`DataItem`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .data_item import DataItem


@dataclass(frozen=True)
class FilterDecision:
    """Filter decision for a QA pair or entire item.

    Attributes:
        qa_idx: QA pair index (0-based), or -1 for a global decision.
        rejected: Whether this QA pair (or entire item) should be rejected.
        reason: Explanation for the decision.
    """

    qa_idx: int
    rejected: bool
    reason: str = ""


@dataclass(frozen=True)
class RewriteDecision:
    """Rewrite decision for a single QA pair.

    Attributes:
        qa_idx: QA pair index (0-based).
        new_question: Rewritten question text (None = no change).
        new_answer: Rewritten answer text (None = no change).
        message: Description of what was changed.
    """

    qa_idx: int
    new_question: Optional[str] = None
    new_answer: Optional[str] = None
    message: str = ""


@dataclass
class Result:
    """
    Operator execution result.

    Responsibilities:
    1. Store decision lists
    2. Apply decisions to DataItem
    3. Manage records

    Can contain both filter and rewrite decisions, allowing a single
    operator (e.g., MLLM) to both filter and rewrite in one pass.
    """

    item_idx: int
    filter_decisions: List[FilterDecision] = field(default_factory=list)
    rewrite_decisions: List[RewriteDecision] = field(default_factory=list)

    # === Add decisions ===

    def add_filter(self, qa_idx: int, rejected: bool, reason: str = ""):
        """Add a filter decision."""
        self.filter_decisions.append(FilterDecision(qa_idx, rejected, reason))

    def add_rewrite(
        self,
        qa_idx: int,
        new_question: Optional[str] = None,
        new_answer: Optional[str] = None,
        message: str = "",
    ):
        """Add a rewrite decision."""
        self.rewrite_decisions.append(
            RewriteDecision(qa_idx, new_question, new_answer, message)
        )

    # === Properties ===

    @property
    def has_filter(self) -> bool:
        """Whether any QA pair is marked for filtering."""
        return any(d.rejected for d in self.filter_decisions)

    @property
    def has_rewrite(self) -> bool:
        """Whether any QA pair has rewrite changes."""
        return any(
            d.new_question is not None or d.new_answer is not None
            for d in self.rewrite_decisions
        )

    # === Apply to DataItem ===

    def apply_to(
        self,
        item: "DataItem",
        op_name: str,
    ) -> Tuple[Optional["DataItem"], Optional["DataItem"]]:
        """
        Apply all decisions to a data item.

        Order: rewrite first, then filter (to preserve rewritten content).

        Args:
            item: DataItem to modify.
            op_name: Operator name for records.

        Returns:
            (kept_item, rejected_item)
        """
        # 1. Apply rewrites
        self._apply_rewrites(item, op_name)

        # 2. Apply filters
        return self._apply_filters(item, op_name)

    def _apply_rewrites(self, item: "DataItem", op_name: str):
        """Apply rewrite decisions to modify item content."""
        for d in self.rewrite_decisions:
            if d.new_answer is not None:
                item.set_answer(d.qa_idx, d.new_answer, save_original=True)
            if d.new_question is not None:
                item.set_question(d.qa_idx, d.new_question, save_original=True)
            if d.message:
                item.add_rewrite_record(op_name, d.qa_idx, d.message)

    def _apply_filters(
        self,
        item: "DataItem",
        op_name: str,
    ) -> Tuple[Optional["DataItem"], Optional["DataItem"]]:
        """
        Apply filter decisions.

        Returns:
            - (item, None): Keep everything
            - (None, item): Reject everything
            - (kept, rejected): Split item
        """
        # No filter decisions = keep everything
        if not self.filter_decisions:
            item.mark_kept()
            return item, None

        # Check for global decision (qa_idx=-1) - rejects entire item
        for d in self.filter_decisions:
            if d.qa_idx == -1 and d.rejected:
                item.mark_rejected()
                item.add_filter_record(op_name, -1, d.reason)
                return None, item

        # Get rejected indices (excluding global decisions)
        rejected_indices = [
            d.qa_idx for d in self.filter_decisions if d.rejected and d.qa_idx >= 0
        ]

        # No QA rejected = keep everything
        if not rejected_indices:
            item.mark_kept()
            # Record keep reasons if any
            for d in self.filter_decisions:
                if d.reason and not d.rejected:
                    item.add_keep_record(op_name, d.qa_idx, d.reason)
            return item, None

        # Build kept indices
        all_indices = set(range(item.qa_count))
        kept_indices = sorted(all_indices - set(rejected_indices))

        # All QA rejected = reject entire item
        if not kept_indices:
            item.mark_rejected()
            for d in self.filter_decisions:
                if d.rejected:
                    item.add_filter_record(op_name, d.qa_idx, d.reason)
            return None, item

        # Partial rejection = split the item
        kept_item, rejected_item = item.split(kept_indices, rejected_indices)

        # Record reasons (with re-indexed qa_idx)
        for d in self.filter_decisions:
            if d.rejected and d.qa_idx in rejected_indices:
                new_idx = rejected_indices.index(d.qa_idx)
                rejected_item.add_filter_record(op_name, new_idx, d.reason)
            elif d.reason and not d.rejected and d.qa_idx in kept_indices:
                new_idx = kept_indices.index(d.qa_idx)
                kept_item.add_keep_record(op_name, new_idx, d.reason)

        return kept_item, rejected_item


# Backward compatibility alias
OperatorResult = Result
