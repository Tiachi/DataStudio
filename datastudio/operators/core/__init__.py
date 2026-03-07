"""Core abstractions for operators."""

from .data_item import QA, DataItem
from .operator import Filter, Operator, Rewriter
from .result import FilterDecision, OperatorResult, Result, RewriteDecision

__all__ = [
    # Results
    "FilterDecision",
    "RewriteDecision",
    "Result",
    "OperatorResult",  # Backward compatibility alias
    # Data
    "QA",
    "DataItem",
    # Base classes
    "Operator",
    "Filter",
    "Rewriter",
]
