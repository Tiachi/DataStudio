"""Operators for data filtering, rewriting, and MLLM-powered processing.

Submodules:
    filters: Rule-based quality control filters.
    rewriters: Content transformation rewriters.
    mllm: MLLM-powered filter and rewrite operators.
"""

# Core abstractions
from .core import OperatorResult  # Backward compatibility
from .core import (
    QA,
    DataItem,
    Filter,
    FilterDecision,
    Operator,
    Result,
    RewriteDecision,
    Rewriter,
)

# Filters
from .filters import (
    ConvLengthFilter,
    ImageAspectRatioFilter,
    ImageExtFilter,
    ImageSizeFilter,
    LengthAnomalyFilter,
    ResponseTagFilter,
    TextRepeatFilter,
)

# MLLM operators
from .mllm import (
    MLLMFilter,
    MLLMOperator,
    MLLMRewriter,
    RequestBuilder,
    SelectiveMLLMRewriter,
)

# Rewriters
from .rewriters import (
    AddNoThinkRewriter,
    NormImageTagRewriter,
    NormMultiTurnPromptRewriter,
    NormPromptRewriter,
    NormThinkRewriter,
    RemoveAnswerRewriter,
    RemoveReasonRewriter,
    RemoveThinkRewriter,
    SplitRewriter,
)

__all__ = [
    # Core
    "FilterDecision",
    "RewriteDecision",
    "Result",
    "OperatorResult",
    "QA",
    "DataItem",
    "Operator",
    "Filter",
    "Rewriter",
    # Filters
    "ConvLengthFilter",
    "ImageSizeFilter",
    "ImageAspectRatioFilter",
    "ImageExtFilter",
    "LengthAnomalyFilter",
    "ResponseTagFilter",
    "TextRepeatFilter",
    # Rewriters
    "RemoveThinkRewriter",
    "NormThinkRewriter",
    "AddNoThinkRewriter",
    "NormImageTagRewriter",
    "NormPromptRewriter",
    "NormMultiTurnPromptRewriter",
    "RemoveAnswerRewriter",
    "RemoveReasonRewriter",
    "SplitRewriter",
    # MLLM
    "RequestBuilder",
    "MLLMOperator",
    "MLLMFilter",
    "MLLMRewriter",
    "SelectiveMLLMRewriter",
]
