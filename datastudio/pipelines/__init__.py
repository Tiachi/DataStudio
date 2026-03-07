"""Pipeline orchestration for composing sub-pipelines with priority."""

from .pipeline import Pipeline
from .sub_pipeline import SubPipeline, unwrap_items, wrap_items

__all__ = [
    "SubPipeline",
    "Pipeline",
    "wrap_items",
    "unwrap_items",
]
