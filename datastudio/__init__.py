"""DataStudio - A toolkit for multimodal data processing and filtering.

Provides a config-driven pipeline for filtering and rewriting multimodal
datasets using rule-based and MLLM-powered operators.

Example::

    from datastudio.datasets import StandardDataLoader, StandardDataSaver
    from datastudio.operators import ConvLengthFilter, NormPromptRewriter
    from datastudio.pipelines import Pipeline

    loader = StandardDataLoader(data_root='./data', dataset='train.jsonl', ...)
    pipeline = Pipeline(operations, logger)
    result = pipeline(data_list)
"""

__version__ = "0.1.0"
__author__ = "DataStudio Team"

from . import datasets, models, operators, pipelines, utils

__all__ = [
    "__version__",
    "datasets",
    "models",
    "operators",
    "pipelines",
    "utils",
]
