"""API model backends for MLLM inference.

Provides :class:`OpenAIAPI` (async single-process) and :class:`MPOpenAIAPI`
(multiprocessing) wrappers for OpenAI-compatible endpoints.
"""

from .mp_openai_api import MPOpenAIAPI
from .openai_api import OpenAIAPI

__all__ = ["OpenAIAPI", "MPOpenAIAPI"]
