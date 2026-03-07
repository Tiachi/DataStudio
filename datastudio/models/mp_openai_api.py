"""Multiprocessing OpenAI-compatible API wrapper for large-scale workloads.

Wraps :class:`OpenAIAPI` with ``fork``-based multiprocessing so that
CPU-bound image encoding and JSON serialization run in true parallelism
across cores, each worker owning an independent async event loop.

Memory is shared via fork + Copy-on-Write: the main process stores
pre-encoded messages in a module-level global, and forked workers read
their slices without copying (N workers ≈ 1x memory).
"""

import math
import multiprocessing as mp
import os
from typing import Any, List, Optional

from tqdm import tqdm

from datastudio.utils.registry import MODELS

from .base import BaseAPI
from .openai_api import OpenAIAPI

# Module-level globals shared with forked workers via Copy-on-Write.
# Workers read these without modification, so physical pages stay shared.
_SHARED_MESSAGES: Optional[List[Any]] = None
_SHARED_MODEL_KWARGS: Optional[dict] = None


def _init_shared_data(messages: List[Any], model_kwargs: dict):
    """Set module-level globals before forking workers.

    Must be called in the main process before Pool creation.

    Args:
        messages: The full pre-encoded messages list.
        model_kwargs: Dict of OpenAIAPI constructor arguments.
    """
    global _SHARED_MESSAGES, _SHARED_MODEL_KWARGS
    _SHARED_MESSAGES = messages
    _SHARED_MODEL_KWARGS = model_kwargs


def _clear_shared_data():
    """Clear module-level globals after workers finish."""
    global _SHARED_MESSAGES, _SHARED_MODEL_KWARGS
    _SHARED_MESSAGES = None
    _SHARED_MODEL_KWARGS = None


def _worker_fn(args):
    """Worker function executed in each forked subprocess.

    Reads its message slice from the inherited ``_SHARED_MESSAGES`` global
    (zero-copy via COW) and runs an independent :class:`OpenAIAPI` instance.

    Args:
        args: Tuple of (worker_id, start_idx, end_idx, worker_concurrency).

    Returns:
        List of response strings for the slice ``[start_idx, end_idx)``.
    """
    worker_id, start_idx, end_idx, worker_concurrency = args

    if start_idx >= end_idx:
        return []

    # Read slice from inherited global — no copy, COW shared pages
    messages_slice = _SHARED_MESSAGES[start_idx:end_idx]

    # Build per-worker OpenAIAPI from inherited kwargs
    model_kwargs = dict(_SHARED_MODEL_KWARGS)
    model_kwargs["thread_num"] = worker_concurrency
    # Suppress per-worker logging to avoid noisy output
    model_kwargs.pop("logger", None)

    api = OpenAIAPI(**model_kwargs)
    results = api.generate(messages_slice)
    api.shutdown()
    return results


@MODELS.register_module()
class MPOpenAIAPI(BaseAPI):
    """Multiprocessing OpenAI-compatible API using fork + COW shared memory.

    Distributes requests across multiple worker processes, each running an
    independent :class:`OpenAIAPI` with its own async event loop.

    Designed for workloads where single-process async saturates CPU on
    image encoding / JSON serialization, total request count is large,
    and payload size makes pickle-based IPC infeasible.

    Memory model:
        Pre-encoded messages are stored in a module-level global before
        ``fork()``.  Workers inherit the parent's address space via
        Copy-on-Write and only read shared data, so N workers ≈ 1x memory.
    """

    def __init__(
        self,
        model: str,
        key: str = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 1.0,
        repetition_penalty: float = 1.0,
        api_base: str = None,
        port: int = None,
        retry: int = 10,
        wait: int = 3,
        timeout: tuple = (30, 1800),
        max_tokens: int = 16384,
        thread_num: int = 8192,
        return_dict: bool = False,
        logger=None,
        max_connections: int = None,
        num_workers: int = None,
        worker_concurrency: int = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        """Initialize multiprocessing OpenAI API wrapper.

        Args:
            model: Model name, e.g., gpt-4, Qwen2-VL-72B.
            key: API key (uses OPENAI_API_KEY env var if not provided).
            temperature: Generation temperature.
            top_p: Nucleus sampling threshold (0~1).
            top_k: Top-K sampling (0 to disable). vLLM/SGLang extra param.
            min_p: Minimum probability threshold. vLLM/SGLang extra param.
            presence_penalty: Penalize tokens already present in the output.
            repetition_penalty: Penalize repeated tokens. vLLM/SGLang extra param.
            api_base: API base URL (full URL to chat/completions endpoint).
            port: Port number for local deployments.
            retry: Number of retry attempts on API failure.
            wait: Max wait time between retries (seconds).
            timeout: Request timeout as (connect_timeout, read_timeout) tuple.
            max_tokens: Maximum tokens in response.
            thread_num: Total maximum concurrent requests across all workers.
            return_dict: Whether to parse response as dict.
            logger: Logger instance.
            max_connections: Max TCP connections per worker. Defaults to
                min(worker_concurrency, 16384).
            num_workers: Number of worker processes. Defaults to
                min(cpu_count, 8). Set based on available CPU cores.
            worker_concurrency: Async concurrency limit per worker. Defaults to
                thread_num // num_workers.
            enable_thinking: Whether to enable thinking mode for models that support it
                (e.g., Qwen3 on vLLM/SGLang). Default False.
            **kwargs: Additional API parameters passed to payload.
        """
        super().__init__(
            retry=retry,
            wait=wait,
            timeout=timeout,
            logger=logger,
            openai_format=True,
            image_key_name="image_url",
            thread_num=thread_num,
            **kwargs,
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.return_dict = return_dict
        self.enable_thinking = enable_thinking

        # Resolve API key early so workers don't need env vars
        self.key = key or os.environ.get("OPENAI_API_KEY", "")

        # Store API URL construction params
        self.api_base = api_base
        self.port = port

        # Build resolved API URL (same logic as OpenAIAPI)
        if not api_base:
            self._resolved_api_base = os.environ.get("OPENAI_API_BASE", "")
        else:
            if port is not None:
                self._resolved_api_base = f"{api_base}:{port}/v1/chat/completions"
            else:
                self._resolved_api_base = api_base

        # Multiprocessing configuration
        cpu_count = os.cpu_count() or 4
        self.num_workers = num_workers or min(cpu_count, 8)

        # Per-worker concurrency: split total concurrency across workers
        if worker_concurrency is not None:
            self.worker_concurrency = worker_concurrency
        else:
            self.worker_concurrency = max(1, thread_num // self.num_workers)

        # Per-worker max connections
        if max_connections is not None:
            self.worker_max_connections = max_connections
        else:
            self.worker_max_connections = min(self.worker_concurrency, 16384)

        if logger:
            logger.info(
                f"MPOpenAIAPI initialized: "
                f"workers={self.num_workers}, "
                f"concurrency_per_worker={self.worker_concurrency}, "
                f"total_concurrency={self.num_workers * self.worker_concurrency}"
            )
            logger.info(f"Using API URL: {self._resolved_api_base}")

    def _build_model_kwargs(self) -> dict:
        """Build kwargs dict to reconstruct OpenAIAPI in worker processes.

        Returns:
            Dict of constructor arguments for OpenAIAPI.
        """
        kwargs = {
            "model": self.model,
            "key": self.key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "repetition_penalty": self.repetition_penalty,
            "api_base": self._resolved_api_base,
            "retry": self.retry,
            "wait": self.wait,
            "timeout": self.timeout,
            "max_tokens": self.max_tokens,
            "thread_num": self.worker_concurrency,
            "return_dict": self.return_dict,
            "max_connections": self.worker_max_connections,
            "enable_thinking": self.enable_thinking,
        }
        kwargs.update(self.default_kwargs)
        return kwargs

    def generate(self, messages: List[Any], **kwargs) -> List[str]:
        """Generate responses using fork-based multiprocessing.

        Args:
            messages: List of input messages.
            **kwargs: Additional generation arguments.

        Returns:
            List of response strings, in same order as input.
        """
        if not messages:
            return []

        total = len(messages)

        # Pre-encode PIL images to base64 so they are fork-safe and COW-shared
        if self.logger:
            self.logger.info(f"Pre-encoding images for {total} messages...")
        self._pre_encode_images(messages)

        # Store in module-level global for COW sharing after fork
        model_kwargs = self._build_model_kwargs()
        model_kwargs.update(kwargs)
        _init_shared_data(messages, model_kwargs)

        # Compute index ranges for each worker
        num_workers = min(self.num_workers, total)
        chunk_size = math.ceil(total / num_workers)

        worker_args = []
        for worker_id in range(num_workers):
            start = worker_id * chunk_size
            end = min(start + chunk_size, total)
            worker_args.append((worker_id, start, end, self.worker_concurrency))

        if self.logger:
            self.logger.info(
                f"Forking {num_workers} workers for {total} messages "
                f"(~{chunk_size}/worker, concurrency={self.worker_concurrency}/worker)"
            )

        # Fork workers (safe: no threads/event loops exist yet in main process)
        ctx = mp.get_context("fork")

        all_results = []
        try:
            with ctx.Pool(processes=num_workers) as pool:
                with tqdm(total=total, desc="MPOpenAIAPI Progress") as pbar:
                    for chunk_results in pool.imap(_worker_fn, worker_args):
                        all_results.extend(chunk_results)
                        pbar.update(len(chunk_results))
        finally:
            # Always clear shared data to release memory
            _clear_shared_data()

        if self.logger:
            fail_count = sum(1 for r in all_results if r == self.fail_msg)
            self.logger.info(
                f"Completed: {total} requests, "
                f"{total - fail_count} succeeded, {fail_count} failed"
            )

        return all_results

    def generate_inner(self, inputs, **kwargs):
        """Sync version for single request (BaseAPI compatibility).

        Delegates to a temporary OpenAIAPI instance.

        Args:
            inputs: Preprocessed message list (OpenAI format).
            **kwargs: Additional parameters.

        Returns:
            Tuple of (ret_code, answer, response).
        """
        model_kwargs = self._build_model_kwargs()
        api = OpenAIAPI(**model_kwargs)
        return api.generate_inner(inputs, **kwargs)

    def shutdown(self):
        """Shutdown and release resources.

        Workers are ephemeral (created per generate() call), nothing to clean up.
        """
        pass
