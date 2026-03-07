"""Async OpenAI-compatible API wrapper.

Uses aiohttp with semaphore-based concurrency control and connection pooling
to handle high-throughput batch inference against OpenAI-compatible endpoints.
"""

import asyncio
import os
import time
from typing import Any, List, Optional, Tuple

import aiohttp
import requests
from tqdm.asyncio import tqdm as async_tqdm

from datastudio.utils.registry import MODELS

from .base import BaseAPI


@MODELS.register_module()
class OpenAIAPI(BaseAPI):
    """Async OpenAI-compatible API wrapper.

    Uses a shared :class:`aiohttp.ClientSession` with connection pooling
    and ``asyncio.Semaphore`` for concurrency control.  Each request handles
    retries independently with exponential backoff.
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
        enable_thinking: bool = False,
        **kwargs,
    ):
        """Initialize OpenAI-compatible API wrapper.

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
            thread_num: Maximum number of concurrent requests (semaphore limit).
            return_dict: Whether to parse response as dict via load_str_to_dict.
            logger: Logger instance.
            max_connections: Max TCP connections in the pool. Defaults to
                min(thread_num, 16384).
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
        self.concurrency_limit = thread_num
        self.enable_thinking = enable_thinking

        # Connection pool size
        if max_connections is None:
            self.max_connections = min(thread_num, 16384)
        else:
            self.max_connections = max_connections

        # Get API key
        self.key = key or os.environ.get("OPENAI_API_KEY", "")

        # Build API URL
        if not api_base:
            self.api_base = os.environ.get("OPENAI_API_BASE", "")
        else:
            if port is not None:
                self.api_base = f"{api_base}:{port}/v1/chat/completions"
            else:
                if logger:
                    logger.warning("Port not specified, using api_base directly")
                self.api_base = api_base

        if logger:
            logger.info(f"Using API URL: {self.api_base}")
            logger.info(
                f"Concurrency: {self.concurrency_limit}, "
                f"Connections: {self.max_connections}"
            )

    def _build_openai_headers(self) -> dict:
        """Build HTTP headers using instance API key.

        Overrides BaseAPI._build_openai_headers to use self.key directly.

        Returns:
            Headers dict with Content-Type and optional Authorization.
        """
        return super()._build_openai_headers(self.key)

    # Sampling parameter names managed explicitly (used by _build_payload)
    _SAMPLING_PARAM_KEYS = (
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    )

    def _build_payload(self, message: List[dict], **kwargs) -> dict:
        """Build request payload for chat completions API.

        Args:
            message: Preprocessed message list (OpenAI format).
            **kwargs: Override parameters.

        Returns:
            Request payload dict.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        min_p = kwargs.get("min_p", self.min_p)
        presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
        repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)

        # Filter out known keys to avoid duplicates in payload
        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in self._SAMPLING_PARAM_KEYS
        }

        payload = {
            "model": self.model,
            "messages": message,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            **extra_kwargs,
        }

        # Request structured JSON output when return_dict is enabled
        if self.return_dict:
            payload["response_format"] = {"type": "json_object"}

        # Add thinking mode config (for vLLM/SGLang with Qwen3 etc.)
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {}
        if "enable_thinking" not in payload["chat_template_kwargs"]:
            payload["chat_template_kwargs"]["enable_thinking"] = self.enable_thinking

        return payload

    async def _request_non_streaming(
        self,
        session: aiohttp.ClientSession,
        message: List[dict],
        **kwargs,
    ) -> Tuple[int, Any]:
        """Make a non-streaming API request.

        Args:
            session: Shared aiohttp session.
            message: Preprocessed message list.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (ret_code, response_content).
        """
        headers = self._build_openai_headers()
        payload = self._build_payload(message, **kwargs)

        async with session.post(
            self.api_base, json=payload, headers=headers
        ) as response:
            resp_status = response.status
            if resp_status == 200:
                data = await response.json(content_type=None)
            else:
                error_text = await response.text()

        if resp_status != 200:
            return 1, f"HTTP {resp_status}: {error_text[:200]}"

        # Post-process OUTSIDE connection scope
        answer = self._parse_openai_response(data)
        return 0, answer

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        message: List[dict],
        **kwargs,
    ) -> Tuple[int, Any]:
        """Make an API request, delegating to the non-streaming handler."""
        return await self._request_non_streaming(session, message, **kwargs)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with full jitter.

        Delegates to base class with default max_delay of 2.0 seconds.

        Args:
            attempt: Current retry attempt (0-indexed).

        Returns:
            Delay in seconds.
        """
        return super()._calculate_retry_delay(attempt, max_delay=2.0)

    async def _process_single_request(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        processed_msg: Any,
        all_kwargs: dict,
        pbar: Optional[async_tqdm] = None,
        error_counter: Optional[dict] = None,
    ) -> str:
        """Process a single request with concurrency control and retry.

        Args:
            session: Shared aiohttp session.
            semaphore: Concurrency limiter.
            processed_msg: Preprocessed message (OpenAI format).
            all_kwargs: Additional generation arguments.
            pbar: Optional progress bar.
            error_counter: Optional error tracking dict.

        Returns:
            Response content string or fail_msg.
        """
        ret_code = None
        response_struct = None

        # Per-attempt timeout
        _, read_timeout = self.timeout
        per_attempt_timeout = read_timeout + 60

        connect_timeout, _ = self.timeout

        async with semaphore:
            for i in range(self.retry):
                try:
                    if error_counter is not None:
                        error_counter["active"] += 1
                        pbar.set_postfix(**error_counter)
                    t0 = asyncio.get_event_loop().time()
                    ret_code, response_struct = await asyncio.wait_for(
                        self._make_request(session, processed_msg, **all_kwargs),
                        timeout=per_attempt_timeout,
                    )
                    if error_counter is not None:
                        error_counter["active"] -= 1

                    if ret_code == 0 and response_struct and response_struct != "":
                        if pbar:
                            if error_counter is not None:
                                error_counter["ok"] += 1
                                pbar.set_postfix(**error_counter)
                            pbar.update(1)
                        return response_struct
                    else:
                        raise Exception(f"Invalid response: {response_struct}")

                except aiohttp.ServerTimeoutError:
                    elapsed = asyncio.get_event_loop().time() - t0
                    ret_code = 1
                    if elapsed < connect_timeout + 5:
                        timeout_type = (
                            f"connect timeout (sock_connect={connect_timeout}s)"
                        )
                    else:
                        timeout_type = (
                            f"read timeout (sock_read={per_attempt_timeout - 60}s)"
                        )
                    response_struct = f"aiohttp {timeout_type}"
                    if error_counter is not None:
                        error_counter["active"] -= 1
                        error_counter["timeout"] += 1
                        pbar.set_postfix(**error_counter)
                    if self.logger and i + 1 > self.retry * 2 // 3:
                        self.logger.warning(
                            f"Attempt {i+1}/{self.retry} {timeout_type} "
                            f"(elapsed {elapsed:.1f}s). "
                            f"Server may be overloaded or unreachable."
                        )
                except asyncio.TimeoutError:
                    elapsed = asyncio.get_event_loop().time() - t0
                    ret_code = 1
                    response_struct = "Request timeout"
                    if error_counter is not None:
                        error_counter["active"] -= 1
                        error_counter["timeout"] += 1
                        pbar.set_postfix(**error_counter)
                    if self.logger and i + 1 > self.retry * 2 // 3:
                        self.logger.warning(
                            f"Attempt {i+1}/{self.retry} timed out "
                            f"after {elapsed:.1f}s (limit {per_attempt_timeout}s)"
                        )
                except aiohttp.ClientError as e:
                    ret_code = 1
                    response_struct = f"ClientError: {type(e).__name__}"
                    if error_counter is not None:
                        error_counter["active"] -= 1
                        error_counter["error"] += 1
                        pbar.set_postfix(**error_counter)
                except Exception as e:
                    if error_counter is not None:
                        error_counter["active"] -= 1
                        error_counter["error"] += 1
                        pbar.set_postfix(**error_counter)
                    if i + 1 > self.retry * 2 // 3 and self.logger:
                        self.logger.warning(
                            f"Attempt {i+1}/{self.retry} failed, "
                            f"RetCode: {ret_code}, "
                            f"Error: {type(e).__name__}: {str(e)[:200]}"
                        )

                if i < self.retry - 1:
                    delay = self._calculate_retry_delay(i)
                    await asyncio.sleep(delay)

        if error_counter is not None:
            error_counter["fail"] += 1
            pbar.set_postfix(**error_counter)
        if self.logger:
            self.logger.warning(
                f"Failed after {self.retry} attempts, "
                f"RetCode: {ret_code}, Response: {response_struct}"
            )
        if pbar:
            pbar.update(1)
        return self.fail_msg

    async def _generate_async(
        self,
        processed_messages: List[Any],
        all_kwargs: dict,
    ) -> List[str]:
        """Run async generation over all messages.

        Creates a shared :class:`aiohttp.ClientSession` with connection
        pooling and fires all requests via ``asyncio.gather``.

        Args:
            processed_messages: List of pre-processed messages (OpenAI format).
            all_kwargs: Additional generation arguments.

        Returns:
            List of response strings.
        """
        # Configure timeout
        connect_timeout, read_timeout = self.timeout
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=connect_timeout,
            sock_read=read_timeout,
        )

        # Configure connector with connection pooling
        pool_size = min(self.max_connections, self.concurrency_limit)
        connector = aiohttp.TCPConnector(
            limit=pool_size,
            ttl_dns_cache=300,
            force_close=False,
            keepalive_timeout=300,
        )

        total = len(processed_messages)
        if self.logger:
            self.logger.info(
                f"[OpenAI API] Processing {total} requests with concurrency={self.concurrency_limit}, "
                f"connections={pool_size}"
            )

        error_counter = {"ok": 0, "active": 0, "timeout": 0, "error": 0, "fail": 0}
        pbar = async_tqdm(total=total, desc="Processing API Requests", leave=True)
        pbar.set_postfix(ok=0, active=0, timeout=0, error=0, fail=0)

        # Warm-up: gradually increase concurrency to avoid TCP connection storm
        initial_concurrency = min(64, self.concurrency_limit)
        need_warmup = (
            self.concurrency_limit > initial_concurrency and total > initial_concurrency
        )

        if need_warmup:
            semaphore = asyncio.Semaphore(initial_concurrency)
        else:
            semaphore = asyncio.Semaphore(self.concurrency_limit)

        async with aiohttp.ClientSession(
            timeout=timeout, connector=connector
        ) as session:
            if need_warmup:
                extra_slots = self.concurrency_limit - initial_concurrency

                async def _ramp_up_semaphore():
                    released = 0
                    batch = initial_concurrency
                    while released < extra_slots:
                        await asyncio.sleep(1.0)
                        to_release = min(batch, extra_slots - released)
                        for _ in range(to_release):
                            semaphore.release()
                        released += to_release
                        batch = (
                            min(batch * 2, extra_slots - released)
                            if released < extra_slots
                            else 0
                        )

                ramp_task = asyncio.create_task(_ramp_up_semaphore())

            tasks = [
                self._process_single_request(
                    session, semaphore, msg, all_kwargs, pbar, error_counter
                )
                for msg in processed_messages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            if need_warmup:
                ramp_task.cancel()
                try:
                    await ramp_task
                except asyncio.CancelledError:
                    pass

        pbar.refresh()
        pbar.close()

        # Handle any exceptions that slipped through
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                if self.logger:
                    self.logger.warning(f"Task exception: {r}")
                final_results.append(self.fail_msg)
            elif r is None:
                final_results.append(self.fail_msg)
            else:
                final_results.append(r)

        return final_results

    def generate(self, messages: List[Any], **kwargs) -> List[str]:
        """Main method to generate responses.

        Uses asyncio.run to execute the async generation pipeline.
        Pre-encodes images before async processing.

        Args:
            messages: List of input messages.
            **kwargs: Additional generation arguments.

        Returns:
            List of response strings, in same order as input.
        """
        if not messages:
            return []

        # Phase 1: image pre-encoding
        t_encode = time.time()
        self._pre_encode_images(messages)
        encode_elapsed = time.time() - t_encode

        # Phase 2: request preprocessing
        t_preproc = time.time()
        processed_messages = []
        for msg in messages:
            processed_messages.append(self.pre_process(msg))
        preproc_elapsed = time.time() - t_preproc

        # Phase 3: drop redundant original buffers
        t_cleanup = time.time()
        self._clear_message_buffers(messages)
        cleanup_elapsed = time.time() - t_cleanup

        if self.logger:
            self.logger.info(
                "[OpenAIAPI] prepare phases: "
                f"pre_encode={encode_elapsed:.2f}s, "
                f"preprocess={preproc_elapsed:.2f}s, "
                f"clear_message_buffers={cleanup_elapsed:.2f}s"
            )

        all_kwargs = dict(self.default_kwargs)
        all_kwargs.update(kwargs)

        # Run async pipeline
        try:
            # Check if there's already a running event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g., Jupyter notebook)
            # Use nest_asyncio or create a new thread
            import threading

            self.logger.warning("Running in existing event loop, using thread")
            result = [None]

            def _run():
                result[0] = asyncio.run(
                    self._generate_async(processed_messages, all_kwargs)  # noqa: F821
                )

            thread = threading.Thread(target=_run)
            thread.start()
            thread.join()
            final = result[0]
        else:
            final = asyncio.run(self._generate_async(processed_messages, all_kwargs))

        # Release base64 payloads before next batch
        del processed_messages
        self._collect_memory(trim=True)

        return final

    def generate_inner(self, inputs, **kwargs):
        """Sync version of generate for single request (BaseAPI compatibility).

        Args:
            inputs: Preprocessed message list (OpenAI format).
            **kwargs: Additional parameters.

        Returns:
            Tuple of (ret_code, answer, response).
        """

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        top_k = kwargs.get("top_k", self.top_k)
        min_p = kwargs.get("min_p", self.min_p)
        presence_penalty = kwargs.get("presence_penalty", self.presence_penalty)
        repetition_penalty = kwargs.get("repetition_penalty", self.repetition_penalty)

        headers = self._build_openai_headers()

        extra_kwargs = {
            k: v for k, v in kwargs.items() if k not in self._SAMPLING_PARAM_KEYS
        }

        payload = {
            "model": self.model,
            "messages": inputs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            **extra_kwargs,
        }

        if self.return_dict:
            payload["response_format"] = {"type": "json_object"}

        # Add thinking mode config (for vLLM/SGLang with Qwen3 etc.)
        if "chat_template_kwargs" not in payload:
            payload["chat_template_kwargs"] = {}
        if "enable_thinking" not in payload["chat_template_kwargs"]:
            payload["chat_template_kwargs"]["enable_thinking"] = self.enable_thinking

        response = requests.post(
            self.api_base,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        try:
            answer = self._parse_openai_response(response.json())
        except Exception as e:
            return 1, e, response

        return 0, answer, response

    def shutdown(self):
        """Shutdown and release resources.

        Session is created and destroyed per generate() call.
        """
        pass
