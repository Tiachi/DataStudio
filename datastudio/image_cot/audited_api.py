"""Audited OpenAI-compatible transport, reusing DataStudio's async engine."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from threading import Lock
import time

from PIL import Image

from datastudio.models.openai_api import OpenAIAPI

from .contracts import (
    ConfigurationError,
    GenerationReceipt,
    InvariantError,
    canonical,
    identity,
    normalize_connection,
    require,
    strict_json,
)


class AuditedOpenAIAPI(OpenAIAPI):
    """One active batch per instance; no implicit retries or raw-response logs."""

    def __init__(self, connection, role_config, limits):
        self.connection = normalize_connection(connection)
        self.role_config = role_config
        self.limits = limits
        auth = self.connection["auth"]
        key = (
            os.environ.get(auth.get("key_env", ""), "")
            if auth["mode"] == "env_bearer"
            else ""
        )
        require(auth["mode"] == "none" or key, "api_key_env_missing")
        super().__init__(
            model=self.connection["model"],
            key=key,
            api_base=self.connection["api_base"],
            retry=1,
            timeout=(
                min(30, role_config["read_timeout_s"]),
                role_config["read_timeout_s"],
            ),
            max_tokens=role_config["max_output_tokens"],
            thread_num=limits["max_inflight"],
            logger=None,
        )
        # Upstream falls back to OPENAI_API_KEY even for an explicitly empty key.
        self.key = key
        self._active = Lock()
        self._task_map = {}
        self._received = {}

    def encode_image_directly(self, image):
        # Reuse upstream pre-encoding/dedup; keep text/chart details losslessly.
        if isinstance(image, Image.Image):
            with io.BytesIO() as buffer:
                image.save(buffer, format="PNG")
                return "data:image/png;base64," + base64.b64encode(
                    buffer.getvalue()
                ).decode("ascii")
        require(
            type(image) is str and image.startswith("data:image/"),
            "image_not_predecoded",
        )
        return image

    def _build_payload(self, message, **_kwargs):
        if self.connection["provider_profile"] == "deepseek_official":
            # DeepSeek's system message schema requires a string, while upstream
            # BaseAPI prepares system content as a list of text parts.
            message = [dict(m) for m in message]
            for entry in message:
                if entry.get("role") == "system" and type(entry.get("content")) is list:
                    require(
                        all(
                            p.get("type") == "text" and type(p.get("text")) is str
                            for p in entry["content"]
                        ),
                        "deepseek_system_text_required",
                    )
                    entry["content"] = "\n".join(p["text"] for p in entry["content"])
        return {
            "model": self.model,
            "messages": message,
            "stream": False,
            "max_tokens": self.role_config["max_output_tokens"],
            **self.role_config.get("effective_options", {}),
        }

    async def _request_non_streaming(self, session, message, **kwargs):
        task = self._task_map[id(message)]
        payload = self._build_payload(message)
        encoded = canonical(payload)
        request_digest = identity(payload)
        if len(encoded) > self.limits["max_request_bytes"]:
            receipt = GenerationReceipt(
                task.task_id,
                task.attempt_id,
                "error",
                error="request_body_limit",
                request_digest=request_digest,
            )
            self._received[task.task_id] = receipt
            return 0, receipt
        started = time.monotonic()
        try:
            async with session.post(
                self.api_base, data=encoded, headers=self._build_openai_headers()
            ) as response:
                # Chunked/unknown Content-Length responses need the same hard cap.
                body = bytearray()
                async for block in response.content.iter_chunked(65536):
                    if len(body) + len(block) > self.limits["max_response_bytes"]:
                        receipt = GenerationReceipt(
                            task.task_id,
                            task.attempt_id,
                            "unresolved",
                            http_status=response.status,
                            error="response_body_limit",
                            elapsed_s=time.monotonic() - started,
                            request_digest=request_digest,
                        )
                        self._received[task.task_id] = receipt
                        return 0, receipt
                    body.extend(block)
                try:
                    raw = strict_json(body)
                    require(type(raw) is dict, "response_object_required")
                except ConfigurationError:
                    # Preserve bounded non-JSON responses once, as private evidence.
                    raw = {
                        "_unparsed_body_base64": base64.b64encode(body).decode("ascii")
                    }
                status = "received" if 200 <= response.status < 300 else "error"
                error = None if status == "received" else "http_error"
                if response.status >= 500 or response.status in (408, 429):
                    status, error = "unresolved", "http_execution_unknown"
                receipt = GenerationReceipt(
                    task.task_id,
                    task.attempt_id,
                    status,
                    raw=raw,
                    http_status=response.status,
                    provider_request_id=response.headers.get("x-request-id"),
                    error=error,
                    elapsed_s=time.monotonic() - started,
                    request_digest=request_digest,
                )
                self._received[task.task_id] = receipt
                return 0, receipt
        except (asyncio.TimeoutError, OSError):
            receipt = GenerationReceipt(
                task.task_id,
                task.attempt_id,
                "unresolved",
                error="transport_execution_unknown",
                elapsed_s=time.monotonic() - started,
                request_digest=request_digest,
            )
            self._received[task.task_id] = receipt
            return 0, receipt

    def generate_audited(self, tasks, *, deadline_s=None):
        require(
            self._active.acquire(blocking=False),
            "model_batch_already_active",
            InvariantError,
        )
        messages = []
        deadline = time.monotonic() + min(
            self.role_config["batch_deadline_s"],
            (
                deadline_s
                if deadline_s is not None
                else self.role_config["batch_deadline_s"]
            ),
        )
        try:
            require(
                len({t.task_id for t in tasks}) == len(tasks),
                "duplicate_model_task",
                InvariantError,
            )
            messages = [[dict(part) for part in task.payload] for task in tasks]
            self._pre_encode_images(messages)
            prepared = []
            for task, message in zip(tasks, messages):
                try:
                    processed = self.pre_process(message)
                    self._task_map[id(processed)] = task
                    prepared.append(processed)
                except (ValueError, TypeError, AssertionError, OSError):
                    self._received[task.task_id] = GenerationReceipt(
                        task.task_id,
                        task.attempt_id,
                        "error",
                        error="preprocess_failed_no_dispatch",
                    )

            async def execute():
                try:
                    # Parent method owns the session, semaphore and gather.
                    await asyncio.wait_for(
                        self._generate_async(prepared, {}),
                        timeout=max(0.001, deadline - time.monotonic()),
                    )
                except asyncio.TimeoutError:
                    pass  # Keep any completed receipts; the rest remain unresolved.

            if prepared and time.monotonic() < deadline:
                asyncio.run(execute())
            return [
                self._received.get(task.task_id)
                or GenerationReceipt(
                    task.task_id,
                    task.attempt_id,
                    "unresolved",
                    error="no_durable_response",
                )
                for task in tasks
            ]
        finally:
            self._task_map.clear()
            self._received.clear()
            messages.clear()
            self._active.release()


class MockAPI:
    """Deterministic transport fixture; never evidence of semantic model quality."""

    model = "synthetic-image-cot-mock"

    def generate_audited(self, tasks):
        receipts = []
        for task in tasks:
            if task.operation in {"short", "long"}:
                message = {
                    "content": "The synthetic chart shows two marked groups.\nAnswer: 2"
                }
                if task.operation == "long":
                    message["reasoning_content"] = (
                        "Read the two groups in this synthetic fixture."
                    )
            elif task.operation == "question_check":
                message = {
                    "content": canonical(
                        {
                            "schema_version": "image_cot.question_check/1",
                            "status": "pass",
                            "reason": "synthetic fixture",
                        }
                    ).decode()
                }
            else:
                message = {
                    "content": canonical(
                        {
                            "schema_version": "image_cot.judgment/1",
                            "answer_status": "pass",
                            "visual_status": "pass",
                            "reasoning_status": "pass",
                            "instruction_status": "pass",
                            "critical_errors": [],
                            "uncertainties": [],
                            "checks": [
                                {
                                    "kind": "visual",
                                    "claim": "synthetic groups",
                                    "evidence_source": "image:0",
                                    "finding": "supported",
                                    "reason": "synthetic fixture",
                                }
                            ],
                            "reason": "synthetic fixture; not a real judgment",
                        }
                    ).decode()
                }
            raw = {
                "model": self.model,
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
            receipts.append(
                GenerationReceipt(
                    task.task_id, task.attempt_id, "received", raw=raw, http_status=200
                )
            )
        return receipts
