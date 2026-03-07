"""
Unit tests for datastudio.models.openai_api module.

Tests cover:
- OpenAIAPI initialization and configuration
- Payload building
- Response parsing
- Async request handling (mocked aiohttp)
- Retry logic with exponential backoff
- Sync generate_inner
- Batch generate (async pipeline)
- Edge cases and error handling
"""

import asyncio
import json
import pickle
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from datastudio.models.openai_api import OpenAIAPI

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api():
    """Create an OpenAIAPI instance with test defaults."""
    return OpenAIAPI(
        model="test-model",
        key="test-key",
        api_base="http://localhost:8000/v1/chat/completions",
        retry=3,
        wait=0.01,
        timeout=(5, 30),
        max_tokens=1024,
        thread_num=4,
    )


@pytest.fixture
def api_with_port():
    """Create an OpenAIAPI instance using api_base + port pattern."""
    return OpenAIAPI(
        model="test-model",
        key="test-key",
        api_base="http://localhost",
        port=8000,
        retry=2,
        wait=0.01,
        timeout=(5, 30),
    )


@pytest.fixture
def api_return_dict():
    """Create an OpenAIAPI instance with return_dict=True."""
    return OpenAIAPI(
        model="test-model",
        key="test-key",
        api_base="http://localhost:8000/v1/chat/completions",
        return_dict=True,
        retry=2,
        wait=0.01,
        timeout=(5, 30),
    )


@pytest.fixture
def mock_openai_success_response():
    """Standard successful OpenAI API response."""
    return {"choices": [{"message": {"content": "Test response content"}}]}


@pytest.fixture
def mock_openai_reasoning_response():
    """OpenAI API response with reasoning_content."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Final answer",
                    "reasoning_content": "Step 1: think\nStep 2: reason",
                }
            }
        ]
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestOpenAIAPIInit:
    """Tests for OpenAIAPI initialization."""

    def test_init_basic(self, api):
        assert api.model == "test-model"
        assert api.key == "test-key"
        assert api.temperature == 0.6
        assert api.max_tokens == 1024
        assert api.concurrency_limit == 4
        assert api.return_dict is False

    def test_init_with_port(self, api_with_port):
        assert "8000" in api_with_port.api_base
        assert api_with_port.api_base == "http://localhost:8000/v1/chat/completions"

    def test_init_api_base_direct(self):
        api = OpenAIAPI(
            model="m",
            api_base="http://custom-endpoint.com/api/chat",
        )
        assert api.api_base == "http://custom-endpoint.com/api/chat"

    def test_init_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            api = OpenAIAPI(model="m", api_base="http://x")
            assert api.key == "env-key"

    def test_init_api_base_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_BASE": "http://env-base"}):
            api = OpenAIAPI(model="m")
            assert api.api_base == "http://env-base"

    def test_init_max_connections_default(self, api):
        assert api.max_connections == min(4, 16384)

    def test_init_max_connections_custom(self):
        api = OpenAIAPI(model="m", api_base="http://x", max_connections=100)
        assert api.max_connections == 100

    def test_init_with_logger(self, mock_logger):
        api = OpenAIAPI(
            model="m",
            api_base="http://localhost",
            port=8000,
            logger=mock_logger,
        )
        assert mock_logger.info.called

    def test_registered_in_models_registry(self):
        from datastudio.utils.registry import MODELS

        assert "OpenAIAPI" in MODELS


# =============================================================================
# Payload Building Tests
# =============================================================================


class TestBuildPayload:
    """Tests for _build_payload method."""

    def test_basic_payload(self, api):
        message = [{"role": "user", "content": "Hello"}]
        payload = api._build_payload(message)
        assert payload["model"] == "test-model"
        assert payload["messages"] == message
        assert payload["temperature"] == 0.6
        assert payload["max_tokens"] == 1024

    def test_payload_override_temperature(self, api):
        message = [{"role": "user", "content": "Hello"}]
        payload = api._build_payload(message, temperature=0.7)
        assert payload["temperature"] == 0.7

    def test_payload_override_max_tokens(self, api):
        message = [{"role": "user", "content": "Hello"}]
        payload = api._build_payload(message, max_tokens=512)
        assert payload["max_tokens"] == 512

    def test_payload_extra_kwargs(self, api):
        message = [{"role": "user", "content": "Hello"}]
        payload = api._build_payload(message, top_p=0.9, frequency_penalty=0.5)
        assert payload["top_p"] == 0.9
        assert payload["frequency_penalty"] == 0.5

    def test_payload_return_dict_mode(self, api_return_dict):
        message = [{"role": "user", "content": "Hello"}]
        payload = api_return_dict._build_payload(message)
        assert payload["response_format"] == {"type": "json_object"}

    def test_payload_no_return_dict(self, api):
        message = [{"role": "user", "content": "Hello"}]
        payload = api._build_payload(message)
        assert "response_format" not in payload


# =============================================================================
# Response Parsing Tests
# =============================================================================


class TestParseResponse:
    """Tests for _parse_openai_response method."""

    def test_parse_normal_response(self, api, mock_openai_success_response):
        result = api._parse_openai_response(mock_openai_success_response)
        assert result == "Test response content"

    def test_parse_reasoning_response(self, api, mock_openai_reasoning_response):
        result = api._parse_openai_response(mock_openai_reasoning_response)
        assert "<think>" in result
        assert "Step 1: think" in result
        assert "Final answer" in result

    def test_parse_empty_choices(self, api):
        data = {"choices": []}
        with pytest.raises(IndexError):
            api._parse_openai_response(data)

    def test_parse_missing_content(self, api):
        data = {"choices": [{"message": {}}]}
        result = api._parse_openai_response(data)
        assert result == api.fail_msg

    def test_parse_return_dict_mode(self, api_return_dict):
        data = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
        result = api_return_dict._parse_openai_response(data)
        assert isinstance(result, dict)
        assert result["key"] == "value"


# =============================================================================
# Retry Delay Tests
# =============================================================================


class TestRetryDelay:
    """Tests for _calculate_retry_delay method."""

    def test_delay_is_bounded(self, api):
        for attempt in range(10):
            delay = api._calculate_retry_delay(attempt)
            assert 0 <= delay <= 2.0

    def test_delay_increases_with_attempts(self, api):
        # Statistical test: average delay should increase
        delays_early = [api._calculate_retry_delay(0) for _ in range(100)]
        delays_late = [api._calculate_retry_delay(5) for _ in range(100)]
        # Max possible delay increases, so average should too
        assert (
            sum(delays_early) / len(delays_early)
            <= sum(delays_late) / len(delays_late) + 0.5
        )


# =============================================================================
# Async Request Tests (mocked aiohttp)
# =============================================================================


class TestAsyncRequest:
    """Tests for async request methods using mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_request_non_streaming_success(
        self, api, mock_openai_success_response
    ):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_success_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        message = [{"role": "user", "content": "Hello"}]
        ret_code, result = await api._request_non_streaming(mock_session, message)
        assert ret_code == 0
        assert result == "Test response content"

    @pytest.mark.asyncio
    async def test_request_non_streaming_http_error(self, api):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        message = [{"role": "user", "content": "Hello"}]
        ret_code, result = await api._request_non_streaming(mock_session, message)
        assert ret_code == 1
        assert "500" in result

    @pytest.mark.asyncio
    async def test_request_non_streaming_timeout(self, api):
        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=asyncio.TimeoutError())

        message = [{"role": "user", "content": "Hello"}]
        # _request_non_streaming does not catch TimeoutError; it propagates
        with pytest.raises(asyncio.TimeoutError):
            await api._request_non_streaming(mock_session, message)

    @pytest.mark.asyncio
    async def test_request_non_streaming_client_error(self, api):
        import aiohttp

        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            side_effect=aiohttp.ClientError("Connection refused")
        )

        message = [{"role": "user", "content": "Hello"}]
        # _request_non_streaming does not catch ClientError; it propagates
        with pytest.raises(aiohttp.ClientError):
            await api._request_non_streaming(mock_session, message)


# =============================================================================
# Process Single Request Tests
# =============================================================================


class TestProcessSingleRequest:
    """Tests for _process_single_request with retry logic."""

    @pytest.mark.asyncio
    async def test_process_single_success(self, api, mock_openai_success_response):
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_openai_success_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(10)
        result = await api._process_single_request(mock_session, semaphore, "Hello", {})
        assert result == "Test response content"

    @pytest.mark.asyncio
    async def test_process_single_retry_then_success(
        self, api, mock_openai_success_response
    ):
        call_count = 0

        mock_fail_response = AsyncMock()
        mock_fail_response.status = 500
        mock_fail_response.text = AsyncMock(return_value="Server Error")
        mock_fail_response.__aenter__ = AsyncMock(return_value=mock_fail_response)
        mock_fail_response.__aexit__ = AsyncMock(return_value=False)

        mock_ok_response = AsyncMock()
        mock_ok_response.status = 200
        mock_ok_response.json = AsyncMock(return_value=mock_openai_success_response)
        mock_ok_response.__aenter__ = AsyncMock(return_value=mock_ok_response)
        mock_ok_response.__aexit__ = AsyncMock(return_value=False)

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return mock_fail_response
            return mock_ok_response

        mock_session = AsyncMock()
        mock_session.post = MagicMock(side_effect=side_effect)

        semaphore = asyncio.Semaphore(10)
        result = await api._process_single_request(mock_session, semaphore, "Hello", {})
        assert result == "Test response content"
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_process_single_all_retries_fail(self, api):
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_response)

        semaphore = asyncio.Semaphore(10)
        result = await api._process_single_request(mock_session, semaphore, "Hello", {})
        assert result == api.fail_msg

    @pytest.mark.asyncio
    async def test_process_single_preprocess_failure(self, api):
        mock_session = AsyncMock()
        semaphore = asyncio.Semaphore(10)

        # None input will fail preprocessing
        result = await api._process_single_request(mock_session, semaphore, None, {})
        assert result == api.fail_msg


# =============================================================================
# Sync generate_inner Tests
# =============================================================================


class TestGenerateInner:
    """Tests for synchronous generate_inner method."""

    def test_generate_inner_success(self, api, mock_openai_success_response):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_openai_success_response
            mock_post.return_value = mock_resp

            message = [{"role": "user", "content": "Hello"}]
            ret_code, answer, response = api.generate_inner(message)
            assert ret_code == 0
            assert answer == "Test response content"

    def test_generate_inner_with_kwargs(self, api, mock_openai_success_response):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_openai_success_response
            mock_post.return_value = mock_resp

            message = [{"role": "user", "content": "Hello"}]
            ret_code, answer, _ = api.generate_inner(
                message, temperature=0.5, max_tokens=100
            )
            assert ret_code == 0

            # Verify the payload sent
            call_kwargs = mock_post.call_args
            sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert sent_payload["temperature"] == 0.5
            assert sent_payload["max_tokens"] == 100

    def test_generate_inner_parse_error(self, api):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.side_effect = Exception("JSON parse error")
            mock_post.return_value = mock_resp

            message = [{"role": "user", "content": "Hello"}]
            ret_code, answer, response = api.generate_inner(message)
            assert ret_code == 1

    def test_generate_inner_return_dict(self, api_return_dict):
        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"result": "ok"}'}}]
            }
            mock_post.return_value = mock_resp

            message = [{"role": "user", "content": "Hello"}]
            ret_code, answer, _ = api_return_dict.generate_inner(message)
            assert ret_code == 0
            assert isinstance(answer, dict)

            # Verify response_format was sent
            call_kwargs = mock_post.call_args
            sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
            assert sent_payload["response_format"] == {"type": "json_object"}


# =============================================================================
# Batch Generate Tests
# =============================================================================


class TestGenerate:
    """Tests for the main generate() method."""

    def test_generate_empty_messages(self, api):
        results = api.generate([])
        assert results == []

    def test_generate_single_message(self, api, mock_openai_success_response):
        with patch.object(api, "_generate_async") as mock_async:
            mock_async.return_value = ["Test response content"]
            results = api.generate(["Hello"])
            assert len(results) == 1
            assert results[0] == "Test response content"

    def test_generate_multiple_messages(self, api, mock_openai_success_response):
        with patch.object(api, "_generate_async") as mock_async:
            mock_async.return_value = ["Response 1", "Response 2", "Response 3"]
            results = api.generate(["msg1", "msg2", "msg3"])
            assert len(results) == 3

    def test_generate_preserves_order(self, api):
        expected = [f"Response {i}" for i in range(5)]
        with patch.object(api, "_generate_async") as mock_async:
            mock_async.return_value = expected
            results = api.generate([f"msg{i}" for i in range(5)])
            assert results == expected

    def test_generate_calls_pre_encode_images(self, api):
        with patch.object(api, "_pre_encode_images") as mock_encode:
            with patch.object(api, "_generate_async", return_value=["ok"]):
                api.generate(["Hello"])
                mock_encode.assert_called_once()


# =============================================================================
# Shutdown Tests
# =============================================================================


class TestShutdown:
    """Tests for shutdown method."""

    def test_shutdown_no_error(self, api):
        api.shutdown()  # Should not raise


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_api_key(self):
        api = OpenAIAPI(model="m", api_base="http://x", key="")
        assert api.key == ""

    def test_very_large_thread_num(self):
        api = OpenAIAPI(model="m", api_base="http://x", thread_num=100000)
        assert api.concurrency_limit == 100000
        assert api.max_connections == 16384  # capped

    def test_custom_kwargs_passed_through(self):
        api = OpenAIAPI(
            model="m",
            api_base="http://x",
            top_p=0.9,
            presence_penalty=0.1,
        )
        assert api.top_p == 0.9
        assert api.presence_penalty == 0.1

    def test_unicode_in_response(self, api):
        data = {"choices": [{"message": {"content": "你好世界 🌍"}}]}
        result = api._parse_openai_response(data)
        assert "你好世界" in result

    def test_very_long_response(self, api):
        long_content = "A" * 100000
        data = {"choices": [{"message": {"content": long_content}}]}
        result = api._parse_openai_response(data)
        assert len(result) == 100000

    def test_reasoning_content_stripped(self, api):
        data = {
            "choices": [
                {
                    "message": {
                        "content": "  answer  ",
                        "reasoning_content": "  thinking  ",
                    }
                }
            ]
        }
        result = api._parse_openai_response(data)
        assert "<think>\nthinking\n</think>" in result
        assert "answer" in result
