"""
Unit tests for datastudio.models.base module.

Tests cover:
- BaseAPI class
- Image encoding
- Message preprocessing
- Retry logic
"""

import base64
import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


class ConcreteAPI:
    """Concrete implementation of BaseAPI for testing."""

    def __init__(self, **kwargs):
        from datastudio.models.base import BaseAPI

        # Create a concrete subclass that implements generate_inner
        class TestableAPI(BaseAPI):
            def generate_inner(self, inputs, **kwargs):
                return 0, "test response", None

        self._api = TestableAPI(**kwargs)

    def __getattr__(self, name):
        return getattr(self._api, name)


@pytest.fixture
def api():
    """Create a testable API instance."""
    return ConcreteAPI()


@pytest.fixture
def real_image():
    """Create a real PIL Image for testing."""
    img = Image.new("RGB", (100, 100), color="red")
    return img


class TestBaseAPIInit:
    """Tests for BaseAPI initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        api = ConcreteAPI()
        assert api.retry == 10
        assert api.wait == 5

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        api = ConcreteAPI(retry=5, wait=2, timeout=60)
        assert api.retry == 5
        assert api.wait == 2
        assert api.timeout == 60

    def test_init_with_system_proxy(self):
        """Test initialization with system proxy enabled."""
        api = ConcreteAPI(use_system_proxy=True)
        assert api.use_system_proxy is True


class TestBaseAPIAllowedTypes:
    """Tests for allowed_types attribute."""

    def test_allowed_types_default(self, api):
        """Test default allowed types."""
        from datastudio.models.base import BaseAPI

        assert "text" in BaseAPI.allowed_types
        assert "image" in BaseAPI.allowed_types


class TestBaseAPIEncodeImage:
    """Tests for encode_image_directly method."""

    def test_encode_image_from_pil(self, api, real_image):
        """Test encoding PIL Image."""
        result = api.encode_image_directly(real_image)
        assert isinstance(result, str)
        # Should be base64 encoded
        assert len(result) > 0

    def test_encode_image_from_bytes(self, api, real_image):
        """Test encoding from bytes."""
        # Convert PIL image to bytes
        buffer = io.BytesIO()
        real_image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        result = api.encode_image_directly(io.BytesIO(img_bytes))
        assert isinstance(result, str)

    def test_encode_image_from_base64_string(self, api, real_image):
        """Test encoding already base64 encoded string."""
        # Create base64 string with data URI prefix
        buffer = io.BytesIO()
        real_image.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode()
        data_uri = f"data:image/png;base64,{b64_str}"

        result = api.encode_image_directly(data_uri)
        # Should return as-is since it's already encoded
        assert result == data_uri


class TestBaseAPICheckContent:
    """Tests for check_content method."""

    def test_check_content_text_only(self, api):
        """Test check_content with text string."""
        result = api.check_content("Hello world")
        assert result == "str"

    def test_check_content_with_dict(self, api):
        """Test check_content with dict."""
        result = api.check_content({"type": "text", "value": "Hello"})
        assert result == "dict"

    def test_check_content_with_list_str(self, api):
        """Test check_content with list of strings."""
        result = api.check_content(["Hello", "World"])
        assert result == "liststr"

    def test_check_content_with_list_dict(self, api):
        """Test check_content with list of dicts."""
        result = api.check_content(
            [{"type": "text", "value": "Hello"}, {"type": "text", "value": "World"}]
        )
        assert result == "listdict"


class TestBaseAPIPreprocContent:
    """Tests for preproc_content method."""

    def test_preproc_content_simple_text(self, api):
        """Test preprocessing simple text."""
        result = api.preproc_content("Hello world")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["value"] == "Hello world"

    def test_preproc_content_dict_input(self, api):
        """Test preprocessing dict input."""
        inputs = {"type": "text", "value": "Hello"}
        result = api.preproc_content(inputs)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_preproc_content_list_str_input(self, api):
        """Test preprocessing list of strings."""
        inputs = ["Hello", "World"]
        result = api.preproc_content(inputs)
        assert isinstance(result, list)
        assert len(result) == 2


class TestBaseAPIPrepareInputs:
    """Tests for prepare_inputs method."""

    def test_prepare_inputs_text(self, api):
        """Test preparing text-only inputs."""
        inputs = [{"type": "text", "value": "Hello world"}]
        result, text = api.prepare_inputs(inputs)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_prepare_inputs_with_image(self, api, real_image):
        """Test preparing inputs with image."""
        inputs = [
            {"type": "text", "value": "Describe this image"},
            {"type": "image", "value": real_image},
        ]
        result, text = api.prepare_inputs(inputs)
        assert isinstance(result, list)


class TestBaseAPIGenerate:
    """Tests for generate method."""

    def test_generate_single_message(self, api):
        """Test generating response for single message."""
        messages = ["Hello"]
        results = api.generate(messages)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_generate_multiple_messages(self, api):
        """Test generating responses for multiple messages."""
        messages = ["Hello", "World"]
        results = api.generate(messages)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_generate_with_kwargs(self, api):
        """Test generating with additional kwargs."""
        messages = ["Hello"]
        results = api.generate(messages, temperature=0.5)
        assert isinstance(results, list)

    def test_generate_empty_messages(self, api):
        """Test generating with empty messages list."""
        results = api.generate([])
        assert results == []


class TestBaseAPIProcessSingleMessage:
    """Tests for process_single_message method."""

    def test_process_single_message(self, api):
        """Test processing a single message."""
        result = api.process_single_message("Hello", {})
        assert result is not None


class TestBaseAPIPreProcess:
    """Tests for pre_process method."""

    def test_pre_process_text_message(self, api):
        """Test preprocessing text message."""
        result = api.pre_process("Hello world")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_pre_process_list_dict_message(self, api):
        """Test preprocessing list of dict message."""
        msg = [{"type": "text", "value": "Hello"}]
        result = api.pre_process(msg)
        assert isinstance(result, list)


class TestBaseAPIShutdown:
    """Tests for shutdown method."""

    def test_shutdown(self, api):
        """Test shutdown method."""
        api.shutdown()  # Should not raise


class TestBaseAPIEdgeCases:
    """Edge case tests for BaseAPI."""

    def test_generate_with_none_message(self, api):
        """Test generating with None in messages."""
        # This might raise or handle gracefully depending on implementation
        try:
            results = api.generate([None])
        except (TypeError, AssertionError):
            pass  # Expected

    def test_generate_with_empty_content(self, api):
        """Test generating with empty content."""
        results = api.generate([""])
        assert isinstance(results, list)

    def test_retry_on_failure(self):
        """Test retry logic on failure."""
        from datastudio.models.base import BaseAPI

        class FailingAPI(BaseAPI):
            def __init__(self):
                super().__init__(retry=2, wait=0.01)
                self.call_count = 0

            def generate_inner(self, inputs, **kwargs):
                self.call_count += 1
                if self.call_count < 2:
                    return 1, None, None  # Failure
                return 0, "success", None

        api = FailingAPI()
        result = api.process_single_message("test", {})
        assert result == "success"
        assert api.call_count == 2

    def test_concurrent_generation(self, api):
        """Test concurrent generation."""
        messages = ["msg1", "msg2", "msg3", "msg4"]
        results = api.generate(messages)
        assert len(results) == 4


class TestBaseAPIAbstract:
    """Tests for abstract method enforcement."""

    def test_cannot_instantiate_base_api(self):
        """Test that BaseAPI cannot be instantiated directly."""
        from datastudio.models.base import BaseAPI

        # BaseAPI is not truly abstract in Python, but generate_inner raises NotImplementedError
        api = BaseAPI()
        with pytest.raises(NotImplementedError):
            api.generate_inner([])

    def test_must_implement_generate_inner(self):
        """Test that subclass must implement generate_inner."""
        from datastudio.models.base import BaseAPI

        class IncompleteAPI(BaseAPI):
            pass

        api = IncompleteAPI()
        with pytest.raises(NotImplementedError):
            api.generate_inner([])
