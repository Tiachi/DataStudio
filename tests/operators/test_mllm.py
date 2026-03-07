"""
Unit tests for datastudio.operators.mllm module.

Tests cover:
- RequestBuilder
- MLLMOperator
- MLLMFilter
- MLLMRewriter
- SelectiveMLLMRewriter
"""

from unittest.mock import MagicMock, patch

import pytest

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.mllm import MLLMFilter, MLLMRewriter, SelectiveMLLMRewriter
from datastudio.operators.mllm.request import RequestBuilder, copy_image


class TestCopyImage:
    """Tests for copy_image function."""

    def test_copy_image_none(self):
        """Test copy_image with None."""
        result = copy_image(None)
        assert result is None

    def test_copy_image_bytes(self):
        """Test copy_image with bytes."""
        img_bytes = b"fake image data"
        result = copy_image(img_bytes)
        assert result == img_bytes

    def test_copy_image_pil(self, mock_image):
        """Test copy_image with PIL Image returns the same reference."""
        result = copy_image(mock_image)
        assert result is mock_image


class TestRequestBuilder:
    """Tests for RequestBuilder class."""

    @pytest.fixture
    def builder_default(self):
        """Create builder with default settings."""
        return RequestBuilder()

    @pytest.fixture
    def builder_custom(self, temp_dir):
        """Create builder with custom settings."""
        import os

        prompt_file = os.path.join(temp_dir, "prompt.txt")
        with open(prompt_file, "w") as f:
            f.write("Test prompt: {content}")

        return RequestBuilder(
            prompt=prompt_file,
            with_image=True,
            with_question=True,
            with_answer=True,
            with_original=False,
        )

    def test_init_default(self, builder_default):
        """Test initialization with defaults."""
        assert builder_default.with_image is True
        assert builder_default.with_question is True
        assert builder_default.with_answer is False

    def test_init_with_prompt_string(self):
        """Test initialization with prompt string."""
        builder = RequestBuilder(prompt="Custom prompt: {content}")
        assert "Custom prompt" in builder.prompt_text

    def test_prompt_text_property(self, builder_default):
        """Test prompt_text property."""
        text = builder_default.prompt_text
        assert isinstance(text, str)

    def test_build_request(self, builder_default, data_item):
        """Test build_request creates request dict."""
        request = builder_default.build_request(data_item, qa_idx=0)
        assert isinstance(request, dict)

    def test_build_request_with_image(self, data_item_with_image):
        """Test build_request includes image."""
        builder = RequestBuilder(with_image=True)
        request = builder.build_request(data_item_with_image, qa_idx=0)
        # Should include image in request

    def test_build_request_copy_img(self, data_item_with_image):
        """Test build_request with copy_img=True."""
        builder = RequestBuilder(with_image=True)
        request = builder.build_request(data_item_with_image, qa_idx=0, copy_img=True)

    def test_build_requests(self, builder_default, data_item):
        """Test build_requests creates list of requests."""
        requests = builder_default.build_requests(data_item)
        assert isinstance(requests, list)
        assert len(requests) == data_item.qa_count

    def test_build_requests_batch_qa(self, builder_default, data_item_multi_turn):
        """Test build_requests with batch_qa=True."""
        requests = builder_default.build_requests(data_item_multi_turn, batch_qa=True)
        # With batch_qa, should return single request

    def test_build_requests_selective(self, builder_default, data_item_multi_turn):
        """Test build_requests_selective."""
        requests = builder_default.build_requests_selective(
            data_item_multi_turn, qa_indices=[0]
        )
        assert len(requests) == 1

    def test_format_content(self, builder_default, data_item):
        """Test format_content creates formatted string."""
        content = builder_default.format_content(data_item, qa_idx=0)
        assert isinstance(content, str)

    def test_parse_response_string(self, builder_default):
        """Test parse_response with string response."""
        response = "yes"
        parsed = builder_default.parse_response(response, qa_idx=0)
        assert isinstance(parsed, dict)

    def test_parse_response_dict(self, builder_default):
        """Test parse_response with dict response."""
        response = {"result": "yes", "reason": "test"}
        parsed = builder_default.parse_response(response, qa_idx=0)
        assert isinstance(parsed, dict)

    def test_parse_response_json_string(self, builder_default):
        """Test parse_response with JSON string response."""
        response = '{"result": "yes", "reason": "test"}'
        parsed = builder_default.parse_response(response, qa_idx=0)
        assert isinstance(parsed, dict)


class TestMLLMFilter:
    """Tests for MLLMFilter class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.generate.return_value = ['{"q0": true, "q0_reason": "good quality"}']
        return model

    @pytest.fixture
    def request_builder_cfg(self):
        """Create request builder config."""
        return {
            "type": "RequestBuilder",
            "prompt": "Evaluate this: {content}",
            "with_image": False,
            "with_question": True,
            "with_answer": True,
        }

    @pytest.fixture
    def filter_instance(self, mock_model, request_builder_cfg):
        """Create MLLMFilter instance."""
        return MLLMFilter(model=mock_model, request_builder=request_builder_cfg)

    def test_init(self, mock_model, request_builder_cfg):
        """Test initialization."""
        f = MLLMFilter(model=mock_model, request_builder=request_builder_cfg)
        assert f is not None

    def test_init_requires_request_builder(self, mock_model):
        """Test initialization requires request_builder."""
        with pytest.raises(ValueError, match="request_builder is required"):
            MLLMFilter(model=mock_model)

    def test_init_requires_model(self, request_builder_cfg):
        """Test initialization requires model."""
        with pytest.raises(ValueError, match="model is required"):
            MLLMFilter(model=None, request_builder=request_builder_cfg)

    def test_process(self, filter_instance, data_item):
        """Test process returns Result."""
        result = filter_instance.process(data_item)
        assert result is not None

    def test_process_batch(self, filter_instance):
        """Test process_batch."""
        from datastudio.operators.core.data_item import wrap_data_list

        items = wrap_data_list(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "Q1"},
                        {"from": "gpt", "value": "A1"},
                    ]
                },
                {
                    "conversations": [
                        {"from": "human", "value": "Q2"},
                        {"from": "gpt", "value": "A2"},
                    ]
                },
            ]
        )

        filter_instance.model.generate.return_value = [
            '{"q0": true, "q0_reason": "good"}',
            '{"q0": false, "q0_reason": "bad"}',
        ]

        results = filter_instance.process_batch(items)
        assert len(results) == len(items)

    def test_add_decision_reject(self, filter_instance):
        """Test _add_decision adds rejection."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        parsed = {"result": True, "reason": "bad quality"}

        filter_instance._add_decision(result, qa_idx=0, parsed=parsed)
        assert result.has_filter

    def test_add_decision_keep(self, filter_instance):
        """Test _add_decision adds keep decision (rejected=False)."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        parsed = {"result": False, "reason": "good"}

        filter_instance._add_decision(result, qa_idx=0, parsed=parsed)
        # has_filter only returns True if there's a rejected decision
        # For keep (rejected=False), has_filter should be False
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is False


class TestMLLMRewriter:
    """Tests for MLLMRewriter class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.generate.return_value = ['{"q0": "Rewritten answer"}']
        return model

    @pytest.fixture
    def request_builder_cfg(self):
        """Create request builder config."""
        return {
            "type": "RequestBuilder",
            "prompt": "Rewrite this: {content}",
            "with_image": False,
            "with_question": True,
            "with_answer": True,
        }

    @pytest.fixture
    def rewriter_instance(self, mock_model, request_builder_cfg):
        """Create MLLMRewriter instance."""
        return MLLMRewriter(model=mock_model, request_builder=request_builder_cfg)

    def test_init(self, mock_model, request_builder_cfg):
        """Test initialization."""
        r = MLLMRewriter(model=mock_model, request_builder=request_builder_cfg)
        assert r is not None

    def test_init_with_rewrite_type(self, mock_model, request_builder_cfg):
        """Test initialization with rewrite_type."""
        r = MLLMRewriter(
            model=mock_model,
            request_builder=request_builder_cfg,
            rewrite_type="question",
        )
        assert r.rewrite_type == "question"

    def test_process(self, rewriter_instance, data_item):
        """Test process returns Result."""
        result = rewriter_instance.process(data_item)
        assert result is not None

    def test_add_decision_answer(self, rewriter_instance):
        """Test _add_decision adds answer rewrite."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        parsed = {"result": "New answer"}

        rewriter_instance._add_decision(result, qa_idx=0, parsed=parsed)
        assert result.has_rewrite


class TestSelectiveMLLMRewriter:
    """Tests for SelectiveMLLMRewriter class."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = MagicMock()
        model.generate.return_value = ['{"q0": "Rewritten"}']
        return model

    @pytest.fixture
    def request_builder_cfg(self):
        """Create request builder config."""
        return {
            "type": "RequestBuilder",
            "prompt": "Rewrite: {content}",
            "with_image": False,
        }

    def test_init_default(self, mock_model, request_builder_cfg):
        """Test initialization with default should_rewrite_fn."""
        r = SelectiveMLLMRewriter(model=mock_model, request_builder=request_builder_cfg)
        assert r is not None

    def test_init_with_custom_fn(self, mock_model, request_builder_cfg):
        """Test initialization with custom should_rewrite_fn."""

        def custom_fn(question, answer):
            return len(answer) < 10

        r = SelectiveMLLMRewriter(
            model=mock_model,
            request_builder=request_builder_cfg,
            should_rewrite_fn=custom_fn,
        )
        assert r.should_rewrite_fn == custom_fn


class TestMLLMOperatorEdgeCases:
    """Edge case tests for MLLM operators."""

    @pytest.fixture
    def request_builder_cfg(self):
        """Create request builder config."""
        return {
            "type": "RequestBuilder",
            "prompt": "Test: {content}",
            "with_image": False,
        }

    def test_filter_model_error(self, request_builder_cfg):
        """Test filter handles model error."""
        model = MagicMock()
        model.generate.side_effect = Exception("Model error")

        f = MLLMFilter(model=model, request_builder=request_builder_cfg)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )

        # Should raise error
        with pytest.raises(Exception):
            f.process(item)

    def test_rewriter_empty_response(self, request_builder_cfg):
        """Test rewriter handles empty response."""
        model = MagicMock()
        model.generate.return_value = [""]

        r = MLLMRewriter(model=model, request_builder=request_builder_cfg)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )

        result = r.process(item)
        # Should handle gracefully

    def test_filter_invalid_json_response(self, request_builder_cfg):
        """Test filter handles invalid JSON response."""
        model = MagicMock()
        model.generate.return_value = ["not valid json"]

        f = MLLMFilter(model=model, request_builder=request_builder_cfg)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )

        result = f.process(item)
        # Should handle gracefully with parse_error


class TestMLLMRegistration:
    """Tests for MLLM operator registration."""

    def test_mllm_filter_registered(self):
        """Test MLLMFilter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "MLLMFilter" in OPERATORS

    def test_mllm_rewriter_registered(self):
        """Test MLLMRewriter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "MLLMRewriter" in OPERATORS

    def test_request_builder_registered(self):
        """Test RequestBuilder is registered."""
        from datastudio.utils.registry import REQUESTS

        assert "RequestBuilder" in REQUESTS


class TestMLLMFilterAddDecision:
    """Detailed tests for MLLMFilter._add_decision."""

    @pytest.fixture
    def filter_instance(self):
        """Create MLLMFilter instance for testing."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Test: {content}",
            "with_image": False,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": true}']
        return MLLMFilter(model=model, request_builder=request_builder_cfg)

    def test_parse_error_defaults_to_keep(self, filter_instance):
        """Test with parse_error=True -> should add keep decision (rejected=False) with reason='parse_error'."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(result, qa_idx=0, parsed={"parse_error": True})
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is False
        assert result.filter_decisions[0].reason == "parse_error"

    def test_result_true_string_rejects(self, filter_instance):
        """Test with result='true' string -> should reject."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": "true", "reason": "bad"}
        )
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is True

    def test_result_yes_string_rejects(self, filter_instance):
        """Test with result='yes' string -> should reject."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": "yes", "reason": "bad"}
        )
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is True

    def test_result_false_string_keeps(self, filter_instance):
        """Test with result='false' string -> should NOT reject."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": "false", "reason": "ok"}
        )
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is False

    def test_result_true_bool_rejects(self, filter_instance):
        """Test with result=True bool -> should reject."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": True, "reason": "bad quality"}
        )
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is True

    def test_result_false_bool_keeps(self, filter_instance):
        """Test with result=False bool -> should NOT reject."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": False, "reason": "good"}
        )
        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is False

    def test_reason_stored_correctly(self, filter_instance):
        """Test reason is stored correctly in the decision."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        filter_instance._add_decision(
            result, qa_idx=0, parsed={"result": True, "reason": "low quality content"}
        )
        assert result.filter_decisions[0].reason == "low quality content"


class TestMLLMRewriterAddDecision:
    """Detailed tests for MLLMRewriter._add_decision."""

    @pytest.fixture
    def rewriter_answer(self):
        """Create MLLMRewriter instance with rewrite_type='answer'."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Rewrite: {content}",
            "with_image": False,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": "Rewritten"}']
        return MLLMRewriter(
            model=model, request_builder=request_builder_cfg, rewrite_type="answer"
        )

    @pytest.fixture
    def rewriter_question(self):
        """Create MLLMRewriter instance with rewrite_type='question'."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Rewrite: {content}",
            "with_image": False,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": "Rewritten"}']
        return MLLMRewriter(
            model=model, request_builder=request_builder_cfg, rewrite_type="question"
        )

    def test_parse_error_no_rewrite(self, rewriter_answer):
        """Test with parse_error=True -> no rewrite added."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        rewriter_answer._add_decision(result, qa_idx=0, parsed={"parse_error": True})
        assert len(result.rewrite_decisions) == 0
        assert result.has_rewrite is False

    def test_result_none_no_rewrite(self, rewriter_answer):
        """Test with result=None -> no rewrite added."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        rewriter_answer._add_decision(result, qa_idx=0, parsed={"result": None})
        assert len(result.rewrite_decisions) == 0
        assert result.has_rewrite is False

    def test_result_answer_rewrite(self, rewriter_answer):
        """Test with result='New answer' and rewrite_type='answer' -> add_rewrite with new_answer."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        rewriter_answer._add_decision(result, qa_idx=0, parsed={"result": "New answer"})
        assert len(result.rewrite_decisions) == 1
        assert result.rewrite_decisions[0].new_answer == "New answer"
        assert result.rewrite_decisions[0].new_question is None
        assert result.has_rewrite is True

    def test_result_question_rewrite(self, rewriter_question):
        """Test with result='New question' and rewrite_type='question' -> add_rewrite with new_question."""
        from datastudio.operators.core.result import Result

        result = Result(item_idx=0)
        rewriter_question._add_decision(
            result, qa_idx=0, parsed={"result": "New question"}
        )
        assert len(result.rewrite_decisions) == 1
        assert result.rewrite_decisions[0].new_question == "New question"
        assert result.rewrite_decisions[0].new_answer is None
        assert result.has_rewrite is True


class TestSelectiveMLLMRewriterFiltering:
    """Tests for SelectiveMLLMRewriter selective logic."""

    @pytest.fixture
    def base_kwargs(self):
        """Base kwargs for creating SelectiveMLLMRewriter."""
        return {
            "model": MagicMock(),
            "request_builder": {
                "type": "RequestBuilder",
                "prompt": "Rewrite: {content}",
                "with_image": False,
            },
        }

    def test_should_rewrite_custom_true(self, base_kwargs):
        """Test _should_rewrite with custom function that returns True."""

        def always_true(question, answer):
            return True

        r = SelectiveMLLMRewriter(should_rewrite_fn=always_true, **base_kwargs)
        assert r._should_rewrite("any question", "any answer") is True

    def test_should_rewrite_custom_false(self, base_kwargs):
        """Test _should_rewrite with custom function that returns False."""

        def always_false(question, answer):
            return False

        r = SelectiveMLLMRewriter(should_rewrite_fn=always_false, **base_kwargs)
        assert r._should_rewrite("any question", "any answer") is False

    def test_should_rewrite_none_always_true(self, base_kwargs):
        """Test _should_rewrite with None function -> always True."""
        r = SelectiveMLLMRewriter(should_rewrite_fn=None, **base_kwargs)
        assert r._should_rewrite("any question", "any answer") is True
        assert r._should_rewrite("", "") is True


class TestMLLMOperatorSkipLogic:
    """Tests for MLLMOperator._should_skip."""

    def test_should_skip_with_image_required_but_missing(self):
        """Test _should_skip returns True when with_image=True but item has no image."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Test: {content}",
            "with_image": True,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": true}']
        f = MLLMFilter(model=model, request_builder=request_builder_cfg)

        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        assert f._should_skip(item) is True

    def test_should_skip_with_image_required_and_present(self):
        """Test _should_skip returns False when with_image=True and item has image."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Test: {content}",
            "with_image": True,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": true}']
        f = MLLMFilter(model=model, request_builder=request_builder_cfg)

        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image": "test.jpg",
            }
        )
        assert f._should_skip(item) is False

    def test_should_skip_no_image_required(self):
        """Test _should_skip returns False when with_image=False regardless of image."""
        request_builder_cfg = {
            "type": "RequestBuilder",
            "prompt": "Test: {content}",
            "with_image": False,
        }
        model = MagicMock()
        model.generate.return_value = ['{"q0": true}']
        f = MLLMFilter(model=model, request_builder=request_builder_cfg)

        # Without image
        item_no_img = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        assert f._should_skip(item_no_img) is False

        # With image
        item_with_img = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image": "test.jpg",
            }
        )
        assert f._should_skip(item_with_img) is False


class TestRequestBuilderKeyTemplates:
    """Tests for RequestBuilder key_templates parsing."""

    def test_parse_response_with_key_templates(self):
        """Test parse_response with key_templates extracts fields correctly."""
        builder = RequestBuilder(
            prompt="Test: {content}",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
        )
        response = {"q0": True, "q0_reason": "good quality"}
        parsed = builder.parse_response(response, qa_idx=0)
        assert parsed["result"] is True
        assert parsed["reason"] == "good quality"

    def test_parse_response_extracts_q0_and_q0_reason(self):
        """Test parse_response extracts q0 and q0_reason correctly from JSON response."""
        builder = RequestBuilder(
            prompt="Test: {content}",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
        )
        response = {"q0": "yes", "q0_reason": "relevant content"}
        parsed = builder.parse_response(response, qa_idx=0)
        assert parsed["result"] == "yes"
        assert parsed["reason"] == "relevant content"

    def test_parse_response_with_qa_idx_1(self):
        """Test parse_response with qa_idx=1 extracts q1 and q1_reason."""
        builder = RequestBuilder(
            prompt="Test: {content}",
            key_templates={"result": "q{idx}", "reason": "q{idx}_reason"},
        )
        response = {"q1": False, "q1_reason": "acceptable answer"}
        parsed = builder.parse_response(response, qa_idx=1)
        assert parsed["result"] is False
        assert parsed["reason"] == "acceptable answer"

    def test_parse_response_none_key_templates_returns_raw(self):
        """Test parse_response with None key_templates returns raw string as result."""
        builder = RequestBuilder(
            prompt="Test: {content}",
            key_templates=None,
        )
        response = "This is a raw text response"
        parsed = builder.parse_response(response, qa_idx=0)
        assert parsed["result"] == "This is a raw text response"
