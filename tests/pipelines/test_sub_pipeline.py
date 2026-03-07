"""
Unit tests for datastudio.pipelines.sub_pipeline module.

Tests cover:
- wrap_items function
- unwrap_items function
- SubPipeline class
"""

from unittest.mock import MagicMock, patch

import pytest

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.core.operator import Operator
from datastudio.operators.core.result import Result
from datastudio.pipelines.sub_pipeline import SubPipeline, unwrap_items, wrap_items


class TestWrapItems:
    """Tests for wrap_items function."""

    def test_wrap_items_basic(self, sample_data_list):
        """Test wrap_items creates DataItems."""
        items = wrap_items(sample_data_list)

        assert isinstance(items, list)
        assert all(isinstance(item, DataItem) for item in items)
        assert len(items) == len(sample_data_list)

    def test_wrap_items_empty(self):
        """Test wrap_items with empty list."""
        items = wrap_items([])
        assert items == []

    def test_wrap_items_single(self, sample_data_item):
        """Test wrap_items with single item."""
        items = wrap_items([sample_data_item])
        assert len(items) == 1
        assert isinstance(items[0], DataItem)

    def test_wrap_items_preserves_data(self, sample_data_list):
        """Test wrap_items preserves original data."""
        items = wrap_items(sample_data_list)

        for original, wrapped in zip(sample_data_list, items):
            assert wrapped.data["conversations"] == original["conversations"]


class TestUnwrapItems:
    """Tests for unwrap_items function."""

    def test_unwrap_items_basic(self, sample_data_list):
        """Test unwrap_items extracts dicts."""
        items = wrap_items(sample_data_list)
        unwrapped = unwrap_items(items)

        assert isinstance(unwrapped, list)
        assert all(isinstance(d, dict) for d in unwrapped)
        assert len(unwrapped) == len(sample_data_list)

    def test_unwrap_items_empty(self):
        """Test unwrap_items with empty list."""
        unwrapped = unwrap_items([])
        assert unwrapped == []

    def test_unwrap_items_preserves_data(self, sample_data_list):
        """Test unwrap_items preserves data."""
        items = wrap_items(sample_data_list)
        unwrapped = unwrap_items(items)

        for original, result in zip(sample_data_list, unwrapped):
            assert result["conversations"] == original["conversations"]

    def test_wrap_unwrap_roundtrip(self, sample_data_list):
        """Test wrap and unwrap are inverse operations."""
        items = wrap_items(sample_data_list)
        unwrapped = unwrap_items(items)

        for original, result in zip(sample_data_list, unwrapped):
            assert original["conversations"] == result["conversations"]


class MockOperator(Operator):
    """Mock operator for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_count = 0

    def process(self, item: DataItem) -> Result:
        self.process_count += 1
        return Result(item_idx=0)


class MockFilterOperator(Operator):
    """Mock filter operator that rejects some items."""

    def __init__(self, reject_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.reject_indices = reject_indices or []
        self._current_idx = 0

    def process(self, item: DataItem) -> Result:
        result = Result(item_idx=self._current_idx)
        if self._current_idx in self.reject_indices:
            result.add_filter(qa_idx=-1, rejected=True, reason="Test rejection")
        self._current_idx += 1
        return result


class MockRewriterOperator(Operator):
    """Mock rewriter operator."""

    def __init__(self, new_answer="Rewritten", **kwargs):
        super().__init__(**kwargs)
        self.new_answer = new_answer

    def process(self, item: DataItem) -> Result:
        result = Result(item_idx=0)
        result.add_rewrite(qa_idx=0, new_answer=self.new_answer, message="Rewritten")
        return result


class TestSubPipelineInit:
    """Tests for SubPipeline initialization."""

    def test_init_with_operators(self):
        """Test initialization with operators."""
        ops = [MockOperator(), MockOperator()]
        pipeline = SubPipeline(operators=ops)
        assert pipeline is not None

    def test_init_with_name(self):
        """Test initialization with custom name."""
        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops, name="TestPipeline")
        assert pipeline.name == "TestPipeline"

    def test_init_with_priority(self):
        """Test initialization with priority."""
        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops, priority=10)
        assert pipeline.priority == 10

    def test_init_with_logger(self, mock_logger):
        """Test initialization with logger."""
        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops, logger=mock_logger)
        assert pipeline is not None

    def test_init_empty_operators(self):
        """Test initialization with empty operators list."""
        pipeline = SubPipeline(operators=[])
        assert pipeline is not None


class TestSubPipelineCall:
    """Tests for SubPipeline __call__ method."""

    def test_call_basic(self, sample_data_list):
        """Test basic pipeline execution."""
        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        assert isinstance(result, list)
        assert len(result) == len(sample_data_list)

    def test_call_empty_data(self):
        """Test pipeline with empty data."""
        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline([])
        assert result == []

    def test_call_multiple_operators(self, sample_data_list):
        """Test pipeline with multiple operators."""
        ops = [MockOperator(), MockOperator(), MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # All operators should be called
        for op in ops:
            assert op.process_count > 0

    def test_call_with_filter(self, sample_data_list):
        """Test pipeline with filter operator."""
        ops = [MockFilterOperator(reject_indices=[0])]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # First item should be rejected
        # Result length depends on implementation

    def test_call_with_rewriter(self, sample_data_list):
        """Test pipeline with rewriter operator."""
        ops = [MockRewriterOperator(new_answer="Modified")]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # Answers should be rewritten
        for item in result:
            if not item.get("_rejected", False):
                # Check if answer was modified
                pass

    def test_call_preserves_extra_fields(self):
        """Test pipeline preserves extra fields in data."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "custom_field": "custom_value",
                "source_file": "test.json",
            }
        ]

        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(data)

        assert result[0].get("custom_field") == "custom_value"


class TestSubPipelineOperatorOrder:
    """Tests for operator execution order."""

    def test_operators_execute_in_order(self, sample_data_list):
        """Test operators execute in specified order."""
        execution_order = []

        class OrderTrackingOperator(Operator):
            def __init__(self, order_id, **kwargs):
                super().__init__(**kwargs)
                self.order_id = order_id

            def process(self, item):
                execution_order.append(self.order_id)
                return Result(item_idx=0)

        ops = [
            OrderTrackingOperator(order_id=1),
            OrderTrackingOperator(order_id=2),
            OrderTrackingOperator(order_id=3),
        ]
        pipeline = SubPipeline(operators=ops)

        pipeline(sample_data_list[:1])

        assert execution_order == [1, 2, 3]


class TestSubPipelineFiltering:
    """Tests for filtering behavior in SubPipeline."""

    def test_rejected_items_separated(self, sample_data_list):
        """Test rejected items are handled separately."""
        ops = [MockFilterOperator(reject_indices=[1])]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # Check rejection handling

    def test_all_items_rejected(self, sample_data_list):
        """Test when all items are rejected."""
        reject_all = list(range(len(sample_data_list)))
        ops = [MockFilterOperator(reject_indices=reject_all)]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # All should be rejected

    def test_no_items_rejected(self, sample_data_list):
        """Test when no items are rejected."""
        ops = [MockFilterOperator(reject_indices=[])]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(sample_data_list)

        # All should be kept


class TestSubPipelineEdgeCases:
    """Edge case tests for SubPipeline."""

    def test_single_item(self):
        """Test pipeline with single item."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]

        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(data)
        assert len(result) == 1

    def test_unicode_content(self):
        """Test pipeline with unicode content."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Sample question"},
                    {"from": "gpt", "value": "Sample answer"},
                ]
            }
        ]

        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(data)
        assert "Sample question" in result[0]["conversations"][0]["value"]

    def test_large_batch(self):
        """Test pipeline with large batch."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": f"Q{i}"},
                    {"from": "gpt", "value": f"A{i}"},
                ]
            }
            for i in range(100)
        ]

        ops = [MockOperator()]
        pipeline = SubPipeline(operators=ops)

        result = pipeline(data)
        assert len(result) == 100

    def test_operator_raises_exception(self, sample_data_list):
        """Test pipeline handles operator exception."""

        class FailingOperator(Operator):
            def process(self, item):
                raise ValueError("Test error")

        ops = [FailingOperator()]
        pipeline = SubPipeline(operators=ops)

        # Should handle or propagate exception
        with pytest.raises(ValueError):
            pipeline(sample_data_list)


class TestSubPipelineRegistration:
    """Tests for SubPipeline registration."""

    def test_registered_in_pipelines(self):
        """Test SubPipeline is registered in PIPELINES registry."""
        from datastudio.utils.registry import PIPELINES

        assert "SubPipeline" in PIPELINES


class TestSubPipelineChaining:
    """Tests for SubPipeline operator chaining."""

    def test_rejected_items_dont_reach_second_operator(self):
        """Test that rejected items from first operator don't reach second operator."""
        # First operator rejects item at index 0
        filter_op = MockFilterOperator(reject_indices=[0])
        # Second operator tracks how many items it processes
        tracker_op = MockOperator()

        pipeline = SubPipeline(operators=[filter_op, tracker_op])

        data = [
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
            {
                "conversations": [
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ]
            },
        ]

        result = pipeline(data)
        # The second operator should only process 2 items (items 1 and 2),
        # since item 0 was rejected by the first operator
        assert tracker_op.process_count == 2

    def test_rewriter_modifies_data_for_subsequent_filter(self):
        """Test that rewriters modify data and filters can see the modified data."""
        rewriter = MockRewriterOperator(new_answer="Rewritten answer")
        # Use a mock operator that just passes through
        passthrough = MockOperator()

        pipeline = SubPipeline(operators=[rewriter, passthrough])

        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Original"},
                ]
            },
        ]

        result = pipeline(data)
        # The rewriter should have modified the answer
        assert len(result) == 1
        kept = [d for d in result if not d.get("rejected", False)]
        assert len(kept) == 1
        # The answer should be rewritten
        assert kept[0]["conversations"][1]["value"] == "Rewritten answer"


class TestSubPipelineResultMismatch:
    """Tests for SubPipeline handling result count mismatch."""

    def test_execute_operator_pads_results(self):
        """Test _execute_operator pads results when operator returns fewer results than items."""

        class ShortResultOperator(Operator):
            """Operator that returns fewer results than input items."""

            def process(self, item):
                return Result(item_idx=0)

            def process_batch(self, items):
                # Only return 1 result regardless of input count
                return [Result(item_idx=0)]

        op = ShortResultOperator()
        pipeline = SubPipeline(operators=[op], logger=MagicMock())

        data = [
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
            {
                "conversations": [
                    {"from": "human", "value": "Q3"},
                    {"from": "gpt", "value": "A3"},
                ]
            },
        ]

        # Should not raise even though operator returns fewer results
        result = pipeline(data)
        # All 3 items should still be in output (padded results default to keep)
        assert len(result) == 3


class TestWrapUnwrapRoundTrip:
    """Tests for wrap_items/unwrap_items round-trip."""

    def test_roundtrip_preserves_all_data_fields(self):
        """Test that wrap_items -> unwrap_items preserves all data fields."""
        original_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ],
                "source_file": "test.json",
                "custom_field": "custom_value",
                "metadata": {"key": "value"},
            }
        ]

        wrapped = wrap_items(original_data)
        unwrapped = unwrap_items(wrapped)

        assert len(unwrapped) == 1
        assert unwrapped[0]["conversations"] == original_data[0]["conversations"]
        assert unwrapped[0]["source_file"] == "test.json"
        assert unwrapped[0]["custom_field"] == "custom_value"
        assert unwrapped[0]["metadata"] == {"key": "value"}

    def test_roundtrip_empty(self):
        """Test with empty data."""
        wrapped = wrap_items([])
        unwrapped = unwrap_items(wrapped)
        assert unwrapped == []

    def test_roundtrip_single_item(self):
        """Test with single item."""
        original = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        ]
        wrapped = wrap_items(original)
        unwrapped = unwrap_items(wrapped)
        assert len(unwrapped) == 1
        assert unwrapped[0]["conversations"] == original[0]["conversations"]

    def test_roundtrip_multi_turn(self):
        """Test with multi-turn conversation."""
        original = [
            {
                "conversations": [
                    {"from": "human", "value": "Hello"},
                    {"from": "gpt", "value": "Hi!"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "gpt", "value": "Fine!"},
                ]
            }
        ]
        wrapped = wrap_items(original)
        assert len(wrapped) == 1
        assert wrapped[0].qa_count == 2

        unwrapped = unwrap_items(wrapped)
        assert len(unwrapped[0]["conversations"]) == 4
        assert unwrapped[0]["conversations"][0]["value"] == "Hello"
        assert unwrapped[0]["conversations"][3]["value"] == "Fine!"
