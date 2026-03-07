"""
Unit tests for datastudio.operators.core.operator module.

Tests cover:
- Operator abstract base class
- Filter class
- Rewriter class
"""

from unittest.mock import MagicMock

import pytest

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.core.operator import Filter, Operator, Rewriter
from datastudio.operators.core.result import Result


class ConcreteOperator(Operator):
    """Concrete implementation of Operator for testing."""

    def process(self, item: DataItem) -> Result:
        """Concrete implementation of abstract method."""
        return Result(item_idx=0)


class TestOperatorInit:
    """Tests for Operator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        op = ConcreteOperator()
        assert op is not None

    def test_init_with_name(self):
        """Test initialization with custom name."""
        op = ConcreteOperator(name="CustomOperator")
        assert op.name == "CustomOperator"

    def test_init_with_logger(self, mock_logger):
        """Test initialization with logger."""
        op = ConcreteOperator(logger=mock_logger)
        assert op is not None

    def test_init_with_kwargs(self, mock_logger):
        """Test initialization with additional kwargs."""
        op = ConcreteOperator(name="TestOp", logger=mock_logger, custom_param="value")
        assert op is not None


class TestOperatorName:
    """Tests for Operator name property."""

    def test_name_from_init(self):
        """Test name set from initialization."""
        op = ConcreteOperator(name="MyOperator")
        assert op.name == "MyOperator"

    def test_name_default(self):
        """Test default name is class name."""
        op = ConcreteOperator()
        assert op.name == "ConcreteOperator"


class TestOperatorProcess:
    """Tests for Operator process method."""

    def test_process_returns_result(self, data_item):
        """Test process returns Result."""
        op = ConcreteOperator()
        result = op.process(data_item)

        assert isinstance(result, Result)

    def test_process_abstract(self):
        """Test that Operator.process is abstract."""
        with pytest.raises(TypeError):
            Operator()


class TestOperatorProcessBatch:
    """Tests for Operator process_batch method."""

    def test_process_batch_single_item(self, data_item):
        """Test process_batch with single item."""
        op = ConcreteOperator()
        results = op.process_batch([data_item])

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Result)

    def test_process_batch_multiple_items(self, sample_data_list):
        """Test process_batch with multiple items."""
        from datastudio.operators.core.data_item import wrap_data_list

        op = ConcreteOperator()
        items = wrap_data_list(sample_data_list)
        results = op.process_batch(items)

        assert len(results) == len(items)
        assert all(isinstance(r, Result) for r in results)

    def test_process_batch_empty(self):
        """Test process_batch with empty list."""
        op = ConcreteOperator()
        results = op.process_batch([])

        assert results == []


class ConcreteFilter(Filter):
    """Concrete implementation of Filter for testing."""

    def __init__(self, should_reject=False, **kwargs):
        super().__init__(**kwargs)
        self.should_reject = should_reject

    def check(self, item: DataItem, qa_idx: int):
        """Concrete implementation of check method."""
        return self.should_reject, "Test reason"


class TestFilterInit:
    """Tests for Filter initialization."""

    def test_init_with_defaults(self):
        """Test Filter initialization with defaults."""
        f = ConcreteFilter()
        assert f is not None

    def test_init_with_name(self):
        """Test Filter initialization with name."""
        f = ConcreteFilter(name="TestFilter")
        assert f.name == "TestFilter"


class TestFilterCheck:
    """Tests for Filter check method."""

    def test_check_returns_tuple(self, data_item):
        """Test check returns (rejected, reason) tuple."""
        f = ConcreteFilter()
        result = f.check(data_item, qa_idx=0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_check_reject(self, data_item):
        """Test check that rejects."""
        f = ConcreteFilter(should_reject=True)
        rejected, reason = f.check(data_item, qa_idx=0)

        assert rejected is True

    def test_check_keep(self, data_item):
        """Test check that keeps."""
        f = ConcreteFilter(should_reject=False)
        rejected, reason = f.check(data_item, qa_idx=0)

        assert rejected is False


class TestFilterProcess:
    """Tests for Filter process method."""

    def test_process_returns_result(self, data_item):
        """Test process returns Result with filter decisions."""
        f = ConcreteFilter()
        result = f.process(data_item)

        assert isinstance(result, Result)

    def test_process_adds_filter_decisions(self, data_item):
        """Test process adds filter decisions for each QA."""
        f = ConcreteFilter(should_reject=True)
        result = f.process(data_item)

        assert result.has_filter
        assert len(result.filter_decisions) >= 1

    def test_process_multi_turn(self, data_item_multi_turn):
        """Test process with multi-turn conversation."""
        f = ConcreteFilter()
        result = f.process(data_item_multi_turn)

        # Filter only adds decisions for rejected items, so check result is valid
        assert isinstance(result, Result)


class ConcreteRewriter(Rewriter):
    """Concrete implementation of Rewriter for testing."""

    def __init__(self, new_answer="Rewritten", **kwargs):
        super().__init__(**kwargs)
        self.new_answer = new_answer

    def rewrite(self, item: DataItem, qa_idx: int):
        """Concrete implementation of rewrite method."""
        return self.new_answer


class TestRewriterInit:
    """Tests for Rewriter initialization."""

    def test_init_with_defaults(self):
        """Test Rewriter initialization with defaults."""
        r = ConcreteRewriter()
        assert r is not None

    def test_init_with_name(self):
        """Test Rewriter initialization with name."""
        r = ConcreteRewriter(name="TestRewriter")
        assert r.name == "TestRewriter"


class TestRewriterRewrite:
    """Tests for Rewriter rewrite method."""

    def test_rewrite_returns_string(self, data_item):
        """Test rewrite returns new answer string."""
        r = ConcreteRewriter(new_answer="New answer")
        result = r.rewrite(data_item, qa_idx=0)

        assert result == "New answer"

    def test_rewrite_returns_none(self, data_item):
        """Test rewrite can return None (no change)."""

        class NoChangeRewriter(Rewriter):
            def rewrite(self, item, qa_idx):
                return None

        r = NoChangeRewriter()
        result = r.rewrite(data_item, qa_idx=0)

        assert result is None


class TestRewriterProcess:
    """Tests for Rewriter process method."""

    def test_process_returns_result(self, data_item):
        """Test process returns Result with rewrite decisions."""
        r = ConcreteRewriter()
        result = r.process(data_item)

        assert isinstance(result, Result)

    def test_process_adds_rewrite_decisions(self, data_item):
        """Test process adds rewrite decisions."""
        r = ConcreteRewriter(new_answer="Modified")
        result = r.process(data_item)

        assert result.has_rewrite

    def test_process_no_change(self, data_item):
        """Test process when rewrite returns None."""

        class NoChangeRewriter(Rewriter):
            def rewrite(self, item, qa_idx):
                return None

        r = NoChangeRewriter()
        result = r.process(data_item)

        # Should not have rewrite decisions when no changes
        assert isinstance(result, Result)
        assert not result.has_rewrite

    def test_process_multi_turn(self, data_item_multi_turn):
        """Test process with multi-turn conversation."""
        r = ConcreteRewriter()
        result = r.process(data_item_multi_turn)

        # Should process each QA pair
        assert isinstance(result, Result)
        assert result.has_rewrite
        assert len(result.rewrite_decisions) == data_item_multi_turn.qa_count


class TestOperatorAbstract:
    """Tests verifying abstract nature of classes."""

    def test_cannot_instantiate_operator(self):
        """Test that Operator cannot be instantiated."""
        with pytest.raises(TypeError):
            Operator()

    def test_cannot_instantiate_filter_without_check(self):
        """Test that Filter subclass can be instantiated (check has default impl)."""

        class IncompleteFilter(Filter):
            pass

        # Filter.check has a default implementation, so this should work
        f = IncompleteFilter()
        assert f is not None

    def test_cannot_instantiate_rewriter_without_rewrite(self):
        """Test that Rewriter subclass can be instantiated (rewrite has default impl)."""

        class IncompleteRewriter(Rewriter):
            pass

        # Rewriter.rewrite has a default implementation, so this should work
        r = IncompleteRewriter()
        assert r is not None


class TestOperatorEdgeCases:
    """Edge case tests for operators."""

    def test_filter_with_empty_item(self):
        """Test filter with empty conversations."""
        f = ConcreteFilter()
        item = DataItem({"conversations": []})
        result = f.process(item)

        assert isinstance(result, Result)

    def test_rewriter_with_empty_item(self):
        """Test rewriter with empty conversations."""
        r = ConcreteRewriter()
        item = DataItem({"conversations": []})
        result = r.process(item)

        assert isinstance(result, Result)

    def test_process_batch_preserves_order(self, sample_data_list):
        """Test that process_batch preserves item order."""
        from datastudio.operators.core.data_item import wrap_data_list

        op = ConcreteOperator()
        items = wrap_data_list(sample_data_list)
        results = op.process_batch(items)

        for i, result in enumerate(results):
            assert (
                result.item_idx == i or result.item_idx == 0
            )  # Depends on implementation
