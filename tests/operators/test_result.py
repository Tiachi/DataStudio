"""
Unit tests for datastudio.operators.core.result module.

Tests cover:
- FilterDecision dataclass
- RewriteDecision dataclass
- Result class
"""

import pytest

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.core.result import (
    FilterDecision,
    OperatorResult,
    Result,
    RewriteDecision,
)


class TestFilterDecision:
    """Tests for FilterDecision dataclass."""

    def test_create_filter_decision(self):
        """Test creating FilterDecision."""
        decision = FilterDecision(qa_idx=0, rejected=True, reason="Test reason")
        assert decision.qa_idx == 0
        assert decision.rejected is True
        assert decision.reason == "Test reason"

    def test_filter_decision_global(self):
        """Test FilterDecision with global index (-1)."""
        decision = FilterDecision(qa_idx=-1, rejected=True, reason="Global rejection")
        assert decision.qa_idx == -1

    def test_filter_decision_not_rejected(self):
        """Test FilterDecision for kept item."""
        decision = FilterDecision(qa_idx=0, rejected=False, reason="Passed check")
        assert decision.rejected is False

    def test_filter_decision_is_frozen(self):
        """Test that FilterDecision is immutable."""
        decision = FilterDecision(qa_idx=0, rejected=True, reason="Test")
        with pytest.raises(AttributeError):
            decision.rejected = False

    def test_filter_decision_equality(self):
        """Test FilterDecision equality."""
        d1 = FilterDecision(qa_idx=0, rejected=True, reason="R")
        d2 = FilterDecision(qa_idx=0, rejected=True, reason="R")
        d3 = FilterDecision(qa_idx=1, rejected=True, reason="R")

        assert d1 == d2
        assert d1 != d3


class TestRewriteDecision:
    """Tests for RewriteDecision dataclass."""

    def test_create_rewrite_decision(self):
        """Test creating RewriteDecision."""
        decision = RewriteDecision(
            qa_idx=0, new_question="New Q", new_answer="New A", message="Rewritten"
        )
        assert decision.qa_idx == 0
        assert decision.new_question == "New Q"
        assert decision.new_answer == "New A"
        assert decision.message == "Rewritten"

    def test_rewrite_decision_answer_only(self):
        """Test RewriteDecision with answer only."""
        decision = RewriteDecision(
            qa_idx=0, new_question=None, new_answer="New A", message="Answer rewritten"
        )
        assert decision.new_question is None
        assert decision.new_answer == "New A"

    def test_rewrite_decision_question_only(self):
        """Test RewriteDecision with question only."""
        decision = RewriteDecision(
            qa_idx=0,
            new_question="New Q",
            new_answer=None,
            message="Question rewritten",
        )
        assert decision.new_question == "New Q"
        assert decision.new_answer is None

    def test_rewrite_decision_is_frozen(self):
        """Test that RewriteDecision is immutable."""
        decision = RewriteDecision(qa_idx=0, new_answer="A", message="M")
        with pytest.raises(AttributeError):
            decision.new_answer = "B"

    def test_rewrite_decision_equality(self):
        """Test RewriteDecision equality."""
        d1 = RewriteDecision(qa_idx=0, new_answer="A", message="M")
        d2 = RewriteDecision(qa_idx=0, new_answer="A", message="M")
        d3 = RewriteDecision(qa_idx=0, new_answer="B", message="M")

        assert d1 == d2
        assert d1 != d3


class TestResultInit:
    """Tests for Result initialization."""

    def test_create_result(self):
        """Test creating Result."""
        result = Result(item_idx=0)
        assert result.item_idx == 0
        assert result.filter_decisions == []
        assert result.rewrite_decisions == []

    def test_create_result_with_decisions(self):
        """Test creating Result with existing decisions."""
        filter_d = [FilterDecision(qa_idx=0, rejected=True, reason="R")]
        rewrite_d = [RewriteDecision(qa_idx=0, new_answer="A", message="M")]

        result = Result(
            item_idx=0, filter_decisions=filter_d, rewrite_decisions=rewrite_d
        )
        assert len(result.filter_decisions) == 1
        assert len(result.rewrite_decisions) == 1


class TestResultAddMethods:
    """Tests for Result add methods."""

    @pytest.fixture
    def result(self):
        """Create Result for testing."""
        return Result(item_idx=0)

    def test_add_filter_rejected(self, result):
        """Test add_filter for rejection."""
        result.add_filter(qa_idx=0, rejected=True, reason="Failed check")

        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is True

    def test_add_filter_kept(self, result):
        """Test add_filter for kept item."""
        result.add_filter(qa_idx=0, rejected=False, reason="Passed")

        assert len(result.filter_decisions) == 1
        assert result.filter_decisions[0].rejected is False

    def test_add_filter_multiple(self, result):
        """Test adding multiple filter decisions."""
        result.add_filter(qa_idx=0, rejected=True, reason="R1")
        result.add_filter(qa_idx=1, rejected=False, reason="R2")

        assert len(result.filter_decisions) == 2

    def test_add_rewrite(self, result):
        """Test add_rewrite."""
        result.add_rewrite(qa_idx=0, new_answer="New A", message="Rewritten")

        assert len(result.rewrite_decisions) == 1
        assert result.rewrite_decisions[0].new_answer == "New A"

    def test_add_rewrite_with_question(self, result):
        """Test add_rewrite with new question."""
        result.add_rewrite(
            qa_idx=0, new_question="New Q", new_answer="New A", message="Both rewritten"
        )

        assert result.rewrite_decisions[0].new_question == "New Q"
        assert result.rewrite_decisions[0].new_answer == "New A"

    def test_add_rewrite_multiple(self, result):
        """Test adding multiple rewrite decisions."""
        result.add_rewrite(qa_idx=0, new_answer="A1", message="M1")
        result.add_rewrite(qa_idx=1, new_answer="A2", message="M2")

        assert len(result.rewrite_decisions) == 2


class TestResultHasMethods:
    """Tests for Result has_filter and has_rewrite properties."""

    def test_has_filter_false(self):
        """Test has_filter returns False when empty."""
        result = Result(item_idx=0)
        assert not result.has_filter

    def test_has_filter_true(self):
        """Test has_filter returns True when decisions exist."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=0, rejected=True, reason="R")
        assert result.has_filter

    def test_has_rewrite_false(self):
        """Test has_rewrite returns False when empty."""
        result = Result(item_idx=0)
        assert not result.has_rewrite

    def test_has_rewrite_true(self):
        """Test has_rewrite returns True when decisions exist."""
        result = Result(item_idx=0)
        result.add_rewrite(qa_idx=0, new_answer="A", message="M")
        assert result.has_rewrite


class TestResultApplyTo:
    """Tests for Result apply_to method."""

    @pytest.fixture
    def data_item(self, sample_data_item):
        """Create DataItem for testing."""
        return DataItem(sample_data_item)

    def test_apply_to_no_changes(self, data_item):
        """Test apply_to with no decisions."""
        result = Result(item_idx=0)
        kept, rejected = result.apply_to(data_item, op_name="TestOp")

        assert kept is not None
        assert rejected is None

    def test_apply_to_filter_rejection(self, data_item):
        """Test apply_to with filter rejection."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=0, rejected=True, reason="Failed")

        kept, rejected = result.apply_to(data_item, op_name="TestFilter")

        # Item should be rejected
        assert rejected is not None or (kept is not None and kept.is_rejected)

    def test_apply_to_filter_keep(self, data_item):
        """Test apply_to with filter keep."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=0, rejected=False, reason="Passed")

        kept, rejected = result.apply_to(data_item, op_name="TestFilter")

        assert kept is not None

    def test_apply_to_rewrite(self, data_item):
        """Test apply_to with rewrite."""
        original_answer = data_item.get_answer(0)

        result = Result(item_idx=0)
        result.add_rewrite(qa_idx=0, new_answer="Rewritten answer", message="M")

        kept, rejected = result.apply_to(data_item, op_name="TestRewriter")

        # Answer should be changed
        if kept is not None:
            assert kept.get_answer(0) == "Rewritten answer"

    def test_apply_to_global_rejection(self, data_item):
        """Test apply_to with global rejection (qa_idx=-1)."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=-1, rejected=True, reason="Global rejection")

        kept, rejected = result.apply_to(data_item, op_name="TestFilter")

        # Entire item should be rejected
        assert kept is None
        assert rejected is not None
        assert rejected.is_rejected


class TestResultMultiTurn:
    """Tests for Result with multi-turn conversations."""

    @pytest.fixture
    def multi_turn_item(self, sample_multi_turn_conversation):
        """Create multi-turn DataItem."""
        return DataItem({"conversations": sample_multi_turn_conversation})

    def test_apply_partial_rejection(self, multi_turn_item):
        """Test applying partial rejection to multi-turn."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=0, rejected=True, reason="First turn bad")
        result.add_filter(qa_idx=1, rejected=False, reason="Second turn good")

        kept, rejected = result.apply_to(multi_turn_item, op_name="TestFilter")

        # Should split the item: kept has second turn, rejected has first
        assert kept is not None
        assert rejected is not None
        assert kept.qa_count == 1
        assert rejected.qa_count == 1

    def test_apply_multiple_rewrites(self, multi_turn_item):
        """Test applying multiple rewrites to multi-turn."""
        result = Result(item_idx=0)
        result.add_rewrite(qa_idx=0, new_answer="New A1", message="M1")
        result.add_rewrite(qa_idx=1, new_answer="New A2", message="M2")

        kept, rejected = result.apply_to(multi_turn_item, op_name="TestRewriter")

        if kept is not None:
            assert kept.get_answer(0) == "New A1"
            assert kept.get_answer(1) == "New A2"


class TestOperatorResultAlias:
    """Tests for OperatorResult alias."""

    def test_operator_result_is_result(self):
        """Test that OperatorResult is alias for Result."""
        assert OperatorResult is Result

    def test_create_operator_result(self):
        """Test creating OperatorResult."""
        result = OperatorResult(item_idx=0)
        assert isinstance(result, Result)


class TestResultEdgeCases:
    """Edge case tests for Result."""

    def test_empty_reason(self):
        """Test filter decision with empty reason."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=0, rejected=True, reason="")
        assert result.filter_decisions[0].reason == ""

    def test_empty_message(self):
        """Test rewrite decision with empty message."""
        result = Result(item_idx=0)
        result.add_rewrite(qa_idx=0, new_answer="A", message="")
        assert result.rewrite_decisions[0].message == ""

    def test_both_new_values_none(self):
        """Test rewrite with both new_question and new_answer None."""
        result = Result(item_idx=0)
        result.add_rewrite(
            qa_idx=0, new_question=None, new_answer=None, message="No change"
        )
        # Should handle gracefully

    def test_negative_item_idx(self):
        """Test Result with negative item_idx."""
        result = Result(item_idx=-1)
        assert result.item_idx == -1

    def test_large_qa_idx(self):
        """Test with large qa_idx."""
        result = Result(item_idx=0)
        result.add_filter(qa_idx=1000, rejected=True, reason="R")
        assert result.filter_decisions[0].qa_idx == 1000
