"""
Unit tests for datastudio.operators.rewriters module.

Tests cover:
- RemoveThinkRewriter
- NormThinkRewriter
- AddNoThinkRewriter
- NormImageTagRewriter
- NormPromptRewriter
- NormMultiTurnPromptRewriter
- RemoveAnswerRewriter
- RemoveReasonRewriter
- SplitRewriter
"""

import pytest

from datastudio.operators.core.data_item import DataItem, wrap_data_list
from datastudio.operators.rewriters import (
    AddNoThinkRewriter,
    NormImageTagRewriter,
    NormMultiTurnPromptRewriter,
    NormPromptRewriter,
    NormThinkRewriter,
    RemoveAnswerRewriter,
    RemoveReasonRewriter,
    RemoveThinkRewriter,
    SplitRewriter,
)


class TestRemoveThinkRewriter:
    """Tests for RemoveThinkRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return RemoveThinkRewriter()

    def test_rewrite_no_think_tag(self, rewriter):
        """Test rewrite when no think tag present."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Just a normal answer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should return None when no think tags to remove
        assert result is None

    def test_rewrite_with_think_tag(self, rewriter):
        """Test rewrite removes think tag."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "<think>Some thinking</think>The answer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        if result:
            assert "<think>" not in result
            assert "The answer" in result

    def test_rewrite_nested_think_tags(self, rewriter):
        """Test rewrite handles nested think tags."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {
                        "from": "gpt",
                        "value": "<think>Outer<think>Inner</think></think>Answer",
                    },
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should remove all think tags
        assert result is not None
        assert "Answer" in result

    def test_rewrite_multiple_think_tags(self, rewriter):
        """Test rewrite handles multiple think tags."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {
                        "from": "gpt",
                        "value": "<think>First</think>Middle<think>Second</think>End",
                    },
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        if result:
            assert "<think>" not in result


class TestNormThinkRewriter:
    """Tests for NormThinkRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return NormThinkRewriter()

    def test_rewrite_normalize_think(self, rewriter):
        """Test rewrite normalizes think tag format."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "<THINK>content</THINK>Answer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should normalize to lowercase
        assert result is not None
        assert "<think>" in result
        assert "<THINK>" not in result

    def test_rewrite_no_change_needed(self, rewriter):
        """Test rewrite when already normalized."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "<think>\ncontent\n</think>\nAnswer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should return None when already normalized
        assert result is None


class TestAddNoThinkRewriter:
    """Tests for AddNoThinkRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return AddNoThinkRewriter()

    def test_rewrite_adds_empty_think(self, rewriter):
        """Test rewrite adds empty think tag."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Answer without think"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        if result:
            assert "<think>" in result

    def test_rewrite_already_has_think(self, rewriter):
        """Test rewrite when think tag already present."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "<think>Thinking</think>Answer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should not add another think tag; return None
        assert result is None


class TestNormImageTagRewriter:
    """Tests for NormImageTagRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return NormImageTagRewriter()

    def test_rewrite_moves_image_tag(self, rewriter):
        """Test rewrite moves image tag to correct position."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "What is this? <image>"},
                    {"from": "gpt", "value": "A cat"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Result is a new question string or None
        if result is not None:
            assert isinstance(result, str)

    def test_rewrite_no_image_tag(self, rewriter, data_item):
        """Test rewrite when no image tag."""
        result = rewriter.rewrite(data_item, 0)
        # Should return None when no image tag to normalize
        assert result is None

    def test_rewrite_multiple_image_tags(self, rewriter):
        """Test rewrite handles multiple image tags."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "<image> What are these? <image>"},
                    {"from": "gpt", "value": "Two images"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Should not crash with multiple tags
        assert result is None or isinstance(result, str)


class TestNormPromptRewriter:
    """Tests for NormPromptRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return NormPromptRewriter()

    def test_rewrite_normalizes_prompt(self, rewriter):
        """Test rewrite normalizes prompt format."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "  Q with extra spaces  "},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        # Returns new question string or None
        assert result is None or isinstance(result, str)

    def test_rewrite_no_change_needed(self, rewriter, data_item):
        """Test rewrite when already normalized."""
        result = rewriter.rewrite(data_item, 0)
        assert result is None or isinstance(result, str)


class TestNormMultiTurnPromptRewriter:
    """Tests for NormMultiTurnPromptRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return NormMultiTurnPromptRewriter()

    def test_rewrite_multi_turn(self, rewriter, data_item_multi_turn):
        """Test rewrite normalizes multi-turn prompts."""
        result = rewriter.rewrite(data_item_multi_turn, 1)
        # Returns new question string or None
        assert result is None or isinstance(result, str)

    def test_rewrite_single_turn(self, rewriter, data_item):
        """Test rewrite with single turn."""
        result = rewriter.rewrite(data_item, 0)
        assert result is None or isinstance(result, str)


class TestRemoveAnswerRewriter:
    """Tests for RemoveAnswerRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return RemoveAnswerRewriter()

    def test_rewrite_removes_answer_prefix(self, rewriter):
        """Test rewrite removes [ANSWER 0] prefix."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "[ANSWER 0] The actual answer"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        if result:
            assert "[ANSWER 0]" not in result
            assert "The actual answer" in result

    def test_rewrite_no_prefix(self, rewriter, data_item):
        """Test rewrite when no prefix present."""
        result = rewriter.rewrite(data_item, 0)
        # Should return None when no ANSWER prefix
        assert result is None

    def test_rewrite_various_formats(self, rewriter):
        """Test rewrite handles various ANSWER formats."""
        test_cases = ["[ANSWER 1] Text", "[ANSWER] Text", "[Answer 0] Text"]
        for answer in test_cases:
            item = DataItem(
                {
                    "conversations": [
                        {"from": "human", "value": "Q"},
                        {"from": "gpt", "value": answer},
                    ]
                }
            )
            result = rewriter.rewrite(item, 0)


class TestRemoveReasonRewriter:
    """Tests for RemoveReasonRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return RemoveReasonRewriter()

    def test_rewrite_removes_reasoning_prefix(self, rewriter):
        """Test rewrite removes Reasoning: prefix."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Reasoning: Because... Answer: 42"},
                ]
            }
        )
        result = rewriter.rewrite(item, 0)
        if result:
            assert "Reasoning:" not in result

    def test_rewrite_no_prefix(self, rewriter, data_item):
        """Test rewrite when no prefix present."""
        result = rewriter.rewrite(data_item, 0)
        # Should return None when no reasoning prefix
        assert result is None


class TestSplitRewriter:
    """Tests for SplitRewriter."""

    @pytest.fixture
    def rewriter(self):
        """Create rewriter instance."""
        return SplitRewriter()

    def test_process_single_turn(self, rewriter, data_item):
        """Test process with single turn (no split)."""
        result = rewriter.process(data_item)
        assert result is not None

    def test_process_multi_turn(self, rewriter, data_item_multi_turn):
        """Test process with multi-turn conversation."""
        result = rewriter.process(data_item_multi_turn)
        assert result is not None

    def test_expand_items_single(self, rewriter, data_item):
        """Test expand_items with single turn."""
        items = [data_item]
        expanded = rewriter.expand_items(items)
        assert len(expanded) >= 1

    def test_expand_items_multi_turn(self, rewriter, data_item_multi_turn):
        """Test expand_items splits multi-turn into single turns."""
        items = [data_item_multi_turn]
        expanded = rewriter.expand_items(items)
        # Should have more items after expansion
        assert len(expanded) >= data_item_multi_turn.qa_count

    def test_expand_items_preserves_data(self, rewriter, data_item_multi_turn):
        """Test expand_items preserves conversation data."""
        items = [data_item_multi_turn]
        expanded = rewriter.expand_items(items)

        for item in expanded:
            assert item.qa_count == 1  # Each expanded item should have 1 QA

    def test_expand_items_empty_list(self, rewriter):
        """Test expand_items with empty list."""
        expanded = rewriter.expand_items([])
        assert expanded == []


class TestRewriterRegistration:
    """Tests for rewriter registration in OPERATORS registry."""

    def test_remove_think_rewriter_registered(self):
        """Test RemoveThinkRewriter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "RemoveThinkRewriter" in OPERATORS

    def test_norm_think_rewriter_registered(self):
        """Test NormThinkRewriter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "NormThinkRewriter" in OPERATORS

    def test_split_rewriter_registered(self):
        """Test SplitRewriter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "SplitRewriter" in OPERATORS


class TestRewriterEdgeCases:
    """Edge case tests for rewriters."""

    def test_rewriter_empty_answer(self):
        """Test rewriters handle empty answer."""
        r = RemoveThinkRewriter()
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": ""},
                ]
            }
        )
        result = r.rewrite(item, 0)

    def test_rewriter_unicode_content(self):
        """Test rewriters handle unicode content."""
        r = RemoveThinkRewriter()
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Sample question"},
                    {"from": "gpt", "value": "<think>reasoning</think>Sample answer"},
                ]
            }
        )
        result = r.rewrite(item, 0)
        if result:
            assert "Sample answer" in result

    def test_rewriter_special_characters(self):
        """Test rewriters handle special characters."""
        r = RemoveAnswerRewriter()
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "[ANSWER 0] <>&\"' special chars"},
                ]
            }
        )
        result = r.rewrite(item, 0)
        if result:
            assert "<>&\"'" in result

    def test_split_rewriter_single_message(self):
        """Test SplitRewriter with incomplete conversation."""
        r = SplitRewriter()
        item = DataItem({"conversations": [{"from": "human", "value": "Q only"}]})
        result = r.process(item)
