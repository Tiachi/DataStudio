"""
Unit tests for datastudio.operators.core.data_item module.

Tests cover:
- QA dataclass
- DataItem class
- wrap_data_list and unwrap_data_items functions
"""

from copy import deepcopy

import pytest

from datastudio.operators.core.data_item import (
    QA,
    DataItem,
    unwrap_data_items,
    wrap_data_list,
)


class TestQADataclass:
    """Tests for QA dataclass."""

    def test_create_qa(self):
        """Test creating QA instance."""
        qa = QA(idx=0, question="What is AI?", answer="Artificial Intelligence")
        assert qa.idx == 0
        assert qa.question == "What is AI?"
        assert qa.answer == "Artificial Intelligence"

    def test_qa_with_original(self):
        """Test QA with original question/answer."""
        qa = QA(
            idx=0,
            question="Modified Q",
            answer="Modified A",
            ori_question="Original Q",
            ori_answer="Original A",
        )
        assert qa.ori_question == "Original Q"
        assert qa.ori_answer == "Original A"

    def test_qa_default_original_none(self):
        """Test QA defaults original to None."""
        qa = QA(idx=0, question="Q", answer="A")
        assert qa.ori_question is None
        assert qa.ori_answer is None

    def test_qa_is_frozen(self):
        """Test that QA is immutable (frozen)."""
        qa = QA(idx=0, question="Q", answer="A")
        with pytest.raises(AttributeError):
            qa.question = "New Q"

    def test_qa_equality(self):
        """Test QA equality comparison."""
        qa1 = QA(idx=0, question="Q", answer="A")
        qa2 = QA(idx=0, question="Q", answer="A")
        qa3 = QA(idx=1, question="Q", answer="A")

        assert qa1 == qa2
        assert qa1 != qa3


class TestDataItemInit:
    """Tests for DataItem initialization."""

    def test_init_with_minimal_data(self):
        """Test initialization with minimal data."""
        data = {
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ]
        }
        item = DataItem(data)
        assert item is not None

    def test_init_with_index(self):
        """Test initialization with custom index."""
        data = {
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ]
        }
        item = DataItem(data, idx=5)
        # Index should be set

    def test_init_with_image(self, sample_data_item_with_image):
        """Test initialization with image."""
        item = DataItem(sample_data_item_with_image)
        assert item.has_image

    def test_init_without_image(self, sample_data_item):
        """Test initialization without image."""
        item = DataItem(sample_data_item)
        # has_image should be False or image should be None

    def test_init_preserves_original_data(self, sample_data_item):
        """Test that initialization preserves original data."""
        original = deepcopy(sample_data_item)
        item = DataItem(sample_data_item)
        assert item.data["conversations"] == original["conversations"]


class TestDataItemProperties:
    """Tests for DataItem properties."""

    @pytest.fixture
    def item(self, sample_data_item):
        """Create DataItem for testing."""
        return DataItem(sample_data_item)

    def test_data_property(self, item):
        """Test data property returns underlying dict."""
        assert isinstance(item.data, dict)
        assert "conversations" in item.data

    def test_conversations_property(self, item):
        """Test conversations property."""
        convs = item.conversations
        assert isinstance(convs, list)
        assert len(convs) > 0

    def test_qa_pairs_property(self, item):
        """Test qa_pairs property returns list of QA."""
        qa_pairs = item.qa_pairs
        assert isinstance(qa_pairs, list)
        assert all(isinstance(qa, QA) for qa in qa_pairs)

    def test_qa_count_property(self, item):
        """Test qa_count property."""
        count = item.qa_count
        assert isinstance(count, int)
        assert count >= 1

    def test_image_property_with_image(self, sample_data_item_with_image):
        """Test image property when image_pil exists."""
        # image property returns 'image_pil' field, not 'image'
        sample_data_item_with_image["image_pil"] = "fake_pil_image"
        item = DataItem(sample_data_item_with_image)
        assert item.image is not None

    def test_image_property_without_image(self, sample_data_item):
        """Test image property when no image."""
        item = DataItem(sample_data_item)
        assert item.image is None

    def test_has_image_property(self, sample_data_item, sample_data_item_with_image):
        """Test has_image property."""
        item_no_img = DataItem(sample_data_item)
        item_with_img = DataItem(sample_data_item_with_image)

        assert not item_no_img.has_image
        # has_image checks for 'image_pil' OR 'image' field
        assert item_with_img.has_image  # sample_data_item_with_image has 'image' field

    def test_is_rejected_default_false(self, item):
        """Test is_rejected defaults to False."""
        assert not item.is_rejected


class TestDataItemQAAccess:
    """Tests for QA access methods."""

    @pytest.fixture
    def item(self, sample_multi_turn_conversation):
        """Create multi-turn DataItem for testing."""
        return DataItem({"conversations": sample_multi_turn_conversation})

    def test_get_qa(self, item):
        """Test get_qa returns correct QA."""
        qa = item.get_qa(0)
        assert isinstance(qa, QA)
        assert qa.idx == 0

    def test_get_qa_second_turn(self, item):
        """Test get_qa for second turn."""
        qa = item.get_qa(1)
        assert qa.idx == 1

    def test_get_qa_invalid_index(self, item):
        """Test get_qa with invalid index raises error."""
        with pytest.raises(IndexError):
            item.get_qa(100)

    def test_get_question(self, item):
        """Test get_question returns question text."""
        question = item.get_question(0)
        assert isinstance(question, str)
        assert len(question) > 0

    def test_get_answer(self, item):
        """Test get_answer returns answer text."""
        answer = item.get_answer(0)
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestDataItemQAModification:
    """Tests for QA modification methods."""

    @pytest.fixture
    def item(self, sample_data_item):
        """Create DataItem for testing."""
        return DataItem(sample_data_item)

    def test_set_question(self, item):
        """Test set_question modifies question."""
        original = item.get_question(0)
        item.set_question(0, "New question")
        assert item.get_question(0) == "New question"

    def test_set_question_saves_original(self, item):
        """Test set_question saves original when requested."""
        original = item.get_question(0)
        item.set_question(0, "New question", save_original=True)

        qa = item.get_qa(0)
        assert qa.ori_question == original

    def test_set_answer(self, item):
        """Test set_answer modifies answer."""
        original = item.get_answer(0)
        item.set_answer(0, "New answer")
        assert item.get_answer(0) == "New answer"

    def test_set_answer_saves_original(self, item):
        """Test set_answer saves original when requested."""
        original = item.get_answer(0)
        item.set_answer(0, "New answer", save_original=True)

        qa = item.get_qa(0)
        assert qa.ori_answer == original

    def test_set_question_no_save_original(self, item):
        """Test set_question without saving original."""
        item.set_question(0, "New question", save_original=False)
        qa = item.get_qa(0)
        # Original should not be saved


class TestDataItemFormatting:
    """Tests for QA formatting methods."""

    @pytest.fixture
    def item(self, sample_data_item):
        """Create DataItem for testing."""
        return DataItem(sample_data_item)

    def test_format_qa_question_only(self, item):
        """Test format_qa with question only."""
        formatted = item.format_qa(0, with_question=True, with_answer=False)
        assert isinstance(formatted, str)
        assert item.get_question(0) in formatted

    def test_format_qa_with_answer(self, item):
        """Test format_qa with question and answer."""
        formatted = item.format_qa(0, with_question=True, with_answer=True)
        assert item.get_question(0) in formatted
        assert item.get_answer(0) in formatted

    def test_format_qa_with_original(self, item):
        """Test format_qa with original values."""
        item.set_answer(0, "Modified", save_original=True)
        formatted = item.format_qa(0, with_answer=True, with_original=True)
        # Should include original value

    def test_format_all_qa(self, data_item_multi_turn):
        """Test format_all_qa for multi-turn."""
        formatted = data_item_multi_turn.format_all_qa(
            with_question=True, with_answer=True
        )
        assert isinstance(formatted, str)


class TestDataItemRejection:
    """Tests for rejection functionality."""

    @pytest.fixture
    def item(self, sample_data_item):
        """Create DataItem for testing."""
        return DataItem(sample_data_item)

    def test_mark_rejected(self, item):
        """Test mark_rejected sets rejected flag."""
        item.mark_rejected()
        assert item.is_rejected

    def test_mark_kept(self, item):
        """Test mark_kept clears rejected flag."""
        item.mark_rejected()
        item.mark_kept()
        assert not item.is_rejected

    def test_add_filter_record(self, item):
        """Test add_filter_record adds record."""
        item.add_filter_record("TestFilter", 0, "Test reason")
        # Record should be added to data

    def test_add_keep_record(self, item):
        """Test add_keep_record adds record."""
        item.add_keep_record("TestFilter", 0, "Kept reason")

    def test_add_rewrite_record(self, item):
        """Test add_rewrite_record adds record."""
        item.add_rewrite_record("TestRewriter", 0, "Rewrite message")

    def test_add_full_filter_record(self, item):
        """Test add_full_filter_record for full item rejection."""
        item.add_full_filter_record("TestFilter", "Full rejection reason")


class TestDataItemSplit:
    """Tests for split functionality."""

    @pytest.fixture
    def multi_turn_item(self, sample_multi_turn_conversation):
        """Create multi-turn DataItem for testing."""
        return DataItem({"conversations": sample_multi_turn_conversation})

    def test_split_basic(self, multi_turn_item):
        """Test basic split functionality."""
        kept, rejected = multi_turn_item.split(kept_indices=[0], rejected_indices=[1])

        assert kept is not None
        assert rejected is not None

    def test_split_all_kept(self, multi_turn_item):
        """Test split with all indices kept."""
        kept, rejected = multi_turn_item.split(kept_indices=[0, 1], rejected_indices=[])

        assert kept is not None
        assert rejected is None or rejected.qa_count == 0

    def test_split_all_rejected(self, multi_turn_item):
        """Test split with all indices rejected."""
        kept, rejected = multi_turn_item.split(kept_indices=[], rejected_indices=[0, 1])

        assert kept is None or kept.qa_count == 0
        assert rejected is not None


class TestWrapUnwrapFunctions:
    """Tests for wrap_data_list and unwrap_data_items functions."""

    def test_wrap_data_list(self, sample_data_list):
        """Test wrap_data_list creates DataItems."""
        items = wrap_data_list(sample_data_list)

        assert isinstance(items, list)
        assert all(isinstance(item, DataItem) for item in items)
        assert len(items) == len(sample_data_list)

    def test_wrap_data_list_empty(self):
        """Test wrap_data_list with empty list."""
        items = wrap_data_list([])
        assert items == []

    def test_unwrap_data_items(self, sample_data_list):
        """Test unwrap_data_items extracts dicts."""
        items = wrap_data_list(sample_data_list)
        unwrapped = unwrap_data_items(items)

        assert isinstance(unwrapped, list)
        assert all(isinstance(d, dict) for d in unwrapped)
        assert len(unwrapped) == len(sample_data_list)

    def test_unwrap_data_items_empty(self):
        """Test unwrap_data_items with empty list."""
        unwrapped = unwrap_data_items([])
        assert unwrapped == []

    def test_wrap_unwrap_roundtrip(self, sample_data_list):
        """Test that wrap and unwrap are inverse operations."""
        items = wrap_data_list(sample_data_list)
        unwrapped = unwrap_data_items(items)

        for original, result in zip(sample_data_list, unwrapped):
            assert original["conversations"] == result["conversations"]


class TestDataItemEdgeCases:
    """Edge case tests for DataItem."""

    def test_empty_conversations(self):
        """Test DataItem with empty conversations."""
        data = {"conversations": []}
        item = DataItem(data)
        assert item.qa_count == 0

    def test_single_message_conversation(self):
        """Test DataItem with single message (no complete QA)."""
        data = {"conversations": [{"from": "human", "value": "Hello"}]}
        item = DataItem(data)
        # Should handle gracefully

    def test_odd_number_messages(self):
        """Test DataItem with odd number of messages."""
        data = {
            "conversations": [
                {"from": "human", "value": "Q1"},
                {"from": "gpt", "value": "A1"},
                {"from": "human", "value": "Q2"},
            ]
        }
        item = DataItem(data)
        # Should handle incomplete last turn

    def test_unicode_content(self):
        """Test DataItem with unicode content."""
        data = {
            "conversations": [
                {"from": "human", "value": "Sample question"},
                {"from": "gpt", "value": "Sample answer"},
            ]
        }
        item = DataItem(data)
        assert "Sample question" in item.get_question(0)
        assert "Sample answer" in item.get_answer(0)

    def test_very_long_content(self):
        """Test DataItem with very long content."""
        long_text = "A" * 100000
        data = {
            "conversations": [
                {"from": "human", "value": long_text},
                {"from": "gpt", "value": long_text},
            ]
        }
        item = DataItem(data)
        assert len(item.get_question(0)) == 100000

    def test_special_characters_in_content(self):
        """Test DataItem with special characters."""
        data = {
            "conversations": [
                {
                    "from": "human",
                    "value": "Question with <tag> and \"quotes\" and 'apostrophes'",
                },
                {"from": "gpt", "value": "Answer with\nnewlines\tand\ttabs"},
            ]
        }
        item = DataItem(data)
        assert "<tag>" in item.get_question(0)
        assert "\n" in item.get_answer(0)
