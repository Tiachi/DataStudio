"""
Unit tests for datastudio.operators.filters module.

Tests cover:
- ConvLengthFilter
- ImageSizeFilter
- ImageAspectRatioFilter
- ImageExtFilter
- LengthAnomalyFilter
- ResponseTagFilter
- TextRepeatFilter
"""

from unittest.mock import MagicMock, patch

import pytest

from datastudio.operators.core.data_item import DataItem
from datastudio.operators.filters import (
    ConvLengthFilter,
    ImageAspectRatioFilter,
    ImageExtFilter,
    ImageSizeFilter,
    LengthAnomalyFilter,
    ResponseTagFilter,
    TextRepeatFilter,
)


class TestConvLengthFilter:
    """Tests for ConvLengthFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return ConvLengthFilter()

    @pytest.fixture
    def filter_custom(self):
        """Create filter with custom settings."""
        return ConvLengthFilter(min_length=2, max_length=10)

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert filter_default.min_length == 1
        assert filter_default.max_length == 36

    def test_init_custom(self, filter_custom):
        """Test initialization with custom values."""
        assert filter_custom.min_length == 2
        assert filter_custom.max_length == 10

    def test_check_valid_length(self, filter_default):
        """Test check passes for valid length."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        rejected, reason = filter_default.check(item, 0)
        assert rejected is False

    def test_check_too_short(self, filter_custom):
        """Test process rejects too short conversation (qa_count < min_length)."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
            }
        )
        # Single turn (qa_count=1) should be rejected with min_length=2
        result = filter_custom.process(item)
        assert result.has_filter

    def test_check_too_long(self):
        """Test process rejects too long conversation (qa_count > max_length)."""
        f = ConvLengthFilter(max_length=1)
        convs = []
        for i in range(5):
            convs.append({"from": "human", "value": f"Q{i}"})
            convs.append({"from": "gpt", "value": f"A{i}"})
        item = DataItem({"conversations": convs})
        result = f.process(item)
        assert result.has_filter

    def test_process_returns_result(self, filter_default, data_item):
        """Test process returns Result."""
        result = filter_default.process(data_item)
        assert result is not None


class TestImageSizeFilter:
    """Tests for ImageSizeFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return ImageSizeFilter()

    @pytest.fixture
    def filter_custom(self):
        """Create filter with custom settings."""
        return ImageSizeFilter(min_size=100, max_ratio=10.0, require_image=True)

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert filter_default.min_size == 28
        assert filter_default.max_ratio == 20.0

    def test_init_custom(self, filter_custom):
        """Test initialization with custom values."""
        assert filter_custom.min_size == 100
        assert filter_custom.max_ratio == 10.0
        assert filter_custom.require_image is True

    def test_check_no_image_not_required(self, filter_default, data_item):
        """Test check passes when no image and not required."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_no_image_required(self, filter_custom, data_item):
        """Test check rejects when no image but required."""
        # ImageSizeFilter.process is item-level, not per-QA
        result = filter_custom.process(data_item)
        # Should reject since require_image=True and no image_pil
        assert result.has_filter

    def test_check_valid_image(self, filter_default):
        """Test check passes for valid image size."""
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="red")
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image_pil": img,
            }
        )

        result = filter_default.process(item)
        # Should pass with valid size
        assert not result.has_filter

    def test_check_image_too_small(self):
        """Test check rejects too small image."""
        from PIL import Image

        f = ImageSizeFilter(min_size=100)
        img = Image.new("RGB", (50, 50), color="red")
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image_pil": img,
            }
        )

        result = f.process(item)
        # Should be rejected
        assert result.has_filter


class TestImageAspectRatioFilter:
    """Tests for ImageAspectRatioFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return ImageAspectRatioFilter()

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert filter_default.max_aspect_ratio == 64.0

    def test_init_custom(self):
        """Test initialization with custom value."""
        f = ImageAspectRatioFilter(max_aspect_ratio=4.0)
        assert f.max_aspect_ratio == 4.0

    def test_check_no_image(self, filter_default, data_item):
        """Test check passes when no image."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_valid_ratio(self, filter_default):
        """Test check passes for valid aspect ratio."""
        # ImageAspectRatioFilter uses item.data.get('img_size')
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "img_size": [[100, 100]],
            }
        )

        result = filter_default.process(item)
        # 1:1 ratio should pass
        assert not result.has_filter

    def test_check_extreme_ratio(self):
        """Test check rejects extreme aspect ratio."""
        f = ImageAspectRatioFilter(max_aspect_ratio=2.0)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "img_size": [[1000, 100]],
            }
        )

        result = f.process(item)
        # 10:1 ratio should be rejected
        assert result.has_filter


class TestImageExtFilter:
    """Tests for ImageExtFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return ImageExtFilter()

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert ".gif" in filter_default.excluded_exts

    def test_init_custom(self):
        """Test initialization with custom extensions."""
        f = ImageExtFilter(excluded_exts=[".bmp", ".tiff"])
        assert ".bmp" in f.excluded_exts
        assert ".tiff" in f.excluded_exts

    def test_check_no_image(self, filter_default, data_item):
        """Test check passes when no image (item-level filter returns False)."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_valid_extension(self, filter_default):
        """Test check passes for valid extension."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image": "test.jpg",
            }
        )
        rejected, reason = filter_default.check(item, 0)
        assert rejected is False

    def test_check_excluded_extension(self, filter_default):
        """Test check rejects excluded extension."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image": "test.gif",
            }
        )
        # ImageExtFilter.process is item-level
        result = filter_default.process(item)
        assert result.has_filter

    def test_check_empty_extension(self, filter_default):
        """Test check handles empty extension."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "image": "noextension",
            }
        )
        # Empty extension is in default excluded_exts
        result = filter_default.process(item)
        assert result.has_filter


class TestLengthAnomalyFilter:
    """Tests for LengthAnomalyFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return LengthAnomalyFilter()

    @pytest.fixture
    def filter_check_all(self):
        """Create filter checking all fields."""
        return LengthAnomalyFilter(
            min_length=10,
            max_length=1000,
            check_question=True,
            check_answer=True,
            check_ori_answer=True,
        )

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert filter_default.min_length == 1
        assert filter_default.max_length == 8192

    def test_check_valid_length(self, filter_default, data_item):
        """Test check passes for valid length."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_too_short(self):
        """Test check rejects too short text."""
        f = LengthAnomalyFilter(min_length=100, check_answer=True)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Short"},
                ]
            }
        )
        rejected, reason = f.check(item, 0)
        assert rejected is True

    def test_check_too_long(self):
        """Test check rejects too long text."""
        f = LengthAnomalyFilter(max_length=10, check_answer=True)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A" * 100},
                ]
            }
        )
        rejected, reason = f.check(item, 0)
        assert rejected is True

    def test_check_question(self):
        """Test checking question length."""
        f = LengthAnomalyFilter(min_length=5, check_question=True)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Hi"},
                    {"from": "gpt", "value": "Hello there!"},
                ]
            }
        )
        rejected, reason = f.check(item, 0)
        # Short question (2 chars < min_length=5) should be rejected
        assert rejected is True


class TestResponseTagFilter:
    """Tests for ResponseTagFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return ResponseTagFilter()

    def test_check_no_tags(self, filter_default, data_item):
        """Test check passes when no image/video tags."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_with_image_tag(self, filter_default):
        """Test check rejects response with image tag."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Here is an image: <image>"},
                ],
                "image": "test.jpg",
            }
        )  # Need image for filter to check
        rejected, reason = filter_default.check(item, 0)
        assert rejected is True

    def test_check_with_video_tag(self, filter_default):
        """Test check rejects response with video tag."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "Watch this: <video>"},
                ],
                "image": "test.jpg",
            }
        )  # Need image for filter to check
        rejected, reason = filter_default.check(item, 0)
        assert rejected is True

    def test_check_tag_in_question_ok(self, filter_default):
        """Test check allows tags in question."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "What is in this <image>?"},
                    {"from": "gpt", "value": "It shows a cat."},
                ]
            }
        )
        rejected, reason = filter_default.check(item, 0)
        assert rejected is False


class TestTextRepeatFilter:
    """Tests for TextRepeatFilter."""

    @pytest.fixture
    def filter_default(self):
        """Create filter with default settings."""
        return TextRepeatFilter()

    @pytest.fixture
    def filter_check_all(self):
        """Create filter checking all fields."""
        return TextRepeatFilter(
            check_question=True, check_answer=True, check_ori_answer=True
        )

    def test_init_default(self, filter_default):
        """Test initialization with defaults."""
        assert filter_default.check_question is True
        assert filter_default.check_answer is True

    def test_check_no_repetition(self, filter_default, data_item):
        """Test check passes for non-repetitive text."""
        rejected, reason = filter_default.check(data_item, 0)
        assert rejected is False

    def test_check_with_repetition(self, filter_default):
        """Test check rejects repetitive text (needs 1000+ chars with 10+ repeats)."""
        repeated_text = (
            "This is a repeated sentence. " * 100
        )  # ~2900 chars, 100 repeats
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": repeated_text},
                ]
            }
        )
        rejected, reason = filter_default.check(item, 0)
        # Should detect repetition for highly repetitive text over 1000 chars
        assert rejected is True

    def test_check_consecutive_chars(self, filter_default):
        """Test check detects consecutive character repetition."""
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A" * 1000},
                ]
            }
        )
        rejected, reason = filter_default.check(item, 0)
        assert rejected is True


class TestFilterRegistration:
    """Tests for filter registration in OPERATORS registry."""

    def test_conv_length_filter_registered(self):
        """Test ConvLengthFilter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "ConvLengthFilter" in OPERATORS

    def test_image_size_filter_registered(self):
        """Test ImageSizeFilter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "ImageSizeFilter" in OPERATORS

    def test_length_anomaly_filter_registered(self):
        """Test LengthAnomalyFilter is registered."""
        from datastudio.utils.registry import OPERATORS

        assert "LengthAnomalyFilter" in OPERATORS


class TestFilterEdgeCases:
    """Edge case tests for filters."""

    def test_filter_empty_conversations(self):
        """Test filters handle empty conversations."""
        f = ConvLengthFilter()
        item = DataItem({"conversations": []})
        result = f.process(item)
        assert result is not None

    def test_filter_unicode_content(self):
        """Test filters handle unicode content."""
        f = LengthAnomalyFilter(check_answer=True)
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Sample question"},
                    {"from": "gpt", "value": "Sample answer"},
                ]
            }
        )
        rejected, reason = f.check(item, 0)
        assert rejected is False

    def test_filter_special_characters(self):
        """Test filters handle special characters."""
        f = ResponseTagFilter()
        item = DataItem(
            {
                "conversations": [
                    {"from": "human", "value": "Q <>&\"'"},
                    {"from": "gpt", "value": "A <>&\"'"},
                ]
            }
        )
        rejected, reason = f.check(item, 0)
        # Should not false positive on HTML entities
        assert rejected is False
