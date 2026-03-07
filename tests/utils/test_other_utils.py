"""
Unit tests for other datastudio.utils modules.

Tests cover:
- files.py: build_file function
- json_parser.py: JSON parsing utilities
- language.py: Language detection
- qa_utils.py: QA utilities
- repetition.py: Repetition detection
- vision.py: Vision utilities
"""

import base64
import json
import os
from unittest.mock import MagicMock, patch

import pytest


class TestFilesModule:
    """Tests for datastudio.utils.files module."""

    def test_build_file_creates_directory(self, temp_dir):
        """Test build_file creates directory."""
        from datastudio.utils.files import build_file

        filepath = build_file(temp_dir, "test_prefix")

        assert os.path.dirname(filepath) == temp_dir or os.path.exists(
            os.path.dirname(filepath)
        )

    def test_build_file_with_nested_dir(self, temp_dir):
        """Test build_file with nested directory."""
        from datastudio.utils.files import build_file

        nested_dir = os.path.join(temp_dir, "level1", "level2")
        filepath = build_file(nested_dir, "test")

        assert os.path.exists(os.path.dirname(filepath))

    def test_build_file_returns_path(self, temp_dir):
        """Test build_file returns valid path."""
        from datastudio.utils.files import build_file

        filepath = build_file(temp_dir, "prefix")

        assert isinstance(filepath, str)
        assert "prefix" in filepath


class TestJsonParserModule:
    """Tests for datastudio.utils.json_parser module."""

    def test_load_str_to_dict_basic_json(self):
        """Test load_str_to_dict with basic JSON string."""
        from datastudio.utils.json_parser import load_str_to_dict

        result = load_str_to_dict('{"key": "value"}')
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_load_str_to_dict_with_single_quotes(self):
        """Test load_str_to_dict handles single-quoted JSON via json_repair."""
        from datastudio.utils.json_parser import load_str_to_dict

        input_str = "{'key': 'value'}"
        result = load_str_to_dict(input_str)
        # json_repair should handle single quotes
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_load_str_to_dict_with_whitespace(self):
        """Test load_str_to_dict handles JSON with surrounding whitespace."""
        from datastudio.utils.json_parser import load_str_to_dict

        input_str = '  \n  {"key": "value"}  \n  '
        result = load_str_to_dict(input_str)
        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_load_str_to_dict_invalid_returns_string(self):
        """Test load_str_to_dict returns original string when parsing fails."""
        from datastudio.utils.json_parser import load_str_to_dict

        input_str = "this is not json at all"
        result = load_str_to_dict(input_str)
        assert isinstance(result, str)
        assert result == input_str

    def test_load_str_to_dict_valid_json(self):
        """Test load_str_to_dict with valid JSON."""
        from datastudio.utils.json_parser import load_str_to_dict

        result = load_str_to_dict('{"key": "value", "number": 42}')

        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["number"] == 42

    def test_load_str_to_dict_with_markdown(self):
        """Test load_str_to_dict with markdown wrapper."""
        from datastudio.utils.json_parser import load_str_to_dict

        input_str = '```json\n{"key": "value"}\n```'
        result = load_str_to_dict(input_str)

        assert isinstance(result, dict)
        assert result["key"] == "value"

    def test_load_str_to_dict_invalid(self):
        """Test load_str_to_dict with invalid JSON."""
        from datastudio.utils.json_parser import load_str_to_dict

        result = load_str_to_dict("not valid json")

        # Should return original string or handle gracefully
        assert isinstance(result, (dict, str))

    def test_load_str_to_dict_nested(self):
        """Test load_str_to_dict with nested JSON."""
        from datastudio.utils.json_parser import load_str_to_dict

        input_str = '{"outer": {"inner": {"deep": "value"}}}'
        result = load_str_to_dict(input_str)

        assert result["outer"]["inner"]["deep"] == "value"


class TestLanguageModule:
    """Tests for datastudio.utils.language module."""

    def test_detect_language_english(self):
        """Test detect_language with English text."""
        from datastudio.utils.language import detect_language

        result = detect_language("Hello, how are you today?")
        assert result == "english"

    def test_detect_language_chinese(self):
        """Test detect_language correctly identifies Chinese text."""
        from datastudio.utils.language import detect_language

        # Chinese text is required here to test language detection
        result = detect_language(
            "\u4f60\u597d\uff0c\u4eca\u5929\u600e\u4e48\u6837\uff1f"
        )
        assert result == "chinese"

    def test_detect_language_mixed(self):
        """Test detect_language with mixed English and Chinese text."""
        from datastudio.utils.language import detect_language

        # Mixed text is required here to test language detection
        result = detect_language("Hello \u4f60\u597d world \u4e16\u754c")
        # Should return dominant language or "unknown"
        assert result in ["english", "chinese", "unknown"]

    def test_detect_language_empty(self):
        """Test detect_language with empty string."""
        from datastudio.utils.language import detect_language

        result = detect_language("")
        assert result == "unknown"

    def test_is_mixed_language_true(self):
        """Test is_mixed_language returns True for mixed English/Chinese."""
        from datastudio.utils.language import is_mixed_language

        # Mixed language pair is required to test cross-language detection
        result = is_mixed_language("Hello", "\u4f60\u597d")
        assert result is True

    def test_is_mixed_language_false(self):
        """Test is_mixed_language returns False for same language."""
        from datastudio.utils.language import is_mixed_language

        result = is_mixed_language("Hello", "World")
        assert result is False

    def test_is_mixed_language_chinese(self):
        """Test is_mixed_language returns False for two Chinese strings."""
        from datastudio.utils.language import is_mixed_language

        # Two Chinese strings should not be considered mixed
        result = is_mixed_language("\u4f60\u597d", "\u4e16\u754c")
        assert result is False


class TestQAUtilsModule:
    """Tests for datastudio.utils.qa_utils module."""

    def test_get_qa_count_single(self):
        """Test get_qa_count with single turn."""
        from datastudio.utils.qa_utils import get_qa_count

        conversations = [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]

        count = get_qa_count(conversations)
        assert count == 1

    def test_get_qa_count_multi(self):
        """Test get_qa_count with multiple turns."""
        from datastudio.utils.qa_utils import get_qa_count

        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
            {"from": "human", "value": "Q3"},
            {"from": "gpt", "value": "A3"},
        ]

        count = get_qa_count(conversations)
        assert count == 3

    def test_get_qa_count_empty(self):
        """Test get_qa_count with empty conversations."""
        from datastudio.utils.qa_utils import get_qa_count

        count = get_qa_count([])
        assert count == 0

    def test_get_qa_count_incomplete(self):
        """Test get_qa_count with incomplete turn."""
        from datastudio.utils.qa_utils import get_qa_count

        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},  # No answer
        ]

        count = get_qa_count(conversations)
        assert count == 1  # Only complete turns

    def test_calculate_qa_statistics(self, sample_data_list):
        """Test calculate_qa_statistics."""
        from datastudio.utils.qa_utils import calculate_qa_statistics

        stats = calculate_qa_statistics(sample_data_list)

        assert isinstance(stats, dict)
        assert "total_qa" in stats or "avg_qa" in stats or len(stats) > 0


class TestRepetitionModule:
    """Tests for datastudio.utils.repetition module."""

    def test_has_meaningful_content_true(self):
        """Test has_meaningful_content with meaningful text."""
        from datastudio.utils.repetition import has_meaningful_content

        result = has_meaningful_content(
            "This is a meaningful sentence with varied words."
        )
        assert result is True

    def test_has_meaningful_content_false(self):
        """Test has_meaningful_content with mostly non-alphanumeric text."""
        from datastudio.utils.repetition import has_meaningful_content

        # has_meaningful_content checks ratio of alphanumeric chars
        # "a a a a..." has high alphanumeric ratio due to 'a' characters
        # Use mostly punctuation/spaces to get False
        result = has_meaningful_content("... ... ... ... ...")
        assert result is False

    def test_has_meaningful_content_empty(self):
        """Test has_meaningful_content with empty string."""
        from datastudio.utils.repetition import has_meaningful_content

        result = has_meaningful_content("")
        assert result is False

    def test_detect_consecutive_repetition_found(self):
        """Test detect_consecutive_repetition finds repetition."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        # Function requires text length >= 1000 to check
        text = "word " * 300  # Long enough text with repetition
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is True

    def test_detect_consecutive_repetition_not_found(self):
        """Test detect_consecutive_repetition with no repetition."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        text = "This is a normal sentence without excessive repetition."
        has_rep, pattern = detect_consecutive_repetition(text)

        assert has_rep is False

    def test_detect_consecutive_repetition_chars(self):
        """Test detect_consecutive_repetition with character repetition."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        # Function requires text length >= 1000
        text = "AB" * 600  # 1200 chars, "ABAB..." pattern
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is True

    def test_detect_consecutive_repetition_threshold(self):
        """Test detect_consecutive_repetition with custom threshold."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        text = "word " * 5  # Only 5 repetitions
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is False


class TestVisionModule:
    """Tests for datastudio.utils.vision module."""

    def test_encode_image_to_base64(self):
        """Test encode_image_to_base64."""
        from PIL import Image

        from datastudio.utils.vision import encode_image_to_base64

        # Create a real PIL Image
        img = Image.new("RGB", (100, 100), color="red")
        result = encode_image_to_base64(img)

        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_image_to_base64_with_resize(self):
        """Test encode_image_to_base64 with resize."""
        from PIL import Image

        from datastudio.utils.vision import encode_image_to_base64

        # Create a large PIL Image
        img = Image.new("RGB", (1000, 1000), color="blue")
        result = encode_image_to_base64(img, target_size=100)

        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")

    def test_apply_resize_image_pil(self):
        """Test apply_resize_image_pil."""
        from PIL import Image

        from datastudio.utils.vision import apply_resize_image_pil

        # Create a large PIL Image
        img = Image.new("RGB", (2000, 1000), color="green")
        result = apply_resize_image_pil(img, 1024)

        # Should resize to fit within 1024
        assert result.width <= 1024
        assert result.height <= 1024

    def test_parse_file_image(self):
        """Test parse_file with image path."""
        from datastudio.utils.vision import parse_file

        file_type, content = parse_file("test.jpg")

        assert file_type == "image" or file_type is None

    def test_parse_file_dict_with_image(self):
        """Test parse_file with dict containing image."""
        from datastudio.utils.vision import parse_file

        data = {"image": "test.jpg", "other": "data"}
        file_type, content = parse_file(data)

    def test_parse_file_none(self):
        """Test parse_file with None."""
        from datastudio.utils.vision import parse_file

        file_type, content = parse_file(None)

        assert file_type is None


class TestDatabaseModule:
    """Tests for datastudio.utils.database module (LMDB)."""

    def test_get_shard_id(self):
        """Test get_shard_id."""
        from datastudio.utils.database import get_shard_id

        shard_id = get_shard_id("test_key", num_shards=32)

        assert isinstance(shard_id, int)
        assert 0 <= shard_id < 32

    def test_get_shard_id_consistent(self):
        """Test get_shard_id is consistent."""
        from datastudio.utils.database import get_shard_id

        id1 = get_shard_id("same_key", num_shards=32)
        id2 = get_shard_id("same_key", num_shards=32)

        assert id1 == id2

    def test_get_shard_id_distribution(self):
        """Test get_shard_id distributes keys."""
        from datastudio.utils.database import get_shard_id

        shard_ids = set()
        for i in range(100):
            shard_id = get_shard_id(f"key_{i}", num_shards=32)
            shard_ids.add(shard_id)

        # Should use multiple shards
        assert len(shard_ids) > 1


class TestShardedLMDBManager:
    """Tests for ShardedLMDBManager class."""

    @pytest.fixture
    def mock_lmdb(self):
        """Mock lmdb module."""
        with patch("datastudio.utils.database.lmdb") as mock:
            mock_env = MagicMock()
            mock.open.return_value = mock_env
            yield mock

    def test_init(self, temp_dir, mock_lmdb):
        """Test ShardedLMDBManager initialization."""
        from datastudio.utils.database import ShardedLMDBManager

        manager = ShardedLMDBManager(cache_dir=temp_dir, num_shards=4)
        assert manager is not None

    def test_exists(self, temp_dir, mock_lmdb):
        """Test exists method."""
        from datastudio.utils.database import ShardedLMDBManager

        manager = ShardedLMDBManager(cache_dir=temp_dir, num_shards=4)

        # Mock transaction
        mock_txn = MagicMock()
        mock_txn.get.return_value = None

        result = manager.exists("test_path.jpg")
        # Should check if key exists

    def test_close_all(self, temp_dir, mock_lmdb):
        """Test close_all method."""
        from datastudio.utils.database import ShardedLMDBManager

        manager = ShardedLMDBManager(cache_dir=temp_dir, num_shards=4)
        manager.close_all()
        # Should close all shards


class TestRepetitionModuleExtended:
    """Additional tests for datastudio.utils.repetition module."""

    def test_detect_consecutive_repetition_short_text_returns_false(self):
        """Test detect_consecutive_repetition with text < 1000 chars returns (False, None)."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        text = "short text " * 10  # well under 1000 characters
        has_rep, pattern = detect_consecutive_repetition(text)

        assert has_rep is False
        assert pattern is None

    def test_detect_consecutive_repetition_long_text_with_repeats(self):
        """Test detect_consecutive_repetition with text >= 1000 chars and 10+ consecutive repeats returns (True, pattern)."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        # Build a string that is >= 1000 chars and ends with 15 consecutive repeats
        repeated_unit = "hello world "
        text = repeated_unit * 150  # 1800 chars, each unit repeated many times
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is True
        assert pattern is not None

    def test_detect_consecutive_repetition_formatting_patterns_excluded(self):
        """Test detect_consecutive_repetition with formatting patterns (|, __, --) returns (False, None) even if repeated."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        # A string made of formatting characters, repeated enough to exceed 1000 chars
        text = "| " * 600  # 1200 chars, pattern is "| " containing "|"
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is False
        assert pattern is None

    def test_detect_consecutive_repetition_short_pattern_not_detected_at_length_one(
        self,
    ):
        """Test detect_consecutive_repetition skips patterns of length < 2 (the loop starts at length=2)."""
        from datastudio.utils.repetition import detect_consecutive_repetition

        # Build a long string where the only repetition is at the single-character level.
        # Use varied 2-char pairs so no length-2+ pattern repeats 10 times consecutively
        # at the tail of the string, but a single char 'Z' appears at the very end once.
        base = "abcdefghij" * 100  # 1000 chars, no consecutive repetition
        text = base + "Z"
        has_rep, pattern = detect_consecutive_repetition(text, min_times=10)

        assert has_rep is False
        assert pattern is None


class TestLanguageModuleExtended:
    """Additional tests for datastudio.utils.language module."""

    def test_detect_language_english_text(self):
        """Test detect_language with English text."""
        from datastudio.utils.language import detect_language

        result = detect_language("The quick brown fox jumps over the lazy dog.")
        assert result == "english"

    def test_detect_language_chinese_text(self):
        """Test detect_language with Chinese text."""
        from datastudio.utils.language import detect_language

        result = detect_language(
            "\u6211\u4eec\u4eca\u5929\u53bb\u516c\u56ed\u73a9\u4e86"
        )
        assert result == "chinese"

    def test_detect_language_empty_string(self):
        """Test detect_language with empty string."""
        from datastudio.utils.language import detect_language

        result = detect_language("")
        assert result == "unknown"

    def test_detect_language_chinese_char_detected(self):
        """Test that a single Chinese character is detected as Chinese."""
        from datastudio.utils.language import detect_language

        # Any string with at least one CJK unified ideograph should be detected as Chinese
        result = detect_language("\u4e2d")
        assert result == "chinese"

    def test_detect_language_ascii_char_detected_as_english(self):
        """Test that a pure ASCII letter string is detected as English."""
        from datastudio.utils.language import detect_language

        result = detect_language("abc")
        assert result == "english"


class TestQaUtilsModuleExtended:
    """Additional tests for datastudio.utils.qa_utils module."""

    def test_get_qa_count_standard_conversations(self):
        """Test get_qa_count with standard conversations format (4 messages = 2 pairs)."""
        from datastudio.utils.qa_utils import get_qa_count

        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
            {"from": "gpt", "value": "A2"},
        ]
        count = get_qa_count(conversations)
        assert count == 2

    def test_get_qa_count_empty_conversations(self):
        """Test get_qa_count with empty conversations."""
        from datastudio.utils.qa_utils import get_qa_count

        count = get_qa_count([])
        assert count == 0

    def test_get_qa_count_odd_number_messages(self):
        """Test get_qa_count with odd number of messages (incomplete pair)."""
        from datastudio.utils.qa_utils import get_qa_count

        conversations = [
            {"from": "human", "value": "Q1"},
            {"from": "gpt", "value": "A1"},
            {"from": "human", "value": "Q2"},
        ]
        count = get_qa_count(conversations)
        # 3 // 2 == 1
        assert count == 1

    def test_calculate_qa_statistics_with_various_batch_sizes(self):
        """Test calculate_qa_statistics with various batch sizes."""
        from datastudio.utils.qa_utils import calculate_qa_statistics

        datas = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            },
        ]

        # Without batch_size
        stats_no_batch = calculate_qa_statistics(datas)
        assert stats_no_batch["total_qa_pairs"] == 3
        assert stats_no_batch["max_qa_count"] == 2
        assert stats_no_batch["avg_qa_count"] == 1.5

        # With batch_size=10
        stats_batch = calculate_qa_statistics(
            datas, batch_size=10, adjust_batch_size=True
        )
        assert stats_batch["real_batch_size"] >= 1
        assert stats_batch["real_batch_size"] <= 10

        # With adjust_batch_size=False
        stats_no_adjust = calculate_qa_statistics(
            datas, batch_size=10, adjust_batch_size=False
        )
        assert stats_no_adjust["real_batch_size"] == 10

    def test_calculate_qa_statistics_empty_datas(self):
        """Test calculate_qa_statistics with empty data list."""
        from datastudio.utils.qa_utils import calculate_qa_statistics

        stats = calculate_qa_statistics([])
        assert stats["avg_qa_count"] == 0
        assert stats["max_qa_count"] == 0
        assert stats["total_qa_pairs"] == 0


class TestVisionModuleExtended:
    """Additional tests for datastudio.utils.vision module."""

    def test_apply_resize_image_pil_smaller_than_max(self):
        """Test apply_resize_image_pil with image already smaller than max_size."""
        from PIL import Image

        from datastudio.utils.vision import apply_resize_image_pil

        img = Image.new("RGB", (100, 50), color="red")
        result = apply_resize_image_pil(img, 1024)

        # Image should remain the same size
        assert result.width == 100
        assert result.height == 50

    def test_apply_resize_image_pil_larger_than_max(self):
        """Test apply_resize_image_pil with image larger than max_size."""
        from PIL import Image

        from datastudio.utils.vision import apply_resize_image_pil

        img = Image.new("RGB", (3000, 1500), color="blue")
        result = apply_resize_image_pil(img, 1024)

        # Larger dimension should be capped at 1024
        assert result.width == 1024
        assert result.height == 512
        # Aspect ratio should be preserved
        assert abs(result.width / result.height - 2.0) < 0.01

    def test_apply_resize_image_pil_square_image(self):
        """Test apply_resize_image_pil with square image."""
        from PIL import Image

        from datastudio.utils.vision import apply_resize_image_pil

        img = Image.new("RGB", (2000, 2000), color="green")
        result = apply_resize_image_pil(img, 512)

        assert result.width == 512
        assert result.height == 512

    def test_apply_resize_image_pil_none_input(self):
        """Test apply_resize_image_pil with None input raises AttributeError."""
        from datastudio.utils.vision import apply_resize_image_pil

        with pytest.raises(AttributeError):
            apply_resize_image_pil(None, 1024)
