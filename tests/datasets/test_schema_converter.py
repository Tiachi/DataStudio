"""
Unit tests for datastudio.datasets.formatters.schema_converter module.

Tests cover:
- SchemaType constants
- SchemaConverter.detect_schema
- SchemaConverter.detect_schema_from_list
- SchemaConverter.normalize_item / normalize
- SchemaConverter.denormalize_item / denormalize
- Round-trip conversion fidelity
- Convenience functions normalize_data / denormalize_data
"""

import pytest

from datastudio.datasets.formatters.schema_converter import (
    ROLE_MAP_TO_MESSAGES,
    ROLE_MAP_TO_STANDARD,
    SchemaConverter,
    SchemaType,
    denormalize_data,
    normalize_data,
)


class TestSchemaType:
    """Tests for SchemaType constants."""

    def test_standard_type(self):
        assert SchemaType.STANDARD == "standard"

    def test_messages_type(self):
        assert SchemaType.MESSAGES == "messages"


class TestDetectSchema:
    """Tests for SchemaConverter.detect_schema."""

    def test_detect_standard(self):
        item = {"conversations": [{"from": "human", "value": "Q"}]}
        assert SchemaConverter.detect_schema(item) == SchemaType.STANDARD

    def test_detect_messages(self):
        item = {"messages": [{"role": "user", "content": "Q"}]}
        assert SchemaConverter.detect_schema(item) == SchemaType.MESSAGES

    def test_detect_no_conversation_field(self):
        """Items without conversation fields default to standard."""
        item = {"text": "some data"}
        assert SchemaConverter.detect_schema(item) == SchemaType.STANDARD

    def test_detect_both_fields_prefers_messages(self):
        """When both fields exist, messages takes priority."""
        item = {
            "messages": [{"role": "user", "content": "Q"}],
            "conversations": [{"from": "human", "value": "Q"}],
        }
        assert SchemaConverter.detect_schema(item) == SchemaType.MESSAGES


class TestDetectSchemaFromList:
    """Tests for SchemaConverter.detect_schema_from_list."""

    def test_detect_from_standard_list(self):
        data = [
            {"conversations": [{"from": "human", "value": "Q"}]},
            {"conversations": [{"from": "human", "value": "Q2"}]},
        ]
        assert SchemaConverter.detect_schema_from_list(data) == SchemaType.STANDARD

    def test_detect_from_messages_list(self):
        data = [
            {"messages": [{"role": "user", "content": "Q"}]},
        ]
        assert SchemaConverter.detect_schema_from_list(data) == SchemaType.MESSAGES

    def test_detect_from_empty_list(self):
        assert SchemaConverter.detect_schema_from_list([]) == SchemaType.STANDARD

    def test_detect_skips_items_without_fields(self):
        data = [
            {"text": "no conv field"},
            {"messages": [{"role": "user", "content": "Q"}]},
        ]
        assert SchemaConverter.detect_schema_from_list(data) == SchemaType.MESSAGES


class TestNormalizeItem:
    """Tests for SchemaConverter.normalize_item."""

    def test_normalize_standard_item(self):
        item = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
            ],
            "image": "test.jpg",
        }
        result = SchemaConverter.normalize_item(item)
        assert result["_original_schema"] == SchemaType.STANDARD
        assert "conversations" in result
        assert result["image"] == "test.jpg"

    def test_normalize_messages_item(self):
        item = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "images": ["test.jpg"],
        }
        result = SchemaConverter.normalize_item(item)
        assert result["_original_schema"] == SchemaType.MESSAGES
        assert "conversations" in result
        assert "messages" not in result
        # Roles should be mapped
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"
        # Single image in list -> unwrapped
        assert result["image"] == "test.jpg"

    def test_normalize_messages_multiple_images(self):
        item = {
            "messages": [{"role": "user", "content": "Q"}],
            "images": ["a.jpg", "b.jpg"],
        }
        result = SchemaConverter.normalize_item(item)
        assert result["image"] == ["a.jpg", "b.jpg"]

    def test_normalize_messages_empty_images(self):
        item = {
            "messages": [{"role": "user", "content": "Q"}],
            "images": [],
        }
        result = SchemaConverter.normalize_item(item)
        assert "image" not in result

    def test_normalize_messages_preserves_extra_fields(self):
        item = {
            "messages": [{"role": "user", "content": "Q"}],
            "metadata": {"source": "test"},
        }
        result = SchemaConverter.normalize_item(item)
        assert result["metadata"] == {"source": "test"}

    def test_normalize_system_role(self):
        item = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = SchemaConverter.normalize_item(item)
        assert result["conversations"][0]["from"] == "system"
        assert result["conversations"][1]["from"] == "human"


class TestDenormalizeItem:
    """Tests for SchemaConverter.denormalize_item."""

    def test_denormalize_standard_item(self):
        item = {
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
            "_original_schema": SchemaType.STANDARD,
        }
        result = SchemaConverter.denormalize_item(item)
        assert "_original_schema" not in result
        assert "conversations" in result

    def test_denormalize_messages_item(self):
        item = {
            "conversations": [
                {"from": "human", "value": "Q"},
                {"from": "gpt", "value": "A"},
            ],
            "image": "test.jpg",
            "_original_schema": SchemaType.MESSAGES,
        }
        result = SchemaConverter.denormalize_item(item)
        assert "messages" in result
        assert "conversations" not in result
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"
        assert result["images"] == ["test.jpg"]

    def test_denormalize_messages_multiple_images(self):
        item = {
            "conversations": [{"from": "human", "value": "Q"}],
            "image": ["a.jpg", "b.jpg"],
            "_original_schema": SchemaType.MESSAGES,
        }
        result = SchemaConverter.denormalize_item(item)
        assert result["images"] == ["a.jpg", "b.jpg"]

    def test_denormalize_no_schema_key(self):
        """Items without _original_schema default to standard."""
        item = {
            "conversations": [{"from": "human", "value": "Q"}],
        }
        result = SchemaConverter.denormalize_item(item)
        assert "conversations" in result


class TestRoundTrip:
    """Tests for round-trip conversion fidelity."""

    def test_standard_round_trip(self):
        original = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there"},
            ],
            "image": "photo.jpg",
            "metadata": {"id": 123},
        }
        normalized = SchemaConverter.normalize_item(original)
        restored = SchemaConverter.denormalize_item(normalized)

        assert restored["conversations"] == original["conversations"]
        assert restored["image"] == original["image"]
        assert restored["metadata"] == original["metadata"]

    def test_messages_round_trip(self):
        original = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            "images": ["math.png"],
            "id": "q1",
        }
        normalized = SchemaConverter.normalize_item(original)
        restored = SchemaConverter.denormalize_item(normalized)

        assert restored["messages"] == original["messages"]
        assert restored["images"] == original["images"]
        assert restored["id"] == original["id"]

    def test_batch_round_trip(self):
        data = [
            {"conversations": [{"from": "human", "value": "Q1"}]},
            {"messages": [{"role": "user", "content": "Q2"}]},
        ]
        normalized = SchemaConverter.normalize(data)
        assert all("conversations" in item for item in normalized)

        restored = SchemaConverter.denormalize(normalized)
        assert "conversations" in restored[0]
        assert "messages" in restored[1]


class TestConvenienceFunctions:
    """Tests for normalize_data and denormalize_data."""

    def test_normalize_data(self):
        data = [{"messages": [{"role": "user", "content": "Q"}]}]
        result = normalize_data(data)
        assert len(result) == 1
        assert "conversations" in result[0]

    def test_denormalize_data(self):
        data = [
            {
                "conversations": [{"from": "human", "value": "Q"}],
                "_original_schema": SchemaType.MESSAGES,
            }
        ]
        result = denormalize_data(data)
        assert "messages" in result[0]

    def test_normalize_empty_list(self):
        assert normalize_data([]) == []

    def test_denormalize_empty_list(self):
        assert denormalize_data([]) == []


class TestRoleMappings:
    """Tests for role mapping constants."""

    def test_role_map_to_standard(self):
        assert ROLE_MAP_TO_STANDARD["user"] == "human"
        assert ROLE_MAP_TO_STANDARD["assistant"] == "gpt"
        assert ROLE_MAP_TO_STANDARD["system"] == "system"

    def test_role_map_to_messages(self):
        assert ROLE_MAP_TO_MESSAGES["human"] == "user"
        assert ROLE_MAP_TO_MESSAGES["gpt"] == "assistant"
        assert ROLE_MAP_TO_MESSAGES["system"] == "system"

    def test_unknown_role_passthrough(self):
        """Unknown roles should pass through unchanged."""
        item = {
            "messages": [{"role": "custom_role", "content": "text"}],
        }
        result = SchemaConverter.normalize_item(item)
        assert result["conversations"][0]["from"] == "custom_role"


class TestEdgeCases:
    """Edge case tests."""

    def test_normalize_item_without_conversations_or_messages(self):
        item = {"text": "raw text", "label": 1}
        result = SchemaConverter.normalize_item(item)
        # Should return with STANDARD schema marker
        assert result["_original_schema"] == SchemaType.STANDARD

    def test_empty_conversations(self):
        item = {"conversations": []}
        result = SchemaConverter.normalize_item(item)
        assert result["conversations"] == []

    def test_empty_messages(self):
        item = {"messages": []}
        result = SchemaConverter.normalize_item(item)
        assert result["conversations"] == []

    def test_missing_content_in_message(self):
        item = {"messages": [{"role": "user"}]}
        result = SchemaConverter.normalize_item(item)
        assert result["conversations"][0]["value"] == ""

    def test_missing_role_in_message(self):
        item = {"messages": [{"content": "text"}]}
        result = SchemaConverter.normalize_item(item)
        # Default role is "user" -> "human"
        assert result["conversations"][0]["from"] == "human"

    def test_images_as_non_list(self):
        """images field as string instead of list."""
        item = {
            "messages": [{"role": "user", "content": "Q"}],
            "images": "single_image.jpg",
        }
        result = SchemaConverter.normalize_item(item)
        assert result["image"] == "single_image.jpg"
