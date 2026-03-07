"""Extended tests for FormatRegistry and schema handling edge cases."""

import json
import os

import pytest

from datastudio.datasets.formatters import FormatRegistry


class TestFormatRegistryAutoNormalize:
    """Tests for auto_normalize behavior in FormatRegistry.load."""

    def test_load_with_auto_normalize_converts_messages_to_conversations(
        self, temp_dir
    ):
        """Test load with auto_normalize=True converts messages to conversations."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                "images": ["img.jpg"],
            }
        ]
        filepath = os.path.join(temp_dir, "messages_schema.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(
            filepath, auto_normalize=True, add_source_file=False
        )

        assert len(loaded) == 1
        item = loaded[0]
        # Should have been converted to standard schema
        assert "conversations" in item
        assert "messages" not in item
        assert item["conversations"][0]["from"] == "human"
        assert item["conversations"][0]["value"] == "Hello"
        assert item["conversations"][1]["from"] == "gpt"
        assert item["conversations"][1]["value"] == "Hi there!"
        # images -> image conversion
        assert "image" in item
        assert item["image"] == "img.jpg"

    def test_load_with_auto_normalize_false_preserves_messages_schema(self, temp_dir):
        """Test load with auto_normalize=False preserves messages schema."""
        data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
            }
        ]
        filepath = os.path.join(temp_dir, "messages_schema.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(
            filepath, auto_normalize=False, add_source_file=False
        )

        assert len(loaded) == 1
        item = loaded[0]
        # Should preserve the original messages schema
        assert "messages" in item
        assert item["messages"][0]["role"] == "user"

    def test_save_with_auto_denormalize_converts_back_to_original_schema(
        self, temp_dir
    ):
        """Test save with auto_denormalize=True converts back to original schema."""
        from datastudio.datasets.formatters import SchemaType

        # Build normalized data that originally came from messages schema
        normalized_data = [
            {
                "conversations": [
                    {"from": "human", "value": "What is AI?"},
                    {"from": "gpt", "value": "Artificial Intelligence."},
                ],
                "image": "photo.jpg",
                "_original_schema": SchemaType.MESSAGES,
            }
        ]
        filepath = os.path.join(temp_dir, "denorm_output.json")
        FormatRegistry.save(normalized_data, filepath, auto_denormalize=True)

        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert len(saved) == 1
        item = saved[0]
        # Should have been converted back to messages schema
        assert "messages" in item
        assert "conversations" not in item
        assert item["messages"][0]["role"] == "user"
        assert item["messages"][1]["role"] == "assistant"
        # image -> images
        assert "images" in item
        assert item["images"] == ["photo.jpg"]

    def test_save_with_auto_denormalize_false_preserves_conversations_schema(
        self, temp_dir
    ):
        """Test save with auto_denormalize=False preserves conversations schema."""
        from datastudio.datasets.formatters import SchemaType

        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Question"},
                    {"from": "gpt", "value": "Answer"},
                ],
                "_original_schema": SchemaType.MESSAGES,
            }
        ]
        filepath = os.path.join(temp_dir, "no_denorm_output.json")
        FormatRegistry.save(data, filepath, auto_denormalize=False)

        with open(filepath, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert len(saved) == 1
        item = saved[0]
        # Should still have conversations since we did not denormalize
        assert "conversations" in item
        assert item["conversations"][0]["from"] == "human"


class TestFormatRegistryRemoveRejected:
    """Tests for remove_rejected behavior."""

    def test_load_with_remove_rejected_true_removes_rejected_field(self, temp_dir):
        """Test load with remove_rejected=True removes 'rejected' field."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "rejected": True,
            }
        ]
        filepath = os.path.join(temp_dir, "rejected.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(
            filepath, remove_rejected=True, add_source_file=False
        )

        assert len(loaded) == 1
        assert "rejected" not in loaded[0]

    def test_load_with_remove_rejected_true_removes_filtered_field(self, temp_dir):
        """Test load with remove_rejected=True removes 'filtered' field."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "filtered": True,
            }
        ]
        filepath = os.path.join(temp_dir, "filtered.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(
            filepath, remove_rejected=True, add_source_file=False
        )

        assert len(loaded) == 1
        assert "filtered" not in loaded[0]

    def test_load_with_remove_rejected_false_preserves_rejected_field(self, temp_dir):
        """Test load with remove_rejected=False preserves 'rejected' field."""
        data = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ],
                "rejected": True,
                "filtered": False,
            }
        ]
        filepath = os.path.join(temp_dir, "keep_rejected.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(
            filepath, remove_rejected=False, add_source_file=False
        )

        assert len(loaded) == 1
        assert "rejected" in loaded[0]
        assert loaded[0]["rejected"] is True
        assert "filtered" in loaded[0]
        assert loaded[0]["filtered"] is False


class TestFormatRegistryRoundTrip:
    """End-to-end round-trip tests."""

    def test_load_json_save_json_preserves_data(self, temp_dir, temp_json_file):
        """Test load JSON -> save JSON preserves data."""
        loaded = FormatRegistry.load(
            temp_json_file,
            add_source_file=False,
            remove_rejected=False,
            auto_normalize=False,
        )
        output_path = os.path.join(temp_dir, "roundtrip_output.json")
        FormatRegistry.save(loaded, output_path, auto_denormalize=False)

        # Re-load and compare
        reloaded = FormatRegistry.load(
            output_path,
            add_source_file=False,
            remove_rejected=False,
            auto_normalize=False,
        )
        assert len(reloaded) == len(loaded)
        for orig, rt in zip(loaded, reloaded):
            assert orig["conversations"] == rt["conversations"]

    def test_load_jsonl_save_jsonl_preserves_data(self, temp_dir, temp_jsonl_file):
        """Test load JSONL -> save JSONL preserves data."""
        loaded = FormatRegistry.load(
            temp_jsonl_file,
            add_source_file=False,
            remove_rejected=False,
            auto_normalize=False,
        )
        output_path = os.path.join(temp_dir, "roundtrip_output.jsonl")
        FormatRegistry.save(loaded, output_path, auto_denormalize=False)

        # Re-load and compare
        reloaded = FormatRegistry.load(
            output_path,
            add_source_file=False,
            remove_rejected=False,
            auto_normalize=False,
        )
        assert len(reloaded) == len(loaded)
        for orig, rt in zip(loaded, reloaded):
            assert orig["conversations"] == rt["conversations"]

    def test_load_messages_schema_save_load_back_preserves_data(self, temp_dir):
        """Test load messages schema -> save -> load back preserves data."""
        original_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is Python?"},
                    {
                        "role": "assistant",
                        "content": "Python is a programming language.",
                    },
                ],
                "images": ["diagram.png"],
            },
            {
                "messages": [
                    {"role": "user", "content": "Explain ML."},
                    {
                        "role": "assistant",
                        "content": "Machine Learning is a subset of AI.",
                    },
                ],
            },
        ]

        # Write original messages-schema data
        input_path = os.path.join(temp_dir, "messages_input.json")
        with open(input_path, "w") as f:
            json.dump(original_data, f)

        # Load with auto_normalize (converts to standard schema)
        loaded = FormatRegistry.load(
            input_path,
            add_source_file=False,
            remove_rejected=False,
            auto_normalize=True,
        )

        # Verify normalization happened
        for item in loaded:
            assert "conversations" in item
            assert "_original_schema" in item

        # Save with auto_denormalize (converts back to messages schema)
        output_path = os.path.join(temp_dir, "messages_roundtrip.json")
        FormatRegistry.save(loaded, output_path, auto_denormalize=True)

        # Read the raw saved file and verify it matches original structure
        with open(output_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        assert len(saved) == 2
        # First item should have messages schema
        assert "messages" in saved[0]
        assert saved[0]["messages"][0]["role"] == "user"
        assert saved[0]["messages"][0]["content"] == "What is Python?"
        assert saved[0]["messages"][1]["role"] == "assistant"
        assert saved[0]["messages"][1]["content"] == "Python is a programming language."
        assert "images" in saved[0]
        assert saved[0]["images"] == ["diagram.png"]

        # Second item
        assert "messages" in saved[1]
        assert saved[1]["messages"][0]["content"] == "Explain ML."
        assert (
            saved[1]["messages"][1]["content"] == "Machine Learning is a subset of AI."
        )
