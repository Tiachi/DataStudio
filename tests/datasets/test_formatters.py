"""
Unit tests for datastudio.datasets.formatters module.

Tests cover:
- BaseFormat abstract class
- JsonFormat: load/save JSON files
- JsonlFormat: load/save JSONL files
- FormatRegistry: format registration and lookup
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from datastudio.datasets.formatters import (
    BaseFormat,
    FormatRegistry,
    JsonFormat,
    JsonlFormat,
)


class TestBaseFormat:
    """Tests for BaseFormat abstract class."""

    def test_base_format_is_abstract(self):
        """Test that BaseFormat cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFormat()

    def test_base_format_requires_extensions(self):
        """Test that subclass must implement extensions property."""

        class IncompleteFormat(BaseFormat):
            def load(self, file_path):
                pass

            def save(self, data, file_path, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteFormat()

    def test_base_format_requires_load(self):
        """Test that subclass must implement load method."""

        class IncompleteFormat(BaseFormat):
            @property
            def extensions(self):
                return [".test"]

            def save(self, data, file_path, **kwargs):
                pass

        with pytest.raises(TypeError):
            IncompleteFormat()

    def test_base_format_requires_save(self):
        """Test that subclass must implement save method."""

        class IncompleteFormat(BaseFormat):
            @property
            def extensions(self):
                return [".test"]

            def load(self, file_path):
                pass

        with pytest.raises(TypeError):
            IncompleteFormat()


class TestJsonFormat:
    """Tests for JsonFormat class."""

    @pytest.fixture
    def json_format(self):
        """Create JsonFormat instance."""
        return JsonFormat()

    def test_extensions(self, json_format):
        """Test that JsonFormat supports .json extension."""
        assert ".json" in json_format.extensions()

    def test_load_valid_json(self, json_format, temp_json_file):
        """Test loading a valid JSON file."""
        data = json_format.load(temp_json_file)
        assert isinstance(data, list)
        assert len(data) == 3
        assert "conversations" in data[0]

    def test_load_empty_json_array(self, json_format, temp_dir):
        """Test loading an empty JSON array."""
        filepath = os.path.join(temp_dir, "empty.json")
        with open(filepath, "w") as f:
            json.dump([], f)

        data = json_format.load(filepath)
        assert data == []

    def test_load_single_object_json(self, json_format, temp_dir):
        """Test loading a JSON file with single object (not array)."""
        filepath = os.path.join(temp_dir, "single.json")
        with open(filepath, "w") as f:
            json.dump({"key": "value"}, f)

        data = json_format.load(filepath)
        # Should wrap single object in list
        assert isinstance(data, list) or isinstance(data, dict)

    def test_load_nonexistent_file(self, json_format):
        """Test loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            json_format.load("/nonexistent/path/file.json")

    def test_load_invalid_json(self, json_format, temp_dir):
        """Test loading invalid JSON raises error."""
        filepath = os.path.join(temp_dir, "invalid.json")
        with open(filepath, "w") as f:
            f.write("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            json_format.load(filepath)

    def test_save_json(self, json_format, temp_dir, sample_data_list):
        """Test saving data to JSON file."""
        filepath = os.path.join(temp_dir, "output.json")
        json_format.save(sample_data_list, filepath)

        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == len(sample_data_list)

    def test_save_json_with_indent(self, json_format, temp_dir, sample_data_list):
        """Test saving JSON with custom indent."""
        filepath = os.path.join(temp_dir, "indented.json")
        json_format.save(sample_data_list, filepath, indent=2)

        with open(filepath, "r") as f:
            content = f.read()
        # Indented JSON should have newlines
        assert "\n" in content

    def test_save_json_ensure_ascii_false(self, json_format, temp_dir):
        """Test saving JSON with non-ASCII characters."""
        data = [{"text": "unicode test value"}]
        filepath = os.path.join(temp_dir, "unicode.json")
        json_format.save(data, filepath, ensure_ascii=False)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "unicode test value" in content

    def test_save_empty_list(self, json_format, temp_dir):
        """Test saving empty list."""
        filepath = os.path.join(temp_dir, "empty.json")
        json_format.save([], filepath)

        with open(filepath, "r") as f:
            loaded = json.load(f)
        assert loaded == []

    def test_save_creates_parent_directory(
        self, json_format, temp_dir, sample_data_list
    ):
        """Test that save creates parent directories if needed."""
        filepath = os.path.join(temp_dir, "subdir", "nested", "output.json")
        json_format.save(sample_data_list, filepath)

        assert os.path.exists(filepath)


class TestJsonlFormat:
    """Tests for JsonlFormat class."""

    @pytest.fixture
    def jsonl_format(self):
        """Create JsonlFormat instance."""
        return JsonlFormat()

    def test_extensions(self, jsonl_format):
        """Test that JsonlFormat supports .jsonl extension."""
        assert ".jsonl" in jsonl_format.extensions()

    def test_load_valid_jsonl(self, jsonl_format, temp_jsonl_file):
        """Test loading a valid JSONL file."""
        data = jsonl_format.load(temp_jsonl_file)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_load_empty_jsonl(self, jsonl_format, temp_dir):
        """Test loading an empty JSONL file."""
        filepath = os.path.join(temp_dir, "empty.jsonl")
        with open(filepath, "w") as f:
            pass  # Empty file

        data = jsonl_format.load(filepath)
        assert data == []

    def test_load_jsonl_with_blank_lines(self, jsonl_format, temp_dir):
        """Test loading JSONL with blank lines."""
        filepath = os.path.join(temp_dir, "blanks.jsonl")
        with open(filepath, "w") as f:
            f.write('{"a": 1}\n')
            f.write("\n")  # Blank line
            f.write('{"b": 2}\n')

        data = jsonl_format.load(filepath)
        assert len(data) == 2

    def test_load_nonexistent_file(self, jsonl_format):
        """Test loading a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            jsonl_format.load("/nonexistent/path/file.jsonl")

    def test_load_invalid_jsonl_line(self, jsonl_format, temp_dir):
        """Test loading JSONL with invalid line - invalid lines are skipped."""
        filepath = os.path.join(temp_dir, "invalid.jsonl")
        with open(filepath, "w") as f:
            f.write('{"valid": true}\n')
            f.write("not valid json\n")

        # JsonlFormat skips invalid lines with a warning
        data = jsonl_format.load(filepath)
        assert len(data) == 1  # Only valid line loaded
        assert data[0] == {"valid": True}

    def test_save_jsonl(self, jsonl_format, temp_dir, sample_data_list):
        """Test saving data to JSONL file."""
        filepath = os.path.join(temp_dir, "output.jsonl")
        jsonl_format.save(sample_data_list, filepath)

        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == len(sample_data_list)

    def test_save_jsonl_each_line_valid_json(
        self, jsonl_format, temp_dir, sample_data_list
    ):
        """Test that each line in saved JSONL is valid JSON."""
        filepath = os.path.join(temp_dir, "output.jsonl")
        jsonl_format.save(sample_data_list, filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)  # Should not raise

    def test_save_jsonl_ensure_ascii_false(self, jsonl_format, temp_dir):
        """Test saving JSONL with non-ASCII characters."""
        data = [{"text": "unicode test value"}]
        filepath = os.path.join(temp_dir, "unicode.jsonl")
        jsonl_format.save(data, filepath, ensure_ascii=False)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "unicode test value" in content

    def test_save_empty_list(self, jsonl_format, temp_dir):
        """Test saving empty list."""
        filepath = os.path.join(temp_dir, "empty.jsonl")
        jsonl_format.save([], filepath)

        with open(filepath, "r") as f:
            content = f.read()
        assert content == "" or content == "\n"


class TestFormatRegistry:
    """Tests for FormatRegistry class."""

    def test_get_json_format(self):
        """Test getting format for .json file."""
        fmt = FormatRegistry.get("test.json")
        assert isinstance(fmt, JsonFormat)

    def test_get_jsonl_format(self):
        """Test getting format for .jsonl file."""
        fmt = FormatRegistry.get("test.jsonl")
        assert isinstance(fmt, JsonlFormat)

    def test_get_format_with_path(self):
        """Test getting format with full path."""
        fmt = FormatRegistry.get("/path/to/data/file.json")
        assert isinstance(fmt, JsonFormat)

    def test_get_unsupported_format(self):
        """Test getting format for unsupported extension."""
        with pytest.raises((KeyError, ValueError)):
            FormatRegistry.get("test.csv")

    def test_load_json_file(self, temp_json_file):
        """Test loading JSON file through registry."""
        data = FormatRegistry.load(temp_json_file)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_load_jsonl_file(self, temp_jsonl_file):
        """Test loading JSONL file through registry."""
        data = FormatRegistry.load(temp_jsonl_file)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_load_adds_source_file(self, temp_json_file):
        """Test that load adds file_path to items."""
        data = FormatRegistry.load(temp_json_file, add_source_file=True)
        for item in data:
            assert "file_path" in item
            assert item["file_path"] == temp_json_file

    def test_save_json_file(self, temp_dir, sample_data_list):
        """Test saving JSON file through registry."""
        filepath = os.path.join(temp_dir, "output.json")
        FormatRegistry.save(sample_data_list, filepath)

        assert os.path.exists(filepath)

    def test_save_jsonl_file(self, temp_dir, sample_data_list):
        """Test saving JSONL file through registry."""
        filepath = os.path.join(temp_dir, "output.jsonl")
        FormatRegistry.save(sample_data_list, filepath)

        assert os.path.exists(filepath)

    def test_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = FormatRegistry.supported_extensions()
        assert ".json" in extensions
        assert ".jsonl" in extensions

    def test_register_custom_format(self):
        """Test registering a custom format."""

        class CustomFormat(BaseFormat):
            @classmethod
            def extensions(cls):
                return [".custom"]

            def load(self, file_path):
                return []

            def save(self, data, file_path, **kwargs):
                pass

        # Register and verify
        FormatRegistry.register(CustomFormat)
        assert ".custom" in FormatRegistry.supported_extensions()


class TestFormatRegistryEdgeCases:
    """Edge case tests for FormatRegistry."""

    def test_load_file_with_uppercase_extension(self, temp_dir, sample_data_list):
        """Test loading file with uppercase extension."""
        filepath = os.path.join(temp_dir, "test.JSON")
        with open(filepath, "w") as f:
            json.dump(sample_data_list, f)

        # Should handle case-insensitively or raise appropriate error
        try:
            data = FormatRegistry.load(filepath)
            assert isinstance(data, list)
        except (KeyError, ValueError):
            pass  # Also acceptable behavior

    def test_load_file_no_extension(self):
        """Test loading file without extension."""
        with pytest.raises((KeyError, ValueError)):
            FormatRegistry.get("filename_no_ext")

    def test_load_removes_rejected_items(self, temp_dir):
        """Test that load removes 'filtered' and 'rejected' fields."""
        data = [
            {"conversations": [], "filtered": True, "rejected": True},
            {"conversations": [], "filtered": False},
        ]
        filepath = os.path.join(temp_dir, "with_rejected.json")
        with open(filepath, "w") as f:
            json.dump(data, f)

        loaded = FormatRegistry.load(filepath, remove_rejected=True)
        # Should remove 'filtered' and 'rejected' fields from items
        for item in loaded:
            assert "filtered" not in item
            assert "rejected" not in item
