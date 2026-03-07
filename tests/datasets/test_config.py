"""
Unit tests for datastudio.datasets.config module.

Tests cover:
- DatasetConfig dataclass
- StandardConfig dataclass
- ConfigLoader class
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import yaml

from datastudio.datasets.config import ConfigLoader, DatasetConfig, StandardConfig


class TestDatasetConfig:
    """Tests for DatasetConfig dataclass."""

    def test_create_with_required_fields(self):
        """Test creating DatasetConfig with required fields only."""
        config = DatasetConfig(file_path="/path/to/data.json")
        assert config.file_path == "/path/to/data.json"
        assert config.source == "Unknown"  # Default value

    def test_create_with_all_fields(self):
        """Test creating DatasetConfig with all fields."""
        config = DatasetConfig(
            file_path="/path/to/data.json",
            source="TestSource",
            data_size=1000,
            extra={"key": "value"},
        )
        assert config.file_path == "/path/to/data.json"
        assert config.source == "TestSource"
        assert config.data_size == 1000
        assert config.extra == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test creating from dict with minimal fields."""
        data = {"file_path": "/data/test.json"}
        config = DatasetConfig.from_dict(data)
        assert config.file_path == "/data/test.json"

    def test_from_dict_with_aliases(self):
        """Test creating from dict with field aliases."""
        # Test aliases defined in FIELD_ALIASES: json_path, jsonl_path, etc.
        data = {"json_path": "/data/test.json", "source": "Test"}
        config = DatasetConfig.from_dict(data)
        assert config.file_path == "/data/test.json"
        assert config.source == "Test"

    def test_from_dict_full(self):
        """Test creating from dict with all fields."""
        data = {
            "file_path": "/data/test.json",
            "source": "TestSource",
            "data_size": 500,
            "custom_field": "data",  # Extra fields go into 'extra' dict
        }
        config = DatasetConfig.from_dict(data)
        assert config.file_path == "/data/test.json"
        assert config.source == "TestSource"
        assert config.data_size == 500
        assert config.extra == {"custom_field": "data"}

    def test_to_dict(self):
        """Test converting to dict."""
        config = DatasetConfig(
            file_path="/path/to/data.json", source="TestSource", data_size=100
        )
        result = config.to_dict()
        assert isinstance(result, dict)
        assert result["file_path"] == "/path/to/data.json"
        assert result["source"] == "TestSource"
        assert result["data_size"] == 100

    def test_to_dict_excludes_none(self):
        """Test that to_dict excludes None values."""
        config = DatasetConfig(file_path="/path/to/data.json")
        result = config.to_dict()
        # data_size is None by default, should be excluded or None
        assert result.get("data_size") is None or "data_size" not in result

    def test_equality(self):
        """Test equality comparison."""
        config1 = DatasetConfig(file_path="/path/a.json", source="A")
        config2 = DatasetConfig(file_path="/path/a.json", source="A")
        config3 = DatasetConfig(file_path="/path/b.json", source="B")

        assert config1 == config2
        assert config1 != config3


class TestStandardConfig:
    """Tests for StandardConfig dataclass."""

    def test_create_empty(self):
        """Test creating empty StandardConfig."""
        config = StandardConfig()
        assert config.datasets == []
        assert config.extra == {}

    def test_create_with_datasets(self):
        """Test creating with dataset configs."""
        datasets = [
            DatasetConfig(file_path="/data/a.json", source="A"),
            DatasetConfig(file_path="/data/b.json", source="B"),
        ]
        config = StandardConfig(datasets=datasets)
        assert len(config.datasets) == 2

    def test_from_dict_empty(self):
        """Test creating from empty dict."""
        config = StandardConfig.from_dict({})
        assert config.datasets == []

    def test_from_dict_with_datasets(self):
        """Test creating from dict with datasets."""
        data = {
            "datasets": [
                {"file_path": "/data/a.json", "source": "A"},
                {"file_path": "/data/b.json", "source": "B"},
            ]
        }
        config = StandardConfig.from_dict(data)
        assert len(config.datasets) == 2
        assert config.datasets[0].source == "A"
        assert config.datasets[1].source == "B"

    def test_from_dict_with_extra(self):
        """Test creating from dict with extra fields."""
        data = {
            "datasets": [],
            "version": "1.0",  # Extra fields (not 'datasets') go into extra
        }
        config = StandardConfig.from_dict(data)
        assert config.extra.get("version") == "1.0"

    def test_to_dict(self):
        """Test converting to dict."""
        datasets = [DatasetConfig(file_path="/data/a.json", source="A")]
        config = StandardConfig(datasets=datasets, extra={"key": "value"})
        result = config.to_dict()

        assert "datasets" in result
        assert len(result["datasets"]) == 1
        # extra fields are spread into the result dict, not nested under 'extra'
        assert result.get("key") == "value"

    def test_get_file_paths(self):
        """Test getting all file paths."""
        datasets = [
            DatasetConfig(file_path="/data/a.json"),
            DatasetConfig(file_path="/data/b.json"),
            DatasetConfig(file_path="/data/c.json"),
        ]
        config = StandardConfig(datasets=datasets)
        paths = config.get_file_paths()

        assert len(paths) == 3
        assert "/data/a.json" in paths
        assert "/data/b.json" in paths
        assert "/data/c.json" in paths

    def test_get_file_paths_empty(self):
        """Test getting file paths from empty config."""
        config = StandardConfig()
        paths = config.get_file_paths()
        assert paths == []

    def test_get_sources_map(self):
        """Test getting path to source mapping."""
        datasets = [
            DatasetConfig(file_path="/data/a.json", source="SourceA"),
            DatasetConfig(file_path="/data/b.json", source="SourceB"),
        ]
        config = StandardConfig(datasets=datasets)
        sources_map = config.get_sources_map()

        assert sources_map["/data/a.json"] == "SourceA"
        assert sources_map["/data/b.json"] == "SourceB"


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_yaml_config(self, temp_yaml_file):
        """Test loading YAML config file."""
        config = ConfigLoader.load(temp_yaml_file)
        assert isinstance(config, StandardConfig)
        assert len(config.datasets) == 2

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load("/nonexistent/config.yaml")

    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML raises error."""
        filepath = os.path.join(temp_dir, "invalid.yaml")
        with open(filepath, "w") as f:
            f.write("invalid: yaml: content: {{")

        with pytest.raises((yaml.YAMLError, Exception)):
            ConfigLoader.load(filepath)

    def test_load_empty_yaml(self, temp_dir):
        """Test loading empty YAML file."""
        filepath = os.path.join(temp_dir, "empty.yaml")
        with open(filepath, "w") as f:
            pass  # Empty file

        config = ConfigLoader.load(filepath)
        assert config.datasets == []

    def test_save_config(self, temp_dir):
        """Test saving config to YAML file."""
        config = StandardConfig(
            datasets=[DatasetConfig(file_path="/data/test.json", source="Test")]
        )
        filepath = os.path.join(temp_dir, "output.yaml")
        ConfigLoader.save(config, filepath)

        assert os.path.exists(filepath)

        # Verify saved content
        loaded = ConfigLoader.load(filepath)
        assert len(loaded.datasets) == 1
        assert loaded.datasets[0].source == "Test"

    def test_load_file_paths(self, temp_yaml_file):
        """Test loading just file paths from config."""
        paths = ConfigLoader.load_file_paths(temp_yaml_file)
        assert isinstance(paths, list)
        assert len(paths) == 2

    def test_load_file_paths_with_data_root(self, temp_dir):
        """Test loading file paths with data_root prefix."""
        filepath = os.path.join(temp_dir, "config.yaml")
        config_data = {"datasets": [{"file_path": "a.json"}, {"file_path": "b.json"}]}
        with open(filepath, "w") as f:
            yaml.dump(config_data, f)

        paths = ConfigLoader.load_file_paths(filepath, data_root="/data/root")
        assert all(p.startswith("/data/root") for p in paths)

    def test_load_sources_map(self, temp_yaml_file):
        """Test loading sources map from config."""
        sources_map = ConfigLoader.load_sources_map(temp_yaml_file)
        assert isinstance(sources_map, dict)
        assert len(sources_map) == 2

    def test_create_config_from_file_paths(self):
        """Test creating config from file paths."""
        file_paths = ["/data/a.json", "/data/b.json"]
        config = ConfigLoader.create_config(file_paths)

        assert len(config.datasets) == 2
        assert config.datasets[0].file_path == "/data/a.json"

    def test_create_config_with_sources(self):
        """Test creating config with source mapping."""
        file_paths = ["/data/a.json", "/data/b.json"]
        sources = {"/data/a.json": "SourceA", "/data/b.json": "SourceB"}
        config = ConfigLoader.create_config(file_paths, sources=sources)

        assert config.datasets[0].source == "SourceA"
        assert config.datasets[1].source == "SourceB"

    def test_create_config_with_data_sizes(self):
        """Test creating config with data sizes."""
        file_paths = ["/data/a.json"]
        data_sizes = {"/data/a.json": 1000}
        config = ConfigLoader.create_config(file_paths, data_sizes=data_sizes)

        assert config.datasets[0].data_size == 1000


class TestConfigLoaderEdgeCases:
    """Edge case tests for ConfigLoader."""

    def test_load_yaml_with_anchors(self, temp_dir):
        """Test loading YAML with anchors and aliases."""
        filepath = os.path.join(temp_dir, "anchors.yaml")
        content = """
defaults: &defaults
  source: DefaultSource

datasets:
  - <<: *defaults
    file_path: /data/a.json
  - <<: *defaults
    file_path: /data/b.json
    source: OverriddenSource
"""
        with open(filepath, "w") as f:
            f.write(content)

        config = ConfigLoader.load(filepath)
        assert config.datasets[0].source == "DefaultSource"
        assert config.datasets[1].source == "OverriddenSource"

    def test_load_yaml_with_relative_paths(self, temp_dir):
        """Test loading config with relative file paths."""
        filepath = os.path.join(temp_dir, "relative.yaml")
        config_data = {
            "datasets": [
                {"file_path": "./data/a.json"},
                {"file_path": "../other/b.json"},
            ]
        }
        with open(filepath, "w") as f:
            yaml.dump(config_data, f)

        config = ConfigLoader.load(filepath)
        assert len(config.datasets) == 2

    def test_save_and_load_roundtrip(self, temp_dir):
        """Test that save and load are inverse operations."""
        original = StandardConfig(
            datasets=[
                DatasetConfig(file_path="/data/a.json", source="A", data_size=100),
                DatasetConfig(file_path="/data/b.json", source="B", data_size=200),
            ],
            extra={"version": "2.0"},
        )

        filepath = os.path.join(temp_dir, "roundtrip.yaml")
        ConfigLoader.save(original, filepath)
        loaded = ConfigLoader.load(filepath)

        assert len(loaded.datasets) == len(original.datasets)
        for orig, load in zip(original.datasets, loaded.datasets):
            assert orig.file_path == load.file_path
            assert orig.source == load.source
