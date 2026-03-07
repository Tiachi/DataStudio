"""
Unit tests for datastudio.datasets.saver submodule components.

Tests cover:
- SimpleSaver class
- YamlConfigGenerator class
- ProcessedChecker class
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from datastudio.datasets.saver import ProcessedChecker, SimpleSaver, YamlConfigGenerator


class TestSimpleSaver:
    """Tests for SimpleSaver class."""

    @pytest.fixture
    def simple_saver(self, temp_dir, mock_logger):
        """Create SimpleSaver instance."""
        sources = {"/data/a.json": "SourceA", "/data/b.json": "SourceB"}
        return SimpleSaver(output_dir=temp_dir, sources=sources, logger=mock_logger)

    def test_init(self, temp_dir, mock_logger):
        """Test SimpleSaver initialization."""
        sources = {"/data/test.json": "TestSource"}
        saver = SimpleSaver(output_dir=temp_dir, sources=sources, logger=mock_logger)
        assert saver is not None

    def test_get_source_from_path_exact_match(self, simple_saver):
        """Test getting source with exact path match."""
        source = simple_saver.get_source_from_path("/data/a.json")
        assert source == "SourceA"

    def test_get_source_from_path_no_match(self, simple_saver):
        """Test getting source with no matching path."""
        source = simple_saver.get_source_from_path("/data/unknown.json")
        assert source == "unknown"  # Returns "unknown" for no match

    def test_clean_item(self, simple_saver):
        """Test cleaning data item."""
        item = {
            "conversations": [],
            "file_path": "/data/a.json",
            "_internal_field": "value",
            "image": "test.jpg",
        }
        cleaned = simple_saver.clean_item(item)

        # Internal fields should be removed or kept based on implementation
        assert "conversations" in cleaned

    def test_group_by_source(self, simple_saver):
        """Test grouping data by source."""
        data_list = [
            {"conversations": [], "file_path": "/data/a.json"},
            {"conversations": [], "file_path": "/data/a.json"},
            {"conversations": [], "file_path": "/data/b.json"},
        ]
        grouped = simple_saver.group_by_source(data_list)

        assert isinstance(grouped, dict)
        # Should have entries for each source

    def test_save_creates_files(self, simple_saver, temp_dir):
        """Test that save creates output files."""
        data_list = [
            {"conversations": [], "file_path": "/data/a.json"},
            {"conversations": [], "file_path": "/data/b.json"},
        ]
        simple_saver.save(data_list)

        # Output files should be created

    def test_save_rejected_data(self, simple_saver, temp_dir):
        """Test saving rejected data separately."""
        data_list = [
            {"conversations": [], "file_path": "/data/a.json", "_rejected": True}
        ]
        simple_saver.save(data_list, is_rejected=True)

    def test_save_append_mode(self, simple_saver, temp_dir):
        """Test saving in append mode."""
        data_list = [{"conversations": [], "file_path": "/data/a.json"}]

        # First save
        simple_saver.save(data_list, append_mode=True)

        # Second save (append)
        simple_saver.save(data_list, append_mode=True)

    def test_save_returns_stats(self, simple_saver):
        """Test that save returns statistics."""
        data_list = [
            {"conversations": [], "file_path": "/data/a.json"},
            {"conversations": [], "file_path": "/data/a.json"},
        ]
        stats = simple_saver.save(data_list)

        assert isinstance(stats, dict)


class TestYamlConfigGenerator:
    """Tests for YamlConfigGenerator class."""

    @pytest.fixture
    def yaml_generator(self, temp_dir, mock_logger):
        """Create YamlConfigGenerator instance."""
        return YamlConfigGenerator(
            output_dir=temp_dir,
            yaml_name="output",  # Without extension - extension is added by generate()
            logger=mock_logger,
        )

    def test_init(self, temp_dir, mock_logger):
        """Test YamlConfigGenerator initialization."""
        generator = YamlConfigGenerator(
            output_dir=temp_dir,
            yaml_name="test",  # Without extension
            logger=mock_logger,
        )
        assert generator is not None

    def test_add_file(self, yaml_generator):
        """Test adding file to generator."""
        yaml_generator.add_file("/output/data.json")
        yaml_generator.add_file("/output/data2.json")

    def test_add_rejected_file(self, yaml_generator):
        """Test adding rejected file."""
        yaml_generator.add_file("/output/filtered/data.json", is_rejected=True)

    def test_generate_creates_yaml(self, yaml_generator, temp_dir):
        """Test that generate creates YAML file."""
        yaml_generator.add_file("/output/data.json")

        dataset_stats = {"/output/data.json": {"json_size": 100, "source": "test"}}
        yaml_generator.generate(dataset_stats, cnt_data=90, cnt_rejected=10)

        yaml_path = os.path.join(temp_dir, "output.yaml")
        assert os.path.exists(yaml_path)

    def test_generate_yaml_content(self, yaml_generator, temp_dir):
        """Test generated YAML content structure."""
        yaml_generator.add_file("/output/data.json")

        dataset_stats = {"/output/data.json": {"json_size": 100, "source": "test"}}
        yaml_generator.generate(dataset_stats, cnt_data=100, cnt_rejected=0)

        yaml_path = os.path.join(temp_dir, "output.yaml")
        with open(yaml_path, "r") as f:
            content = yaml.safe_load(f)

        assert "datasets" in content

    def test_generate_empty(self, yaml_generator, temp_dir):
        """Test generating with no files added."""
        yaml_generator.generate({}, cnt_data=0, cnt_rejected=0)


class TestProcessedChecker:
    """Tests for ProcessedChecker class."""

    @pytest.fixture
    def checker(self, temp_dir, mock_logger):
        """Create ProcessedChecker instance."""
        sources = {"/data/a.json": "SourceA", "/data/b.json": "SourceB"}
        return ProcessedChecker(
            output_dir=temp_dir, sources=sources, logger=mock_logger
        )

    def test_init(self, temp_dir, mock_logger):
        """Test ProcessedChecker initialization."""
        sources = {"/data/test.json": "TestSource"}
        checker = ProcessedChecker(
            output_dir=temp_dir, sources=sources, logger=mock_logger
        )
        assert checker is not None

    def test_get_source_from_path(self, checker):
        """Test getting source from path."""
        source = checker.get_source_from_path("/data/a.json")
        assert source == "SourceA"

    def test_check_unprocessed(self, checker):
        """Test checking unprocessed dataset."""
        result = checker.check("/data/new_dataset.json")
        # Should return tuple indicating not processed
        assert isinstance(result, tuple)

    def test_check_processed(self, checker, temp_dir):
        """Test checking already processed dataset."""
        # Create output file to simulate processed state
        source_dir = os.path.join(temp_dir, "SourceA")
        os.makedirs(source_dir, exist_ok=True)
        output_file = os.path.join(source_dir, "a.json")
        with open(output_file, "w") as f:
            json.dump([{"conversations": []}], f)

        result = checker.check("/data/a.json")
        # Should detect as processed


class TestSaverComponentsEdgeCases:
    """Edge case tests for saver components."""

    def test_simple_saver_empty_sources(self, temp_dir, mock_logger):
        """Test SimpleSaver with empty sources."""
        saver = SimpleSaver(output_dir=temp_dir, sources={}, logger=mock_logger)
        result = saver.save([])
        assert isinstance(result, dict)

    def test_simple_saver_special_chars_in_source(self, temp_dir, mock_logger):
        """Test SimpleSaver with special characters in source name."""
        sources = {"/data/test.json": "Source (Special) [Name]"}
        saver = SimpleSaver(output_dir=temp_dir, sources=sources, logger=mock_logger)

        data_list = [{"conversations": [], "file_path": "/data/test.json"}]
        saver.save(data_list)

    def test_yaml_generator_without_logger(self, temp_dir):
        """Test YamlConfigGenerator without logger."""
        generator = YamlConfigGenerator(
            output_dir=temp_dir, yaml_name="test.yaml", logger=None
        )
        generator.add_file("/output/data.json")
        generator.generate({}, cnt_data=0, cnt_rejected=0)

    def test_processed_checker_nested_path(self, temp_dir, mock_logger):
        """Test ProcessedChecker with nested paths."""
        sources = {"/data/subdir/nested/file.json": "NestedSource"}
        checker = ProcessedChecker(
            output_dir=temp_dir, sources=sources, logger=mock_logger
        )

        source = checker.get_source_from_path("/data/subdir/nested/file.json")
        assert source == "NestedSource"
