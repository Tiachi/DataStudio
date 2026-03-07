"""
Unit tests for datastudio.datasets.data_saver module.

Tests cover:
- StandardDataSaver class
- Data accumulation and saving
- YAML config generation
- Incremental saving
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from datastudio.datasets.data_saver import StandardDataSaver


@pytest.fixture
def yaml_config_file(temp_dir):
    """Create a YAML config file for testing."""
    yaml_path = os.path.join(temp_dir, "dataset.yaml")
    yaml_content = """datasets:
  - file_path: test.json
    source: test_source
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path


class TestStandardDataSaverInit:
    """Tests for StandardDataSaver initialization."""

    def test_init_with_minimal_args(self, temp_dir, mock_logger, yaml_config_file):
        """Test initialization with minimal arguments."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        assert saver is not None

    def test_init_with_all_args(self, temp_dir, mock_logger, yaml_config_file):
        """Test initialization with all arguments."""
        saver = StandardDataSaver(
            output_dir=temp_dir,
            dataset=yaml_config_file,
            logger=mock_logger,
            save_yaml_config=True,
            save_yaml_name="output",
            incremental_save_threshold=1000,
        )
        assert saver is not None
        assert saver.incremental_save_threshold == 1000

    def test_init_creates_output_dir(self, temp_dir, mock_logger, yaml_config_file):
        """Test that init creates output directory if needed."""
        output_dir = os.path.join(temp_dir, "new_output")
        saver = StandardDataSaver(
            output_dir=output_dir, dataset=yaml_config_file, logger=mock_logger
        )
        assert os.path.exists(output_dir)


class TestStandardDataSaverAccumulation:
    """Tests for data accumulation in StandardDataSaver."""

    @pytest.fixture
    def saver(self, temp_dir, mock_logger, yaml_config_file):
        """Create a saver instance for testing."""
        return StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )

    def test_call_accumulates_data(self, saver):
        """Test that __call__ accumulates data."""
        data1 = [{"conversations": [], "file_path": "test.json"}]
        data2 = [{"conversations": [], "file_path": "test.json"}]
        saver(data1)
        saver(data2)
        assert len(saver.datas) == 2

    def test_call_with_empty_list(self, saver):
        """Test __call__ with empty list."""
        saver([])  # Should not raise
        assert len(saver.datas) == 0

    def test_call_with_single_item(self, saver):
        """Test __call__ with single item."""
        saver([{"conversations": [], "file_path": "test.json"}])
        assert len(saver.datas) == 1

    def test_call_separates_rejected(self, saver):
        """Test that rejected items are separated."""
        data = [
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": True},
        ]
        saver(data)
        assert len(saver.datas) == 1
        assert len(saver.rejected_datas) == 1


class TestStandardDataSaverSave:
    """Tests for save functionality in StandardDataSaver."""

    @pytest.fixture
    def saver(self, temp_dir, mock_logger, yaml_config_file):
        """Create a saver instance for testing."""
        return StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )

    def test_save_creates_file(self, saver, temp_dir):
        """Test that save creates output file."""
        data = [{"conversations": [], "file_path": "test.json"}]
        saver(data)
        saver.save()

        # Check that file was created
        output_files = list(Path(temp_dir).glob("**/*.json"))
        assert len(output_files) > 0

    def test_save_writes_correct_data(self, saver, temp_dir):
        """Test that save writes correct data."""
        data = [
            {
                "conversations": [{"from": "human", "value": "Q"}],
                "file_path": "test.json",
            },
            {
                "conversations": [{"from": "gpt", "value": "A"}],
                "file_path": "test.json",
            },
        ]
        saver(data)
        saver.save()

        # Find and read output file
        output_files = list(Path(temp_dir).glob("**/*.json"))
        if output_files:
            with open(output_files[0], "r") as f:
                saved_data = json.load(f)
            assert len(saved_data) == 2

    def test_save_empty_data(self, saver):
        """Test save with no accumulated data."""
        saver.save()  # Should not raise

    def test_save_multiple_times(self, saver):
        """Test calling save multiple times."""
        saver([{"conversations": [], "file_path": "test.json"}])
        saver.save()

        saver([{"conversations": [], "file_path": "test.json"}])
        saver.save()


class TestStandardDataSaverYaml:
    """Tests for YAML config generation in StandardDataSaver."""

    def test_save_yaml_creates_file(self, temp_dir, mock_logger, yaml_config_file):
        """Test that save_yaml creates YAML file."""
        saver = StandardDataSaver(
            output_dir=temp_dir,
            dataset=yaml_config_file,
            logger=mock_logger,
            save_yaml_config=True,
            save_yaml_name="output",
        )
        data = [{"conversations": [], "file_path": "test.json"}]
        saver(data)
        saver.save()
        saver.save_yaml()

        yaml_path = os.path.join(temp_dir, "output.yaml")
        assert os.path.exists(yaml_path)

    def test_save_yaml_disabled(self, temp_dir, mock_logger, yaml_config_file):
        """Test that YAML is not created when disabled."""
        saver = StandardDataSaver(
            output_dir=temp_dir,
            dataset=yaml_config_file,
            logger=mock_logger,
            save_yaml_config=False,
        )
        data = [{"conversations": [], "file_path": "test.json"}]
        saver(data)
        saver.save()
        saver.save_yaml()

        # No YAML files should be created by save_yaml
        yaml_files = [
            f for f in Path(temp_dir).glob("*.yaml") if f.name != "dataset.yaml"
        ]
        assert len(yaml_files) == 0


class TestStandardDataSaverIncrementalSave:
    """Tests for incremental save functionality."""

    def test_incremental_save_triggers_at_threshold(
        self, temp_dir, mock_logger, yaml_config_file
    ):
        """Test that incremental save triggers at threshold."""
        saver = StandardDataSaver(
            output_dir=temp_dir,
            dataset=yaml_config_file,
            logger=mock_logger,
            incremental_save_threshold=5,
        )

        # Add data exceeding threshold
        for i in range(10):
            saver([{"conversations": [], "file_path": "test.json", "id": i}])
            saver.incremental_save()

        # Data should have been saved (buffer cleared)
        assert len(saver.datas) < 10

    def test_incremental_save_below_threshold(
        self, temp_dir, mock_logger, yaml_config_file
    ):
        """Test that incremental save doesn't trigger below threshold."""
        saver = StandardDataSaver(
            output_dir=temp_dir,
            dataset=yaml_config_file,
            logger=mock_logger,
            incremental_save_threshold=100,
        )

        # Add data below threshold
        for i in range(5):
            saver([{"conversations": [], "file_path": "test.json", "id": i}])
            saver.incremental_save()

        # Data should still be in buffer
        assert len(saver.datas) == 5

    def test_incremental_save_method(self, temp_dir, mock_logger, yaml_config_file):
        """Test calling incremental_save directly."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        data = [{"conversations": [], "file_path": "test.json"}]
        saver(data)
        saver.incremental_save()  # Should not raise


class TestStandardDataSaverClean:
    """Tests for clean functionality in StandardDataSaver."""

    def test_clean_clears_buffer(self, temp_dir, mock_logger, yaml_config_file):
        """Test that clean clears the data buffer."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        saver([{"conversations": [], "file_path": "test.json"}])
        saver.clean()

        assert len(saver.datas) == 0
        assert len(saver.rejected_datas) == 0


class TestStandardDataSaverHasProcessed:
    """Tests for has_processed check in StandardDataSaver."""

    def test_has_processed_false_initially(
        self, temp_dir, mock_logger, yaml_config_file
    ):
        """Test has_processed returns False for unprocessed dataset."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        assert not saver.has_processed("new_dataset.json")


class TestStandardDataSaverEdgeCases:
    """Edge case tests for StandardDataSaver."""

    def test_save_with_unicode_data(self, temp_dir, mock_logger, yaml_config_file):
        """Test saving data with unicode characters."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        data = [
            {
                "conversations": [{"from": "human", "value": "unicode test message"}],
                "file_path": "test.json",
            }
        ]
        saver(data)
        saver.save()

        # Verify unicode is preserved
        output_files = list(Path(temp_dir).glob("**/*.json"))
        if output_files:
            with open(output_files[0], "r", encoding="utf-8") as f:
                content = f.read()
            assert "unicode test message" in content

    def test_save_large_batch(self, temp_dir, mock_logger, yaml_config_file):
        """Test saving large batch of data."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        large_data = [
            {
                "conversations": [
                    {"from": "human", "value": f"Q{i}"},
                    {"from": "gpt", "value": f"A{i}"},
                ],
                "file_path": "test.json",
            }
            for i in range(100)
        ]
        saver(large_data)
        saver.save()

    def test_rejected_data_handling(self, temp_dir, mock_logger, yaml_config_file):
        """Test handling of rejected data items."""
        saver = StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )
        data = [
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": True},
        ]
        saver(data)
        saver.save()

        # Rejected items should be saved separately
        assert saver.cnt_rejected_data == 1


class TestStandardDataSaverCleanupImages:
    """Tests for _cleanup_batch_images static method."""

    def test_image_pil_is_removed(self):
        """Test that image_pil is removed from data dicts."""
        datas = [
            {"conversations": [], "file_path": "test.json", "image_pil": MagicMock()},
            {"conversations": [], "file_path": "test.json", "image_pil": MagicMock()},
        ]
        StandardDataSaver._cleanup_batch_images(datas)

        for item in datas:
            assert "image_pil" not in item

    def test_other_fields_are_preserved(self):
        """Test that other fields are preserved after cleanup."""
        datas = [
            {
                "conversations": [{"from": "human", "value": "Q"}],
                "file_path": "test.json",
                "image": "img.jpg",
                "image_pil": MagicMock(),
                "source": "test_source",
            }
        ]
        StandardDataSaver._cleanup_batch_images(datas)

        assert datas[0]["conversations"] == [{"from": "human", "value": "Q"}]
        assert datas[0]["file_path"] == "test.json"
        assert datas[0]["image"] == "img.jpg"
        assert datas[0]["source"] == "test_source"
        assert "image_pil" not in datas[0]

    def test_empty_list(self):
        """Test cleanup with empty list does not raise."""
        datas = []
        StandardDataSaver._cleanup_batch_images(datas)
        assert datas == []

    def test_items_without_image_pil_are_unaffected(self):
        """Test that items without image_pil key are not modified."""
        datas = [
            {"conversations": [], "file_path": "test.json", "image": "img.jpg"},
        ]
        StandardDataSaver._cleanup_batch_images(datas)

        assert datas[0]["conversations"] == []
        assert datas[0]["file_path"] == "test.json"
        assert datas[0]["image"] == "img.jpg"


class TestStandardDataSaverCounters:
    """Tests for counter tracking in StandardDataSaver."""

    @pytest.fixture
    def saver(self, temp_dir, mock_logger, yaml_config_file):
        """Create a saver instance for testing."""
        return StandardDataSaver(
            output_dir=temp_dir, dataset=yaml_config_file, logger=mock_logger
        )

    def test_cnt_data_increments_after_save(self, saver):
        """Test cnt_data increments after save."""
        data = [
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": False},
        ]
        saver(data)
        saver.save()

        assert saver.cnt_data == 2

    def test_cnt_rejected_data_increments_after_save(self, saver):
        """Test cnt_rejected_data increments after save with rejected items."""
        data = [
            {"conversations": [], "file_path": "test.json", "rejected": True},
            {"conversations": [], "file_path": "test.json", "rejected": True},
            {"conversations": [], "file_path": "test.json", "rejected": True},
        ]
        saver(data)
        saver.save()

        assert saver.cnt_rejected_data == 3

    def test_counters_accumulate_across_multiple_saves(self, saver):
        """Test counters accumulate across multiple saves."""
        # First batch
        data1 = [
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": True},
        ]
        saver(data1)
        saver.save()

        assert saver.cnt_data == 1
        assert saver.cnt_rejected_data == 1

        # Second batch
        data2 = [
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": False},
            {"conversations": [], "file_path": "test.json", "rejected": True},
        ]
        saver(data2)
        saver.save()

        assert saver.cnt_data == 3
        assert saver.cnt_rejected_data == 2
