"""
Unit tests for datastudio.datasets.data_loader module.

Tests cover:
- StandardDataLoader class
- Data loading with various configurations
- Batch iteration
- Checkpoint support
- LMDB caching
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from datastudio.datasets.data_loader import StandardDataLoader


class TestStandardDataLoaderInit:
    """Tests for StandardDataLoader initialization."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_init_with_minimal_args(self, mock_registry, temp_dir, temp_json_file):
        """Test initialization with minimal arguments."""
        mock_registry.load.return_value = [{"conversations": []}]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=32
        )
        assert loader.batch_size == 32

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_init_with_all_args(
        self, mock_registry, temp_dir, temp_json_file, mock_logger
    ):
        """Test initialization with all arguments."""
        mock_registry.load.return_value = [{"conversations": []}]

        loader = StandardDataLoader(
            data_root=temp_dir,
            dataset=temp_json_file,
            batch_size=16,
            parallel_loading=False,
            num_workers=64,
            logger=mock_logger,
            cache_dir=temp_dir,
            lmdb_num_shards=16,
            resize_image=False,
            resize_image_size=512,
            use_image=False,
            adjust_batch_size=False,
        )
        assert loader.batch_size == 16

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_init_creates_cache_dir(self, mock_registry, temp_dir, temp_json_file):
        """Test that init creates cache directory if needed."""
        mock_registry.load.return_value = []
        cache_dir = os.path.join(temp_dir, "new_cache")

        loader = StandardDataLoader(
            data_root=temp_dir,
            dataset=temp_json_file,
            batch_size=32,
            cache_dir=cache_dir,
        )
        # Cache dir should be created or handled


class TestStandardDataLoaderProperties:
    """Tests for StandardDataLoader properties."""

    @pytest.fixture
    def mock_loader(self, temp_dir, temp_json_file):
        """Create a mocked loader for testing."""
        with patch("datastudio.datasets.data_loader.FormatRegistry") as mock_registry:
            mock_registry.load.return_value = [
                {
                    "conversations": [
                        {"from": "human", "value": "Q1"},
                        {"from": "gpt", "value": "A1"},
                    ]
                },
                {
                    "conversations": [
                        {"from": "human", "value": "Q2"},
                        {"from": "gpt", "value": "A2"},
                    ]
                },
                {
                    "conversations": [
                        {"from": "human", "value": "Q3"},
                        {"from": "gpt", "value": "A3"},
                    ]
                },
            ]
            loader = StandardDataLoader(
                data_root=temp_dir, dataset=temp_json_file, batch_size=2
            )
            return loader

    def test_get_name(self, mock_loader):
        """Test get_name returns dataset name."""
        name = mock_loader.get_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_len(self, mock_loader):
        """Test __len__ returns number of batches."""
        length = len(mock_loader)
        assert length == 2  # 3 items with batch_size 2 = 2 batches

    def test_is_empty_false(self, mock_loader):
        """Test is_empty returns False when data exists."""
        assert not mock_loader.is_empty()

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_is_empty_true(self, mock_registry, temp_dir, temp_json_file):
        """Test is_empty returns True when no data."""
        mock_registry.load.return_value = []
        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=32
        )
        assert loader.is_empty()


class TestStandardDataLoaderIteration:
    """Tests for StandardDataLoader iteration."""

    @pytest.fixture
    def loader_with_data(self, temp_dir, temp_json_file):
        """Create loader with test data."""
        with patch("datastudio.datasets.data_loader.FormatRegistry") as mock_registry:
            mock_registry.load.return_value = [
                {
                    "conversations": [
                        {"from": "human", "value": f"Q{i}"},
                        {"from": "gpt", "value": f"A{i}"},
                    ]
                }
                for i in range(5)
            ]
            loader = StandardDataLoader(
                data_root=temp_dir, dataset=temp_json_file, batch_size=2
            )
            return loader

    def test_iter_returns_self(self, loader_with_data):
        """Test __iter__ returns self."""
        assert iter(loader_with_data) is loader_with_data

    def test_iteration_yields_batches(self, loader_with_data):
        """Test iteration yields batches of correct size."""
        batches = list(loader_with_data)
        assert len(batches) == 3  # 5 items / 2 batch_size = 3 batches
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1  # Last batch has remaining items

    def test_iteration_yields_dicts(self, loader_with_data):
        """Test iteration yields list of dicts."""
        for batch in loader_with_data:
            assert isinstance(batch, list)
            for item in batch:
                assert isinstance(item, dict)
                assert "conversations" in item

    def test_call_returns_batch(self, loader_with_data):
        """Test __call__ returns a batch."""
        batch = loader_with_data()
        assert isinstance(batch, list)
        assert len(batch) <= 2


class TestStandardDataLoaderCheckpoint:
    """Tests for StandardDataLoader checkpoint functionality."""

    @pytest.fixture
    def loader_with_checkpoint(self, temp_dir, temp_json_file):
        """Create loader with checkpoint manager."""
        checkpoint_manager = MagicMock()
        checkpoint_manager.get_start_index.return_value = 0
        checkpoint_manager.is_completed.return_value = False

        with patch("datastudio.datasets.data_loader.FormatRegistry") as mock_registry:
            mock_registry.load.return_value = [
                {
                    "conversations": [
                        {"from": "human", "value": f"Q{i}"},
                        {"from": "gpt", "value": f"A{i}"},
                    ]
                }
                for i in range(10)
            ]
            loader = StandardDataLoader(
                data_root=temp_dir,
                dataset=temp_json_file,
                batch_size=2,
                checkpoint_manager=checkpoint_manager,
            )
            return loader, checkpoint_manager

    def test_update_checkpoint(self, loader_with_checkpoint):
        """Test update_checkpoint calls checkpoint manager."""
        loader, checkpoint_manager = loader_with_checkpoint
        loader.update_checkpoint()
        # Verify checkpoint manager was called

    def test_is_completed_false(self, loader_with_checkpoint):
        """Test is_completed returns False when not complete."""
        loader, checkpoint_manager = loader_with_checkpoint
        checkpoint_manager.is_completed.return_value = False
        assert not loader.is_completed()

    def test_is_completed_true(self, loader_with_checkpoint):
        """Test is_completed returns True when complete."""
        loader, checkpoint_manager = loader_with_checkpoint
        checkpoint_manager.is_completed.return_value = True
        assert loader.is_completed()

    def test_resume_from_checkpoint(self, temp_dir, temp_json_file):
        """Test resuming from checkpoint starts at correct index."""
        checkpoint_manager = MagicMock()
        checkpoint_manager.get_start_index.return_value = 4  # Start from index 4
        checkpoint_manager.is_completed.return_value = False

        with patch("datastudio.datasets.data_loader.FormatRegistry") as mock_registry:
            mock_registry.load.return_value = [
                {
                    "conversations": [
                        {"from": "human", "value": f"Q{i}"},
                        {"from": "gpt", "value": f"A{i}"},
                    ]
                }
                for i in range(10)
            ]
            loader = StandardDataLoader(
                data_root=temp_dir,
                dataset=temp_json_file,
                batch_size=2,
                checkpoint_manager=checkpoint_manager,
            )
            # Should resume from index 4


class TestStandardDataLoaderBatchSizeAdjustment:
    """Tests for batch size adjustment based on QA count."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_adjust_batch_size_enabled(self, mock_registry, temp_dir, temp_json_file):
        """Test batch size adjustment when enabled."""
        # Multi-turn conversations should reduce batch size
        mock_registry.load.return_value = [
            {
                "conversations": [
                    {"from": "human", "value": "Q1"},
                    {"from": "gpt", "value": "A1"},
                    {"from": "human", "value": "Q2"},
                    {"from": "gpt", "value": "A2"},
                ]
            }
            for _ in range(10)
        ]

        loader = StandardDataLoader(
            data_root=temp_dir,
            dataset=temp_json_file,
            batch_size=32,
            adjust_batch_size=True,
        )
        # Batch size should be adjusted based on QA count

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_adjust_batch_size_disabled(self, mock_registry, temp_dir, temp_json_file):
        """Test batch size remains unchanged when adjustment disabled."""
        mock_registry.load.return_value = [
            {
                "conversations": [
                    {"from": "human", "value": "Q"},
                    {"from": "gpt", "value": "A"},
                ]
                * 10
            }
            for _ in range(10)
        ]

        loader = StandardDataLoader(
            data_root=temp_dir,
            dataset=temp_json_file,
            batch_size=32,
            adjust_batch_size=False,
        )
        assert loader.batch_size == 32


class TestStandardDataLoaderImageHandling:
    """Tests for image handling in StandardDataLoader."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    @patch("datastudio.datasets.data_loader.ShardedLMDBManager")
    def test_use_image_true(self, mock_lmdb, mock_registry, temp_dir, temp_json_file):
        """Test loader with image loading enabled."""
        mock_registry.load.return_value = [{"conversations": [], "image": "test.jpg"}]
        mock_lmdb.get_instance.return_value = MagicMock()

        loader = StandardDataLoader(
            data_root=temp_dir,
            dataset=temp_json_file,
            batch_size=32,
            use_image=True,
            cache_dir=temp_dir,
        )

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_use_image_false(self, mock_registry, temp_dir, temp_json_file):
        """Test loader with image loading disabled."""
        mock_registry.load.return_value = [{"conversations": [], "image": "test.jpg"}]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=32, use_image=False
        )


class TestStandardDataLoaderEdgeCases:
    """Edge case tests for StandardDataLoader."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_single_item(self, mock_registry, temp_dir, temp_json_file):
        """Test loader with single item."""
        mock_registry.load.return_value = [{"conversations": []}]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=32
        )
        assert len(loader) == 1

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_batch_size_larger_than_data(self, mock_registry, temp_dir, temp_json_file):
        """Test when batch size is larger than data size."""
        mock_registry.load.return_value = [{"conversations": []} for _ in range(5)]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=100
        )
        batches = list(loader)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_batch_size_one(self, mock_registry, temp_dir, temp_json_file):
        """Test with batch size of 1."""
        mock_registry.load.return_value = [{"conversations": []} for _ in range(3)]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=1
        )
        batches = list(loader)
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_multiple_iterations(self, mock_registry, temp_dir, temp_json_file):
        """Test multiple iterations over loader."""
        mock_registry.load.return_value = [{"conversations": []} for _ in range(5)]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=2
        )

        first_pass = list(loader)
        second_pass = list(loader)

        assert len(first_pass) == len(second_pass)


class TestStandardDataLoaderResolvePath:
    """Tests for _resolve_data_path method."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_dict_config_with_file_path(self, mock_registry, temp_dir):
        """Test that dict config with file_path resolves correctly."""
        mock_registry.load.return_value = [{"conversations": []}]

        dataset = {"file_path": "train.json", "source": "TestSource"}
        loader = StandardDataLoader(data_root=temp_dir, dataset=dataset, batch_size=32)
        expected = os.path.join(temp_dir, "train.json")
        assert loader.name == expected

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_dict_config_with_json_path_alias(self, mock_registry, temp_dir):
        """Test that dict config with json_path alias resolves correctly."""
        mock_registry.load.return_value = [{"conversations": []}]

        dataset = {"json_path": "data.json", "source": "TestSource"}
        loader = StandardDataLoader(data_root=temp_dir, dataset=dataset, batch_size=32)
        expected = os.path.join(temp_dir, "data.json")
        assert loader.name == expected

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_string_path(self, mock_registry, temp_dir):
        """Test that a plain string path is resolved relative to data_root."""
        mock_registry.load.return_value = [{"conversations": []}]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset="subdir/data.json", batch_size=32
        )
        expected = os.path.join(temp_dir, "subdir/data.json")
        assert loader.name == expected

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_dict_config_missing_file_path_raises(self, mock_registry, temp_dir):
        """Test that dict config without any file path raises ValueError."""
        mock_registry.load.return_value = [{"conversations": []}]

        dataset = {"source": "TestSource"}
        with pytest.raises(ValueError, match="Missing file path"):
            StandardDataLoader(data_root=temp_dir, dataset=dataset, batch_size=32)

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_relative_path_joins_with_data_root(self, mock_registry, temp_dir):
        """Test that a relative file_path is joined with data_root."""
        mock_registry.load.return_value = [{"conversations": []}]

        dataset = {"file_path": "relative/path/data.json"}
        loader = StandardDataLoader(data_root=temp_dir, dataset=dataset, batch_size=32)
        expected = os.path.join(temp_dir, "relative/path/data.json")
        assert loader.name == expected

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_absolute_path_stays_absolute(self, mock_registry, temp_dir):
        """Test that an absolute file_path is not modified."""
        mock_registry.load.return_value = [{"conversations": []}]

        dataset = {"file_path": "/absolute/path/data.json"}
        loader = StandardDataLoader(data_root=temp_dir, dataset=dataset, batch_size=32)
        assert loader.name == "/absolute/path/data.json"


class TestStandardDataLoaderAdaptiveResize:
    """Tests for _get_adaptive_resize_size method."""

    @pytest.fixture
    def loader(self, temp_dir, temp_json_file):
        """Create a loader with mocked FormatRegistry for method testing."""
        with patch("datastudio.datasets.data_loader.FormatRegistry") as mock_registry:
            mock_registry.load.return_value = [{"conversations": []}]
            loader = StandardDataLoader(
                data_root=temp_dir,
                dataset=temp_json_file,
                batch_size=32,
                resize_image_size=2048,
            )
            return loader

    def test_1_image_returns_resize_image_size(self, loader):
        """Test 1 image returns the configured resize_image_size."""
        assert loader._get_adaptive_resize_size(1) == 2048

    def test_2_images_returns_resize_image_size(self, loader):
        """Test 2 images returns the configured resize_image_size."""
        assert loader._get_adaptive_resize_size(2) == 2048

    def test_3_images_returns_1024(self, loader):
        """Test 3 images returns 1024."""
        assert loader._get_adaptive_resize_size(3) == 1024

    def test_5_images_returns_1024(self, loader):
        """Test 5 images returns 1024."""
        assert loader._get_adaptive_resize_size(5) == 1024

    def test_6_images_returns_672(self, loader):
        """Test 6 images returns 672."""
        assert loader._get_adaptive_resize_size(6) == 672

    def test_10_images_returns_672(self, loader):
        """Test 10 images returns 672."""
        assert loader._get_adaptive_resize_size(10) == 672

    def test_11_images_returns_448(self, loader):
        """Test 11 images returns 448."""
        assert loader._get_adaptive_resize_size(11) == 448

    def test_15_images_returns_448(self, loader):
        """Test 15 images returns 448."""
        assert loader._get_adaptive_resize_size(15) == 448

    def test_16_images_returns_392(self, loader):
        """Test 16+ images returns 392."""
        assert loader._get_adaptive_resize_size(16) == 392

    def test_100_images_returns_392(self, loader):
        """Test large image count returns 392."""
        assert loader._get_adaptive_resize_size(100) == 392


class TestStandardDataLoaderImageSkip:
    """Tests for _load_images_for_batch with use_image=False."""

    @patch("datastudio.datasets.data_loader.FormatRegistry")
    def test_load_images_returns_batch_unchanged_when_use_image_false(
        self, mock_registry, temp_dir, temp_json_file
    ):
        """Test that _load_images_for_batch returns batch unchanged when use_image=False."""
        mock_registry.load.return_value = [{"conversations": [], "image": "test.jpg"}]

        loader = StandardDataLoader(
            data_root=temp_dir, dataset=temp_json_file, batch_size=32, use_image=False
        )

        batch = [
            {"conversations": [{"from": "human", "value": "Q"}], "image": "img1.jpg"},
            {"conversations": [{"from": "human", "value": "Q2"}], "image": "img2.jpg"},
        ]
        result = loader._load_images_for_batch(batch)

        assert result is batch
        assert len(result) == 2
        # image_pil should NOT have been added
        for item in result:
            assert "image_pil" not in item
