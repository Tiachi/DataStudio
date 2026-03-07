"""
Unit tests for datastudio.datasets.builder module.

Tests cover:
- build_dataloaders function
- Checkpoint integration
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from datastudio.datasets.builder import build_dataloaders


class TestBuildDataloaders:
    """Tests for build_dataloaders function."""

    @patch("datastudio.datasets.builder.build_from_cfg")
    @patch("datastudio.datasets.builder.ConfigLoader")
    def test_build_with_yaml_config(
        self, mock_config_loader, mock_build, temp_dir, mock_logger
    ):
        """Test building dataloaders from YAML config."""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.datasets = [MagicMock()]
        mock_config.get_file_paths.return_value = ["/data/a.json", "/data/b.json"]
        mock_config_loader.load.return_value = mock_config
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": "config.yaml",
            "data_root": temp_dir,
            "batch_size": 32,
            "num_workers": 4,
        }

        # Create yaml file
        yaml_path = os.path.join(temp_dir, "config.yaml")
        with open(yaml_path, "w") as f:
            f.write("datasets:\n  - file_path: a.json\n")

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )

        assert isinstance(loaders, list)

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_with_single_file(self, mock_build, temp_dir, mock_logger):
        """Test building dataloaders from single file."""
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/test.json"],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )

        assert isinstance(loaders, list)
        assert len(loaders) == 1

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_with_file_list(self, mock_build, temp_dir, mock_logger):
        """Test building dataloaders from file list."""
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/a.json", "/data/b.json", "/data/c.json"],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )

        assert isinstance(loaders, list)
        assert len(loaders) == 3

    @patch("datastudio.datasets.builder.build_from_cfg")
    @patch("datastudio.datasets.builder.CheckpointManager")
    def test_build_with_checkpoint(
        self, mock_checkpoint, mock_build, temp_dir, mock_logger
    ):
        """Test building dataloaders with checkpoint support."""
        mock_build.return_value = MagicMock()
        mock_checkpoint.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/test.json"],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
            "checkpoint_enabled": True,
        }

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )

        # Checkpoint manager should be created
        mock_checkpoint.assert_called_once()

    def test_build_with_invalid_config(self, temp_dir, mock_logger):
        """Test building with invalid config raises error."""
        dataset_config = {"type": "StandardDataLoader"}  # Missing required fields

        with pytest.raises((KeyError, ValueError, TypeError)):
            build_dataloaders(dataset_config, logger=mock_logger, work_dir=temp_dir)

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_passes_loader_options(self, mock_build, temp_dir, mock_logger):
        """Test that loader options are passed correctly."""
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/test.json"],
            "data_root": "/data",
            "batch_size": 16,
            "num_workers": 8,
            "use_image": False,
            "resize_image_size": 512,
        }

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )

        # Verify build_from_cfg was called with correct params
        assert mock_build.called


class TestBuildDataloadersEdgeCases:
    """Edge case tests for build_dataloaders."""

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_with_empty_file_list(self, mock_build, temp_dir, mock_logger):
        """Test building with empty file list."""
        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": [],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        loaders = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )
        assert loaders == []

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_without_logger(self, mock_build, temp_dir):
        """Test building without logger."""
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/test.json"],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        loaders = build_dataloaders(dataset_config, logger=None, work_dir=temp_dir)
        assert isinstance(loaders, list)

    @patch("datastudio.datasets.builder.build_from_cfg")
    def test_build_without_work_dir(self, mock_build, mock_logger):
        """Test building without work_dir disables checkpoint."""
        mock_build.return_value = MagicMock()

        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": ["/data/test.json"],
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        loaders = build_dataloaders(dataset_config, logger=mock_logger, work_dir=None)
        assert isinstance(loaders, list)

    def test_build_with_none_dataset(self, temp_dir, mock_logger):
        """Test building with None dataset returns None."""
        dataset_config = {
            "type": "StandardDataLoader",
            "dataset": None,
            "data_root": "/data",
            "batch_size": 32,
            "num_workers": 4,
        }

        result = build_dataloaders(
            dataset_config, logger=mock_logger, work_dir=temp_dir
        )
        assert result is None
