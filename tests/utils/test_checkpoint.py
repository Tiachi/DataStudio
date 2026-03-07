"""
Unit tests for datastudio.utils.checkpoint module.

Tests cover:
- CheckpointManager class
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from datastudio.utils.checkpoint import CheckpointManager


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_init_basic(self, temp_dir):
        """Test basic initialization."""
        manager = CheckpointManager(work_dir=temp_dir)
        assert manager is not None

    def test_init_with_custom_name(self, temp_dir):
        """Test initialization with custom checkpoint name."""
        manager = CheckpointManager(
            work_dir=temp_dir, checkpoint_name="custom_checkpoint.json"
        )
        assert manager is not None

    def test_init_with_logger(self, temp_dir, mock_logger):
        """Test initialization with logger."""
        manager = CheckpointManager(work_dir=temp_dir, logger=mock_logger)
        assert manager is not None

    def test_init_creates_work_dir(self, temp_dir):
        """Test that init creates work_dir if needed."""
        new_dir = os.path.join(temp_dir, "new_work_dir")
        manager = CheckpointManager(work_dir=new_dir)

        assert os.path.exists(new_dir)


class TestCheckpointManagerGetStartIndex:
    """Tests for CheckpointManager.get_start_index method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_get_start_index_no_checkpoint(self, manager):
        """Test get_start_index with no existing checkpoint."""
        index = manager.get_start_index("/data/test.json", total_items=100)
        assert index == 0

    def test_get_start_index_with_checkpoint(self, manager, temp_dir):
        """Test get_start_index with existing checkpoint."""
        # Create checkpoint - processed_idx is the last processed index (exclusive)
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        index = manager.get_start_index("/data/test.json", total_items=100)
        assert index == 50  # Returns processed_idx to resume from

    def test_get_start_index_completed(self, manager):
        """Test get_start_index for completed file."""
        manager.mark_completed("/data/test.json", total_items=100)

        index = manager.get_start_index("/data/test.json", total_items=100)
        # Should indicate completion


class TestCheckpointManagerIsCompleted:
    """Tests for CheckpointManager.is_completed method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_is_completed_false(self, manager):
        """Test is_completed returns False for incomplete file."""
        assert not manager.is_completed("/data/test.json")

    def test_is_completed_true(self, manager):
        """Test is_completed returns True for completed file."""
        manager.mark_completed("/data/test.json")
        assert manager.is_completed("/data/test.json")

    def test_is_completed_partial(self, manager):
        """Test is_completed for partially processed file."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)
        assert not manager.is_completed("/data/test.json")


class TestCheckpointManagerUpdate:
    """Tests for CheckpointManager.update method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_update_creates_checkpoint(self, manager, temp_dir):
        """Test update creates checkpoint file."""
        manager.update("/data/test.json", processed_idx=10, total_items=100)

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        assert os.path.exists(checkpoint_path)

    def test_update_stores_progress(self, manager):
        """Test update stores progress correctly."""
        manager.update("/data/test.json", processed_idx=25, total_items=100)

        progress = manager.get_progress("/data/test.json")
        assert progress is not None
        assert progress["processed_idx"] == 25

    def test_update_multiple_files(self, manager):
        """Test update handles multiple files."""
        manager.update("/data/file1.json", processed_idx=10, total_items=50)
        manager.update("/data/file2.json", processed_idx=20, total_items=100)

        progress1 = manager.get_progress("/data/file1.json")
        progress2 = manager.get_progress("/data/file2.json")

        assert progress1["processed_idx"] == 10
        assert progress2["processed_idx"] == 20

    def test_update_overwrites_previous(self, manager):
        """Test update overwrites previous progress."""
        manager.update("/data/test.json", processed_idx=10, total_items=100)
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        progress = manager.get_progress("/data/test.json")
        assert progress["processed_idx"] == 50


class TestCheckpointManagerMarkCompleted:
    """Tests for CheckpointManager.mark_completed method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_mark_completed(self, manager):
        """Test mark_completed sets completed status."""
        manager.mark_completed("/data/test.json", total_items=100)

        assert manager.is_completed("/data/test.json")

    def test_mark_completed_with_total(self, manager):
        """Test mark_completed stores total items."""
        manager.mark_completed("/data/test.json", total_items=100)

        progress = manager.get_progress("/data/test.json")
        assert progress["total_items"] == 100


class TestCheckpointManagerGetProgress:
    """Tests for CheckpointManager.get_progress method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_get_progress_no_checkpoint(self, manager):
        """Test get_progress with no checkpoint."""
        progress = manager.get_progress("/data/test.json")
        assert progress is None

    def test_get_progress_with_checkpoint(self, manager):
        """Test get_progress with checkpoint."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        progress = manager.get_progress("/data/test.json")

        assert progress is not None
        assert "processed_idx" in progress
        assert "total_items" in progress


class TestCheckpointManagerClear:
    """Tests for CheckpointManager.clear method."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_clear_specific_file(self, manager):
        """Test clearing checkpoint for specific file."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)
        manager.clear("/data/test.json")

        progress = manager.get_progress("/data/test.json")
        assert progress is None

    def test_clear_all(self, manager):
        """Test clearing all checkpoints."""
        manager.update("/data/file1.json", processed_idx=10, total_items=50)
        manager.update("/data/file2.json", processed_idx=20, total_items=100)

        manager.clear()

        assert manager.get_progress("/data/file1.json") is None
        assert manager.get_progress("/data/file2.json") is None

    def test_clear_nonexistent(self, manager):
        """Test clearing nonexistent checkpoint."""
        manager.clear("/data/nonexistent.json")  # Should not raise


class TestCheckpointManagerPersistence:
    """Tests for checkpoint persistence."""

    def test_checkpoint_persists_across_instances(self, temp_dir):
        """Test checkpoint persists across manager instances."""
        # First instance
        manager1 = CheckpointManager(work_dir=temp_dir)
        manager1.update("/data/test.json", processed_idx=50, total_items=100)

        # Second instance
        manager2 = CheckpointManager(work_dir=temp_dir)
        progress = manager2.get_progress("/data/test.json")

        assert progress is not None
        assert progress["processed_idx"] == 50

    def test_checkpoint_file_format(self, temp_dir):
        """Test checkpoint file is valid JSON."""
        manager = CheckpointManager(work_dir=temp_dir)
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)


class TestCheckpointManagerEdgeCases:
    """Edge case tests for CheckpointManager."""

    def test_special_characters_in_path(self, temp_dir):
        """Test handling paths with special characters."""
        manager = CheckpointManager(work_dir=temp_dir)

        path = "/data/test (copy) [1].json"
        manager.update(path, processed_idx=10, total_items=50)

        progress = manager.get_progress(path)
        assert progress is not None

    def test_very_long_path(self, temp_dir):
        """Test handling very long file paths."""
        manager = CheckpointManager(work_dir=temp_dir)

        path = "/data/" + "a" * 200 + ".json"
        manager.update(path, processed_idx=10, total_items=50)

        progress = manager.get_progress(path)
        assert progress is not None

    def test_unicode_path(self, temp_dir):
        """Test handling unicode in file paths."""
        manager = CheckpointManager(work_dir=temp_dir)

        path = "/data/unicode_file.json"
        manager.update(path, processed_idx=10, total_items=50)

        progress = manager.get_progress(path)
        assert progress is not None

    def test_concurrent_updates(self, temp_dir):
        """Test handling concurrent updates (basic)."""
        manager = CheckpointManager(work_dir=temp_dir)

        # Simulate rapid updates
        for i in range(100):
            manager.update("/data/test.json", processed_idx=i, total_items=100)

        progress = manager.get_progress("/data/test.json")
        assert progress["processed_idx"] == 99

    def test_zero_total_items(self, temp_dir):
        """Test handling zero total items."""
        manager = CheckpointManager(work_dir=temp_dir)

        manager.mark_completed("/data/empty.json", total_items=0)

        assert manager.is_completed("/data/empty.json")


class TestCheckpointManagerGetStartIndexCompleted:
    """Tests for get_start_index when file is already completed."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_get_start_index_completed(self, manager):
        """Test get_start_index returns -1 when file is completed."""
        manager.mark_completed("/data/test.json", total_items=100)

        index = manager.get_start_index("/data/test.json", total_items=100)
        assert index == -1


class TestCheckpointManagerItemCountChanged:
    """Tests for checkpoint reset when total_items changes between saves."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_item_count_changed_resets_to_zero(self, manager):
        """Test that changing total_items between saves resets start index to 0."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        # Now query with a different total_items
        index = manager.get_start_index("/data/test.json", total_items=200)
        assert index == 0

    def test_item_count_unchanged_resumes(self, manager):
        """Test that same total_items resumes from saved index."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        index = manager.get_start_index("/data/test.json", total_items=100)
        assert index == 50

    def test_item_count_decreased_resets(self, manager):
        """Test that decreasing total_items also resets to 0."""
        manager.update("/data/test.json", processed_idx=50, total_items=100)

        index = manager.get_start_index("/data/test.json", total_items=80)
        assert index == 0


class TestCheckpointManagerAutoComplete:
    """Tests for automatic completion when processed_idx >= total_items."""

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager for testing."""
        return CheckpointManager(work_dir=temp_dir)

    def test_update_auto_completes_at_total(self, manager):
        """Test that update with processed_idx == total_items marks as completed."""
        manager.update("/data/test.json", processed_idx=100, total_items=100)

        assert manager.is_completed("/data/test.json")

    def test_update_auto_completes_beyond_total(self, manager):
        """Test that update with processed_idx > total_items marks as completed."""
        manager.update("/data/test.json", processed_idx=150, total_items=100)

        assert manager.is_completed("/data/test.json")

    def test_update_not_completed_below_total(self, manager):
        """Test that update with processed_idx < total_items does not complete."""
        manager.update("/data/test.json", processed_idx=99, total_items=100)

        assert not manager.is_completed("/data/test.json")

    def test_auto_complete_sets_completed_flag_in_progress(self, manager):
        """Test that auto-completed progress dict has completed=True."""
        manager.update("/data/test.json", processed_idx=100, total_items=100)

        progress = manager.get_progress("/data/test.json")
        assert progress is not None
        assert progress["completed"] is True
        assert progress["processed_idx"] == 100
        assert progress["total_items"] == 100


class TestCheckpointManagerCorruptFile:
    """Tests for handling corrupt checkpoint files gracefully."""

    def test_corrupt_json_file_handled_gracefully(self, temp_dir):
        """Test that a corrupt checkpoint file results in empty checkpoints."""
        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write("{invalid json content!!! not valid")

        # Should not raise; should gracefully recover
        manager = CheckpointManager(work_dir=temp_dir)

        # Checkpoints should be empty after failed load
        assert manager.get_progress("/data/test.json") is None

    def test_corrupt_file_allows_new_updates(self, temp_dir):
        """Test that after corrupt file, new updates work normally."""
        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write("not json at all {{{")

        manager = CheckpointManager(work_dir=temp_dir)

        # Should be able to update and retrieve progress normally
        manager.update("/data/test.json", processed_idx=25, total_items=100)
        progress = manager.get_progress("/data/test.json")
        assert progress is not None
        assert progress["processed_idx"] == 25

    def test_corrupt_file_with_logger_warns(self, temp_dir, mock_logger):
        """Test that a corrupt file logs a warning when logger is provided."""
        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write("{{broken}}")

        manager = CheckpointManager(work_dir=temp_dir, logger=mock_logger)

        # Logger should have received a warning about the failed load
        mock_logger.warning.assert_called()

    def test_empty_file_handled_gracefully(self, temp_dir):
        """Test that an empty checkpoint file is handled gracefully."""
        checkpoint_path = os.path.join(temp_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write("")

        manager = CheckpointManager(work_dir=temp_dir)

        assert manager.get_progress("/data/test.json") is None
