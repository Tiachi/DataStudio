"""Checkpoint manager for item-level resume support."""

import json
import os
import time
from typing import Dict, Optional


class CheckpointManager:
    """Tracks per-file processing progress for resumable execution.

    Persists a JSON file mapping each data file to its last processed
    index, allowing the pipeline to resume from the exact interruption
    point.
    """

    def __init__(
        self, work_dir: str, checkpoint_name: str = "checkpoint.json", logger=None
    ):
        """Initialize checkpoint manager.

        Args:
            work_dir: Working directory for checkpoint file.
            checkpoint_name: Name of checkpoint file.
            logger: Logger instance.
        """
        self.work_dir = work_dir
        self.checkpoint_file = os.path.join(work_dir, checkpoint_name)
        self.logger = logger
        self.checkpoints: Dict[str, dict] = {}

        os.makedirs(work_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load checkpoint from file."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    self.checkpoints = json.load(f)
                if self.logger:
                    self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load checkpoint: {e}")
                self.checkpoints = {}

    def _save(self) -> None:
        """Save checkpoint to file."""
        try:
            # Write to temp file first, then rename for atomicity
            temp_file = self.checkpoint_file + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.checkpoints, f, indent=2, ensure_ascii=False)
            try:
                os.replace(temp_file, self.checkpoint_file)
            except OSError:
                # Fallback for cross-device moves: copy and delete
                import shutil

                shutil.copy2(temp_file, self.checkpoint_file)
                os.remove(temp_file)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to save checkpoint: {e}")

    def get_start_index(self, file_path: str, total_items: int) -> int:
        """Get the starting index for processing a file.

        Args:
            file_path: Path to the data file.
            total_items: Total number of items in the file.

        Returns:
            Index to start processing from (0 if new file).
        """
        if file_path in self.checkpoints:
            ckpt = self.checkpoints[file_path]
            if ckpt.get("completed", False):
                return -1  # Already completed, skip entirely

            saved_total = ckpt.get("total_items", 0)
            if saved_total != total_items:
                if self.logger:
                    self.logger.warning(
                        f"File {file_path} item count changed ({saved_total} -> {total_items}), "
                        "restarting from beginning"
                    )
                return 0

            start_idx = ckpt.get("processed_idx", 0)
            if self.logger:
                self.logger.info(
                    f"Resuming {file_path} from index {start_idx}/{total_items}"
                )
            return start_idx

        return 0

    def is_completed(self, file_path: str) -> bool:
        """Check if a file has been fully processed.

        Args:
            file_path: Path to the data file.

        Returns:
            True if file is completed.
        """
        if file_path in self.checkpoints:
            return self.checkpoints[file_path].get("completed", False)
        return False

    def update(self, file_path: str, processed_idx: int, total_items: int) -> None:
        """Update checkpoint for a file.

        Args:
            file_path: Path to the data file.
            processed_idx: Index of last processed item (exclusive).
            total_items: Total number of items in the file.
        """
        completed = processed_idx >= total_items

        self.checkpoints[file_path] = {
            "processed_idx": processed_idx,
            "total_items": total_items,
            "last_update": time.time(),
            "completed": completed,
        }
        self._save()

        if self.logger and completed:
            self.logger.info(f"File {file_path} processing completed")

    def mark_completed(self, file_path: str, total_items: int = 0) -> None:
        """Mark a file as fully processed.

        Args:
            file_path: Path to the data file.
            total_items: Total number of items in the file (optional, for accuracy).
        """
        if file_path in self.checkpoints:
            ckpt = self.checkpoints[file_path]
            ckpt["completed"] = True
            ckpt["last_update"] = time.time()
            if total_items > 0:
                ckpt["processed_idx"] = total_items
                ckpt["total_items"] = total_items
        else:
            self.checkpoints[file_path] = {
                "processed_idx": total_items,
                "total_items": total_items,
                "last_update": time.time(),
                "completed": True,
            }
        self._save()

    def get_progress(self, file_path: str) -> Optional[Dict]:
        """Get progress info for a file.

        Args:
            file_path: Path to the data file.

        Returns:
            Progress dict or None if not tracked.
        """
        return self.checkpoints.get(file_path)

    def clear(self, file_path: Optional[str] = None) -> None:
        """Clear checkpoint(s).

        Args:
            file_path: Specific file to clear, or None to clear all.
        """
        if file_path:
            self.checkpoints.pop(file_path, None)
        else:
            self.checkpoints = {}
        self._save()
