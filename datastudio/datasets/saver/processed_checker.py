"""Processed data detection."""

import os
from typing import Optional

from datastudio.datasets.formatters import FormatRegistry


class ProcessedChecker:
    """Checker for detecting already processed datasets.

    Checks if processed output files exist in the output directory.
    Used for skipping already completed files during batch processing.

    Note:
        When using item-level checkpointing, this checker only detects
        fully completed files. Partially processed files are managed
        by CheckpointManager.

    Attributes:
        output_dir: Output directory path.
        sources: Mapping from file_path to source name.
        logger: Logger instance.
    """

    def __init__(self, output_dir: str, sources: dict, logger=None):
        """Initialize the checker.

        Args:
            output_dir: Output directory path.
            sources: Mapping from file_path to source name.
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.sources = sources
        self.logger = logger

    def get_source_from_path(self, file_path: str) -> Optional[str]:
        """Get source name from file path.

        Args:
            file_path: Path to data file.

        Returns:
            Source name or None if not found.
        """
        if file_path in self.sources:
            return self.sources[file_path]

        filename = os.path.basename(file_path)
        for path, source in self.sources.items():
            if os.path.basename(path) == filename:
                return source

        return None

    def check(self, dataset) -> tuple:
        """Check if dataset has already been processed.

        Args:
            dataset: Dataset object (must have 'datas' attribute).

        Returns:
            tuple: (is_processed, stats)
                - is_processed: True if already processed
                - stats: Dict containing main_data, rejected_data, operator_counts, etc.
        """
        try:
            if not dataset.datas:
                return False, {}
            source_file = dataset.datas[0].get("file_path")
            if not source_file:
                return False, {}

            basename = os.path.basename(source_file)
            source = self.get_source_from_path(source_file)
            if not source:
                return False, {}

            output_path = os.path.join(self.output_dir, source, basename)

            if not os.path.exists(output_path):
                return False, {}

            # File exists, load statistics
            if self.logger:
                self.logger.info(f"Dataset [{basename}] already processed, skipping...")

            stats = self._load_stats(output_path, source)

            # Check for rejected file
            rejected_path = os.path.join(self.output_dir, source, "rejected", basename)
            if os.path.exists(rejected_path):
                rejected_stats = self._load_stats(rejected_path, source)
                stats["rejected"] = rejected_stats

            return True, stats

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking processed status: {e}")
            return False, {}

    def _load_stats(self, file_path: str, source: str) -> dict:
        """Load statistics from processed file.

        Args:
            file_path: Path to processed file.
            source: Source name.

        Returns:
            dict: Statistics including path, json_size, source, and data.
        """
        try:
            data = FormatRegistry.load(file_path, add_source_file=False)
            return {
                "path": file_path,
                "json_size": len(data),
                "source": source,
                "data": data,
            }
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load stats from {file_path}: {e}")
            return {}
