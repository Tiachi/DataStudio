"""Core data saving logic."""

import os
from typing import Dict, List, Optional

from datastudio.datasets.formatters import FormatRegistry


class SimpleSaver:
    """Saves processed data grouped by source file.

    Groups items by their ``_source_file``, removes temporary fields,
    and delegates to :class:`FormatRegistry` for format-aware saving.
    """

    def __init__(
        self,
        output_dir: str,
        sources: Dict[str, str],
        logger=None,
    ):
        """Initialize the saver.

        Args:
            output_dir: Output root directory.
            sources: Mapping from json_path to source name.
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.sources = sources
        self.logger = logger

        os.makedirs(self.output_dir, exist_ok=True)

    def get_source_from_path(self, file_path: str) -> Optional[str]:
        """Get source name from file path.

        Args:
            file_path: Path to data file.

        Returns:
            Source name, or 'unknown' if not found.
        """
        # Direct full path match
        if file_path in self.sources:
            return self.sources[file_path]

        # Match by filename
        filename = os.path.basename(file_path)
        for path, source in self.sources.items():
            if os.path.basename(path) == filename:
                return source

        return "unknown"

    def clean_item(self, item: dict) -> dict:
        """Clean a single data item by removing temporary fields.

        Args:
            item: Data item dictionary.

        Returns:
            dict: Cleaned copy of the item.
        """
        data_copy = item.copy()

        # Remove image_pil field
        if "image_pil" in data_copy:
            try:
                img = data_copy.pop("image_pil")
                if hasattr(img, "close"):
                    img.close()
            except Exception:
                pass

        return data_copy

    def group_by_source(self, data_list: List[dict]) -> Dict[str, List[dict]]:
        """Group data by source file.

        Args:
            data_list: List of data items.

        Returns:
            dict: Mapping from source file path to list of items.
        """
        grouped = {}
        for data in data_list:
            source_file = data.get("file_path")
            if source_file is None:
                if self.logger:
                    self.logger.warning("Data without source file field, skipping")
                continue

            if source_file not in grouped:
                grouped[source_file] = []
            grouped[source_file].append(data)

        return grouped

    def save(
        self,
        data_list: List[dict],
        is_rejected: bool = False,
        append_mode: bool = True,
    ) -> Dict[str, dict]:
        """Save data to files.

        Args:
            data_list: List of data items.
            is_rejected: Whether this is rejected data.
            append_mode: Whether to append to existing files (for checkpointing).

        Returns:
            dict: Save statistics {output_path: {"json_size": N, "source": S}}
        """
        if not data_list:
            return {}

        grouped = self.group_by_source(data_list)
        stats = {}

        for source_file, items in grouped.items():
            # Get original file info
            basename = os.path.basename(source_file)
            source = self.get_source_from_path(source_file)

            # Build output directory
            output_dir = os.path.join(self.output_dir, source)
            if is_rejected:
                output_dir = os.path.join(output_dir, "rejected")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, basename)

            # Clean data
            clean_items = [self.clean_item(item) for item in items]

            # Check if append mode needed
            file_existed = os.path.exists(output_path)

            # If append mode and file exists, load existing data first
            if append_mode and file_existed:
                try:
                    # Load without auto_normalize to preserve original schema metadata
                    # Keep remove_rejected=False to preserve rejected field in append mode
                    existing_data = FormatRegistry.load(
                        output_path,
                        add_source_file=False,
                        auto_normalize=False,
                        remove_rejected=False,
                    )
                    clean_items = existing_data + clean_items
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"Failed to load existing data from {output_path}: {e}"
                        )

            # Save using FormatRegistry (auto-detects format)
            FormatRegistry.save(clean_items, output_path)

            # Record statistics
            stats[output_path] = {
                "json_size": len(clean_items),
                "source": source,
            }

            if self.logger:
                prefix = "Rejected " if is_rejected else ""
                mode_str = "(appended)" if append_mode and file_existed else ""
                self.logger.info(
                    f"{prefix}Saved {len(clean_items)} items to {output_path} {mode_str}"
                )

        return stats
