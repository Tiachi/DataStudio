"""JSON data format handler."""

import json
import os
import tempfile

from .base import BaseFormat


class JsonFormat(BaseFormat):
    """JSON data format handler.

    Supports loading and saving data in JSON format.
    Single objects are wrapped in a list for consistency.
    """

    @classmethod
    def extensions(cls) -> list:
        """Return supported file extensions.

        Returns:
            list: ['.json']
        """
        return [".json"]

    def load(self, file_path: str) -> list:
        """Load data from JSON file.

        Args:
            file_path: Path to JSON file.

        Returns:
            list: Data as list of dicts (single objects are wrapped in list).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def save(
        self, data: list, file_path: str, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Save data to JSON file.

        Args:
            data: List of data dictionaries.
            file_path: Output file path.
            indent: JSON indentation level (default: 4).
            ensure_ascii: Whether to escape non-ASCII characters (default: False).
        """
        dir_path = os.path.dirname(file_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        # Write to temp file first, then atomic rename to avoid data loss on disk full
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, file_path)
        except BaseException:
            # Clean up temp file on failure, preserve original file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise
