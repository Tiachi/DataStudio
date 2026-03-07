"""JSONL (JSON Lines) data format handler."""

import json
import logging
import os
import tempfile

from .base import BaseFormat

logger = logging.getLogger(__name__)


class JsonlFormat(BaseFormat):
    """JSONL (JSON Lines) data format handler.

    Each line contains one JSON object. Invalid lines are skipped with a warning.
    """

    @classmethod
    def extensions(cls) -> list:
        """Return supported file extensions.

        Returns:
            list: ['.jsonl']
        """
        return [".jsonl"]

    def load(self, file_path: str) -> list:
        """Load data from JSONL file.

        Args:
            file_path: Path to JSONL file.

        Returns:
            list: List of data dictionaries.

        Note:
            Invalid JSON lines are skipped with a warning message.
        """
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Skipping invalid JSON at line %d in %s: %s",
                            line_num,
                            file_path,
                            e,
                        )
        return data

    def save(self, data: list, file_path: str, ensure_ascii: bool = False) -> None:
        """Save data to JSONL file.

        Args:
            data: List of data dictionaries.
            file_path: Output file path.
            ensure_ascii: Whether to escape non-ASCII characters (default: False).
        """
        dir_path = os.path.dirname(file_path) or "."
        os.makedirs(dir_path, exist_ok=True)
        # Write to temp file first, then atomic rename to avoid data loss on disk full
        fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", errors="surrogatepass") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=ensure_ascii) + "\n")
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
