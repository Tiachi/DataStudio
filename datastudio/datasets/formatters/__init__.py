"""Unified data loading/saving with automatic schema conversion.

Example::

    from datastudio.datasets.formatters import FormatRegistry
    data = FormatRegistry.load('/path/to/data.jsonl')
    FormatRegistry.save(data, '/path/to/output.jsonl')
"""

import os

from .base import BaseFormat
from .schema_converter import (
    SchemaConverter,
    SchemaType,
    denormalize_data,
    normalize_data,
)


class FormatRegistry:
    """Registry for data format handlers with automatic schema conversion.

    Detects file format by extension, converts between ``conversations``
    and ``messages`` schemas automatically, and preserves original schema
    metadata for round-trip fidelity.
    """

    _formats: dict = {}  # ext -> format_class

    @classmethod
    def register(cls, format_class):
        """Register a format handler (decorator).

        Args:
            format_class: Format class to register.

        Returns:
            The registered format class.
        """
        for ext in format_class.extensions():
            cls._formats[ext.lower()] = format_class
        return format_class

    @classmethod
    def get(cls, file_path: str) -> BaseFormat:
        """Get format handler instance based on file extension.

        Args:
            file_path: Path to data file.

        Returns:
            BaseFormat: Format handler instance.

        Raises:
            ValueError: If file extension is not supported.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in cls._formats:
            raise ValueError(
                f"Unsupported format: '{ext}'. Available: {list(cls._formats.keys())}"
            )
        return cls._formats[ext]()

    @classmethod
    def load(
        cls,
        file_path: str,
        add_source_file: bool = True,
        remove_rejected: bool = True,
        auto_normalize: bool = True,
    ) -> list:
        """Load data from file.

        Args:
            file_path: Path to data file.
            add_source_file: Whether to add 'file_path' field to each item.
            remove_rejected: Whether to remove 'filtered'/'rejected' fields.
            auto_normalize: Whether to auto-convert to standard schema (default: True).
                           Original schema is stored in '_original_schema' for round-trip.

        Returns:
            list: List of data dictionaries in standard schema.

        Note:
            This method modifies the loaded data items in place by:
            - Converting to standard schema if auto_normalize=True
            - Adding 'file_path' field if add_source_file=True
            - Removing 'filtered' and 'rejected' fields if remove_rejected=True

            If you need to preserve the original data, make a deep copy after loading.
        """
        data = cls.get(file_path).load(file_path)

        # Auto-normalize to standard schema
        if auto_normalize:
            data = normalize_data(data)

        for item in data:
            if add_source_file:
                item["file_path"] = file_path
            if remove_rejected:
                item.pop("filtered", None)  # Legacy field
                item.pop("rejected", None)  # Current field
        return data

    @classmethod
    def save(
        cls,
        data: list,
        file_path: str,
        auto_denormalize: bool = True,
        **kwargs,
    ) -> None:
        """Save data to file.

        Args:
            data: List of data dictionaries.
            file_path: Output file path.
            auto_denormalize: Whether to auto-convert back to original schema (default: True).
            **kwargs: Format-specific options.
        """
        # Auto-denormalize to original schema
        if auto_denormalize:
            data = denormalize_data(data)

        cls.get(file_path).save(data, file_path, **kwargs)

    @classmethod
    def supported_extensions(cls) -> list:
        """Get all supported file extensions.

        Returns:
            list: List of supported extensions.
        """
        return list(cls._formats.keys())


# Register built-in formats
from .json_formatter import JsonFormat  # noqa: E402
from .jsonl_formatter import JsonlFormat  # noqa: E402

FormatRegistry.register(JsonFormat)
FormatRegistry.register(JsonlFormat)

__all__ = [
    "BaseFormat",
    "FormatRegistry",
    "JsonFormat",
    "JsonlFormat",
    "SchemaConverter",
    "SchemaType",
    "normalize_data",
    "denormalize_data",
]
