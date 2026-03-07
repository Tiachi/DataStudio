"""Dataset configuration with field alias normalization.

Example::

    config = ConfigLoader.load('config.yaml')
    paths = config.get_file_paths()
    sources = config.get_sources_map()
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

# Field alias mapping: old field name -> standard field name
FIELD_ALIASES = {
    "json_path": "file_path",
    "jsonl_path": "file_path",
    "parquet_path": "file_path",
    "data_path": "file_path",
    "json_size": "data_size",
    "num_samples": "data_size",
}

_CORE_FIELDS = {"file_path", "source", "data_size"}


@dataclass
class DatasetConfig:
    """Standard configuration for a single dataset.

    Attributes:
        file_path: Path to the data file.
        source: Data source name/identifier.
        data_size: Number of samples (optional).
        extra: Additional metadata fields.
    """

    file_path: str
    source: str = "Unknown"
    data_size: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        """Create config from dict, automatically handling field aliases.

        Args:
            data: Raw configuration dictionary.

        Returns:
            DatasetConfig instance with standardized fields.
        """
        seen = {}
        for k, v in data.items():
            std_key = FIELD_ALIASES.get(k, k)
            if std_key not in seen:
                seen[std_key] = v

        return cls(
            file_path=seen.get("file_path", ""),
            source=seen.get("source", "Unknown"),
            data_size=seen.get("data_size"),
            extra={k: v for k, v in seen.items() if k not in _CORE_FIELDS},
        )

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            dict: Configuration as dictionary.
        """
        result = {"file_path": self.file_path, "source": self.source}
        if self.data_size is not None:
            result["data_size"] = self.data_size
        result.update(self.extra)
        return result


@dataclass
class StandardConfig:
    """Standardized complete configuration containing multiple datasets.

    Attributes:
        datasets: List of DatasetConfig objects.
        extra: Additional top-level configuration fields.
    """

    datasets: List[DatasetConfig] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "StandardConfig":
        """Create standard config from dictionary.

        Args:
            data: Raw configuration dictionary with 'datasets' key.

        Returns:
            StandardConfig instance.
        """
        datasets = [DatasetConfig.from_dict(item) for item in data.get("datasets", [])]
        extra = {k: v for k, v in data.items() if k != "datasets"}
        return cls(datasets=datasets, extra=extra)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            dict: Configuration as dictionary.
        """
        return {"datasets": [ds.to_dict() for ds in self.datasets], **self.extra}

    def get_file_paths(self) -> List[str]:
        """Get all data file paths.

        Returns:
            List of file paths from all datasets.
        """
        return [ds.file_path for ds in self.datasets]

    def get_sources_map(self) -> dict:
        """Get file_path to source mapping.

        Returns:
            dict: Mapping from file_path to source name.
        """
        return {ds.file_path: ds.source for ds in self.datasets}


class ConfigLoader:
    """Configuration loader for loading and saving YAML configs.

    Provides static methods for loading YAML configurations and
    converting them to standardized format.

    Example:
        >>> config = ConfigLoader.load("config.yaml")
        >>> paths = config.get_file_paths()
        >>> sources = config.get_sources_map()
    """

    @classmethod
    def load(cls, yaml_path: str) -> StandardConfig:
        """Load YAML configuration file.

        Args:
            yaml_path: Path to YAML file.

        Returns:
            StandardConfig instance.
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return StandardConfig.from_dict(data or {})

    @classmethod
    def save(cls, config: StandardConfig, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            config: StandardConfig to save.
            yaml_path: Output file path.
        """
        os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    @classmethod
    def load_file_paths(
        cls, yaml_path: str, data_root: Optional[str] = None
    ) -> List[str]:
        """Load config and return all data file paths.

        Args:
            yaml_path: Path to YAML config file.
            data_root: Optional root directory to prepend to relative paths.

        Returns:
            List of absolute file paths.
        """
        paths = cls.load(yaml_path).get_file_paths()
        if data_root:
            paths = [
                os.path.join(data_root, p) if not os.path.isabs(p) else p for p in paths
            ]
        return paths

    @classmethod
    def load_sources_map(cls, yaml_path: str) -> dict:
        """Load config and return file_path to source mapping.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            dict: Mapping from file_path to source name.
        """
        return cls.load(yaml_path).get_sources_map()

    @classmethod
    def create_config(
        cls,
        file_paths: List[str],
        sources: Optional[dict] = None,
        data_sizes: Optional[dict] = None,
    ) -> StandardConfig:
        """Create a standard configuration from file paths.

        Args:
            file_paths: List of data file paths.
            sources: Optional mapping from path to source name.
            data_sizes: Optional mapping from path to data size.

        Returns:
            StandardConfig instance.
        """
        sources, data_sizes = sources or {}, data_sizes or {}
        datasets = [
            DatasetConfig(
                file_path=p,
                source=sources.get(p, "Unknown"),
                data_size=data_sizes.get(p),
            )
            for p in file_paths
        ]
        return StandardConfig(datasets=datasets)
