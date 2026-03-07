"""YAML configuration generator."""

import os
from typing import Dict, List

import yaml


class YamlConfigGenerator:
    """Generates training dataset YAML configs from processed output.

    Attributes:
        output_dir: Output directory path.
        yaml_name: YAML file name (without extension).
        logger: Logger instance.
    """

    def __init__(self, output_dir: str, yaml_name: str, logger=None):
        """Initialize the generator.

        Args:
            output_dir: Output directory path.
            yaml_name: YAML file name (without extension).
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.yaml_name = yaml_name
        self.logger = logger

        self.data_files: List[str] = []
        self.rejected_data_files: List[str] = []

    def add_file(self, file_path: str, is_rejected: bool = False):
        """Add file to configuration list.

        Args:
            file_path: Path to data file.
            is_rejected: Whether this is a rejected data file.
        """
        if is_rejected:
            if file_path not in self.rejected_data_files:
                self.rejected_data_files.append(file_path)
        else:
            if file_path not in self.data_files:
                self.data_files.append(file_path)

    def generate(
        self, dataset_stats: Dict[str, dict], cnt_data: int, cnt_rejected: int
    ):
        """Generate YAML configuration files.

        Args:
            dataset_stats: Dataset statistics {path: {"json_size": N, "source": S}}.
            cnt_data: Total data count.
            cnt_rejected: Rejected data count.
        """
        # Generate main YAML
        if self.data_files:
            yaml_path = os.path.join(self.output_dir, f"{self.yaml_name}.yaml")
            self._save_yaml(self.data_files, yaml_path, dataset_stats)
            if self.logger:
                self.logger.info(
                    f"Saved {self.yaml_name}.yaml, remain: {cnt_data}, rejected: {cnt_rejected}"
                )

        # Generate rejected YAML
        if self.rejected_data_files:
            rejected_yaml_path = os.path.join(
                self.output_dir, f"{self.yaml_name}_rejected.yaml"
            )
            self._save_yaml(self.rejected_data_files, rejected_yaml_path, dataset_stats)
            if self.logger:
                self.logger.info(f"Saved {self.yaml_name}_rejected.yaml")

    def _save_yaml(
        self, data_files: List[str], yaml_path: str, dataset_stats: Dict[str, dict]
    ):
        """Save YAML configuration file.

        Args:
            data_files: List of data file paths.
            yaml_path: Output YAML file path.
            dataset_stats: Dataset statistics for metadata.
        """
        yaml_config = {"datasets": []}

        for data_file in data_files:
            dataset_info = {
                "file_path": data_file,  # Keep field name for compatibility
                "sampling_strategy": "all",
            }

            if data_file in dataset_stats:
                stats = dataset_stats[data_file]
                dataset_info["source"] = stats.get("source", "unknown")
                dataset_info["json_size"] = stats.get("json_size", 0)

            yaml_config["datasets"].append(dataset_info)

        yaml_dir = os.path.dirname(yaml_path)
        if yaml_dir:
            os.makedirs(yaml_dir, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
