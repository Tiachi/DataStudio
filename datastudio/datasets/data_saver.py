"""Data saver with composition-based design.

Components:
    - :class:`SimpleSaver`: Core save logic
    - :class:`YamlConfigGenerator`: YAML config generation
    - :class:`ProcessedChecker`: Processed file detection
    - :class:`StatsCollector`: Statistics collection
"""

import os

from datastudio.datasets.config import ConfigLoader
from datastudio.datasets.saver import ProcessedChecker, SimpleSaver, YamlConfigGenerator
from datastudio.utils.registry import DATASAVER
from datastudio.utils.statistics import StatsCollector


@DATASAVER.register_module()
class StandardDataSaver:
    """Dataset saver that groups output by source and tracks statistics.

    Output directory structure::

        output_dir/
        ├── source1/
        │   ├── file1.json
        │   └── rejected/
        │       └── file1.json
        ├── source2/
        │   └── file2.jsonl
        └── config.yaml
    """

    def __init__(
        self,
        output_dir: str,
        dataset: str,
        logger,
        save_yaml_config: bool = False,
        save_yaml_name: str = None,
        incremental_save_threshold: int = 8192,  # Save every N items
    ):
        """Initialize the data saver.

        Args:
            output_dir: Output directory path.
            dataset: Original dataset YAML path.
            logger: Logger instance.
            save_yaml_config: Whether to save YAML config.
            save_yaml_name: YAML config file name.
            incremental_save_threshold: Number of items before triggering incremental save.
        """
        self.logger = logger
        self.output_dir = output_dir
        self.save_yaml_config = save_yaml_config
        self.save_yaml_name = save_yaml_name
        self.incremental_save_threshold = incremental_save_threshold

        # Data buffers
        self.datas = []
        self.rejected_datas = []

        # Statistics
        self.cnt_data = 0
        self.cnt_rejected_data = 0
        self.dataset_stats = {}

        # Load source configuration
        self.sources = self._load_sources(dataset)

        # Initialize components
        self.saver = SimpleSaver(output_dir, self.sources, logger)
        self.checker = ProcessedChecker(output_dir, self.sources, logger)
        self.yaml_gen = (
            YamlConfigGenerator(output_dir, save_yaml_name, logger)
            if save_yaml_config
            else None
        )
        self.stats_collector = StatsCollector(logger=logger, work_dir=output_dir)
        self.stats_collector.start_pipeline()

        os.makedirs(output_dir, exist_ok=True)

    def _load_sources(self, dataset_yaml: str) -> dict:
        """Load source mapping from YAML using standardized config loader."""
        config = ConfigLoader.load(dataset_yaml)
        return config.get_sources_map()

    def has_processed(self, dataset) -> bool:
        """Check if dataset has already been processed."""
        is_processed, stats = self.checker.check(dataset)

        if is_processed and stats:
            # Update statistics information
            main_path = stats.get("path")
            if main_path:
                self.dataset_stats[main_path] = {
                    "json_size": stats.get("json_size", 0),
                    "source": stats.get("source", "unknown"),
                }
                kept_count = stats.get("json_size", 0)
                self.cnt_data += kept_count

                # Record dataset stats (dataset-level only)
                self.stats_collector.record_dataset(
                    main_path,
                    source=stats.get("source", "unknown"),
                    total=kept_count,
                    rejected=0,
                )

                if self.yaml_gen:
                    self.yaml_gen.add_file(main_path, is_rejected=False)

            # Process rejected data
            if "rejected" in stats:
                rejected_stats = stats["rejected"]
                rejected_path = rejected_stats.get("path")
                if rejected_path:
                    self.dataset_stats[rejected_path] = {
                        "json_size": rejected_stats.get("json_size", 0),
                        "source": rejected_stats.get("source", "unknown"),
                    }
                    rejected_count = rejected_stats.get("json_size", 0)
                    self.cnt_rejected_data += rejected_count

                    # Record dataset stats for rejected file
                    self.stats_collector.record_dataset(
                        rejected_path,
                        source=rejected_stats.get("source", "unknown"),
                        total=rejected_count,
                        rejected=rejected_count,
                    )

                    if self.yaml_gen:
                        self.yaml_gen.add_file(rejected_path, is_rejected=True)

        return is_processed

    @staticmethod
    def _cleanup_batch_images(datas: list) -> None:
        for data in datas:
            data.pop("image_pil", None)

    def __call__(self, datas: list):
        """Accumulate data for batch saving.

        Args:
            datas: List of data dicts. Rejected items have 'rejected=True'.
        """
        self._cleanup_batch_images(datas)

        for d in datas:
            if d.get("rejected", False):
                self.rejected_datas.append(d)
            else:
                self.datas.append(d)

    def save(self):
        """Save accumulated data to disk."""
        # Save rejected data
        if self.rejected_datas:
            stats = self.saver.save(self.rejected_datas, is_rejected=True)
            self._update_stats(stats, is_rejected=True, count=len(self.rejected_datas))

        # Save normal data
        if self.datas:
            stats = self.saver.save(self.datas, is_rejected=False)
            self._update_stats(stats, is_rejected=False, count=len(self.datas))

        # Update counts
        self.cnt_data += len(self.datas)
        self.cnt_rejected_data += len(self.rejected_datas)

        self.clean()

    def _update_stats(self, stats: dict, is_rejected: bool, count: int):
        """Update dataset-level statistics."""
        for path, info in stats.items():
            self.dataset_stats[path] = info

            # Record dataset-level stats
            rejected_count = count if is_rejected else 0
            self.stats_collector.record_dataset(
                path,
                source=info.get("source", "unknown"),
                total=count,
                rejected=rejected_count,
            )

            # Add to YAML generator
            if self.yaml_gen:
                self.yaml_gen.add_file(path, is_rejected=is_rejected)

    def save_yaml(self):
        """Save YAML config and statistics."""
        if self.yaml_gen:
            self.yaml_gen.generate(
                self.dataset_stats, self.cnt_data, self.cnt_rejected_data
            )

        # Save statistics
        if self.save_yaml_config and self.save_yaml_name:
            self.stats_collector.end_pipeline()
            csv_path = os.path.join(self.output_dir, f"{self.save_yaml_name}_stats.csv")
            self.stats_collector.export_csv(csv_path)
            self.stats_collector.print_summary()

    def incremental_save(self):
        """Incrementally save data if threshold is reached.

        This enables item-level resume by saving progress periodically
        without waiting for the entire file to be processed.
        """
        total_pending = len(self.datas) + len(self.rejected_datas)
        if total_pending >= self.incremental_save_threshold:
            self.save()

    def clean(self):
        """Clear data buffers."""
        self.datas = []
        self.rejected_datas = []
