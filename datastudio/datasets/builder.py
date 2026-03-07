"""Factory functions for creating dataloaders with checkpoint support."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from datastudio.datasets.config import ConfigLoader
from datastudio.utils.checkpoint import CheckpointManager
from datastudio.utils.registry import DATALOADER, build_from_cfg


def build_dataloaders(
    dataset_config, logger=None, work_dir=None, parallel_load=True, max_workers=4
):
    """Build dataloaders with checkpoint support.

    Args:
        dataset_config: Dataloader configuration dict.
        logger: Logger instance.
        work_dir: Working directory for checkpoint file.
        parallel_load: Whether to load dataloaders in parallel (default: True).
        max_workers: Maximum number of threads for parallel loading (default: 4).

    Config options:
        - checkpoint_enabled: Whether to enable item-level checkpoint (default: True)
        - checkpoint_name: Checkpoint file name (default: "processing_checkpoint.json")
        - parallel_load: Whether to load dataloaders in parallel (default: True)
        - parallel_load_workers: Number of workers for parallel loading (default: 4)

    Note:
        This function modifies the dataset_config dict by popping certain keys.
        Pass a copy if you need to preserve the original config.
    """
    # Make a copy to avoid modifying the original config
    config = dataset_config.copy()
    num_workers = config.pop("num_workers")
    batch_size = config.pop("batch_size")

    # Checkpoint configuration
    checkpoint_enabled = config.pop("checkpoint_enabled", True)
    checkpoint_name = config.pop("checkpoint_name", "processing_checkpoint.json")

    # Parallel loading configuration (can be overridden from config)
    parallel_load = config.pop("parallel_load", parallel_load)
    max_workers = config.pop("parallel_load_workers", max_workers)

    # Initialize checkpoint manager if enabled
    checkpoint_manager = None
    if checkpoint_enabled and work_dir:
        checkpoint_manager = CheckpointManager(
            work_dir=work_dir,
            checkpoint_name=checkpoint_name,
            logger=logger,
        )

    dataset_paths = config.get("dataset", None)
    data_root = config.get("data_root", "")

    if isinstance(dataset_paths, str) and dataset_paths.endswith(".yaml"):
        yaml_path = os.path.join(data_root, dataset_paths)
        config_obj = ConfigLoader.load(yaml_path)
        if config_obj.datasets:
            dataset_paths = config_obj.get_file_paths()
        else:
            if logger:
                logger.error(f"No 'datasets' key found in YAML config {yaml_path}")
            return None
    elif dataset_paths is None:
        if logger:
            logger.error("No dataset found in config")
        return None

    # Filter out already completed files
    pending_datasets = []
    for dataset_path in dataset_paths:
        if checkpoint_manager:
            if isinstance(dataset_path, dict):
                file_path = (
                    dataset_path.get("file_path")
                    or dataset_path.get("json_path")
                    or dataset_path.get("jsonl_path")
                )
            else:
                file_path = dataset_path

            if file_path and not os.path.isabs(file_path):
                full_path = os.path.join(data_root, file_path)
            else:
                full_path = file_path

            if checkpoint_manager.is_completed(full_path):
                if logger:
                    logger.info(
                        f"Skipping already completed file (from checkpoint): {full_path}"
                    )
                continue

        pending_datasets.append(dataset_path)

    if not pending_datasets:
        return []

    def create_single_dataloader(dataset_path):
        """Create a single dataloader for the given dataset path."""
        current_config = config.copy()
        current_config["dataset"] = dataset_path
        return build_from_cfg(
            current_config,
            DATALOADER,
            logger=logger,
            batch_size=batch_size,
            num_workers=num_workers,
            checkpoint_manager=checkpoint_manager,
        )

    # Parallel or sequential loading
    if parallel_load and len(pending_datasets) > 1:
        if logger:
            logger.info(
                f"Loading {len(pending_datasets)} dataloaders in parallel (max_workers={max_workers})"
            )

        dataloaders = [None] * len(pending_datasets)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(create_single_dataloader, dataset_path): idx
                for idx, dataset_path in enumerate(pending_datasets)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    dataloaders[idx] = future.result()
                except Exception as e:
                    if logger:
                        logger.error(
                            f"Failed to load dataloader for {pending_datasets[idx]}: {e}"
                        )
                    raise

        if logger:
            logger.info(f"All {len(dataloaders)} dataloaders loaded successfully")
    else:
        # Sequential loading
        dataloaders = []
        for dataset_path in pending_datasets:
            dataloaders.append(create_single_dataloader(dataset_path))

    return dataloaders
