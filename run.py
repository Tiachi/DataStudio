"""Main entry point for DataStudio pipeline execution.

Usage:
    python run.py -c configs/my_config.py
    python run.py -c configs/my_config.py --cache-images
"""

import argparse
import time
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datastudio.datasets.builder import build_dataloaders
from datastudio.utils.files import build_file
from datastudio.utils.config import get_cfg, init_cfg, print_cfg, save_cfg
from datastudio.utils.registry import (
    HOOKS,
    MODELS,
    PIPELINES,
    DATASAVER,
    build_from_cfg,
)


class Runner:
    """Main pipeline runner that orchestrates data loading, processing, and saving.

    Initializes all components from the global config:
    - Logger
    - Model (optional, for MLLM operators)
    - DataLoaders (with checkpoint support)
    - DataSaver
    - Pipeline (with statistics collection)

    Supports two execution modes:
    - Normal pipeline mode: load -> process -> save
    - Cache-images-only mode: iterate data to populate LMDB image cache
    """

    def __init__(self, cache_images_only: bool = False):
        cfg = get_cfg()

        print_cfg()

        self.cfg = cfg
        self.work_dir = cfg.work_dir
        self.cache_images_only = cache_images_only

        self.logger = build_from_cfg(
            self.cfg.logger,
            HOOKS,
            work_dir=self.work_dir,
            wandb_project="datastudio",
            wandb_config=self.cfg,
        )

        if cfg.get("model", None):
            self.model = build_from_cfg(cfg.model, MODELS, logger=self.logger)
        else:
            self.model = None
            self.logger.warning("No model is used in this pipeline")

        # Checkpoint manager is now created inside build_dataloaders
        self.dataloaders = build_dataloaders(
            cfg.dataloader,
            logger=self.logger,
            work_dir=self.work_dir,
        )
        self.datasaver = build_from_cfg(cfg.datasaver, DATASAVER, logger=self.logger)

        # Pass stats_collector from datasaver to pipeline for unified statistics
        stats_collector = getattr(self.datasaver, 'stats_collector', None)
        self.pipeline = build_from_cfg(
            cfg.pipeline, PIPELINES, logger=self.logger, model=self.model,
            stats_collector=stats_collector
        )

        save_file = build_file(self.work_dir, prefix="config.yaml")
        save_cfg(save_file)

    def run(self):
        """Execute the runner in the configured mode."""
        if self.cache_images_only:
            self._run_cache_images()
        else:
            self._run_pipeline()

    def _run_cache_images(self):
        """Cache images only mode: iterate all data and write images to LMDB cache."""
        self.logger.info("Start caching images")

        total_images = 0
        start_time = time.time()

        for dataloader in self.dataloaders:
            if dataloader.is_empty():
                self.logger.info(f"No items to process for: {dataloader.name}")
                continue

            self.logger.info(f"Caching images for: {dataloader.name}")

            pbar = tqdm(dataloader, desc=f"Caching {dataloader.name}")
            for batch in pbar:
                batch_images = sum(
                    len(item.get("image")) if isinstance(item.get("image"), list) else 1
                    for item in batch
                    if item.get("image")
                )
                total_images += batch_images
                pbar.set_description(f"Cached {total_images} images")

                # Free image_pil to prevent memory accumulation
                for item in batch:
                    item.pop("image_pil", None)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Image caching completed! Total: {total_images}, Time: {elapsed:.2f}s"
        )

        if self.model is not None:
            self.model.shutdown()

    def _run_pipeline(self):
        """Normal pipeline execution mode."""
        self.logger.info("Start running")

        for dataloader in self.dataloaders:
            # Skip if no items to process (partial resume case)
            if dataloader.is_empty():
                self.logger.info(f"No items to process for: {dataloader.name}")
                continue

            # Skip if already completed (file-level check, fallback when no checkpoint)
            if not dataloader.has_checkpoint() and self.datasaver.has_processed(
                dataloader
            ):
                continue

            pbar = tqdm(dataloader, desc=f"Processing {dataloader.name}")
            for datas in pbar:
                t1 = time.time()
                datas = self.pipeline(datas)
                elapsed_time = time.time() - t1
                pbar.set_description(f"Pipeline time: {elapsed_time:.2f} seconds")
                self.logger.info("")
                self.datasaver(datas)

                # Update checkpoint after each batch
                dataloader.update_checkpoint()

                # Incremental save after each batch for large files
                self.datasaver.incremental_save()

            self.datasaver.save()

        self.datasaver.save_yaml()
        if self.model is not None:
            self.model.shutdown()
        self.logger.info("Finish running")


def main():
    """CLI entry point for DataStudio pipeline."""
    parser = argparse.ArgumentParser(description="DataStudio Pipeline Runner")
    parser.add_argument(
        "-c",
        "--config",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cache-images",
        action="store_true",
        help="Only cache images to LMDB, skip pipeline execution",
    )

    args = parser.parse_args()

    if args.config:
        init_cfg(args.config)

    runner = Runner(cache_images_only=args.cache_images)

    runner.run()


if __name__ == "__main__":
    main()
