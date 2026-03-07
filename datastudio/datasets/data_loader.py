"""Dataset loader with LMDB image caching and checkpoint resume support."""

import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Union

from PIL import Image
from tqdm import tqdm

from datastudio.datasets.formatters import FormatRegistry
from datastudio.utils.database import ShardedLMDBManager, write_images_to_sharded_lmdb
from datastudio.utils.qa_utils import calculate_qa_statistics
from datastudio.utils.registry import DATALOADER
from datastudio.utils.vision import apply_resize_image_pil, load_and_resize_image


@DATALOADER.register_module()
class StandardDataLoader:
    """Dataset loader with LMDB image caching and checkpoint resume.

    Supports automatic format detection, parallel image loading, adaptive
    batch sizing, and item-level checkpoint for resumable processing.

    Example::

        loader = StandardDataLoader(
            data_root='/data/datasets',
            dataset={'file_path': 'train.jsonl'},
            batch_size=32,
            logger=logger,
        )
        for batch in loader:
            process(batch)
    """

    def __init__(
        self,
        data_root,
        dataset,
        batch_size,
        parallel_loading=True,
        num_workers=256,
        logger=None,
        cache_dir="~/cache/images_lmdb_sharded",
        lmdb_num_shards=32,
        lmdb_map_size_per_shard=1024 * 1024 * 1024 * 1024,  # 1TB per shard
        lmdb_readonly=False,
        lmdb_lock=False,
        resize_image=True,
        resize_image_size=1024,
        use_image=True,
        use_lmdb_cache=True,  # Whether to cache images in LMDB; False = read from disk directly
        adjust_batch_size=True,  # Whether to adjust batch_size based on turns
        checkpoint_manager=None,  # Checkpoint manager for item-level resume
        **kwargs,  # Accept but ignore deprecated parameters like pre_load_img
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.original_config = None
        self.parallel_loading = parallel_loading
        self.num_workers = num_workers
        self.logger = logger
        self.resize_image = resize_image
        self.resize_image_size = resize_image_size
        self.use_image = use_image
        self.use_lmdb_cache = use_lmdb_cache
        self.adjust_batch_size = adjust_batch_size
        self.checkpoint_manager = checkpoint_manager
        self.name = ""

        # Store LMDB config for lazy initialization
        self._cache_dir = os.path.expanduser(cache_dir) if cache_dir else None
        self._lmdb_num_shards = lmdb_num_shards
        self._lmdb_map_size_per_shard = lmdb_map_size_per_shard
        self._lmdb_readonly = lmdb_readonly
        self._lmdb_lock = lmdb_lock
        self.sharded_lmdb = None

        # Resolve dataset path and load data (supports JSON/JSONL formats)
        data_file = self._resolve_data_path(dataset, data_root)
        self.data_file = data_file
        self.all_datas = FormatRegistry.load(
            data_file, add_source_file=True, remove_rejected=True
        )
        self.total_items = len(self.all_datas)

        # Apply checkpoint: skip already processed items
        self.start_idx = 0
        if self.checkpoint_manager:
            self.start_idx = self.checkpoint_manager.get_start_index(
                data_file, self.total_items
            )
            if self.start_idx == -1:
                # File already completed
                self.datas = []
                self.start_idx = self.total_items
            else:
                self.datas = self.all_datas[self.start_idx :]
        else:
            self.datas = self.all_datas

        # Calculate QA statistics and adjust batch_size
        stats = calculate_qa_statistics(
            self.datas,
            batch_size=batch_size,
            adjust_batch_size=adjust_batch_size,
            logger=logger,
        )
        self.avg_qa_count = stats["avg_qa_count"]
        self.max_qa_count = stats["max_qa_count"]
        self.total_qa_pairs = stats["total_qa_pairs"]
        self.real_batch_size = stats["real_batch_size"]

        if self.logger:
            self.logger.info(
                f"Dataset size: {len(self.datas)}/{self.total_items} (start_idx={self.start_idx}), "
                f"avg turns: {self.avg_qa_count:.2f}, max turns: {self.max_qa_count}, "
                f"adjusted batch_size: {self.real_batch_size}, "
                f"resize: {self.resize_image}, image_max_size: {self.resize_image_size}"
            )

        self.idx = 0
        # Track current absolute index for checkpoint updates
        self.current_abs_idx = self.start_idx

    def _ensure_lmdb_initialized(self) -> None:
        """Ensure LMDB manager is initialized (lazy initialization)."""
        if self.sharded_lmdb is None:
            self.sharded_lmdb = ShardedLMDBManager.get_instance(
                cache_dir=self._cache_dir,
                num_shards=self._lmdb_num_shards,
                map_size_per_shard=self._lmdb_map_size_per_shard,
                readonly=self._lmdb_readonly,
                lock=self._lmdb_lock,
                logger=self.logger,
            )

    def _resolve_data_path(self, dataset: Union[str, Dict], data_root: str) -> str:
        """Resolve dataset file path (supports JSON/JSONL formats).

        Args:
            dataset: Dataset path string or config dict.
            data_root: Root directory for relative paths.

        Returns:
            Absolute path to the data file.
        """
        if isinstance(dataset, dict):
            # Support multiple field names: file_path, json_path, jsonl_path, etc.
            from datastudio.datasets.config import DatasetConfig

            config = DatasetConfig.from_dict(dataset)

            if not config.file_path:
                if self.logger:
                    self.logger.error(
                        f"Missing file path in dataset configuration: {dataset}"
                    )
                raise ValueError("Missing file path (file_path/json_path/jsonl_path)")

            data_file = config.file_path
            if not os.path.isabs(data_file):
                data_file = os.path.join(data_root, data_file)
        else:
            data_file = os.path.join(data_root, dataset)
        self.name = data_file
        return data_file

    def _get_adaptive_resize_size(self, image_count: int) -> int:
        """Get adaptive resize size based on image count.

        Args:
            image_count: Number of images in the item.

        Returns:
            Resize size (max edge length) based on image count:
            - 1-5 images: use original resize_image_size
            - 6-10 images: 448
            - 11-15 images: 512
            - 15+ images: 392
        """
        if image_count > 15:
            return 392
        elif image_count > 10:
            return 448
        elif image_count > 5:
            return 672
        elif image_count > 2:
            return 1024
        else:
            return self.resize_image_size

    def _load_images_for_batch(self, batch: List[Dict]) -> List[Dict]:
        """Load images for a batch from LMDB cache or disk.

        When ``use_lmdb_cache=True``, images are cached to LMDB first and
        then read back.  When ``use_lmdb_cache=False``, images are read
        directly from disk.
        """
        if not self.use_image:
            return batch

        if not self.use_lmdb_cache:
            # Direct mode: multi-threaded loading from disk
            # Collect all (item_index, img_path, resize_size) tasks
            tasks = []
            for i, item in enumerate(batch):
                img_path = item.get("image")
                if img_path and isinstance(img_path, str):
                    actual_path = (
                        img_path
                        if os.path.isabs(img_path)
                        else os.path.join(self.data_root, img_path)
                    )
                    resize_size = self.resize_image_size if self.resize_image else 0
                    tasks.append((i, actual_path, resize_size, False))
                elif img_path and isinstance(img_path, list):
                    adaptive_size = (
                        self._get_adaptive_resize_size(len(img_path))
                        if self.resize_image
                        else 0
                    )
                    for j, img in enumerate(img_path):
                        actual_img = (
                            img
                            if os.path.isabs(img)
                            else os.path.join(self.data_root, img)
                        )
                        tasks.append((i, actual_img, adaptive_size, True, j))

            if tasks:
                num_workers = min(
                    self.num_workers if self.parallel_loading else 1, len(tasks)
                )
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {}
                    for task in tasks:
                        future = executor.submit(
                            load_and_resize_image, task[1], task[2]
                        )
                        futures[future] = task

                    for future in futures:
                        task = futures[future]
                        img_pil = future.result()
                        item_idx = task[0]
                        is_list = task[3]
                        if not is_list:
                            batch[item_idx]["image_pil"] = img_pil
                        else:
                            # Initialize list if needed
                            if "image_pil" not in batch[item_idx]:
                                batch[item_idx]["image_pil"] = [None] * len(
                                    batch[item_idx]["image"]
                                )
                            batch[item_idx]["image_pil"][task[4]] = img_pil

            return batch

        self._ensure_lmdb_initialized()

        # Step 1: Collect unique image paths for this batch
        image_paths_set = set()
        for item in batch:
            img_path = item.get("image")
            if img_path:
                if isinstance(img_path, list):
                    image_paths_set.update(img_path)
                else:
                    image_paths_set.add(img_path)
        image_paths = list(image_paths_set)

        if image_paths:
            # Step 2: Write batch images to LMDB (skips already-cached ones internally)
            write_images_to_sharded_lmdb(
                self.sharded_lmdb,
                image_paths,
                data_root=self.data_root,
                batch_size=2000,
                resize_image=self.resize_image,
                resize_image_size=self.resize_image_size,
                num_workers=self.num_workers if self.parallel_loading else 1,
            )

        # Step 3: Batch read unique images from LMDB
        if image_paths:
            img_bytes_map = self.sharded_lmdb.batch_get(image_paths)
        else:
            img_bytes_map = {}

        # Step 4: Decode each unique image ONCE (avoid decoding same image N times)
        def _decode_one(path_and_bytes):
            """Decode PNG bytes to PIL Image."""
            path, raw_bytes = path_and_bytes
            try:
                img_pil = Image.open(io.BytesIO(raw_bytes))
                img_pil.load()
                return path, img_pil
            except Exception:
                return path, None

        decode_items = [(p, b) for p, b in img_bytes_map.items() if b is not None]
        img_pil_map = {}
        if decode_items:
            num_workers = min(
                self.num_workers if self.parallel_loading else 1,
                len(decode_items),
                32,
            )
            show_decode_progress = len(decode_items) >= 1000 and self.logger is not None
            if show_decode_progress:
                decode_pbar = tqdm(
                    total=len(decode_items),
                    desc="Image decode",
                    unit="img",
                )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for path, img_pil in executor.map(_decode_one, decode_items):
                    if img_pil is not None:
                        img_pil_map[path] = img_pil
                    if show_decode_progress:
                        decode_pbar.update(1)
            if show_decode_progress:
                decode_pbar.close()

        # Step 5: Pre-resize images and cache by (path, size) to avoid repeated resize
        # Items share the same PIL Image object (read-only in downstream pipeline)
        _resized_cache = {}

        def _get_resized(path, resize_size):
            """Get resized image, caching by (path, size) to avoid redundant work."""
            cache_key = (path, resize_size)
            if cache_key not in _resized_cache:
                img_pil = img_pil_map.get(path)
                if img_pil is None:
                    _resized_cache[cache_key] = None
                elif self.resize_image:
                    _resized_cache[cache_key] = apply_resize_image_pil(
                        img_pil, resize_size
                    )
                else:
                    _resized_cache[cache_key] = img_pil
            return _resized_cache[cache_key]

        # Step 6: Distribute to batch items (shared references, no copy)
        for item in batch:
            img_path = item.get("image")
            if img_path and isinstance(img_path, str):
                item["image_pil"] = _get_resized(img_path, self.resize_image_size)
            elif img_path and isinstance(img_path, list):
                image_count = len(img_path)
                adaptive_size = self._get_adaptive_resize_size(image_count)
                item["image_pil"] = [
                    _get_resized(img, adaptive_size) for img in img_path
                ]

        return batch

    def get_name(self) -> str:
        """Get the dataset name."""
        return self.name

    def __len__(self) -> int:
        """Return the number of batches."""
        return (len(self.datas) + self.real_batch_size - 1) // self.real_batch_size

    def __iter__(self) -> Iterator[List[Dict]]:
        """Initialize iteration."""
        self.idx = 0
        self.current_abs_idx = self.start_idx
        return self

    def __next__(self) -> List[Dict]:
        if self.idx >= len(self.datas):
            raise StopIteration

        if self.idx + self.real_batch_size > len(self.datas):
            batch_data = self.datas[self.idx :]
            self.idx = len(self.datas)
        else:
            batch_data = self.datas[self.idx : self.idx + self.real_batch_size]
            self.idx += self.real_batch_size

        # Update absolute index for checkpoint
        self.current_abs_idx = self.start_idx + self.idx

        # Load images for this batch on-demand
        batch_data = self._load_images_for_batch(batch_data)

        return batch_data

    def update_checkpoint(self) -> None:
        """Update checkpoint with current progress."""
        if self.checkpoint_manager:
            self.checkpoint_manager.update(
                self.name, self.current_abs_idx, self.total_items
            )

    def is_completed(self) -> bool:
        """Check if this dataloader has been fully processed."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.is_completed(self.name)
        return False

    def has_checkpoint(self) -> bool:
        """Check if this dataloader has checkpoint tracking enabled and has progress."""
        if self.checkpoint_manager:
            return self.checkpoint_manager.get_progress(self.name) is not None
        return False

    def is_empty(self) -> bool:
        """Check if there are no items to process."""
        return len(self.datas) == 0

    def __call__(self):
        if not self.datas:
            return []

        if self.idx >= len(self.datas):
            self.idx = 0
        if self.idx + self.real_batch_size > len(self.datas):
            batch_data = self.datas[self.idx :]
            self.idx = len(self.datas)
        else:
            batch_data = self.datas[self.idx : self.idx + self.real_batch_size]
            self.idx += self.real_batch_size

        # Update absolute index for checkpoint
        self.current_abs_idx = self.start_idx + self.idx

        # Load images for this batch on-demand
        batch_data = self._load_images_for_batch(batch_data)

        return batch_data
