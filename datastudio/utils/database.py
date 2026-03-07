"""Sharded LMDB manager for image caching.

Distributes data across multiple LMDB files using hash-based sharding
with on-demand shard loading and LRU eviction.

Example:
    >>> manager = ShardedLMDBManager.get_instance(cache_dir='./cache')
    >>> manager.put('image.jpg', image_bytes)
    >>> data = manager.get('image.jpg')
"""

import hashlib
import io
import os
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# Suppress libpng ICC profile warnings before importing cv2/PIL
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
os.environ["QT_LOGGING_RULES"] = "*=false"

import cv2  # noqa: E402
import lmdb  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402

# Allow loading truncated images and suppress decompression warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from datastudio.utils.vision import apply_resize_image_cv2  # noqa: E402

# Suppress OpenCV/libpng warnings
os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)


def get_shard_id(key: str, num_shards: int) -> int:
    """Calculate shard ID based on key hash.

    Args:
        key: Key string to hash.
        num_shards: Total number of shards.

    Returns:
        int: Shard ID (0 to num_shards-1).
    """
    hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_val % num_shards


def init_sharded_lmdb(
    shard_path: str,
    map_size: int = 1024 * 1024 * 1024 * 1024,
    readonly: bool = False,
    lock: bool = True,
    readahead: bool = False,
):
    """Initialize a single LMDB shard environment.

    Args:
        shard_path: Path to shard directory.
        map_size: Maximum database size in bytes.
        readonly: Whether to open in read-only mode.
        lock: Whether to use file locking.
        readahead: Whether to enable OS readahead for mmap pages.
            False (default) is better for random writes/existence checks.
            True is better for batch sequential reads.

    Returns:
        lmdb.Environment: Initialized LMDB environment.
    """
    os.makedirs(shard_path, exist_ok=True)
    env = lmdb.open(
        shard_path,
        map_size=map_size,
        subdir=True,
        max_readers=16384,
        readonly=readonly,
        meminit=False,
        map_async=not readonly,
        lock=lock,
        readahead=readahead,
    )
    return env


class ShardedLMDBManager:
    """Sharded LMDB manager with singleton pattern per cache directory.

    Distributes images across LMDB shards using hash-based partitioning.
    Loads shards on-demand with LRU eviction to minimize memory pressure.

    Example:
        >>> manager = ShardedLMDBManager.get_instance('./cache', num_shards=32)
        >>> manager.put('path/to/image.jpg', image_bytes)
        >>> exists = manager.exists('path/to/image.jpg')
        >>> data = manager.get('path/to/image.jpg')
    """

    _instances: Dict[str, "ShardedLMDBManager"] = {}

    def __init__(
        self,
        cache_dir: str,
        num_shards: int = 32,
        map_size_per_shard: int = 1024 * 1024 * 1024 * 1024,  # 1TB per shard
        readonly: bool = False,
        lock: bool = True,
        logger=None,
        max_open_shards: int = 0,  # 0 means no limit
    ):
        """Initialize the sharded LMDB manager.

        Args:
            cache_dir: Cache directory path.
            num_shards: Number of shards to use.
            map_size_per_shard: Maximum size per shard in bytes.
            readonly: Whether to open in read-only mode.
            lock: Whether to use file locking.
            logger: Logger instance.
            max_open_shards: Maximum open shards (0 = no limit).
        """
        self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        self.num_shards = num_shards
        self.map_size_per_shard = map_size_per_shard
        self.readonly = readonly
        self.lock = lock
        self.logger = logger
        # 0 means no limit, default allows all shards open
        self.max_open_shards = max_open_shards if max_open_shards > 0 else num_shards

        # Use OrderedDict for LRU-style shard management
        self._envs: OrderedDict[int, lmdb.Environment] = OrderedDict()
        self._lock = threading.Lock()
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.logger:
            self.logger.info(
                f"ShardedLMDBManager initialized: cache_dir={self.cache_dir}, "
                f"num_shards={num_shards}, map_size_per_shard={map_size_per_shard/1024**3:.1f}GB"
            )

    @classmethod
    def get_instance(
        cls,
        cache_dir: str,
        num_shards: int = 32,
        map_size_per_shard: int = 1024 * 1024 * 1024 * 1024,
        readonly: bool = False,
        lock: bool = True,
        logger=None,
        max_open_shards: int = 0,
    ) -> "ShardedLMDBManager":
        """Get or create ShardedLMDBManager singleton.

        Args:
            cache_dir: Cache directory path.
            num_shards: Number of shards.
            map_size_per_shard: Maximum size per shard.
            readonly: Read-only mode flag.
            lock: File locking flag.
            logger: Logger instance.
            max_open_shards: Maximum open shards (0 = no limit).

        Returns:
            ShardedLMDBManager instance for the given cache_dir.
        """
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))

        if cache_dir not in cls._instances:
            cls._instances[cache_dir] = cls(
                cache_dir=cache_dir,
                num_shards=num_shards,
                map_size_per_shard=map_size_per_shard,
                readonly=readonly,
                lock=lock,
                logger=logger,
                max_open_shards=max_open_shards,
            )
        return cls._instances[cache_dir]

    def _get_shard_path(self, shard_id: int) -> str:
        """Get path to shard directory."""
        return os.path.join(self.cache_dir, f"shard_{shard_id:03d}")

    def _get_shard_id(self, image_path: str) -> int:
        """Get shard ID for an image path."""
        return get_shard_id(image_path, self.num_shards)

    def _get_env(self, shard_id: int) -> lmdb.Environment:
        """Lazy-load shard environment with LRU eviction (thread-safe).

        Args:
            shard_id: Shard ID to load.

        Returns:
            lmdb.Environment for the shard.
        """
        with self._lock:
            if shard_id in self._envs:
                # Move to end to mark as recently used
                self._envs.move_to_end(shard_id)
                return self._envs[shard_id]

            # Evict oldest shard if at capacity
            while len(self._envs) >= self.max_open_shards:
                oldest_shard_id, oldest_env = self._envs.popitem(last=False)
                try:
                    oldest_env.close()
                except Exception:
                    pass

            shard_path = self._get_shard_path(shard_id)

            # Check existing file size, dynamically adjust map_size
            data_file = os.path.join(shard_path, "data.mdb")
            actual_map_size = self.map_size_per_shard
            if os.path.exists(data_file):
                actual_size = os.path.getsize(data_file)
                if actual_size > self.map_size_per_shard:
                    actual_map_size = actual_size * 2

            self._envs[shard_id] = init_sharded_lmdb(
                shard_path,
                map_size=actual_map_size,
                readonly=self.readonly,
                lock=self.lock,
            )
            return self._envs[shard_id]

    def _get_key(self, image_path: str) -> bytes:
        """Get LMDB key for an image path."""
        return hashlib.md5(image_path.encode()).hexdigest().encode()

    def _get_shard_id_and_key(self, image_path: str) -> Tuple[int, bytes]:
        """Get shard ID and LMDB key in one MD5 call (avoid double hashing).

        Args:
            image_path: Image path string.

        Returns:
            Tuple of (shard_id, key_bytes).
        """
        hex_digest = hashlib.md5(image_path.encode()).hexdigest()
        shard_id = int(hex_digest, 16) % self.num_shards
        return shard_id, hex_digest.encode()

    def exists(self, image_path: str) -> bool:
        """Check if image exists in cache.

        Args:
            image_path: Image path to check.

        Returns:
            bool: True if image exists.
        """
        shard_id, key = self._get_shard_id_and_key(image_path)
        env = self._get_env(shard_id)

        with env.begin(write=False) as txn:
            return txn.get(key) is not None

    def put(self, image_path: str, img_bytes: bytes) -> bool:
        """Write a single image to cache.

        Args:
            image_path: Image path (used as key).
            img_bytes: Image data as bytes.

        Returns:
            bool: True if successful.
        """
        shard_id = self._get_shard_id(image_path)
        env = self._get_env(shard_id)
        key = self._get_key(image_path)

        try:
            with env.begin(write=True) as txn:
                txn.put(key, img_bytes)
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to write to LMDB [{image_path}]: {e}")
            return False

    def get(self, image_path: str) -> Optional[bytes]:
        """Read image data from cache.

        Args:
            image_path: Image path to read.

        Returns:
            bytes: Image data, or None if not found.
        """
        shard_id = self._get_shard_id(image_path)
        env = self._get_env(shard_id)
        key = self._get_key(image_path)

        with env.begin(write=False) as txn:
            return txn.get(key)

    def batch_exists(self, image_paths: List[str]) -> Dict[str, bool]:
        """Batch check existence of multiple images, one transaction per shard.

        Uses cursor.set_key() for fast key-only existence checks (avoids
        reading values), single MD5 per path, and parallel shard access.

        Args:
            image_paths: List of image paths to check.

        Returns:
            Dict mapping image_path -> bool (exists or not).
        """
        # Group paths by shard using single MD5 per path
        shard_groups: Dict[int, List[Tuple[str, bytes]]] = defaultdict(list)
        for path in image_paths:
            shard_id, key = self._get_shard_id_and_key(path)
            shard_groups[shard_id].append((path, key))

        result = {}
        result_lock = threading.Lock()

        def _check_shard(shard_id_and_items):
            shard_id, items = shard_id_and_items
            env = self._get_env(shard_id)
            shard_result = {}
            with env.begin(write=False, buffers=True) as txn:
                cursor = txn.cursor()
                for path, key in items:
                    shard_result[path] = cursor.set_key(key)
            with result_lock:
                result.update(shard_result)

        num_shards_to_check = len(shard_groups)
        if num_shards_to_check > 1:
            workers = min(num_shards_to_check, 8)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_check_shard, shard_groups.items()))
        else:
            for item in shard_groups.items():
                _check_shard(item)

        return result

    def batch_get(self, image_paths: List[str]) -> Dict[str, Optional[bytes]]:
        """Batch read multiple images, one transaction per shard.

        Opens temporary LMDB envs with readahead=True for better sequential
        read performance, and reads shards in parallel threads.

        Args:
            image_paths: List of image paths to read.

        Returns:
            Dict mapping image_path -> bytes (or None if not found).
        """
        # Group paths by shard using single MD5 per path
        shard_groups: Dict[int, List[Tuple[str, bytes]]] = defaultdict(list)
        for path in image_paths:
            shard_id, key = self._get_shard_id_and_key(path)
            shard_groups[shard_id].append((path, key))

        total = len(image_paths)
        show_progress = total >= 1000 and self.logger is not None

        result = {}
        result_lock = threading.Lock()

        if show_progress:
            pbar = tqdm(
                total=total,
                desc="LMDB read",
                unit="img",
            )

        def _open_read_env(shard_id):
            """Open a temporary env with readahead=True for batch reading."""
            shard_path = self._get_shard_path(shard_id)
            data_file = os.path.join(shard_path, "data.mdb")
            actual_map_size = self.map_size_per_shard
            if os.path.exists(data_file):
                actual_size = os.path.getsize(data_file)
                if actual_size > self.map_size_per_shard:
                    actual_map_size = actual_size * 2
            return init_sharded_lmdb(
                shard_path,
                map_size=actual_map_size,
                readonly=True,
                lock=False,
                readahead=True,
            )

        def _read_shard(shard_id_and_items):
            shard_id, items = shard_id_and_items
            env = _open_read_env(shard_id)
            try:
                shard_result = {}
                with env.begin(write=False) as txn:
                    for path, key in items:
                        shard_result[path] = txn.get(key)
                with result_lock:
                    result.update(shard_result)
                    if show_progress:
                        pbar.update(len(items))
            finally:
                env.close()

        num_shards_to_read = len(shard_groups)
        if num_shards_to_read > 1:
            # Parallel read across shards (each shard is an independent file)
            workers = min(num_shards_to_read, 8)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_read_shard, shard_groups.items()))
        else:
            for item in shard_groups.items():
                _read_shard(item)

        if show_progress:
            pbar.close()
            if self.logger:
                self.logger.info(
                    f"LMDB batch read complete: {total} images from {num_shards_to_read} shards"
                )

        return result

    def close_shard(self, shard_id: int):
        """Close a specific shard to release memory.

        Args:
            shard_id: Shard ID to close.
        """
        with self._lock:
            if shard_id in self._envs:
                try:
                    self._envs[shard_id].close()
                except Exception:
                    pass
                del self._envs[shard_id]

    def close_all(self):
        """Close all shards."""
        with self._lock:
            for shard_id in list(self._envs.keys()):
                try:
                    self._envs[shard_id].close()
                except Exception:
                    pass
            self._envs.clear()

    @classmethod
    def close_all_instances(cls):
        """Close all manager instances."""
        for instance in cls._instances.values():
            instance.close_all()
        cls._instances.clear()


def write_images_to_sharded_lmdb(
    sharded_manager: ShardedLMDBManager,
    image_paths: List[str],
    data_root: str = "",
    batch_size: int = 1000,
    resize_image: bool = True,
    resize_image_size: int = 1024,
    num_workers: int = 32,
) -> int:
    """Batch write images to sharded LMDB with optimized per-shard grouping.

    Args:
        sharded_manager: ShardedLMDBManager instance.
        image_paths: List of image paths to write.
        data_root: Root directory for relative paths.
        batch_size: Number of images per batch.
        resize_image: Whether to resize images.
        resize_image_size: Maximum dimension for resizing.
        num_workers: Number of parallel workers.

    Returns:
        int: Number of images successfully written.
    """
    logger = sharded_manager.logger

    def load_and_encode_image(image_path: str) -> Optional[Tuple[str, int, bytes]]:
        """Load image from disk and encode to bytes (no LMDB access)."""
        try:
            shard_id = sharded_manager._get_shard_id(image_path)

            # Process path
            actual_path = image_path
            if not os.path.isabs(image_path) and data_root:
                actual_path = os.path.join(data_root, image_path)

            # Read and process image
            img = cv2.imread(actual_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if resize_image:
                img = apply_resize_image_cv2(img, resize_image_size)
            img_pil = Image.fromarray(img)

            # Convert to bytes
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()

            return (image_path, shard_id, img_bytes)
        except Exception as e:
            if logger:
                logger.warning(f"Error processing image {image_path}: {e}")
            return None

    processed_count = 0

    # Fast check: one batch_exists over ALL paths at once (parallel shards,
    # cursor.set_key, single-MD5). When everything is cached this returns
    # in seconds instead of minutes.
    exists_map = sharded_manager.batch_exists(image_paths)
    new_paths = [p for p in image_paths if not exists_map.get(p, False)]
    skipped_count = len(image_paths) - len(new_paths)
    del exists_map

    if not new_paths:
        if logger:
            logger.info(
                f"Sharded LMDB write complete: 0 new, {skipped_count} skipped (already exist)"
            )
        return 0

    total_batches = (len(new_paths) + batch_size - 1) // batch_size

    pbar = tqdm(
        total=len(new_paths),
        desc="LMDB write",
        unit="img",
        disable=logger is None,
    )

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(new_paths))
        current_batch = new_paths[start_idx:end_idx]

        # Load images from disk in parallel (no LMDB access in threads)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(load_and_encode_image, current_batch))

        # Group by shard and batch write (one txn per shard)
        grouped: Dict[int, List[Tuple[str, bytes]]] = defaultdict(list)
        for r in results:
            if r is not None:
                path, shard_id, img_bytes = r
                grouped[shard_id].append((path, img_bytes))
        del results  # Release memory immediately after grouping

        for shard_id, items in grouped.items():
            env = sharded_manager._get_env(shard_id)
            with env.begin(write=True) as txn:
                for path, img_bytes in items:
                    key = sharded_manager._get_key(path)
                    txn.put(key, img_bytes)
            processed_count += len(items)
        del grouped  # Release memory after writing

        pbar.update(len(current_batch))

    pbar.close()

    if logger:
        logger.info(
            f"Sharded LMDB write complete: {processed_count} new, {skipped_count} skipped (already exist)"
        )

    return processed_count
