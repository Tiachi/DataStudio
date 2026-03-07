"""Image file extension filter."""

import os.path
from typing import Any, List, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter, Result


@OPERATORS.register_module()
class ImageExtFilter(Filter):
    """
    Filter data items based on image file extensions.

    Removes items containing images with excluded file extensions.
    """

    def __init__(
        self,
        excluded_exts: Optional[List[str]] = None,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            excluded_exts: List of excluded extensions (with dot, e.g., ['.gif', '']).
                          Defaults to ['.gif', ''].
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.excluded_exts = excluded_exts or [".gif", ""]
        self._excluded_lower = [e.lower() for e in self.excluded_exts]

        if logger:
            logger.info(f"[{self.name}] Initialized: excluded={self.excluded_exts}")

    def _is_excluded(self, image_path: str) -> Tuple[bool, str]:
        """Check if image extension is in the exclusion list."""
        if not image_path:
            return True, ""
        _, ext = os.path.splitext(image_path)
        return ext.lower() in self._excluded_lower, ext

    def process(self, item: DataItem) -> Result:
        """Check if any images have excluded extensions (item-level)."""
        image = item.data.get("image")
        if image is None:
            return Result(item_idx=item.idx)

        # Normalize to list
        images = image if isinstance(image, list) else [image]
        if not images:
            return Result(item_idx=item.idx)

        excluded_exts = []
        for img_path in images:
            is_excluded, ext = self._is_excluded(img_path)
            if is_excluded:
                excluded_exts.append(ext or "(empty)")

        if excluded_exts:
            result = Result(item_idx=item.idx)
            result.add_filter(
                qa_idx=-1,
                rejected=True,
                reason=(
                    f"Found {len(excluded_exts)} image(s) with excluded extensions: {excluded_exts}"
                ),
            )
            return result

        return Result(item_idx=item.idx)

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Not used - item-level filter."""
        return False, ""
