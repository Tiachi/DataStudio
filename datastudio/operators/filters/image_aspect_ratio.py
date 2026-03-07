"""Image aspect ratio filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter, Result


@OPERATORS.register_module()
class ImageAspectRatioFilter(Filter):
    """
    Filter data items based on image aspect ratio.

    Removes items containing images with extreme aspect ratios.
    """

    def __init__(
        self,
        max_aspect_ratio: float = 64.0,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            max_aspect_ratio: Maximum allowed aspect ratio (width/height or height/width).
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.max_aspect_ratio = max_aspect_ratio

        if logger:
            logger.info(f"[{self.name}] Initialized: max_ratio={max_aspect_ratio}")

    def process(self, item: DataItem) -> Result:
        """Check if all images have acceptable aspect ratios (item-level)."""
        img_sizes = item.data.get("img_size")
        if img_sizes is None:
            return Result(item_idx=item.idx)

        for img_size in img_sizes:
            width, height = img_size[0], img_size[1]
            if height == 0 or width == 0:
                continue
            ratio = max(width / height, height / width)

            if ratio > self.max_aspect_ratio:
                result = Result(item_idx=item.idx)
                result.add_filter(
                    qa_idx=-1,
                    rejected=True,
                    reason=(
                        f"Aspect ratio {ratio:.2f} > max {self.max_aspect_ratio} (size: {img_size})"
                    ),
                )
                return result

        return Result(item_idx=item.idx)

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Not used - item-level filter."""
        return False, ""
