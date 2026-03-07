"""Image size filter."""

from typing import Any, Optional, Tuple

from datastudio.utils.registry import OPERATORS

from ..core import DataItem, Filter, Result


@OPERATORS.register_module()
class ImageSizeFilter(Filter):
    """
    Filter data items based on image dimensions.

    Rejects images that are too small or have extreme aspect ratios.

    Example:
        filter = ImageSizeFilter(min_size=100, max_ratio=10.0)
        pipeline = Pipeline([filter])
        kept, rejected = pipeline(data_list)
    """

    def __init__(
        self,
        min_size: int = 28,
        max_ratio: float = 20.0,
        require_image: bool = False,
        logger: Optional[Any] = None,
        **kwargs,
    ):
        """
        Initialize the filter.

        Args:
            min_size: Minimum dimension (width or height).
            max_ratio: Maximum aspect ratio (larger / smaller).
            require_image: If True, reject items without images.
            logger: Logger instance.
        """
        super().__init__(logger=logger, **kwargs)
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.require_image = require_image

        if logger:
            logger.info(
                f"[{self.name}] Initialized: min_size={min_size}, max_ratio={max_ratio}"
            )

    def process(self, item: DataItem) -> Result:
        """Check if image meets size requirements (item-level)."""
        image = item.image

        # No image
        if image is None:
            result = Result(item_idx=item.idx)
            if self.require_image:
                result.add_filter(qa_idx=-1, rejected=True, reason="no image")
            return result

        # Get dimensions
        try:
            if hasattr(image, "size"):
                width, height = image.size
            elif hasattr(image, "shape"):
                height, width = image.shape[:2]
            else:
                return Result(item_idx=item.idx)
        except Exception:
            return Result(item_idx=item.idx)

        # Check minimum size
        if width < self.min_size or height < self.min_size:
            result = Result(item_idx=item.idx)
            result.add_filter(
                qa_idx=-1,
                rejected=True,
                reason=f"image too small: {width}x{height} < {self.min_size}",
            )
            return result

        # Check aspect ratio
        ratio = (
            max(width, height) / min(width, height)
            if min(width, height) > 0
            else float("inf")
        )
        if ratio > self.max_ratio:
            result = Result(item_idx=item.idx)
            result.add_filter(
                qa_idx=-1,
                rejected=True,
                reason=f"aspect ratio {ratio:.1f} > max {self.max_ratio}",
            )
            return result

        return Result(item_idx=item.idx)

    def check(self, item: DataItem, qa_idx: int) -> Tuple[bool, str]:
        """Not used - item-level filter."""
        return False, ""
