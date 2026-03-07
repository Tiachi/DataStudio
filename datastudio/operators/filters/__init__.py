"""Built-in filter operators for data quality control.

All filters inherit from :class:`~datastudio.operators.core.Filter` and
produce :class:`~datastudio.operators.core.result.FilterDecision` results.
"""

from .conv_length import ConvLengthFilter
from .image_aspect_ratio import ImageAspectRatioFilter
from .image_ext import ImageExtFilter
from .image_size import ImageSizeFilter
from .length_anomaly import LengthAnomalyFilter
from .response_tag import ResponseTagFilter
from .text_repeat import TextRepeatFilter

__all__ = [
    "ConvLengthFilter",
    "ImageSizeFilter",
    "ImageAspectRatioFilter",
    "ImageExtFilter",
    "LengthAnomalyFilter",
    "ResponseTagFilter",
    "TextRepeatFilter",
]
