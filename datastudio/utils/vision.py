"""Vision utilities for image processing, encoding, and validation."""

import base64
import io
import os.path as osp
from typing import Any, Optional, Tuple

import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1e9


# ==============================================================================
# Image encoding
# ==============================================================================


def encode_image_to_base64(img, target_size=-1, fmt="JPEG"):
    """Encode PIL Image to base64 data URL string.

    Args:
        img: PIL Image object.
        target_size: Maximum dimension for thumbnail (-1 to skip resizing).
        fmt: Output format (default: JPEG).

    Returns:
        Base64 encoded data URL string.
    """
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    img_buffer.close()  # Explicitly close BytesIO to free memory
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    del image_data  # Explicitly delete large object
    return f"data:image/jpeg;base64,{image_base64}"


# ==============================================================================
# Image operations
# ==============================================================================


def apply_resize_image_cv2(img_cv2, resize_image_size):
    """Resize CV2 image maintaining aspect ratio.

    Args:
        img_cv2: OpenCV image (numpy array), shape is (height, width, channels).
        resize_image_size: Maximum dimension for the larger side.

    Returns:
        Resized OpenCV image.
    """
    # shape[0] = height, shape[1] = width; cv2.resize takes (width, height)
    h, w = img_cv2.shape[0], img_cv2.shape[1]
    if h > resize_image_size or w > resize_image_size:
        if h > w:
            # Height is larger, scale height to resize_image_size
            new_h = resize_image_size
            new_w = int(w * resize_image_size / h)
        else:
            # Width is larger, scale width to resize_image_size
            new_w = resize_image_size
            new_h = int(h * resize_image_size / w)
        img_cv2 = cv2.resize(img_cv2, (new_w, new_h))
    return img_cv2


def apply_resize_image_pil(img_pil, resize_image_size):
    """Resize PIL image maintaining aspect ratio.

    Args:
        img_pil: PIL Image object.
        resize_image_size: Maximum dimension for the larger side.

    Returns:
        Resized PIL Image.
    """
    if img_pil.width > resize_image_size or img_pil.height > resize_image_size:
        if img_pil.width > img_pil.height:
            img_pil = img_pil.resize(
                (
                    resize_image_size,
                    int(img_pil.height * resize_image_size / img_pil.width),
                )
            )
        else:
            img_pil = img_pil.resize(
                (
                    int(img_pil.width * resize_image_size / img_pil.height),
                    resize_image_size,
                )
            )
    return img_pil


# ==============================================================================
# Validation
# ==============================================================================


def parse_file(s: Any, image_name: str = "image") -> Tuple[Optional[str], Any]:
    """Parse input to determine if it's an image file.

    Args:
        s: Input to parse (string path, PIL Image, list, or other).
        image_name: Name to use for image type identification.

    Returns:
        Tuple of (mime_type, value) where mime_type is None for non-images.
    """
    if isinstance(s, list):
        return (image_name, s)
    elif isinstance(s, Image.Image):
        return (image_name, s)
    elif isinstance(s, str):
        # Check base64 data URL before osp.exists() to avoid stat()
        # syscall on multi-MB strings.
        if s.startswith("data:image"):
            return (image_name, s)
        elif osp.exists(s) and s != "." and osp.isfile(s):
            return (image_name, s)
    return (None, s)


def load_and_resize_image(img_path: str, resize_size: int):
    """Read image from disk and resize.

    Args:
        img_path: Image path (must be an absolute path).
        resize_size: Max edge size for resizing. Set to 0 or negative to skip resizing.

    Returns:
        PIL.Image object, or None if failed.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        if resize_size > 0:
            img_pil = apply_resize_image_pil(img_pil, resize_size)
        return img_pil
    except Exception:
        return None
