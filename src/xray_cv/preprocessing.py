"""Image loading and preprocessing utilities."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


def load_rgb_image(image_bytes: bytes) -> Image.Image:
    """Load uploaded bytes as an RGB PIL image."""

    return Image.open(BytesIO(image_bytes)).convert("RGB")


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Resize and normalize an image for model inference."""

    resized_image = image.resize(image_size)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)
