from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from xray_cv.preprocessing import load_rgb_image, preprocess_image


def test_load_rgb_image_converts_uploaded_bytes() -> None:
    buffer = BytesIO()
    Image.new("L", (8, 8), color=128).save(buffer, format="PNG")

    image = load_rgb_image(buffer.getvalue())

    assert image.mode == "RGB"
    assert image.size == (8, 8)


def test_preprocess_image_resizes_normalizes_and_adds_batch_axis() -> None:
    image = Image.new("RGB", (16, 16), color=(255, 0, 0))

    batch = preprocess_image(image, image_size=(224, 224))

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.isclose(batch.max(), 1.0)
    assert np.isclose(batch.min(), 0.0)
