"""Model loading and prediction service."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from xray_cv.preprocessing import preprocess_image


class PredictiveModel(Protocol):
    def predict(self, image_batch: np.ndarray) -> np.ndarray:
        """Return prediction scores for a batch of images."""


class PneumoniaClassifier:
    """Lazy-loading classifier wrapper around a Keras model."""

    def __init__(
        self,
        model_path: Path,
        image_size: tuple[int, int],
        threshold: float,
        model: Optional[PredictiveModel] = None,
    ) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.threshold = threshold
        self._model = model

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> PredictiveModel:
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)

        return self._model

    def predict(self, image: Image.Image) -> tuple[float, str]:
        model = self.load()
        image_batch = preprocess_image(image, self.image_size)
        prediction = float(model.predict(image_batch)[0][0])
        class_label = "Pneumonia" if prediction > self.threshold else "Normal"
        return prediction, class_label
