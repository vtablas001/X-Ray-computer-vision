"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "cnn_pneumonia_model.keras"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for model inference."""

    model_path: Path = DEFAULT_MODEL_PATH
    prediction_threshold: float = 0.5
    image_size: tuple[int, int] = (224, 224)


def get_settings() -> Settings:
    """Build settings from environment variables."""

    model_path = Path(os.getenv("XRAY_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    threshold = float(os.getenv("XRAY_PREDICTION_THRESHOLD", "0.5"))

    return Settings(model_path=model_path, prediction_threshold=threshold)
