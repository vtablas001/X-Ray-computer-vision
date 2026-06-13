"""API response schemas."""

from pydantic import BaseModel


class PredictionResult(BaseModel):
    prediction: float
    class_label: str
