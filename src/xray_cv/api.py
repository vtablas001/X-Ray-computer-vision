"""FastAPI application for chest X-ray inference."""

from __future__ import annotations

from typing import Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile

from xray_cv.config import Settings, get_settings
from xray_cv.model import PneumoniaClassifier
from xray_cv.preprocessing import load_rgb_image
from xray_cv.schemas import PredictionResult


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()
    classifier = PneumoniaClassifier(
        model_path=settings.model_path,
        image_size=settings.image_size,
        threshold=settings.prediction_threshold,
    )

    app = FastAPI(
        title="Chest X-Ray Pneumonia Detection API",
        version="0.1.0",
        description="Prototype inference API for chest X-ray pneumonia screening.",
    )
    app.state.classifier = classifier

    @app.get("/")
    async def read_root() -> dict[str, str]:
        return {"message": "Pneumonia Detection API"}

    @app.get("/health")
    async def health_check() -> dict[str, Union[bool, str]]:
        classifier = app.state.classifier
        return {
            "status": "ok",
            "model_path": str(classifier.model_path),
            "model_loaded": classifier.is_loaded,
        }

    @app.post("/predict/", response_model=PredictionResult)
    async def predict_image(file: UploadFile = File(...)) -> PredictionResult:
        try:
            classifier = app.state.classifier
            image = load_rgb_image(await file.read())
            prediction, class_label = classifier.predict(image)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error processing image: {exc}") from exc

        return PredictionResult(prediction=prediction, class_label=class_label)

    return app


app = create_app()
