from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from xray_cv.api import create_app


class StubClassifier:
    model_path = "stub-model.keras"
    is_loaded = True

    def predict(self, image: Image.Image) -> tuple[float, str]:
        return 0.91, "Pneumonia"


def test_health_endpoint_reports_model_state() -> None:
    app = create_app()
    app.state.classifier = StubClassifier()

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_returns_prediction() -> None:
    app = create_app()
    app.state.classifier = StubClassifier()
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(buffer, format="PNG")

    response = TestClient(app).post(
        "/predict/",
        files={"file": ("xray.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": 0.91, "class_label": "Pneumonia"}
