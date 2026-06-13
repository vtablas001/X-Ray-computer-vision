# AI-Assisted Chest X-Ray Pneumonia Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![API](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Prototype-green)]()

Prototype computer vision project for chest X-ray pneumonia screening. The repository contains a trained Keras model, a FastAPI inference service, sample images, and the original exploratory notebook.

## Medical Disclaimer

This project is intended for research and development only. It is not a certified medical device and must not be used as the sole basis for diagnosis, triage, or treatment decisions.

## Project Structure

```text
.
├── artifacts/
│   └── models/                  # Trained model artifacts used by the API
├── data/
│   └── sample_images/            # Small sample set for demos and manual checks
├── docs/                         # Supporting documentation and dataset references
├── notebooks/                    # Exploratory analysis and training notebooks
├── src/
│   └── xray_cv/                  # Importable application package
├── tests/                        # Unit and API tests
├── api_dep.py                    # Backward-compatible FastAPI entrypoint
├── pyproject.toml                # Package metadata and optional dependencies
└── requirements.txt              # Legacy dependency file
```

## Quickstart

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the project with API and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For notebook work, include the analysis dependencies:

```bash
python -m pip install -e ".[dev,analysis]"
```

## Configuration

The API reads configuration from environment variables:

```bash
XRAY_MODEL_PATH=artifacts/models/cnn_pneumonia_model.keras
XRAY_PREDICTION_THRESHOLD=0.5
```

Copy `.env.example` if you want to keep local overrides. The default model path already points to the committed model artifact.

## Run The API

```bash
uvicorn xray_cv.api:app --app-dir src --reload
```

Useful endpoints:

- `GET /`: basic API message.
- `GET /health`: service status and model loading state.
- `POST /predict/`: upload a chest X-ray image using multipart form field `file`.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
  -F "file=@data/sample_images/PNEUMONIA/person1946_bacteria_4874.jpeg"
```

## Tests

```bash
pytest
```

The API tests use a stub classifier so they can validate request and response behavior without loading TensorFlow or the Keras model.

## Notebook

The original exploratory and training notebook lives at:

[notebooks/Torax_X_Ray_pneumonia_CNN.ipynb](notebooks/Torax_X_Ray_pneumonia_CNN.ipynb)

## Dataset Reference

The external dataset archive reference is stored in:

[docs/dataset_source.txt](docs/dataset_source.txt)

## Live Demo

[Platform Demo](https://v0-pulmonary-ai-demo.vercel.app/)

<img width="1294" height="1280" alt="Application screenshot" src="https://github.com/user-attachments/assets/d16c3549-3fee-4d1f-a011-89ad3b6b8714" />

<img width="1272" height="767" alt="Application results screenshot" src="https://github.com/user-attachments/assets/51bfb34a-1b05-4c71-a885-03630dab9646" />
