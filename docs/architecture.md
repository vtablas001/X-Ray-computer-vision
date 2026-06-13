# Architecture

This repository is organized around three concerns:

- `src/xray_cv/`: importable application code for inference and API serving.
- `artifacts/models/`: committed model artifacts required by the demo API.
- `notebooks/`: exploratory analysis and model training notebooks.
- `data/sample_images/`: small sample images for manual validation and demos.
- `docs/`: supporting project documentation and external dataset references.

The FastAPI application loads the Keras model lazily on the first prediction request.
This keeps application imports lightweight and makes tests possible without loading
TensorFlow or the model file.

Runtime configuration is read from environment variables:

- `XRAY_MODEL_PATH`: path to the `.keras` model file.
- `XRAY_PREDICTION_THRESHOLD`: binary classification threshold. Defaults to `0.5`.
