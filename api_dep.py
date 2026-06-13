"""Backward-compatible FastAPI entrypoint.

Prefer running the application with:

    uvicorn xray_cv.api:app --app-dir src --reload
"""

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xray_cv.api import app

