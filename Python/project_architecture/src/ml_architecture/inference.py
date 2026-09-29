"""Reuse the full fitted pipeline, including its training-time transforms."""
from collections.abc import Sequence
from pathlib import Path
import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from .features import build_features


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    """Load ONLY trusted local artifacts: joblib/pickle can execute code."""
    return joblib.load(path)


def predict(model: Pipeline, rows: Sequence[Sequence[float]] | NDArray) -> list[int]:
    return np.asarray(model.predict(build_features(rows)), dtype=int).tolist()
