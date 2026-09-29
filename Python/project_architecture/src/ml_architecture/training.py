"""Own estimator construction and fitting, not file paths or CLI parsing."""
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .features import build_features
from .preprocessing import make_preprocessor


def train_model(X: NDArray, y: NDArray, *, regularization_c: float = 1.0) -> Pipeline:
    if not np.isfinite(regularization_c) or regularization_c <= 0:
        raise ValueError("regularization_c must be finite and positive")
    features = build_features(X)
    if np.isnan(features).all(axis=0).any():
        raise ValueError("Each training feature needs at least one observation")
    model = Pipeline([("prepare", make_preprocessor()),
                      ("model", LogisticRegression(C=regularization_c, max_iter=1000))])
    model.fit(features, y)
    return model
