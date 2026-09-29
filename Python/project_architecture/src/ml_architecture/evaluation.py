"""Evaluate an already fitted pipeline on held-out rows. Never fit here."""
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from .features import build_features


def evaluate_model(model: Pipeline, X: NDArray, y: NDArray) -> dict[str, float]:
    predicted = model.predict(build_features(X))
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0))}
