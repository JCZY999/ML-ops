"""Own data loading and basic structural checks, not model fitting."""
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris


def load_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Use sklearn's bundled teaching dataset; no download or credentials."""
    dataset = load_iris()
    X = np.asarray(dataset.data, dtype=np.float64)
    y = np.asarray(dataset.target, dtype=np.int64)
    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError("Expected finite feature rows with matching labels")
    return X, y
