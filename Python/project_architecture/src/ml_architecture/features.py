"""Own the feature contract shared by training and inference."""
from collections.abc import Sequence
import numpy as np
from numpy.typing import NDArray

FEATURE_NAMES = ("sepal_length", "sepal_width", "petal_length", "petal_width")


def build_features(rows: Sequence[Sequence[float]] | NDArray) -> NDArray[np.float64]:
    """Retain the four measurements in a fixed order; NaN means missing."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES) or len(values) == 0:
        raise ValueError("Expected a nonempty matrix with four feature columns")
    if np.isinf(values).any():
        raise ValueError("Infinite measurements are not supported")
    return values
