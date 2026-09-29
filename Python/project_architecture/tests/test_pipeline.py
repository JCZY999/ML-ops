import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from ml_architecture.ingestion import load_data
from ml_architecture.features import build_features
from ml_architecture.training import train_model
from ml_architecture.evaluation import evaluate_model
from ml_architecture.inference import save_model, load_model, predict


@pytest.fixture
def split():
    X, y = load_data()
    return train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)


@pytest.mark.parametrize("rows", [[], [[1, 2]], [[1, 2, 3, np.inf]]])
def test_feature_contract(rows):
    with pytest.raises(ValueError):
        build_features(rows)


def test_training_transforms_do_not_learn_from_test_data(split):
    X_train, X_test, y_train, y_test = split
    model = train_model(X_train, y_train)
    scaler = model.named_steps["prepare"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
    before = scaler.mean_.copy()
    evaluate_model(model, X_test + 100, y_test)
    np.testing.assert_array_equal(scaler.mean_, before)


def test_round_trip_and_missing_values(split, tmp_path):
    X_train, X_test, y_train, y_test = split
    model = train_model(X_train, y_train)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert predict(model, X_test) == predict(loaded, X_test)
    missing = X_test[:1].copy()
    missing[0, 0] = np.nan
    assert len(predict(loaded, missing)) == 1
    metrics = evaluate_model(loaded, X_test, y_test)
    assert set(metrics) == {"accuracy", "macro_f1"}
    assert all(0 <= value <= 1 for value in metrics.values())


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_bad_hyperparameter(split, value):
    with pytest.raises(ValueError):
        train_model(split[0], split[2], regularization_c=value)
