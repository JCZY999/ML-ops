"""Thin orchestration: configuration -> split -> train -> evaluate -> save."""
import argparse
import json
from pathlib import Path
import platform
import tomllib
import sklearn
from sklearn.model_selection import train_test_split
from ml_architecture.ingestion import load_data
from ml_architecture.training import train_model
from ml_architecture.evaluation import evaluate_model
from ml_architecture.inference import save_model
from ml_architecture.features import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    expected = {"seed", "test_size", "regularization_c", "output_dir"}
    if set(config) != expected:
        raise ValueError(f"Configuration must contain exactly {sorted(expected)}")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(config["test_size"]) not in (float, int) or not 0.1 <= config["test_size"] <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4 for this lesson")
    if type(config["regularization_c"]) not in (float, int):
        raise ValueError("regularization_c must be numeric")
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError("output_dir must be a nonempty path")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["seed"], stratify=y)
    model = train_model(X_train, y_train, regularization_c=config["regularization_c"])
    report = {"metrics": evaluate_model(model, X_test, y_test),
              "train_rows": len(y_train), "test_rows": len(y_test),
              "features": FEATURE_NAMES, "config": config,
              "python": platform.python_version(), "sklearn": sklearn.__version__}
    output = (config_path.parent / config["output_dir"]).resolve()
    save_model(model, output / "model.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
