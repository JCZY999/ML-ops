"""Predict four measurements using a trusted artifact from train.py."""
import argparse
from pathlib import Path
from ml_architecture.inference import load_model, predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=float, nargs=4, required=True,
                        metavar=("SEPAL_LENGTH", "SEPAL_WIDTH", "PETAL_LENGTH", "PETAL_WIDTH"))
    args = parser.parse_args()
    print(predict(load_model(args.model), [args.features]))


if __name__ == "__main__":
    main()
