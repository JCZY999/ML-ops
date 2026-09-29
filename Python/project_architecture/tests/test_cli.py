"""Run the documented workflow from an unrelated working directory."""
import json
from pathlib import Path
import subprocess
import sys


def test_train_then_predict_from_another_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "train.toml"
    config.write_text('seed=42\ntest_size=0.2\nregularization_c=1.0\noutput_dir="run"\n', encoding="utf-8")
    subprocess.run([sys.executable, str(root / "scripts/train.py"), "--config", str(config)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert (report["train_rows"], report["test_rows"]) == (120, 30)
    result = subprocess.run([sys.executable, str(root / "scripts/predict.py"),
                             "--model", str(tmp_path / "run/model.joblib"),
                             "--features", "5.1", "3.5", "1.4", "0.2"],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == [0]
