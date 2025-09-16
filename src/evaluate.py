import csv, json, tarfile
from pathlib import Path
import numpy as np
import joblib

MODEL_IN = Path("/opt/ml/processing/model")       # contains model.tar.gz from training
TEST_IN  = Path("/opt/ml/processing/test")        # contains test.csv from preprocess
REPORT   = Path("/opt/ml/processing/evaluation")  # we must write evaluation.json here

def _load_csv_with_header(p: Path):
    with open(p, "r") as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=float)  # skip header
    return data[:, :-1], data[:, -1].astype(int)

def _extract_model(tar_dir: Path, out_dir: Path):
    tar_path = tar_dir / "model.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"Model archive not found: {tar_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(path=out_dir)

def main():
    # Extract trained model
    model_work = Path("/tmp/model")
    _extract_model(MODEL_IN, model_work)
    model_path = model_work / "model.joblib"
    clf = joblib.load(model_path)

    # Load test set
    X, y = _load_csv_with_header(TEST_IN / "test.csv")
    acc = float((clf.predict(X) == y).mean())

    # Emit metrics in the schema your pipeline expects:
    # JsonGet(..., json_path="binary_classification_metrics.accuracy")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "evaluation.json").write_text(json.dumps({
        "binary_classification_metrics": {
            "accuracy": acc
        }
    }))
    print(f"accuracy={acc:.4f}")

if __name__ == "__main__":
    main()

