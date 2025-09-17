# src/evaluate.py
import os
import json
import tarfile
import tempfile
import glob
import numpy as np
import joblib
import sys
import traceback

MODEL_DIR = "/opt/ml/processing/model"   # where the ProcessingInput lands
TEST_DIR  = "/opt/ml/processing/test"
OUT_DIR   = "/opt/ml/processing/evaluation"
REPORT    = os.path.join(OUT_DIR, "evaluation.json")

def _find_model_tar(model_dir: str) -> str:
    # look for any *.tar.gz (SageMaker downloads the Estimator model as model.tar.gz)
    cands = glob.glob(os.path.join(model_dir, "*.tar.gz"))
    if not cands:
        raise FileNotFoundError(f"No .tar.gz found under {model_dir}")
    # prefer model.tar.gz if present
    for p in cands:
        if os.path.basename(p) == "model.tar.gz":
            return p
    return cands[0]

def _extract_and_load_model(model_tar: str):
    tmp = tempfile.mkdtemp()
    with tarfile.open(model_tar, "r:gz") as t:
        t.extractall(tmp)
    # typical path is tmp/model.joblib; but search just in case
    cands = glob.glob(os.path.join(tmp, "**", "model.joblib"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"'model.joblib' not found inside {model_tar}")
    return joblib.load(cands[0])

def _load_csv(path: str):
    # robust loader: handle single row & optional header
    def read(skiprows: int):
        arr = np.genfromtxt(path, delimiter=",", skip_footer=0, skip_header=skiprows)
        if arr.size == 0:
            raise ValueError("empty file")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    try:
        data = read(0)
    except Exception:
        # maybe a header exists; try skipping one line
        data = read(1)

    X, y = data[:, :-1], data[:, -1]
    # If labels look like integers, cast to int (avoids scikit warnings)
    if np.all(np.isfinite(y)) and np.all(np.mod(y, 1) == 0):
        y = y.astype(int)
    return X, y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        model_tar = _find_model_tar(MODEL_DIR)
        clf = _extract_and_load_model(model_tar)

        test_csv = os.path.join(TEST_DIR, "test.csv")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"Test file not found at {test_csv}")

        X, y = _load_csv(test_csv)

        # Compute accuracy
        preds = clf.predict(X)
        acc = float((preds == y).mean())

        # REG pipeline expects this schema:
        report = {"binary_classification_metrics": {"accuracy": acc}}
        with open(REPORT, "w") as f:
            json.dump(report, f)

        # helpful logs
        print(f"[EVAL] X.shape={X.shape}, y.shape={y.shape}, accuracy={acc:.6f}")
        print(f"[EVAL] Wrote report to {REPORT}")

    except Exception as e:
        # Print the full stack to container logs so failure reason is visible
        print("[EVAL] ERROR:", repr(e), file=sys.stderr)
        traceback.print_exc()
        # signal failure to the container
        sys.exit(1)

if __name__ == "__main__":
    main()

