# src/train.py
"""
Compatible with the existing SageMaker pipeline:
- Reads headerless CSV from SM_CHANNEL_TRAIN (default: /opt/ml/input/data/train/train.csv)
- Label is the last column
- Saves model.joblib to SM_MODEL_DIR
- Prints train_accuracy and (if present) val_accuracy
"""

import argparse
import os
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from typing import Tuple

def _resolve_path(base: str, filename: str) -> str:
    """If base is a directory, return base/filename; otherwise treat base as a file path."""
    return os.path.join(base, filename) if os.path.isdir(base) else base

def _load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV with label in the last column.
    Handles headerless files (your default) and tolerates a single header row if present.
    """
    # First try headerless (fast path for your current pipeline)
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        return np.empty((0, 0)), np.empty((0,))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # If the first row produced NaNs (likely a header), retry skipping one header row
    if np.isnan(data).any():
        data = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

    X, y = data[:, :-1], data[:, -1]
    # Cast labels to int if they look integer-like
    if np.all(np.isfinite(y)) and np.all(np.mod(y, 1) == 0):
        y = y.astype(int, copy=False)
    return X, y

def main():
    p = argparse.ArgumentParser()
    # Keep the exact flags your pipeline already uses
    p.add_argument("--max_iter", type=int, default=int(os.environ.get("MAX_ITER", 1000)))
    p.add_argument("--C", type=float, default=float(os.environ.get("C", 2.0)))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    # Optional validation channel (not required by the current pipeline; used if present)
    p.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", ""))
    args, _ = p.parse_known_args()

    # --- Load training set (headerless CSV, label last col) ---
    train_file = _resolve_path(args.train, "train.csv")
    Xtr, ytr = _load_csv(train_file)

    # --- Model: scaler + logistic regression (same as your working version) ---
    clf = make_pipeline(StandardScaler(with_mean=True),
                        LogisticRegression(max_iter=args.max_iter, C=args.C))
    clf.fit(Xtr, ytr)

    # Train accuracy (always printed)
    tr_acc = float((clf.predict(Xtr) == ytr).mean())
    print(f"train_accuracy={tr_acc:.6f}")

    # --- Optional validation accuracy if a val file exists ---
    if args.validation:
        val_file = _resolve_path(args.validation, "val.csv")
        if os.path.exists(val_file):
            Xv, yv = _load_csv(val_file)
            if Xv.size > 0:
                val_acc = float((clf.predict(Xv) == yv).mean())
                print(f"val_accuracy={val_acc:.6f}")

    # --- Save model ---
    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))
    print(f"Saved model to {args.model_dir}")

if __name__ == "__main__":
    main()

