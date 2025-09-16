import argparse
import csv
import os
from pathlib import Path
import sys
import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


TRAIN_CHANNEL = Path(os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
MODEL_DIR     = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def _pick_train_csv() -> Path:
    """
    Return a Path to a CSV file under the train channel.
    Prefer .../train.csv, else first *.csv we can find.
    Raise a clear error if nothing is found.
    """
    p = TRAIN_CHANNEL / "train.csv"
    if p.exists():
        return p
    # fall back to any CSV
    cands = sorted(glob.glob(str(TRAIN_CHANNEL / "**" / "*.csv"), recursive=True))
    if cands:
        return Path(cands[0])
    raise FileNotFoundError(
        f"No CSV found under train channel: {TRAIN_CHANNEL}. "
        "Ensure preprocess step wrote train.csv to the 'train' output and that "
        "the TrainingStep maps that S3 URI to channel name 'train'."
    )


def _load_csv_with_header(p: Path):
    """
    Load CSV with a header row; last column is the label.
    Robust to stray whitespace; raises a clear error if non-numeric cells appear.
    """
    with open(p, "r") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"CSV appears empty: {p}")
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        raise ValueError(f"CSV has header but no data: {p}")

    try:
        data = np.array(data_rows, dtype=float)
    except ValueError as e:
        # Try a whitespace-trimmed pass before failing
        trimmed = [[c.strip() for c in r] for r in data_rows]
        try:
            data = np.array(trimmed, dtype=float)
        except Exception:
            raise ValueError(f"Could not parse numeric values from {p}: {e}") from e

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected tabular data with features+label; got shape {data.shape} from {p}")

    X, y = data[:, :-1], data[:, -1].astype(int)
    if np.unique(y).size < 2:
        raise ValueError("Label column has <2 classes; need a binary classification target.")
    return header, X, y


def main():
    # Be tolerant of unknown args (containers sometimes add their own)
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--C", type=float, default=1.0)
    args, _unknown = ap.parse_known_args()

    print(f"[train.py] TRAIN_CHANNEL={TRAIN_CHANNEL}")
    csv_path = _pick_train_csv()
    print(f"[train.py] Using training CSV: {csv_path}")

    header, X, y = _load_csv_with_header(csv_path)
    print(f"[train.py] Loaded X.shape={X.shape}, y.shape={y.shape}, features={header[:-1]}")

    clf = LogisticRegression(max_iter=args.max_iter, C=args.C)
    clf.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "model.joblib"
    joblib.dump(clf, out_path)
    print(f"[train.py] Saved model to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Print full error and exit non-zero so the container surfaces it
        print(f"[train.py] ERROR: {e}", file=sys.stderr)
        raise

