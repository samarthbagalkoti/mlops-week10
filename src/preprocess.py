# src/preprocess.py
"""
Stratified 60/20/20 split with headerless CSVs (train/test paths unchanged).
Compatible with existing SageMaker pipeline expectations.

Outputs (all headerless):
- /opt/ml/processing/train/train.csv
- /opt/ml/processing/validation/val.csv
- /opt/ml/processing/test/test.csv
"""
import argparse
import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    # Existing pipeline flags (keep these names!)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--features", type=int, default=20)
    ap.add_argument("--informative", type=int, default=15)  # easier by default
    ap.add_argument("--redundant", type=int, default=0)
    ap.add_argument("--repeated", type=int, default=0)
    ap.add_argument("--class-sep", type=float, default=2.0)
    ap.add_argument("--flip-y", type=float, default=0.0)
    ap.add_argument("--clusters-per-class", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)

    # New knobs (safe defaults that reproduce old behavior if ignored)
    ap.add_argument("--train-ratio", type=float, default=0.6)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    # Derive a valid composition for features (guard against bad combos)
    informative = max(0, min(args.informative, args.features))
    # If user asks for too many redundant/repeated, cap them so totals fit n_features
    remaining = max(0, args.features - informative)
    redundant = max(0, min(args.redundant, remaining))
    remaining -= redundant
    repeated = max(0, min(args.repeated, remaining))

    # Build synthetic dataset (binary)
    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=repeated,
        n_classes=2,
        weights=[0.5, 0.5],
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        n_clusters_per_class=args.clusters_per_class,
        random_state=args.random_state,
    ).__iter__()  # type: ignore

    y = y.astype(int, copy=False)

    # Stratified 60/20/20 split (train/val/test), but train/test paths stay identical to the old script
    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    test_ratio = 1.0 - (train_ratio + val_ratio)
    if test_ratio <= 0:
        raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio} -> test <= 0")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=(1.0 - train_ratio), stratify=y, random_state=args.random_state
    )
    rel_val = val_ratio / (val_ratio + test_ratio)  # fraction of the remaining chunk that becomes val
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - rel_val), stratify=y_tmp, random_state=args.random_state
    )

    # Output paths (unchanged for train/test; new validation file added)
    train_path = "/opt/ml/processing/train/train.csv"
    val_path   = "/opt/ml/processing/validation/val.csv"
    test_path  = "/opt/ml/processing/test/test.csv"

    for p in (train_path, val_path, test_path):
        _ensure_dir(p)

    # Write **headerless** CSVs with label as last column (matches old behavior)
    np.savetxt(train_path, np.c_[X_tr, y_tr], delimiter=",")
    np.savetxt(val_path,   np.c_[X_val, y_val], delimiter=",")
    np.savetxt(test_path,  np.c_[X_te, y_te], delimiter=",")

    print(f"Wrote {train_path}, {val_path}, {test_path}")
    print(f"Shapes: train={X_tr.shape}, val={X_val.shape}, test={X_te.shape}")

if __name__ == "__main__":
    main()

