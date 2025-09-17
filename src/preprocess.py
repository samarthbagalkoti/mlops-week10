# src/preprocess.py
import argparse
import os
import numpy as np
from sklearn.datasets import make_classification

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--features", type=int, default=20)
    # make it easier by default
    ap.add_argument("--informative", type=int, default=15)
    ap.add_argument("--redundant", type=int, default=0)
    ap.add_argument("--repeated", type=int, default=0)
    ap.add_argument("--class-sep", type=float, default=2.0)
    ap.add_argument("--flip-y", type=float, default=0.0)
    ap.add_argument("--clusters-per-class", type=int, default=1)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=args.informative,
        n_redundant=args.redundant,
        n_repeated=args.repeated,
        n_classes=2,
        weights=[0.5, 0.5],
        class_sep=args.class_sep,
        flip_y=args.flip_y,
        n_clusters_per_class=args.clusters_per_class,
        random_state=args.random_state,
    )

    n = len(y)
    split = int(0.8 * n)
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    np.savetxt("/opt/ml/processing/train/train.csv", np.c_[X[:split], y[:split]], delimiter=",")
    np.savetxt("/opt/ml/processing/test/test.csv",  np.c_[X[split:], y[split:]], delimiter=",")

if __name__ == "__main__":
    main()

