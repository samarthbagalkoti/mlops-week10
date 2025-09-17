# src/train.py
import argparse
import os
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def load_dataset(train_dir: str):
    path = os.path.join(train_dir, "train.csv")
    data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X, y = data[:, :-1], data[:, -1]
    if np.all(np.mod(y, 1) == 0):
        y = y.astype(int)
    return X, y

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_iter", type=int, default=int(os.environ.get("MAX_ITER", 1000)))
    p.add_argument("--C", type=float, default=float(os.environ.get("C", 2.0)))
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    p.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    args, _ = p.parse_known_args()

    X, y = load_dataset(args.train)
    clf = make_pipeline(StandardScaler(with_mean=True), LogisticRegression(max_iter=args.max_iter, C=args.C))
    clf.fit(X, y)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))

if __name__ == "__main__":
    main()

