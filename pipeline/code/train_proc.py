import argparse, csv, tarfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

def _load_csv(p: Path):
    with open(p, "r") as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=float)  # skip header
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y

def _tar_model(model_path: Path, out_tar: Path):
    with tarfile.open(out_tar, "w:gz") as t:
        t.add(model_path, arcname=model_path.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()

    X, y = _load_csv(Path(args.train_dir) / "train.csv")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    out_dir = Path(args.model_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / "model.joblib"
    _tar = out_dir / "model.tar.gz"
    joblib.dump(clf, model_file)
    _tar_model(model_file, _tar)
    print("Saved model.joblib and model.tar.gz")

if __name__ == "__main__":
    main()
