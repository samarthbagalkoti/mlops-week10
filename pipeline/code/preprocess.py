import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    args = ap.parse_args()

    X, y = load_iris(return_X_y=True)
    data = np.concatenate([X, y.reshape(-1,1)], axis=1)
    tr, te = train_test_split(data, test_size=0.2, stratify=y, random_state=42)

    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    Path(args.test_dir).mkdir(parents=True, exist_ok=True)

    headers = [f"f{i}" for i in range(X.shape[1])] + ["target"]
    with open(Path(args.train_dir) / "train.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(tr.tolist())
    with open(Path(args.test_dir) / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(te.tolist())
    print("Wrote train.csv & test.csv")

if __name__ == "__main__":
    main()
