import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--features", type=int, default=20)
    ap.add_argument("--informative", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    # Output dirs (the ProcessingStep maps these to S3)
    train_dir = Path("/opt/ml/processing/train")
    test_dir  = Path("/opt/ml/processing/test")
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=args.informative,
        n_redundant=max(0, args.features - args.informative),
        n_classes=2,
        random_state=args.random_state,
    )
    data = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    tr, te = train_test_split(data, test_size=0.2, stratify=y, random_state=42)

    headers = [f"f{i}" for i in range(args.features)] + ["label"]
    with open(train_dir / "train.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(tr.tolist())
    with open(test_dir / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(te.tolist())
    print("Wrote train.csv & test.csv")

if __name__ == "__main__":
    main()

