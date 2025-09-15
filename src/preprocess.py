# src/preprocess.py
import os
import argparse
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--informative", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=args.informative,
        n_redundant=args.features - args.informative,
        random_state=args.random_state,
    )
    cols = [f"feature_{i}" for i in range(args.features)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=args.random_state)

    train_out = "/opt/ml/processing/train/train.csv"
    test_out = "/opt/ml/processing/test/test.csv"
    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    os.makedirs(os.path.dirname(test_out), exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Wrote {train_out} and {test_out}")

if __name__ == "__main__":
    main()

