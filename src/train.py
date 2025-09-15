# src/train.py
import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()

    train_path = "/opt/ml/input/data/train/train.csv"
    df = pd.read_csv(train_path)
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    model = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")
    model.fit(X, y)

    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    print(f"train_accuracy={acc:.4f}")

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    print("Saved model to", model_dir)

if __name__ == "__main__":
    main()

