# src/evaluate.py
import os
import json
import tarfile
import argparse
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def extract_model(model_tar_gz, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    with tarfile.open(model_tar_gz) as tar:
        tar.extractall(path=target_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tar", type=str, default="/opt/ml/processing/model/model.tar.gz")
    args = parser.parse_args()

    # inputs mounted by processor:
    test_csv = "/opt/ml/processing/test/test.csv"
    model_dir = "/opt/ml/processing/model_extracted"
    extract_model(args.model_tar, model_dir)
    model_path = os.path.join(model_dir, "model.joblib")

    df = pd.read_csv(test_csv)
    X_test = df.drop(columns=["label"]).values
    y_test = df["label"].values

    model = joblib.load(model_path)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "binary_classification_metrics": {
            "accuracy": float(acc)
        }
    }

    out_dir = "/opt/ml/processing/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)
    print(f"evaluation_accuracy={acc:.4f}")

if __name__ == "__main__":
    main()

