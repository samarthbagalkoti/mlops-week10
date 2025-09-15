import argparse
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def main():
    ap = argparse.ArgumentParser()
    # In SageMaker training, input channel 'train' is mounted to /opt/ml/input/data/train
    ap.add_argument("--channel-train", default="/opt/ml/input/data/train")
    args = ap.parse_args()

    train_csv = Path(args.channel_train) / "train.csv"
    df = pd.read_csv(train_csv)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    model_dir = Path("/opt/ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_dir / "model.joblib")
    print("Saved model.joblib")

if __name__ == "__main__":
    main()
