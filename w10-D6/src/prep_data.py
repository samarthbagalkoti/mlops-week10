import os
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    iris = datasets.load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = X.copy()
    df["label"] = y  # label last column

    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["label"])

    out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    os.makedirs(out_dir, exist_ok=True)

    # XGBoost expects CSV w/o header; label last
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False, header=False)
    val.to_csv(os.path.join(out_dir, "validation.csv"), index=False, header=False)
    test.to_csv(os.path.join(out_dir, "test.csv"), index=False, header=False)

if __name__ == "__main__":
    main()

