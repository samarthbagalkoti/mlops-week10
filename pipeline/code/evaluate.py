import argparse, csv, tarfile, json
from pathlib import Path
import numpy as np
import joblib

def _load_csv(p: Path):
    with open(p, "r") as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=float)
    return data[:, :-1], data[:, -1].astype(int)

def _extract_model(model_tar_dir, out_dir):
    tar_path = Path(model_tar_dir) / "model.tar.gz"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(path=out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--report_dir", required=True)
    args = ap.parse_args()

    work_model = "/opt/ml/processing/model"
    _extract_model(args.model_dir, work_model)
    clf = joblib.load(Path(work_model) / "model.joblib")

    X, y = _load_csv(Path(args.test_dir) / "test.csv")
    acc = float((clf.predict(X) == y).mean())

    out = Path(args.report_dir); out.mkdir(parents=True, exist_ok=True)
    # PROC pipeline uses this JSON path:
    (out / "evaluation.json").write_text(json.dumps({"metrics": {"accuracy": {"value": acc}}}))
    print(f"accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
