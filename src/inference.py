# src/inference.py
import os, io, json
import numpy as np
import joblib

def model_fn(model_dir):
    # Load the model pipeline saved in train.py
    path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model.joblib not found at {path}")
    return joblib.load(path)

def _to_ndarray(csv_text: str) -> np.ndarray:
    buf = io.StringIO(csv_text)
    def read(skip=0):
        arr = np.genfromtxt(buf, delimiter=",", dtype=float, skip_header=skip)
        if arr.size == 0:
            return np.empty((0, 0))
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    try:
        arr = read(0)
    except Exception:
        buf.seek(0)
        arr = read(1)  # try skipping a header

    if arr.size == 0:
        return arr

    # If last column looks like a label (0/1 or all integers), drop it.
    if arr.shape[1] > 1:
        last = arr[:, -1]
        if np.all(np.isfinite(last)) and (np.all(np.mod(last, 1) == 0) or set(np.unique(last)).issubset({0, 1})):
            arr = arr[:, :-1]
    return arr

def input_fn(request_body, content_type="text/csv"):
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    if content_type in ("text/csv", "text/plain", "application/octet-stream"):
        return _to_ndarray(str(request_body))
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "instances" in data:
            data = data["instances"]
        return np.array(data, dtype=float)
    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(input_data, model):
    if input_data.size == 0:
        return np.array([])
    return np.asarray(model.predict(input_data))

def output_fn(prediction, accept="text/csv"):
    if accept == "text/csv":
        flat = np.ravel(prediction)
        lines = [str(int(x)) if float(x).is_integer() else str(float(x)) for x in flat]
        return "\n".join(lines), accept
    if accept == "application/json":
        return json.dumps({"predictions": np.asarray(prediction).tolist()}), accept
    raise ValueError(f"Unsupported accept type: {accept}")

