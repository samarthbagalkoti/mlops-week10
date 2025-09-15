from pathlib import Path
out = Path("/opt/ml/processing/output"); out.mkdir(parents=True, exist_ok=True)
(out / "failed.txt").write_text("metric gate FAILED"); print("Wrote failed.txt")
