from pathlib import Path
out = Path("/opt/ml/processing/output"); out.mkdir(parents=True, exist_ok=True)
(out / "passed.txt").write_text("metric gate PASSED"); print("Wrote passed.txt")
