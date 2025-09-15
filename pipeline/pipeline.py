#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W10 SageMaker Pipeline — preprocess -> train (processing) -> evaluate -> gate
- Pure Processing steps (no Estimator).
- Helper scripts use ONLY stdlib + numpy + scikit-learn (no pandas).
- Managed S3 outputs (no explicit destinations).
- Default instance: ml.t3.medium (override with --proc-instance).
- Prints failing step names & reasons if pipeline fails.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from botocore.exceptions import WaiterError
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("w10.pipeline")
for noisy in ["sagemaker.image_uris", "sagemaker.workflow.utilities", "botocore.credentials"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------- Paths ----------
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR / "code"
PREPROCESS = CODE_DIR / "preprocess.py"
TRAIN = CODE_DIR / "train_proc.py"
EVALUATE = CODE_DIR / "evaluate.py"
MARK_PASS = CODE_DIR / "mark_pass.py"
MARK_FAIL = CODE_DIR / "mark_fail.py"


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def _ensure_codegen():
    """Always (re)write helper scripts to ensure numpy-only versions are used."""
    _write(
        PREPROCESS,
        """\
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    args = ap.parse_args()

    X, y = load_iris(return_X_y=True)
    data = np.concatenate([X, y.reshape(-1,1)], axis=1)
    tr, te = train_test_split(data, test_size=0.2, stratify=y, random_state=42)

    Path(args.train_dir).mkdir(parents=True, exist_ok=True)
    Path(args.test_dir).mkdir(parents=True, exist_ok=True)

    headers = [f"f{i}" for i in range(X.shape[1])] + ["target"]
    with open(Path(args.train_dir) / "train.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(tr.tolist())
    with open(Path(args.test_dir) / "test.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(te.tolist())
    print("Wrote train.csv & test.csv")

if __name__ == "__main__":
    main()
""",
    )

    _write(
        TRAIN,
        """\
import argparse, csv, tarfile
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

def _load_csv(p: Path):
    with open(p, "r") as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=float)  # skip header
    X, y = data[:, :-1], data[:, -1].astype(int)
    return X, y

def _tar_model(model_path: Path, out_tar: Path):
    with tarfile.open(out_tar, "w:gz") as t:
        t.add(model_path, arcname=model_path.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    args = ap.parse_args()

    X, y = _load_csv(Path(args.train_dir) / "train.csv")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    out_dir = Path(args.model_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_file = out_dir / "model.joblib"
    joblib.dump(clf, model_file)
    _tar_model(model_file, out_dir / "model.tar.gz")
    print("Saved model.joblib and model.tar.gz")

if __name__ == "__main__":
    main()
""",
    )

    _write(
        EVALUATE,
        """\
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
    (out / "evaluation.json").write_text(json.dumps({"metrics": {"accuracy": {"value": acc}}}))
    print(f"accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
""",
    )

    _write(
        MARK_PASS,
        """\
from pathlib import Path
out = Path("/opt/ml/processing/output"); out.mkdir(parents=True, exist_ok=True)
(out / "passed.txt").write_text("metric gate PASSED"); print("Wrote passed.txt")
""",
    )

    _write(
        MARK_FAIL,
        """\
from pathlib import Path
out = Path("/opt/ml/processing/output"); out.mkdir(parents=True, exist_ok=True)
(out / "failed.txt").write_text("metric gate FAILED"); print("Wrote failed.txt")
""",
    )


def _pipeline_session(region: str) -> PipelineSession:
    os.environ.setdefault("AWS_DEFAULT_REGION", region)
    return PipelineSession()


def build_pipeline(
    region: str,
    role_arn: str,
    bucket: str,                  # kept for CLI compatibility; outputs are managed
    accuracy_threshold: float,
    framework_version: str,
    proc_instance: str,           # e.g., ml.t3.medium
) -> Pipeline:
    LOG.info(
        "Building pipeline (region=%s, role=%s, bucket=%s, thr=%.4f, fw=%s)",
        region, role_arn, bucket, accuracy_threshold, framework_version
    )
    _ensure_codegen()
    sess = _pipeline_session(region)

    p_region  = ParameterString("Region", default_value=region)
    p_role    = ParameterString("RoleArn", default_value=role_arn)
    p_bucket  = ParameterString("Bucket", default_value=bucket)
    p_acc_thr = ParameterFloat("AccuracyThreshold", default_value=accuracy_threshold)

    # 1) Preprocess
    preprocess = SKLearnProcessor(
        framework_version=framework_version,
        role=p_role,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10-preprocess",
        sagemaker_session=sess,
    )
    step_pre = ProcessingStep(
        name="Preprocess",
        processor=preprocess,
        code=str(PREPROCESS),
        job_arguments=["--train_dir", "/opt/ml/processing/train",
                       "--test_dir", "/opt/ml/processing/test"],
        outputs=[ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
                 ProcessingOutput(output_name="test",  source="/opt/ml/processing/test")],
    )

    # 2) Train
    trainer = SKLearnProcessor(
        framework_version=framework_version,
        role=p_role,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10-train-proc",
        sagemaker_session=sess,
    )
    step_train = ProcessingStep(
        name="TrainProc",
        processor=trainer,
        code=str(TRAIN),
        inputs=[ProcessingInput(
            input_name="train",
            source=step_pre.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            destination="/opt/ml/processing/train")],
        job_arguments=["--train_dir", "/opt/ml/processing/train",
                       "--model_dir", "/opt/ml/processing/model"],
        outputs=[ProcessingOutput(output_name="model", source="/opt/ml/processing/model")],
    )

    # 3) Evaluate
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    evaluator = SKLearnProcessor(
        framework_version=framework_version,
        role=p_role,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10-evaluate",
        sagemaker_session=sess,
    )
    step_eval = ProcessingStep(
        name="Evaluate",
        processor=evaluator,
        code=str(EVALUATE),
        inputs=[
            ProcessingInput(input_name="model",
                            source=step_train.properties.ProcessingOutputConfig.Outputs["model"].S3Output.S3Uri,
                            destination="/opt/ml/processing/model_in"),
            ProcessingInput(input_name="test",
                            source=step_pre.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                            destination="/opt/ml/processing/test"),
        ],
        job_arguments=["--model_dir", "/opt/ml/processing/model_in",
                       "--test_dir", "/opt/ml/processing/test",
                       "--report_dir", "/opt/ml/processing/evaluation"],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[eval_report],
    )

    # 4) Gate
    accuracy_expr = JsonGet(step_name=step_eval.name, property_file=eval_report, json_path="metrics.accuracy.value")
    passer = SKLearnProcessor(framework_version=framework_version, role=p_role,
                              instance_type=proc_instance, instance_count=1,
                              base_job_name="w10-pass", sagemaker_session=sess)
    step_pass = ProcessingStep(
        name="MarkPass", processor=passer, code=str(MARK_PASS),
        outputs=[ProcessingOutput(output_name="passout", source="/opt/ml/processing/output")],
    )
    failer = SKLearnProcessor(framework_version=framework_version, role=p_role,
                              instance_type=proc_instance, instance_count=1,
                              base_job_name="w10-fail", sagemaker_session=sess)
    step_fail = ProcessingStep(
        name="MarkFail", processor=failer, code=str(MARK_FAIL),
        outputs=[ProcessingOutput(output_name="failout", source="/opt/ml/processing/output")],
    )
    gate = ConditionStep(
        name="AccuracyGate",
        conditions=[ConditionGreaterThanOrEqualTo(left=accuracy_expr, right=p_acc_thr)],
        if_steps=[step_pass],
        else_steps=[step_fail],
    )

    return Pipeline(
        name="W10ProcSklearnPipeline",
        parameters=[p_region, p_role, p_bucket, p_acc_thr],
        steps=[step_pre, step_train, step_eval, gate],
        sagemaker_session=sess,
    )


def _print_failure(exe):
    try:
        steps = exe.list_steps()
    except Exception as e:
        LOG.error("Could not list steps: %s", e); return
    failed = [s for s in steps if s.get("StepStatus") == "Failed"]
    if not failed:
        LOG.error("Pipeline failed; steps:\n%s", json.dumps(steps, indent=2, default=str)); return
    LOG.error("Failed steps:")
    for s in failed:
        name = s.get("StepName")
        msg = s.get("FailureReason") or s.get("Metadata", {}).get("ProcessingJob", {}).get("FailureReason")
        LOG.error("  - %s: %s", name, msg)


def main():
    ap = argparse.ArgumentParser(description="W10 SageMaker Pipeline (processing-only; numpy-based scripts)")
    sub = ap.add_subparsers(dest="cmd")

    def add_common(p):
        p.add_argument("--region", required=True)
        p.add_argument("--role-arn", required=True, dest="role_arn")
        p.add_argument("--bucket", required=True)  # kept for compatibility
        p.add_argument("--accuracy-threshold", required=True, type=float, dest="accuracy")
        p.add_argument("--proc-instance", default="ml.t3.medium",
                       help="Processing instance type (default: ml.t3.medium)")

    sub.add_parser("upsert", help="Create/Update the pipeline"); add_common(sub.choices["upsert"])
    sub.add_parser("run", help="Start a pipeline execution");   add_common(sub.choices["run"])

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help(); sys.exit(2)

    pipe = build_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        bucket=args.bucket,
        accuracy_threshold=args.accuracy,
        framework_version="1.2-1",
        proc_instance=args.proc_instance,
    )

    if args.cmd == "upsert":
        pipe.upsert(role_arn=args.role_arn)
        print(f"Upserted pipeline: {pipe.name}")
    elif args.cmd == "run":
        pipe.upsert(role_arn=args.role_arn)
        exe = pipe.start(parameters={
            "Region": args.region,
            "RoleArn": args.role_arn,
            "Bucket": args.bucket,
            "AccuracyThreshold": args.accuracy,
        })
        LOG.info("Execution: %s", exe.arn)
        try:
            exe.wait()
            print(json.dumps(exe.list_steps(), indent=2, default=str))
        except WaiterError:
            _print_failure(exe)
            raise

if __name__ == "__main__":
    main()

