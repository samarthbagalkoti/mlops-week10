#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified W10 SageMaker Pipelines
- proc: processing-only pipeline (preprocess -> train_proc -> evaluate -> accuracy gate)
- reg:  estimator + model registry pipeline (preprocess -> train(estimator) -> evaluate -> gate -> register)

Helper scripts for the proc pipeline are auto-(re)generated (numpy + scikit-learn only).
The reg pipeline expects user-provided src/{preprocess.py,train.py,evaluate.py}.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
import sagemaker
from botocore.exceptions import WaiterError, ClientError

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("w10.pipeline")
for noisy in ["sagemaker.image_uris", "sagemaker.workflow.utilities", "botocore.credentials"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------- Common SageMaker imports ----------
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.fail_step import FailStep
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.step_collections import RegisterModel

# =====================================================================================
#                              Processing-only pipeline (proc)
# =====================================================================================

# ---------- Paths for PROC ----------
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR / "code"
PREPROCESS = CODE_DIR / "preprocess.py"
TRAIN_PROC = CODE_DIR / "train_proc.py"
EVALUATE = CODE_DIR / "evaluate.py"
MARK_PASS = CODE_DIR / "mark_pass.py"
MARK_FAIL = CODE_DIR / "mark_fail.py"

def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def _ensure_codegen():
    """Always (re)write helper scripts for the processing-only pipeline."""
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
        TRAIN_PROC,
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

def build_pipeline_proc(
    region: str,
    role_arn: str,
    bucket: str,                  # kept for CLI compatibility; outputs are managed
    accuracy_threshold: float,
    framework_version: str,
    proc_instance: str,           # e.g., ml.t3.medium
) -> Pipeline:
    LOG.info(
        "Building PROC pipeline (region=%s, role=%s, bucket=%s, thr=%.4f, fw=%s)",
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

    # 2) Train (processing-based)
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
        code=str(TRAIN_PROC),
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

# =====================================================================================
#                    Estimator + Model Registry pipeline (reg)
# =====================================================================================

PIPELINE_NAME_REG = "w10d1-hello-sm-pipeline"

# Minimal allowlist for SKLearn training instance types (subset of long enum from service)
_TRAIN_ALLOWED = {
    "ml.t3.medium", "ml.t3.large", "ml.t3.xlarge", "ml.t3.2xlarge",
    "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
    "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge", "ml.c5.9xlarge", "ml.c5.18xlarge",
    "ml.m4.xlarge", "ml.m4.2xlarge", "ml.m4.4xlarge", "ml.m4.10xlarge", "ml.m4.16xlarge",
    "ml.r5.large", "ml.r5.xlarge", "ml.r5.2xlarge", "ml.r5.4xlarge",
}

def _normalize_training_instance(instance_type: str) -> str:
    """Return a valid training instance type; auto-bump *.large -> *.xlarge for common families."""
    if instance_type in _TRAIN_ALLOWED:
        return instance_type
    # common pitfall: c5.large (not allowed) -> c5.xlarge
    if instance_type.endswith(".large"):
        candidate = instance_type[:-6] + ".xlarge"
        if candidate in _TRAIN_ALLOWED:
            LOG.warning("Training instance '%s' not allowed; using '%s' instead.", instance_type, candidate)
            return candidate
    allowed = ", ".join(sorted(_TRAIN_ALLOWED))
    raise ValueError(
        f"Training instance '{instance_type}' is not supported by the SKLearn training image. "
        f"Choose one of: {allowed}"
    )

def _classic_session(region: str) -> sagemaker.session.Session:
    boto_sess = boto3.Session(region_name=region)
    return sagemaker.session.Session(boto_session=boto_sess)

def ensure_model_package_group(region: str, group_name: str):
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"Model Package Group '{group_name}' exists.")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = (e.response.get("Error", {}).get("Message") or "").lower()
        if code in ("ValidationException", "ResourceNotFoundException") and "does not exist" in msg:
            sm.create_model_package_group(
                ModelPackageGroupName=group_name,
                ModelPackageGroupDescription="W10:D2 demo group"
            )
            print(f"Created Model Package Group '{group_name}'.")
        else:
            raise

def build_pipeline_reg(
    region: str,
    role_arn: str,
    bucket: str,
    acc_threshold: float,
    group_name: str,
    approval_status: str,
    proc_instance: str,        # processing instance type (quota-friendly, e.g., ml.t3.medium)
    train_instance: str,       # training instance type (e.g., ml.t3.medium or ml.m5.large)
) -> Pipeline:
    sm_sess = _classic_session(region)
    default_bucket = sm_sess.default_bucket() if not bucket else bucket
    s3_prefix = "w10d1"
    framework_version = "1.2-1"  # SageMaker SKLearn container

    # Parameters (can override at execution time)
    accuracy_threshold = ParameterFloat("AccuracyThreshold", default_value=acc_threshold)
    model_pkg_group = ParameterString("ModelPackageGroupName", default_value=group_name)
    model_approval = ParameterString("ModelApprovalStatus", default_value=approval_status)

    # ---- Step: Preprocess (Processing) ----
    sklearn_proc = SKLearnProcessor(
        framework_version=framework_version,
        role=role_arn,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10d1-preprocess",
        sagemaker_session=sm_sess,
    )
    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=sklearn_proc,
        code="src/preprocess.py",
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train",
                             destination=f"s3://{default_bucket}/{s3_prefix}/data/train"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",
                             destination=f"s3://{default_bucket}/{s3_prefix}/data/test"),
        ],
        job_arguments=["--samples", "3000", "--features", "20", "--informative", "10", "--random-state", "42"],
    )

    # ---- Step: Train (SKLearn Estimator) ----
    train_instance = _normalize_training_instance(train_instance)
    estimator = SKLearn(
        entry_point="src/train.py",
        role=role_arn,
        instance_type=train_instance,
        instance_count=1,
        framework_version=framework_version,
        sagemaker_session=sm_sess,
        base_job_name="w10d1-train",
        hyperparameters={"max_iter": 200, "C": 1.0},
    )
    step_train = TrainingStep(
        name="Train",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    # ---- Step: Evaluate (Processing) ----
    eval_proc = SKLearnProcessor(
        framework_version=framework_version,
        role=role_arn,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10d1-eval",
        sagemaker_session=sm_sess,
    )
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    step_evaluate = ProcessingStep(
        name="Evaluate",
        processor=eval_proc,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
                input_name="model_artifacts",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                input_name="test_data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{default_bucket}/{s3_prefix}/evaluation",
            )
        ],
        property_files=[eval_report],
    )

    # ---- Gate: accuracy >= threshold ----
    acc_json = JsonGet(
        step_name=step_evaluate.name,
        property_file=eval_report,
        json_path="binary_classification_metrics.accuracy",  # adjust if your src/evaluate.py uses another schema
    )
    step_fail = FailStep(name="FailIfLowAccuracy", error_message="Model accuracy below threshold")
    cond = ConditionGreaterThanOrEqualTo(left=acc_json, right=accuracy_threshold)

    # ---- RegisterModel (runs only if gate passes) ----
    eval_s3_uri = step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    metrics_s3_uri = Join(on="", values=[eval_s3_uri, "/evaluation.json"])
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(s3_uri=metrics_s3_uri, content_type="application/json")
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],  # metadata only; endpoints not created here
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_pkg_group,
        model_metrics=model_metrics,
        approval_status=model_approval,
    )

    step_condition = ConditionStep(
        name="AccuracyGate",
        conditions=[cond],
        if_steps=[register_step],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=PIPELINE_NAME_REG,
        parameters=[accuracy_threshold, model_pkg_group, model_approval],
        steps=[step_preprocess, step_train, step_evaluate, step_condition],
        sagemaker_session=sm_sess,
    )

def _classic_session(region: str) -> sagemaker.session.Session:
    boto_sess = boto3.Session(region_name=region)
    return sagemaker.session.Session(boto_session=boto_sess)

def _ensure_reg_pipeline(region: str,
                         role_arn: str,
                         bucket: str,
                         acc_threshold: float,
                         group_name: str,
                         approval_status: str,
                         proc_instance: str,
                         train_instance: str):
    """Ensure MPG exists and the REG pipeline is upserted."""
    ensure_model_package_group(region, group_name)
    pipe = build_pipeline_reg(
        region, role_arn, bucket, acc_threshold, group_name, approval_status,
        proc_instance=proc_instance, train_instance=train_instance
    )
    pipe.upsert(role_arn=role_arn)
    print(f"Ensured pipeline: {pipe.name}")

def reg_cmd_upsert(args):
    _ensure_reg_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        bucket=args.bucket,
        acc_threshold=args.accuracy_threshold,
        group_name=args.group_name,
        approval_status=args.approval,
        proc_instance=args.proc_instance,
        train_instance=args.train_instance,
    )
    print("Upserted REG pipeline and ensured MPG exists.")

def reg_cmd_run(args):
    sm_sess = _classic_session(args.region)
    try:
        start_resp = sm_sess.sagemaker_client.start_pipeline_execution(PipelineName=PIPELINE_NAME_REG)
        print("Started execution:", start_resp["PipelineExecutionArn"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = (e.response.get("Error", {}).get("Message") or "").lower()
        # Handle missing pipeline by creating it, if we have enough info
        if code in ("ResourceNotFound", "ValidationException") and ("does not exist" in msg or "not found" in msg):
            if not args.role_arn or not args.group_name:
                raise RuntimeError(
                    "REG pipeline not found and --role-arn / --group-name not provided. "
                    "Re-run with: reg run --role-arn <arn> --group-name <mpg>"
                )
            print("REG pipeline missing; creating it now...")
            _ensure_reg_pipeline(region=args.region,
                                 role_arn=args.role_arn,
                                 bucket=args.bucket,
                                 acc_threshold=args.accuracy_threshold,
                                 group_name=args.group_name,
                                 approval_status=args.approval,
                                 proc_instance=args.proc_instance,
                                 train_instance=args.train_instance)
            # Try again
            start_resp = sm_sess.sagemaker_client.start_pipeline_execution(PipelineName=PIPELINE_NAME_REG)
            print("Started execution:", start_resp["PipelineExecutionArn"])
        else:
            raise

def reg_cmd_status(args):
    sm = boto3.client("sagemaker", region_name=args.region)
    execs = sm.list_pipeline_executions(
        PipelineName=PIPELINE_NAME_REG, SortBy="CreationTime", SortOrder="Descending", MaxResults=20
    )
    items = execs.get("PipelineExecutions", []) or []
    if not items:
        print("No executions yet.")
        return

    last = items[0]
    arn = last.get("PipelineExecutionArn")
    status = last.get("PipelineExecutionStatus")
    created = last.get("CreationTime") or last.get("StartTime")
    lastmod = last.get("LastModifiedTime")
    print(f"Last Execution: {arn}\nStatus: {status}\nCreated: {created}\nLastModified: {lastmod}\n")

def reg_cmd_describe(args):
    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.describe_pipeline_execution(PipelineExecutionArn=args.arn)
    out = {
        "PipelineExecutionArn": resp.get("PipelineExecutionArn"),
        "PipelineArn": resp.get("PipelineArn"),
        "PipelineExecutionDisplayName": resp.get("PipelineExecutionDisplayName"),
        "PipelineExecutionStatus": resp.get("PipelineExecutionStatus"),
        "CreationTime": resp.get("CreationTime"),
        "LastModifiedTime": resp.get("LastModifiedTime"),
        "FailureReason": resp.get("FailureReason"),
    }
    print(json.dumps(out, indent=2, default=str))

def reg_cmd_ensure_group(args):
    ensure_model_package_group(args.region, args.group_name)

def reg_cmd_list_packages(args):
    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.list_model_packages(
        ModelPackageGroupName=args.group_name,
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print("No model packages found.")
        return
    for p in pkgs:
        print(f"{p['ModelPackageArn']}  |  v{p['ModelPackageVersion']}  |  {p['ModelApprovalStatus']}  |  {p['CreationTime']}")

def reg_cmd_approve_latest(args):
    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.list_model_packages(
        ModelPackageGroupName=args.group_name,
        SortBy="CreationTime",
        SortOrder="Descending"
    )
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print("No model packages to update.")
        return
    latest = pkgs[0]
    sm.update_model_package(
        ModelPackageArn=latest["ModelPackageArn"],
        ModelApprovalStatus=args.status,
        ApprovalDescription=args.note or ""
    )
    print(f"Set {latest['ModelPackageArn']} to {args.status}")

# =====================================================================================
#                                        CLI
# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description="Unified W10 SageMaker Pipelines")
    sub = ap.add_subparsers(dest="group", required=True)

    # ---------------- PROC group ----------------
    proc = sub.add_parser("proc", help="Processing-only pipeline (original)")
    procs = proc.add_subparsers(dest="cmd", required=True)

    def _proc_common(p):
        p.add_argument("--region", required=True)
        p.add_argument("--role-arn", required=True, dest="role_arn")
        p.add_argument("--bucket", required=True)  # kept for compatibility
        p.add_argument("--accuracy-threshold", required=True, type=float, dest="accuracy")
        p.add_argument("--proc-instance", default="ml.t3.medium",
                       help="Processing instance type (default: ml.t3.medium)")

    p_up = procs.add_parser("upsert", help="Create/Update the PROC pipeline"); _proc_common(p_up)
    p_run = procs.add_parser("run", help="Start a PROC pipeline execution");   _proc_common(p_run)

    # ---------------- REG group ----------------
    reg = sub.add_parser("reg", help="Estimator + Model Registry pipeline")
    regs = reg.add_subparsers(dest="cmd", required=True)

    def _reg_common(p):
        p.add_argument("--region", required=True)

    r_eg = regs.add_parser("ensure-group", help="Create the Model Package Group if missing"); _reg_common(r_eg)
    r_eg.add_argument("--group-name", required=True)

    r_up = regs.add_parser("upsert", help="Create/Update the REG pipeline"); _reg_common(r_up)
    r_up.add_argument("--role-arn", required=True, dest="role_arn")
    r_up.add_argument("--bucket", default="")
    r_up.add_argument("--accuracy-threshold", type=float, default=0.85)
    r_up.add_argument("--group-name", required=True)
    r_up.add_argument("--approval", default="PendingManualApproval")
    r_up.add_argument("--proc-instance", default="ml.t3.medium",
                      help="Processing instance type for Preprocess/Evaluate (quota-friendly)")
    r_up.add_argument("--train-instance", default="ml.t3.medium",
                      help="Training instance type for Estimator (default: ml.t3.medium)")

    r_run = regs.add_parser("run", help="Start a REG pipeline execution (auto-create if missing)"); _reg_common(r_run)
    r_run.add_argument("--role-arn", dest="role_arn", default="", help="(optional) If provided and pipeline is missing, it will be created.")
    r_run.add_argument("--group-name", dest="group_name", default="", help="(optional) MPG name; required if auto-create is needed.")
    r_run.add_argument("--bucket", default="", help="(optional) S3 bucket; used during auto-create.")
    r_run.add_argument("--accuracy-threshold", type=float, dest="accuracy_threshold", default=0.85, help="(optional) Threshold for auto-create.")
    r_run.add_argument("--approval", default="PendingManualApproval", help="(optional) Initial approval for auto-create.")
    r_run.add_argument("--proc-instance", default="ml.t3.medium",
                       help="(optional) Processing instance type for auto-create")
    r_run.add_argument("--train-instance", default="ml.t3.medium",
                       help="(optional) Training instance type for auto-create")

    r_st = regs.add_parser("status", help="Show latest REG pipeline execution status"); _reg_common(r_st)

    r_desc = regs.add_parser("describe", help="Describe a specific pipeline execution by ARN"); _reg_common(r_desc)
    r_desc.add_argument("--arn", required=True, help="PipelineExecutionArn to describe")

    r_ls = regs.add_parser("list-packages", help="List model packages in a group"); _reg_common(r_ls)
    r_ls.add_argument("--group-name", required=True)

    r_ap = regs.add_parser("approve-latest", help="Approve/Reject latest model package"); _reg_common(r_ap)
    r_ap.add_argument("--group-name", required=True)
    r_ap.add_argument("--status", choices=["Approved", "Rejected", "PendingManualApproval"], required=True)
    r_ap.add_argument("--note", default="")

    # ---------------- Parse & dispatch ----------------
    args = ap.parse_args()

    if args.group == "proc":
        if args.cmd == "upsert":
            pipe = build_pipeline_proc(
                region=args.region,
                role_arn=args.role_arn,
                bucket=args.bucket,
                accuracy_threshold=args.accuracy,
                framework_version="1.2-1",
                proc_instance=args.proc_instance,
            )
            pipe.upsert(role_arn=args.role_arn)
            print(f"Upserted pipeline: {pipe.name}")
        elif args.cmd == "run":
            pipe = build_pipeline_proc(
                region=args.region,
                role_arn=args.role_arn,
                bucket=args.bucket,
                accuracy_threshold=args.accuracy,
                framework_version="1.2-1",
                proc_instance=args.proc_instance,
            )
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
                try:
                    steps = exe.list_steps()
                    LOG.error("Pipeline failed; steps:\n%s", json.dumps(steps, indent=2, default=str))
                except Exception:
                    pass
                raise
        else:
            procs.print_help(); sys.exit(2)
    elif args.group == "reg":
        if args.cmd == "ensure-group":
            reg_cmd_ensure_group(args)
        elif args.cmd == "upsert":
            reg_cmd_upsert(args)
        elif args.cmd == "run":
            reg_cmd_run(args)
        elif args.cmd == "status":
            reg_cmd_status(args)
        elif args.cmd == "describe":
            reg_cmd_describe(args)
        elif args.cmd == "list-packages":
            reg_cmd_list_packages(args)
        elif args.cmd == "approve-latest":
            reg_cmd_approve_latest(args)
        else:
            regs.print_help(); sys.exit(2)
    else:
        ap.print_help(); sys.exit(2)

if __name__ == "__main__":
    main()

