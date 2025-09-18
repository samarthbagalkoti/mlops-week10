#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified W10 SageMaker Pipelines
- proc: processing-only pipeline (preprocess -> train_proc -> evaluate -> accuracy gate)
- reg : estimator + HPO + model registry + create model + batch transform

This file merges the previously working script with the new features,
while keeping the original behavior and SDK usage patterns that worked
in your environment (SM SDK v2.251.x).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ---- Friendly import guard so missing deps fail with clear instructions
try:
    import boto3
    import sagemaker
except ModuleNotFoundError as e:
    pkg = str(e).split("'")[1] if "'" in str(e) else "a required package"
    msg = f"""
ERROR: {pkg} is not installed in this Python environment.

Fix:
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install -r requirements.txt

(If you already have .venv active, ensure 'which python3' shows .venv/bin/python3)
"""
    print(msg, file=sys.stderr)
    sys.exit(1)

from botocore.exceptions import WaiterError, ClientError

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOG = logging.getLogger("w10.pipeline")
for noisy in ["sagemaker.image_uris", "sagemaker.workflow.utilities", "botocore.credentials"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------- SDK imports ----------
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep, TuningStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.transformer import Transformer
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.model_step import ModelStep
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

# -------------------------------------------------------------------------------------
# Common helpers
# -------------------------------------------------------------------------------------

def _classic_session(region: str) -> sagemaker.session.Session:
    boto_sess = boto3.Session(region_name=region)
    return sagemaker.session.Session(boto_session=boto_sess)

def _pipeline_session(region: str) -> PipelineSession:
    os.environ.setdefault("AWS_DEFAULT_REGION", region)
    return PipelineSession()

# =====================================================================================
#                              Processing-only pipeline (proc)
# =====================================================================================

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
    """Re(gen) helper scripts for the processing-only pipeline (simple & self-contained)."""
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
    _tar = out_dir / "model.tar.gz"
    joblib.dump(clf, model_file)
    _tar_model(model_file, _tar)
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
    # PROC pipeline uses this JSON path:
    (out / "evaluation.json").write_text(json.dumps({"metrics": {"accuracy": {"value": acc}}}))
    print(f"accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
""",
    )

    _write(
        MARK_PASS,
        "from pathlib import Path\nout=Path('/opt/ml/processing/output');out.mkdir(parents=True,exist_ok=True);(out/'passed.txt').write_text('metric gate PASSED');print('Wrote passed.txt')\n",
    )

    _write(
        MARK_FAIL,
        "from pathlib import Path\nout=Path('/opt/ml/processing/output');out.mkdir(parents=True,exist_ok=True);(out/'failed.txt').write_text('metric gate FAILED');print('Wrote failed.txt')\n",
    )

def build_pipeline_proc(
    region: str,
    role_arn: str,
    bucket: str,
    accuracy_threshold: float,
    framework_version: str,
    proc_instance: str,
) -> Pipeline:
    LOG.info("Building PROC pipeline (region=%s, role=%s, bucket=%s, thr=%.4f, fw=%s)",
             region, role_arn, bucket, accuracy_threshold, framework_version)
    _ensure_codegen()
    sess = _pipeline_session(region)

    p_region  = ParameterString("Region", default_value=region)
    p_role    = ParameterString("RoleArn", default_value=role_arn)
    p_bucket  = ParameterString("Bucket", default_value=bucket)
    p_acc_thr = ParameterFloat("AccuracyThreshold", default_value=accuracy_threshold)

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
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test",  source="/opt/ml/processing/test"),
        ],
    )

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
#                    Estimator + HPO + Model Registry + Batch (reg)
# =====================================================================================

PIPELINE_NAME_REG = "w10d1-hello-sm-pipeline"

_TRAIN_ALLOWED = {
    "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
    "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge", "ml.c5.9xlarge", "ml.c5.18xlarge",
    "ml.m4.xlarge", "ml.m4.2xlarge", "ml.m4.4xlarge", "ml.m4.10xlarge", "ml.m4.16xlarge",
    "ml.r5.large", "ml.r5.xlarge", "ml.r5.2xlarge", "ml.r5.4xlarge",
}
def _normalize_training_instance(instance_type: str) -> str:
    if instance_type in _TRAIN_ALLOWED:
        return instance_type
    if instance_type.endswith(".large"):
        candidate = instance_type[:-6] + ".xlarge"
        if candidate in _TRAIN_ALLOWED:
            LOG.warning("Training instance '%s' not allowed; using '%s' instead.", instance_type, candidate)
            return candidate
    allowed = ", ".join(sorted(_TRAIN_ALLOWED))
    raise ValueError(f"Training instance '{instance_type}' not supported. Choose one of: {allowed}")

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
                ModelPackageGroupDescription="W10:D2/D3/D4 demo group"
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
    proc_instance: str,
    train_instance: str,
    batch_instance: str = "ml.m5.large",
    batch_output_prefix: str = "",
    hpo_max_jobs: int = 6,
    hpo_parallel_jobs: int = 2,
) -> Pipeline:
    sess = _pipeline_session(region)
    default_bucket = sess.default_bucket() if not bucket else bucket
    s3_prefix = "w10d1"
    framework_version = "1.2-1"

    p_accuracy_threshold = ParameterFloat("AccuracyThreshold", default_value=acc_threshold)
    p_model_pkg_group   = ParameterString("ModelPackageGroupName", default_value=group_name)
    p_model_approval    = ParameterString("ModelApprovalStatus", default_value=approval_status)
    p_batch_instance    = ParameterString("BatchInstanceType", default_value=batch_instance)
    p_batch_output      = ParameterString(
        "BatchOutputPrefix",
        default_value=batch_output_prefix or f"s3://{default_bucket}/{s3_prefix}/batch/predictions"
    )

    # Preprocess (now writes train/val/test for HPO + eval)
    sklearn_proc = SKLearnProcessor(
        framework_version=framework_version,
        role=role_arn,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10d1-preprocess",
        sagemaker_session=sess,
    )
    step_preprocess = ProcessingStep(
        name="Preprocess",
        processor=sklearn_proc,
        code="src/preprocess.py",
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train",
                             destination=f"s3://{default_bucket}/{s3_prefix}/data/train"),
            ProcessingOutput(output_name="val_data", source="/opt/ml/processing/validation",
                             destination=f"s3://{default_bucket}/{s3_prefix}/data/validation"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",
                             destination=f"s3://{default_bucket}/{s3_prefix}/data/test"),
        ],
        job_arguments=["--samples", "3000", "--features", "20", "--informative", "10", "--random-state", "42"],
    )

    # Base estimator + metric defs (parsed from your train.py prints)
    metric_defs = [
        {"Name": "val_accuracy", "Regex": r"val_accuracy=([0-9\\.]+)"},
        {"Name": "train_accuracy", "Regex": r"train_accuracy=([0-9\\.]+)"},
    ]
    train_instance = _normalize_training_instance(train_instance)
    estimator = SKLearn(
        entry_point="src/train.py",
        role=role_arn,
        instance_type=train_instance,
        instance_count=1,
        framework_version=framework_version,
        sagemaker_session=sess,
        base_job_name="w10d1-train",
        hyperparameters={"max_iter": 200, "C": 1.0},
        metric_definitions=metric_defs,
    )

    # HPO ranges & tuner
    hpo_ranges = {
        "C": ContinuousParameter(0.01, 10.0, scaling_type="Logarithmic"),
        "max_iter": IntegerParameter(100, 1000),
    }
    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="val_accuracy",
        objective_type="Maximize",
        hyperparameter_ranges=hpo_ranges,
        max_jobs=hpo_max_jobs,
        max_parallel_jobs=hpo_parallel_jobs,
        strategy="Bayesian",
    )
    step_hpo = TuningStep(
        name="HPO",
        tuner=tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                content_type="text/csv"),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri,
                content_type="text/csv"),
        },
    )

    # Evaluate BEST model on TEST
    eval_proc = SKLearnProcessor(
        framework_version=framework_version,
        role=role_arn,
        instance_type=proc_instance,
        instance_count=1,
        base_job_name="w10d1-eval",
        sagemaker_session=sess,
    )
    eval_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_evaluate = ProcessingStep(
        name="Evaluate",
        processor=eval_proc,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_hpo.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket),
                destination="/opt/ml/processing/model",
                input_name="model_artifacts",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
                input_name="test_data",
            ),
        ],
        outputs=[ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{default_bucket}/{s3_prefix}/evaluation",
        )],
        property_files=[eval_report],
    )

    # Gate (REG pipeline expects {"binary_classification_metrics":{"accuracy": <float>}})
    acc_json = JsonGet(
        step_name=step_evaluate.name,
        property_file=eval_report,
        json_path="binary_classification_metrics.accuracy",
    )
    step_fail = FailStep(name="FailIfLowAccuracy", error_message="Model accuracy below threshold")
    cond = ConditionGreaterThanOrEqualTo(left=acc_json, right=p_accuracy_threshold)

    # Script-mode inference model (src/inference.py) based on BEST artifacts
    inference_model = SKLearnModel(
        model_data=step_hpo.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket),
        role=role_arn,
        entry_point="src/inference.py",
        framework_version=framework_version,
        sagemaker_session=sess,
    )

    # Attach metrics to registration
    eval_s3_uri = step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    metrics_s3_uri = Join(on="", values=[eval_s3_uri, "/evaluation.json"])
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(s3_uri=metrics_s3_uri, content_type="application/json")
    )

    # Register in Model Registry (NOTE: hosting instances must be from allowed set; no t3.* here)
    register_step = RegisterModel(
        name="RegisterModel",
        model=inference_model,
        model_metrics=model_metrics,
        model_package_group_name=p_model_pkg_group,
        approval_status=p_model_approval,
        content_types=["text/csv"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
    )

    # Create a deployable Model for Batch Transform (from artifacts) – robust across SDKs
    create_model_args = inference_model.create()
    step_create_model = ModelStep(name="CreateModelForBatch", step_args=create_model_args)

    # Batch Transform (on TEST split)
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_count=1,
        instance_type=p_batch_instance,
        output_path=p_batch_output,
        sagemaker_session=sess,
        assemble_with="Line",
        accept="text/csv",
    )
    transform_input = TransformInput(
        data=step_preprocess.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
        content_type="text/csv",
        split_type="Line",
    )
    step_transform = TransformStep(name="BatchTransform", transformer=transformer, inputs=transform_input)

    step_condition = ConditionStep(
        name="AccuracyGate",
        conditions=[cond],
        if_steps=[register_step, step_create_model, step_transform],
        else_steps=[step_fail],
    )

    return Pipeline(
        name=PIPELINE_NAME_REG,
        parameters=[p_accuracy_threshold, p_model_pkg_group, p_model_approval, p_batch_instance, p_batch_output],
        steps=[step_preprocess, step_hpo, step_evaluate, step_condition],
        sagemaker_session=sess,
    )

def _ensure_reg_pipeline(region: str,
                         role_arn: str,
                         bucket: str,
                         acc_threshold: float,
                         group_name: str,
                         approval_status: str,
                         proc_instance: str,
                         train_instance: str,
                         batch_instance: str,
                         batch_output_prefix: str,
                         hpo_max_jobs: int,
                         hpo_parallel_jobs: int):
    ensure_model_package_group(region, group_name)
    pipe = build_pipeline_reg(
        region, role_arn, bucket, acc_threshold, group_name, approval_status,
        proc_instance=proc_instance, train_instance=train_instance,
        batch_instance=batch_instance, batch_output_prefix=batch_output_prefix,
        hpo_max_jobs=hpo_max_jobs, hpo_parallel_jobs=hpo_parallel_jobs,
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
        batch_instance=args.batch_instance,
        batch_output_prefix=args.batch_output_prefix,
        hpo_max_jobs=args.hpo_max_jobs,
        hpo_parallel_jobs=args.hpo_parallel_jobs,
    )
    print("Upserted REG pipeline and ensured MPG exists.")

def reg_cmd_run(args):
    sm_sess = _classic_session(args.region)  # client-only to start
    try:
        start_resp = sm_sess.sagemaker_client.start_pipeline_execution(PipelineName=PIPELINE_NAME_REG)
        print("Started execution:", start_resp["PipelineExecutionArn"])
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = (e.response.get("Error", {}).get("Message") or "").lower()
        if code in ("ResourceNotFound", "ValidationException") and ("does not exist" in msg or "not found" in msg):
            if not args.role_arn or not args.group_name:
                raise RuntimeError(
                    "REG pipeline not found and --role-arn / --group-name not provided. "
                    "Re-run with: reg run --role-arn <arn> --group-name <mpg>"
                )
            print("REG pipeline missing; creating it now...")
            _ensure_reg_pipeline(
                region=args.region,
                role_arn=args.role_arn,
                bucket=args.bucket,
                acc_threshold=args.accuracy_threshold,
                group_name=args.group_name,
                approval_status=args.approval,
                proc_instance=args.proc_instance,
                train_instance=args.train_instance,
                batch_instance=args.batch_instance,
                batch_output_prefix=args.batch_output_prefix,
                hpo_max_jobs=args.hpo_max_jobs,
                hpo_parallel_jobs=args.hpo_parallel_jobs,
            )
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
        print("No executions yet."); return
    last = items[0]
    print(f"Last Execution: {last.get('PipelineExecutionArn')}\nStatus: {last.get('PipelineExecutionStatus')}\n"
          f"Created: {last.get('CreationTime')}\nLastModified: {last.get('LastModifiedTime')}\n")

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
    resp = sm.list_model_packages(ModelPackageGroupName=args.group_name, SortBy="CreationTime", SortOrder="Descending")
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print("No model packages found."); return
    for p in pkgs:
        print(f"{p['ModelPackageArn']}  |  v{p['ModelPackageVersion']}  |  {p['ModelApprovalStatus']}  |  {p['CreationTime']}")

def reg_cmd_approve_latest(args):
    sm = boto3.client("sagemaker", region_name=args.region)
    resp = sm.list_model_packages(ModelPackageGroupName=args.group_name, SortBy="CreationTime", SortOrder="Descending")
    pkgs = resp.get("ModelPackageSummaryList", [])
    if not pkgs:
        print("No model packages to update."); return
    latest = pkgs[0]
    sm.update_model_package(ModelPackageArn=latest["ModelPackageArn"],
                            ModelApprovalStatus=args.status,
                            ApprovalDescription=args.note or "")
    print(f"Set {latest['ModelPackageArn']} to {args.status}")

# =====================================================================================
#                                        CLI
# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description="Unified W10 SageMaker Pipelines")
    sub = ap.add_subparsers(dest="group", required=True)

    # PROC
    proc = sub.add_parser("proc", help="Processing-only pipeline (preprocess->train_proc->evaluate->gate)")
    procs = proc.add_subparsers(dest="cmd", required=True)

    def _proc_common(p):
        p.add_argument("--region", required=True)
        p.add_argument("--role-arn", required=True, dest="role_arn")
        p.add_argument("--bucket", required=True)
        p.add_argument("--accuracy-threshold", required=True, type=float, dest="accuracy")
        p.add_argument("--proc-instance", default="ml.t3.medium")

    p_up = procs.add_parser("upsert"); _proc_common(p_up)
    p_run = procs.add_parser("run");    _proc_common(p_run)

    # REG
    reg = sub.add_parser("reg", help="Estimator + HPO + Registry + Batch pipeline")
    regs = reg.add_subparsers(dest="cmd", required=True)

    def _reg_common(p):
        p.add_argument("--region", required=True)

    r_eg = regs.add_parser("ensure-group"); _reg_common(r_eg); r_eg.add_argument("--group-name", required=True)

    r_up = regs.add_parser("upsert"); _reg_common(r_up)
    r_up.add_argument("--role-arn", required=True, dest="role_arn")
    r_up.add_argument("--bucket", default="")
    r_up.add_argument("--accuracy-threshold", type=float, default=0.85)
    r_up.add_argument("--group-name", required=True)
    r_up.add_argument("--approval", default="PendingManualApproval")
    r_up.add_argument("--proc-instance", default="ml.t3.medium")
    r_up.add_argument("--train-instance", default="ml.m5.large")
    r_up.add_argument("--batch-instance", default="ml.m5.large")
    r_up.add_argument("--batch-output-prefix", default="")
    r_up.add_argument("--hpo-max-jobs", type=int, default=6)
    r_up.add_argument("--hpo-parallel-jobs", type=int, default=2)

    r_run = regs.add_parser("run"); _reg_common(r_run)
    r_run.add_argument("--role-arn", dest="role_arn", default="")
    r_run.add_argument("--group-name", dest="group_name", default="")
    r_run.add_argument("--bucket", default="")
    r_run.add_argument("--accuracy-threshold", type=float, dest="accuracy_threshold", default=0.85)
    r_run.add_argument("--approval", default="PendingManualApproval")
    r_run.add_argument("--proc-instance", default="ml.t3.medium")
    r_run.add_argument("--train-instance", default="ml.m5.large")
    r_run.add_argument("--batch-instance", default="ml.m5.large")
    r_run.add_argument("--batch-output-prefix", default="")
    r_run.add_argument("--hpo-max-jobs", type=int, default=6)
    r_run.add_argument("--hpo-parallel-jobs", type=int, default=2)

    r_st = regs.add_parser("status"); _reg_common(r_st)
    r_desc = regs.add_parser("describe"); _reg_common(r_desc); r_desc.add_argument("--arn", required=True)
    r_ls = regs.add_parser("list-packages"); _reg_common(r_ls); r_ls.add_argument("--group-name", required=True)
    r_ap = regs.add_parser("approve-latest"); _reg_common(r_ap)
    r_ap.add_argument("--group-name", required=True)
    r_ap.add_argument("--status", choices=["Approved", "Rejected", "PendingManualApproval"], required=True)
    r_ap.add_argument("--note", default="")

    args = ap.parse_args()

    if args.group == "proc":
        if args.cmd == "upsert":
            pipe = build_pipeline_proc(region=args.region, role_arn=args.role_arn, bucket=args.bucket,
                                       accuracy_threshold=args.accuracy, framework_version="1.2-1",
                                       proc_instance=args.proc_instance)
            pipe.upsert(role_arn=args.role_arn)
            print(f"Upserted pipeline: {pipe.name}")
        elif args.cmd == "run":
            pipe = build_pipeline_proc(region=args.region, role_arn=args.role_arn, bucket=args.bucket,
                                       accuracy_threshold=args.accuracy, framework_version="1.2-1",
                                       proc_instance=args.proc_instance)
            pipe.upsert(role_arn=args.role_arn)
            exe = pipe.start(parameters={"Region": args.region, "RoleArn": args.role_arn,
                                         "Bucket": args.bucket, "AccuracyThreshold": args.accuracy})
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

