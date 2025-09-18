# pipeline/pipeline.py
import argparse, json, time, boto3, sagemaker

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString

# Use only widely-available workflow modules
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.endpoint_config_step import EndpointConfigStep
from sagemaker.workflow.endpoint_step import EndpointStep
from sagemaker.workflow.transform_step import TransformStep

from sagemaker.inputs import TrainingInput, TransformInput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.image_uris import retrieve as img
from sagemaker.processing import ProcessingOutput

PIPELINE_NAME = "mini-train-reg-batch-rt"   # no-HPO variant for maximal compatibility

def create_pipeline(region: str, bucket: str, role_arn: str) -> Pipeline:
    sess = PipelineSession(boto3.Session(region_name=region))

    # -------- Parameters --------
    p_region = ParameterString("Region", default_value=region)
    p_bucket = ParameterString("Bucket", default_value=bucket)
    p_role = ParameterString("RoleArn", default_value=role_arn)
    p_train_instance = ParameterString("TrainInstanceType", default_value="ml.m5.large")
    p_proc_instance  = ParameterString("ProcInstanceType",  default_value="ml.m5.large")
    p_xform_instance = ParameterString("TransformInstanceType", default_value="ml.m5.large")
    p_ep_instance    = ParameterString("EndpointInstanceType", default_value="ml.m5.large")
    p_endpoint_name  = ParameterString("EndpointName", default_value=f"mini-endpoint-{int(time.time())}")

    # -------- Step 1: Prep (Iris -> CSVs) --------
    prep = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=p_proc_instance,
        instance_count=1,
        sagemaker_session=sess,
    )
    prep_step = ProcessingStep(
        name="PrepData",
        processor=prep,
        code="src/prep_data.py",
        outputs=[ProcessingOutput(output_name="prepared", source="/opt/ml/processing/output")],
    )

    # -------- Step 2: Train (single job; no HPO) --------
    xgb_image = img("xgboost", region=region, version="1.5-1")
    est = XGBoost(
        image_uri=xgb_image,
        role=role_arn,
        instance_count=1,
        instance_type=p_train_instance,
        sagemaker_session=sess,
        hyperparameters={
            "objective": "multi:softmax",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "max_depth": 4,
        },
    )
    train_step = TrainingStep(
        name="Train",
        estimator=est,
        inputs={
            "train": TrainingInput(
                s3_data=prep_step.properties.ProcessingOutputConfig.Outputs["prepared"].S3Output.S3Uri + "/train.csv",
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=prep_step.properties.ProcessingOutputConfig.Outputs["prepared"].S3Output.S3Uri + "/validation.csv",
                content_type="text/csv",
            ),
        },
    )

    # -------- Step 3: Register (Model Registry) --------
    model_artifact = train_step.properties.ModelArtifacts.S3ModelArtifacts
    reg_model = Model(
        image_uri=xgb_image,
        model_data=model_artifact,
        role=role_arn,
        sagemaker_session=sess,
    )
    create_model_step = CreateModelStep(name="CreateModelFromTrain", model=reg_model)
    register_step = RegisterModel(
        name="RegisterModel",
        model=create_model_step.model,
        model_package_group_name=f"{PIPELINE_NAME}-Group",
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        description="Iris XGBoost model from single training job",
    )

    # -------- Step 4: Batch Transform on test.csv --------
    transformer = Transformer(
        model_name=create_model_step.properties.ModelName,
        instance_type=p_xform_instance,
        instance_count=1,
        output_path=f"s3://{bucket}/batch-output/",
        sagemaker_session=sess,
    )
    transform_step = TransformStep(
        name="BatchTransform",
        transformer=transformer,
        inputs=TransformInput(
            data=prep_step.properties.ProcessingOutputConfig.Outputs["prepared"].S3Output.S3Uri + "/test.csv",
            content_type="text/csv",
            split_type="Line",
        ),
    )

    # -------- Step 5: Real-time Endpoint (create/update) --------
    ep_model = Model(
        image_uri=xgb_image,
        model_data=model_artifact,
        role=role_arn,
        sagemaker_session=sess,
    )
    create_model_for_ep = CreateModelStep(name="CreateModelForEndpoint", model=ep_model)

    # IMPORTANT: use plain dict for production_variants (SDK-version proof)
    ep_cfg_step = EndpointConfigStep(
        name="CreateEndpointConfig",
        endpoint_config_name=f"{PIPELINE_NAME}-cfg-{int(time.time())}",
        production_variants=[{
            "ModelName": create_model_for_ep.properties.ModelName,
            "VariantName": "AllTraffic",
            "InitialInstanceCount": 1,
            "InstanceType": p_ep_instance,
            "InitialVariantWeight": 1.0,
        }],
    )

    ep_step = EndpointStep(
        name="CreateOrUpdateEndpoint",
        endpoint_name=p_endpoint_name,
        endpoint_config_name=ep_cfg_step.properties.EndpointConfigName,
    )

    # -------- Step 6: Invoke endpoint (Processing) --------
    inv = SKLearnProcessor(
        framework_version="1.2-1",
        role=role_arn,
        instance_type=p_proc_instance,
        instance_count=1,
        sagemaker_session=sess,
    )
    invoke_step = ProcessingStep(
        name="RTInvoke",
        processor=inv,
        code="src/rt_invoke.py",
        job_arguments=["--endpoint-name", p_endpoint_name, "--num-rows-test", "3"],
        outputs=[ProcessingOutput(output_name="rt", source="/opt/ml/processing/output")],
        depends_on=[ep_step],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            p_region, p_bucket, p_role,
            p_train_instance, p_proc_instance, p_xform_instance,
            p_ep_instance, p_endpoint_name,
        ],
        steps=[
            prep_step,
            train_step,
            create_model_step,
            register_step,
            transform_step,
            create_model_for_ep,
            ep_cfg_step,
            ep_step,
            invoke_step,
        ],
        sagemaker_session=sess,
    )

# ---------------- CLI ----------------
def upsert(region, bucket, role_arn):
    pipe = create_pipeline(region, bucket, role_arn)
    pipe.upsert(role_arn=role_arn)
    print(f"Upserted pipeline: {pipe.name}")

def run(region, bucket):
    sess = PipelineSession(boto3.Session(region_name=region))
    pipe = Pipeline(name=PIPELINE_NAME, sagemaker_session=sess)
    exe = pipe.start(parameters={})
    print("Started execution:", exe.arn)

def status():
    sm = boto3.client("sagemaker")
    resp = sm.list_pipeline_executions(PipelineName=PIPELINE_NAME, MaxResults=1)
    if not resp.get("PipelineExecutionSummaries"):
        print("No executions yet."); return
    arn = resp["PipelineExecutionSummaries"][0]["PipelineExecutionArn"]
    desc = sm.describe_pipeline_execution(PipelineExecutionArn=arn)
    print(json.dumps({k: desc.get(k) for k in ["PipelineExecutionArn","PipelineExecutionStatus","FailureReason"]}, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upsert","run","status"])
    ap.add_argument("--region", required=True)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--role-arn", required=False)
    a = ap.parse_args()
    if a.cmd == "upsert":
        if not a.role_arn:
            raise SystemExit("--role-arn is required for upsert")
        upsert(a.region, a.bucket, a.role_arn)
    elif a.cmd == "run":
        run(a.region, a.bucket)
    elif a.cmd == "status":
        status()

