# src/rt_deploy_invoke_rollback.py
import argparse
import json
import os
import time
import boto3
from botocore.exceptions import ClientError

def wait_endpoint_in_service(sm_client, endpoint_name):
    while True:
        desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc["EndpointStatus"]
        if status in ("InService", "Failed"):
            return status, desc
        time.sleep(10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-artifact-s3", required=True)
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--num-rows-test", type=int, default=3)
    args = parser.parse_args()

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    sm = boto3.client("sagemaker", region_name=region)
    rt = boto3.client("sagemaker-runtime", region_name=region)

    # Remember prior endpoint config for rollback
    prior_config = None
    try:
        ep_desc = sm.describe_endpoint(EndpointName=args.endpoint_name)
        prior_config = ep_desc.get("EndpointConfigName")
    except ClientError:
        prior_config = None

    # Create/Update: model -> endpoint-config -> endpoint
    # Idempotent model create
    try:
        sm.create_model(
            ModelName=args.model-name,
            PrimaryContainer={
                "Image": args.image_uri,
                "ModelDataUrl": args.model_artifact_s3,
                "Mode": "SingleModel"
            },
            ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ValidationException":
            raise

    ec_name = f"{args.endpoint_name}-cfg-{int(time.time())}"
    sm.create_endpoint_config(
        EndpointConfigName=ec_name,
        ProductionVariants=[{
            "ModelName": args.model_name,
            "VariantName": "AllTraffic",
            "InitialInstanceCount": 1,
            "InstanceType": args.instance_type,
            "InitialVariantWeight": 1.0
        }]
    )

    try:
        # If endpoint exists -> update, else -> create
        sm.describe_endpoint(EndpointName=args.endpoint_name)
        sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=ec_name)
    except ClientError:
        sm.create_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=ec_name)

    status, desc = wait_endpoint_in_service(sm, args.endpoint_name)
    if status != "InService":
        # Rollback if failed and prior exists
        if prior_config:
            sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=prior_config)
        raise RuntimeError(f"Endpoint failed to go InService: {desc}")

    # --- Smoke test: send a few CSV rows ---
    # Simple 4-feature iris row(s)
    payload = "\n".join([
        "5.1,3.5,1.4,0.2",
        "6.0,2.2,5.0,1.5",
        "6.7,3.0,5.2,2.3"
    ])[: args.num_rows_test * 100]  # cap, just in case

    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="text/csv",
        Body=payload.encode("utf-8")
    )
    body = resp["Body"].read().decode("utf-8")

    # Very light sanity check
    if not body.strip():
        # rollback if possible
        if prior_config:
            sm.update_endpoint(EndpointName=args.endpoint_name, EndpointConfigName=prior_config)
        raise RuntimeError("Empty response from endpoint; rolled back to prior config.")

    # Save response to processing output
    out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rt_invoke_response.txt"), "w") as f:
        f.write(body)

    print("OK: Real-time endpoint deployed & invoked. Response saved.")

if __name__ == "__main__":
    main()

