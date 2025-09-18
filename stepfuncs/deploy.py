# stepfuncs/deploy.py
import json
import os
import sys
import boto3
from botocore.exceptions import ClientError

STATE_MACHINE_NAME = "W10-WF-StartAndWatchSageMakerPipeline"

def env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        print(f"Environment variable {name} is required.", file=sys.stderr)
        sys.exit(1)
    return v

def ensure_log_group(logs, name, region, account_id):
    # Create if missing
    try:
        logs.create_log_group(logGroupName=name)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
            raise
    # 14-day retention is sensible
    try:
        logs.put_retention_policy(logGroupName=name, retentionInDays=14)
    except ClientError:
        pass

    # Get ARN
    resp = logs.describe_log_groups(logGroupNamePrefix=name, limit=1)
    for lg in resp.get("logGroups", []):
        if lg["logGroupName"] == name:
            # API returns arn in most regions; build if missing
            return lg.get(
                "arn",
                f"arn:aws:logs:{region}:{account_id}:log-group:{name}"
            )
    return f"arn:aws:logs:{region}:{account_id}:log-group:{name}"

def build_definition(pipeline_name: str):
    """
    ASL using AWS SDK integrations:
      - StartPipelineExecution (with ClientRequestToken)
      - Poll DescribePipelineExecution until terminal
    NOTE: Fail state must use literal strings for Error/Cause.
    """
    return {
        "Comment": f"Start and watch SageMaker Pipeline {pipeline_name}",
        "StartAt": "StartPipeline",
        "States": {
            "StartPipeline": {
                "Type": "Task",
                "Resource": "arn:aws:states:::aws-sdk:sagemaker:startPipelineExecution",
                "Parameters": {
                    "PipelineName": pipeline_name,
                    # Required for idempotency in SFN SDK integrations
                    "ClientRequestToken.$": "$$.Execution.Id"
                },
                "ResultPath": "$.start",
                "Next": "WaitBeforePoll"
            },
            "WaitBeforePoll": {
                "Type": "Wait",
                "Seconds": 20,
                "Next": "Describe"
            },
            "Describe": {
                "Type": "Task",
                "Resource": "arn:aws:states:::aws-sdk:sagemaker:describePipelineExecution",
                "Parameters": {
                    "PipelineExecutionArn.$": "$.start.PipelineExecutionArn"
                },
                "ResultPath": "$.desc",
                "Next": "CheckStatus"
            },
            "CheckStatus": {
                "Type": "Choice",
                "Choices": [
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "Succeeded", "Next": "Success"},
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "Failed",    "Next": "Failure"},
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "Stopped",   "Next": "Failure"},
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "Stopping",  "Next": "WaitBeforePoll"},
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "InProgress","Next": "WaitBeforePoll"},
                    {"Variable": "$.desc.PipelineExecutionStatus", "StringEquals": "Executing", "Next": "WaitBeforePoll"}
                ],
                "Default": "WaitBeforePoll"
            },
            "Success": { "Type": "Succeed" },
            # Fail state cannot use "Cause.$" – only literal strings are allowed.
            # Details remain available in execution output under $.desc.*
            "Failure": {
                "Type": "Fail",
                "Error": "SageMakerPipelineFailed",
                "Cause": "See $.desc.FailureReason and CloudWatch Logs for details."
            }
        }
    }

def upsert_state_machine(sfn, role_arn, name, definition, logging_arn):
    kwargs = {
        "name": name,
        "definition": json.dumps(definition),
        "roleArn": role_arn,
        "loggingConfiguration": {
            "level": "ALL",
            "includeExecutionData": True,
            "destinations": [{"cloudWatchLogsLogGroup": {"logGroupArn": logging_arn}}],
        },
        "tracingConfiguration": {"enabled": False},
        "type": "STANDARD",
    }
    try:
        resp = sfn.create_state_machine(**kwargs)
        return resp["stateMachineArn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "StateMachineAlreadyExists":
            raise
        # update
        upd = sfn.update_state_machine(
            stateMachineArn=f"arn:aws:states:{sfn.meta.region_name}:{sts.get_caller_identity()['Account']}:stateMachine:{name}",
            definition=kwargs["definition"],
            loggingConfiguration=kwargs["loggingConfiguration"],
            roleArn=role_arn,
            tracingConfiguration=kwargs["tracingConfiguration"],
        )
        # return the ARN in the account/region
        return f"arn:aws:states:{sfn.meta.region_name}:{sts.get_caller_identity()['Account']}:stateMachine:{name}"

# ---------- main ----------
session = boto3.Session(region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
region = session.region_name
sts = session.client("sts")
account_id = sts.get_caller_identity()["Account"]
logs = session.client("logs")
sfn = session.client("stepfunctions")

PIPELINE_NAME = env("PIPELINE_NAME", "w10d1-hello-sm-pipeline")
ROLE_ARN      = env("SFN_ROLE_ARN", required=True)
LOG_GROUP     = env("SFN_LOG_GROUP", "/aws/states/" + STATE_MACHINE_NAME)

print(f"Region: {region}")
print(f"Pipeline: {PIPELINE_NAME}")
print(f"State Machine: {STATE_MACHINE_NAME}")
print(f"Role: {ROLE_ARN}")
print(f"Log group: {LOG_GROUP}")

log_arn = ensure_log_group(logs, LOG_GROUP, region, account_id)
definition = build_definition(PIPELINE_NAME)
arn = upsert_state_machine(sfn, ROLE_ARN, STATE_MACHINE_NAME, definition, log_arn)
print("State Machine ARN:", arn)

