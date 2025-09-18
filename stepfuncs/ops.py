import os, time, json, boto3

REGION = os.environ["AWS_REGION"]
SFN_NAME = os.environ.get("SFN_NAME", "w10d1-sm-orchestrator")

# Pull pipeline params from env (falls back to sane defaults)
params = {
  "AccuracyThreshold": float(os.environ.get("ACCURACY_THRESHOLD", "0.85")),
  "ModelPackageGroupName": os.environ.get("MODEL_PKG_GROUP_NAME", "w10d-sm-pg"),
  "ModelApprovalStatus": os.environ.get("MODEL_APPROVAL_STATUS", "PendingManualApproval"),
  "BatchInstanceType": os.environ.get("BATCH_INSTANCE_TYPE", "ml.m5.large"),
  "BatchOutputPrefix": os.environ.get("BATCH_OUTPUT_PREFIX", "")
}

session = boto3.Session(region_name=REGION)
sfn = session.client("stepfunctions")

def get_state_machine_arn():
    paginator = sfn.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page["stateMachines"]:
            if sm["name"] == SFN_NAME:
                return sm["stateMachineArn"]
    raise RuntimeError(f"State machine {SFN_NAME} not found. Run sfn.deploy first.")

def start_and_watch():
    arn = get_state_machine_arn()
    name = f"run-{int(time.time())}"
    resp = sfn.start_execution(
        stateMachineArn=arn,
        name=name,
        input=json.dumps({"params": params})
    )
    exec_arn = resp["executionArn"]
    print("Started execution:", exec_arn)
    status="RUNNING"
    while status == "RUNNING":
        time.sleep(10)
        d = sfn.describe_execution(executionArn=exec_arn)
        status = d["status"]
        print("Status:", status)
    print("Final:", status)
    if "output" in d:
        print("Output:", d["output"])

if __name__ == "__main__":
    start_and_watch()

