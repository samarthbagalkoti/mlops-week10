import argparse
import os
import boto3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint-name", required=True)
    ap.add_argument("--num-rows-test", type=int, default=3)
    args = ap.parse_args()

    rows = [
        "5.1,3.5,1.4,0.2",
        "6.0,2.2,5.0,1.5",
        "6.7,3.0,5.2,2.3",
    ][: args.num_rows_test]
    payload = "\n".join(rows)

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    rt = boto3.client("sagemaker-runtime", region_name=region)
    resp = rt.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="text/csv",
        Body=payload.encode("utf-8"),
    )
    body = resp["Body"].read().decode("utf-8").strip()

    out_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rt_invoke_response.txt"), "w") as f:
        f.write(body + "\n")

    print("Invoke OK. Saved to rt_invoke_response.txt")

if __name__ == "__main__":
    main()

