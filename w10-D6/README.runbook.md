# Mini SageMaker: HPO → Registry → Batch → Real-time (deploy → invoke → rollback)

## Prereqs
- AWS CLI configured with an account that can use SageMaker
- IAM role for SageMaker execution (attach AmazonSageMakerFullAccess for a quick demo)
- Python 3.10+ on your machine

## Quickstart

```bash
make setup
make bucket.create REGION=us-east-1
make pipeline.upsert REGION=us-east-1 ROLE_ARN=arn:aws:iam::<ACCOUNT_ID>:role/<ROLE> BUCKET=<YOUR_UNIQUE_BUCKET>
make pipeline.run REGION=us-east-1 BUCKET=<YOUR_UNIQUE_BUCKET>
# (optionally watch)
make pipeline.status

