Purpose: Hello SageMaker Pipeline with metric gate.

How to run: the four make targets above.

Where to see logs: CloudWatch → SageMaker jobs.

Artifacts: S3 paths printed in step logs; pipeline executions in SageMaker Studio / Console.

Next (W10:D2–D3): add Model Registry + Batch transform + HPO + Step Functions trigger.


make reg.upsert  REG_PROC_INSTANCE=ml.t3.medium REG_TRAIN_INSTANCE=ml.m5.large ACCURACY_THRESHOLD=0.80
make reg.run     REG_PROC_INSTANCE=ml.t3.medium REG_TRAIN_INSTANCE=ml.m5.large ACCURACY_THRESHOLD=0.80
make reg.status

