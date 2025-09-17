.PHONY: setup
setup:
	@./scripts/bootstrap.sh

.PHONY: venv
venv:
	@test -d .venv || python3 -m venv .venv
	@. .venv/bin/activate; python -m pip install --upgrade pip

.PHONY: run-proc-upsert
run-proc-upsert:
	@test -f "pipeline/pipeline.py" || (echo "ERROR: pipeline/pipeline.py not found." && exit 1)
	@. .venv/bin/activate; \
	  python pipeline/pipeline.py proc upsert \
	    --region "us-east-1" \
	    --role-arn "arn:aws:iam::014498620948:role/service-role/AmazonSageMaker-ExecutionRole-20250702T145603" \
	    --bucket "sm-w10" \
	    --accuracy-threshold "0.85" \
	    --proc-instance "ml.t3.medium"

.PHONY: run-proc
run-proc:
	@. .venv/bin/activate; \
	  python pipeline/pipeline.py proc run \
	    --region "us-east-1" \
	    --role-arn "arn:aws:iam::014498620948:role/service-role/AmazonSageMaker-ExecutionRole-20250702T145603" \
	    --bucket "sm-w10" \
	    --accuracy-threshold "0.85" \
	    --proc-instance "ml.t3.medium"

