PY?=python
REGION?=us-east-1
BUCKET?=sm-w10
ROLE_ARN?=arn:aws:iam::014498620948:role/service-role/AmazonSageMaker-ExecutionRole-20250702T145603

ENVVARS=AWS_DEFAULT_REGION=$(REGION) SAGEMAKER_REGION=$(REGION)

.PHONY: setup
setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

.PHONY: bucket.create
bucket.create:
	aws s3 mb s3://$(BUCKET) --region $(REGION) || true

.PHONY: pipeline.upsert
pipeline.upsert:
	. .venv/bin/activate; $(ENVVARS) $(PY) pipeline/pipeline.py upsert \
		--region $(REGION) --bucket $(BUCKET) --role-arn $(ROLE_ARN)

.PHONY: pipeline.run
pipeline.run:
	. .venv/bin/activate; $(ENVVARS) $(PY) pipeline/pipeline.py run \
		--region $(REGION) --bucket $(BUCKET)

.PHONY: pipeline.status
pipeline.status:
	. .venv/bin/activate; $(ENVVARS) $(PY) pipeline/pipeline.py status

.PHONY: clean
clean:
	rm -rf .venv __pycache__ .pytest_cache

