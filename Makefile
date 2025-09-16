# ===== SageMaker Pipeline Makefile =====
SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >

.DEFAULT_GOAL := help

# ---- Config ----
AWS_REGION ?= us-east-1
SM_EXECUTION_ROLE_ARN ?= arn:aws:iam::014498620948:role/service-role/AmazonSageMaker-ExecutionRole-20250702T145603
SM_BUCKET ?= sm-w10
ACCURACY_THRESHOLD ?= 0.85

# Registry defaults
MODEL_PKG_GROUP_NAME ?= W10D2-DemoGroup
MODEL_APPROVAL_STATUS ?= PendingManualApproval

# Instances
PROC_INSTANCE ?= ml.t3.medium            # for PROC pipeline
REG_PROC_INSTANCE ?= ml.t3.medium        # for REG (Preprocess/Evaluate)
REG_TRAIN_INSTANCE ?= ml.m5.large        # for REG (Estimator training)

PY ?= python
VENV ?= .venv
ACTIVATE := source $(VENV)/bin/activate

PIPELINE_PY := pipeline/pipeline.py
PIPELINE_NAME ?= W10ProcSklearnPipeline

# ---- Helpers ----
.PHONY: help show.vars venv deps deps.min clean \
        pipeline.upsert pipeline.run \
        registry.ensure registry.list registry.approve registry.reject \
        reg.upsert reg.run reg.status reg.describe

help:
> echo "Targets:"
> echo "  make venv              # Create local Python venv (.venv)"
> echo "  make deps              # Install deps"
> echo "  make pipeline.upsert   # Create/Update PROC pipeline"
> echo "  make pipeline.run      # Start PROC pipeline"
> echo "  make registry.ensure   # Ensure Model Package Group exists"
> echo "  make registry.list     # List model packages"
> echo "  make registry.approve  # Approve latest model package"
> echo "  make registry.reject   # Reject latest model package"
> echo "  make reg.upsert        # Create/Update REG pipeline (configurable instances)"
> echo "  make reg.run           # Start REG pipeline (auto-create if missing)"
> echo "  make reg.status        # Show latest REG pipeline execution status"
> echo "  make reg.describe PIPELINE_EXEC_ARN=<arn>  # Describe an execution by ARN"

show.vars:
> echo "AWS_REGION=$(AWS_REGION)"
> echo "SM_EXECUTION_ROLE_ARN=$(SM_EXECUTION_ROLE_ARN)"
> echo "SM_BUCKET=$(SM_BUCKET)"
> echo "ACCURACY_THRESHOLD=$(ACCURACY_THRESHOLD)"
> echo "MODEL_PKG_GROUP_NAME=$(MODEL_PKG_GROUP_NAME)"
> echo "MODEL_APPROVAL_STATUS=$(MODEL_APPROVAL_STATUS)"
> echo "PROC_INSTANCE=$(PROC_INSTANCE)"
> echo "REG_PROC_INSTANCE=$(REG_PROC_INSTANCE)"
> echo "REG_TRAIN_INSTANCE=$(REG_TRAIN_INSTANCE)"
> echo "PIPELINE_PY=$(PIPELINE_PY)"
> echo "PIPELINE_NAME=$(PIPELINE_NAME)"
> test -f "$(PIPELINE_PY)" && echo "pipeline.py: OK" || (echo "pipeline.py: MISSING ($(PIPELINE_PY))" && exit 1)
> test -d "$(VENV)" && echo "venv: OK ($(VENV))" || echo "venv: MISSING (run 'make venv && make deps')"

venv:
> test -d "$(VENV)" || python3 -m venv "$(VENV)"
> echo "Virtualenv ready: $(VENV)"

deps:
> $(ACTIVATE); \
>  python -m pip install --upgrade pip wheel setuptools || true; \
>  if [ -f requirements.txt ]; then \
>    pip install -r requirements.txt; \
>  else \
>    echo "requirements.txt not found -> installing minimal deps..."; \
>    pip install --upgrade 'sagemaker>=2.150.0,<3' scikit-learn pandas joblib boto3; \
>  fi
> echo "Dependencies installed."

deps.min:
> $(ACTIVATE); \
>  python -m pip install --upgrade pip wheel setuptools; \
>  pip install --upgrade 'sagemaker>=2.150.0,<3' scikit-learn pandas joblib boto3
> echo "Minimal dependencies installed."

clean:
> rm -rf .pytest_cache **/__pycache__ .mypy_cache .ruff_cache
> find . -name "*.pyc" -delete
> echo "Clean complete."

# ---- PROC Pipeline commands ----
pipeline.upsert:
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" proc upsert \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --bucket "$(SM_BUCKET)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)" \
>    --proc-instance "$(PROC_INSTANCE)"

pipeline.run:
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" proc run \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --bucket "$(SM_BUCKET)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)" \
>    --proc-instance "$(PROC_INSTANCE)"

# ---- REGISTRY helpers ----
registry.ensure:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg ensure-group \
>    --region "$(AWS_REGION)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)"

registry.list:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg list-packages \
>    --region "$(AWS_REGION)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)"

registry.approve:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg approve-latest \
>    --region "$(AWS_REGION)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)" \
>    --status Approved \
>    --note "Approved via W10:D2 demo"

registry.reject:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg approve-latest \
>    --region "$(AWS_REGION)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)" \
>    --status Rejected \
>    --note "Rejected via W10:D2 demo"

# ---- REG Pipeline commands ----
reg.upsert: registry.ensure
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg upsert \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)" \
>    --approval "$(MODEL_APPROVAL_STATUS)" \
>    --bucket "$(SM_BUCKET)" \
>    --proc-instance "$(REG_PROC_INSTANCE)" \
>    --train-instance "$(REG_TRAIN_INSTANCE)"

reg.run:
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg run \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --group-name "$(MODEL_PKG_GROUP_NAME)" \
>    --bucket "$(SM_BUCKET)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)" \
>    --approval "$(MODEL_APPROVAL_STATUS)" \
>    --proc-instance "$(REG_PROC_INSTANCE)" \
>    --train-instance "$(REG_TRAIN_INSTANCE)"

reg.status:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg status \
>    --region "$(AWS_REGION)"

reg.describe:
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" reg describe \
>    --region "$(AWS_REGION)" \
>    --arn "$(PIPELINE_EXEC_ARN)"

