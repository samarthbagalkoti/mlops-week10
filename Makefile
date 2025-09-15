# ===== SageMaker Pipeline Makefile =====
# Works with: python pipeline/pipeline.py upsert|run
# Avoids TAB issues by using a custom recipe prefix.
SHELL := /bin/bash
.ONESHELL:
.RECIPEPREFIX := >

.DEFAULT_GOAL := help

# ---- Config (override via env or CLI: make pipeline.upsert AWS_REGION=ap-south-1) ----
AWS_REGION ?= us-east-1
SM_EXECUTION_ROLE_ARN ?= arn:aws:iam::014498620948:role/service-role/AmazonSageMaker-ExecutionRole-20250702T145603
SM_BUCKET ?= sm-w10
ACCURACY_THRESHOLD ?= 0.85

PY ?= python
VENV ?= .venv
ACTIVATE := source $(VENV)/bin/activate

PIPELINE_PY := pipeline/pipeline.py
PIPELINE_NAME ?= W10ProcSklearnPipeline

# ---- Helpers ----
.PHONY: help show.vars venv deps deps.min clean

help:
> echo "Targets:"
> echo "  make venv            # Create local Python venv (.venv)"
> echo "  make deps            # Install deps from requirements.txt (or fallback to deps.min)"
> echo "  make deps.min        # Minimal deps (sagemaker, sklearn, pandas, joblib, boto3)"
> echo "  make pipeline.upsert # Create/Update SageMaker Pipeline"
> echo "  make pipeline.run    # Start a pipeline execution"
> echo "  make show.vars       # Print current variables"
> echo "  make clean           # Remove caches/temp files"

show.vars:
> echo "AWS_REGION=$(AWS_REGION)"
> echo "SM_EXECUTION_ROLE_ARN=$(SM_EXECUTION_ROLE_ARN)"
> echo "SM_BUCKET=$(SM_BUCKET)"
> echo "ACCURACY_THRESHOLD=$(ACCURACY_THRESHOLD)"
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

# ---- Pipeline commands ----
.PHONY: pipeline.upsert pipeline.run

pipeline.upsert:
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" upsert \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --bucket "$(SM_BUCKET)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)"

pipeline.run:
> test -f "$(PIPELINE_PY)" || (echo "ERROR: $(PIPELINE_PY) not found." && exit 1)
> $(ACTIVATE); \
>  $(PY) "$(PIPELINE_PY)" run \
>    --region "$(AWS_REGION)" \
>    --role-arn "$(SM_EXECUTION_ROLE_ARN)" \
>    --bucket "$(SM_BUCKET)" \
>    --accuracy-threshold "$(ACCURACY_THRESHOLD)"

