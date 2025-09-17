#!/usr/bin/env bash
set -eo pipefail

# always run from repo root
cd "$(dirname "$0")/.."

# create venv if needed
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# activate
# shellcheck disable=SC1091
source .venv/bin/activate

# modern pip
python -m pip install --upgrade pip setuptools wheel

# install deps (retry once in case of pypi hiccups)
pip install -r requirements.txt || { echo "Retrying pip install..."; pip install -r requirements.txt; }

# quick sanity check
python - <<'PY'
import sys
print("Python:", sys.version)
import boto3, sagemaker
print("boto3:", boto3.__version__)
print("sagemaker:", sagemaker.__version__)
PY

echo "✅ Environment ready. To use: source .venv/bin/activate"

