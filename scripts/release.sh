#!/usr/bin/env bash
export MLFLOW_MIGRATION=${1:-0.0.0.dev0}
python -m build
