#!/usr/bin/env bash
# Rebuild everything from scratch: data -> engine -> evaluation
set -e
cd "$(dirname "$0")"
python src/generate_data.py
python src/run_engine.py
python src/evaluate.py
