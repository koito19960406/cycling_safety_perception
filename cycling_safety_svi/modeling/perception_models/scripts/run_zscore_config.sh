#!/bin/bash

# Run perception model with z-score standardization config
# This uses the zscore_config.yaml which is specifically optimized for z-score standardization

cd "$(dirname "$0")"/..
python main.py \
  --config configs/zscore_config.yaml \
  "$@" 