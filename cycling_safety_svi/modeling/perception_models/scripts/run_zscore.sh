#!/bin/bash

# Run perception model with z-score standardization
# This uses the three_category_config as the base configuration

cd "$(dirname "$0")"/..
python main.py \
  --config configs/three_category_config.yaml \
  --dataset.use_z_score true \
  --output_suffix "_zscore" \
  "$@" 