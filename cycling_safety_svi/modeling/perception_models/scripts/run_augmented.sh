#!/bin/bash

# Simple script to train the Augmented model
# Sets up environment and calls the general run_model.sh script

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the general script with three_category_config_v2 (augmented) config
CONFIG_FILE="three_category_config_v2.yaml"

# Make the script executable if needed
chmod +x "${SCRIPT_DIR}/run_model.sh"

# Run the general script with this config
"${SCRIPT_DIR}/run_model.sh" "${CONFIG_FILE}" 