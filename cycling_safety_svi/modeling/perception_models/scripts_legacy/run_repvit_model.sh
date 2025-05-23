#!/bin/bash

# Script to train using the timm/repvit_m2_3.dist_450e_in1k model
# Uses feature extraction from the RepVit model to predict perception ratings

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the repository root directory (3 levels up from script)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Set up base directories
BASE_OUTPUT_DIR="${SCRIPT_DIR}/output"
DATA_DIR="${REPO_ROOT}/data"

# Get the machine name and system
MACHINE_NAME=$(hostname)
SYS_OS=$(uname -s)

# Set the output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/repvit_model_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

# Set paths based on the system OS, following the same logic as main.py
if [[ "$SYS_OS" == "Darwin" ]]; then  # macOS
    # Settings for macOS
    DATA_FILE="/Users/sandervancranenburgh/Documents/Repos_and_data/Data/bicycle_project_roos/perceptionratings.csv"
    IMG_PATH="/Users/sandervancranenburgh/Documents/Repos_and_data/Data/bicycle_project_roos/images"
elif [[ "$SYS_OS" == "Linux" ]]; then
    # Settings for Linux systems
    DATA_FILE="${REPO_ROOT}/data/raw/perceptionratings.csv"
    IMG_PATH="/srv/shared/bicycle_project_roos/images_scaled"
    
    # Custom override for specific machines if needed
    if [[ "$MACHINE_NAME" == "cityai" ]]; then
        # Specific settings for the cityai machine
        IMG_PATH="/srv/shared/bicycle_project_roos/images_scaled"
    fi
else
    # Default fallback
    DATA_FILE="${REPO_ROOT}/data/raw/perceptionratings.csv"
    IMG_PATH="${REPO_ROOT}/data/images"
fi

# Path to the RepVit configuration YAML file
CONFIG_FILE="${SCRIPT_DIR}/configs/repvit_config.yaml"
TMP_CONFIG_FILE="${OUTPUT_DIR}/repvit_config_${TIMESTAMP}.yaml"

# Create a copy of the config file and replace the placeholder paths with actual paths
cp "${CONFIG_FILE}" "${TMP_CONFIG_FILE}"

# Replace placeholder paths with actual paths
sed -i.bak "s|__DATA_FILE__|${DATA_FILE}|g" "${TMP_CONFIG_FILE}"
sed -i.bak "s|__IMG_PATH__|${IMG_PATH}|g" "${TMP_CONFIG_FILE}"
sed -i.bak "s|__OUTPUT_DIR__|${OUTPUT_DIR}|g" "${TMP_CONFIG_FILE}"

echo "Using configuration file: ${TMP_CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Data file: ${DATA_FILE}"
echo "Image path: ${IMG_PATH}"

# Run the training script with RepVit model
python3 main.py \
    --config "${TMP_CONFIG_FILE}" \
    --num_categories 3 \
    --batch_size 8 \
    --no_optuna

echo "Training completed. Results saved to ${OUTPUT_DIR}"

# Run prediction script on test images (uncomment and customize if needed)
# MODEL_PATH="${OUTPUT_DIR}/$(ls -t ${OUTPUT_DIR}/*PerceptionModel*.pt | head -n 1)"  # Get the latest model file
# TEST_IMAGES_DIR="${DATA_DIR}/test_images"  # Update with your test image directory
# OUTPUT_CSV="${OUTPUT_DIR}/predictions.csv"
#
# echo "Running predictions on test images..."
# python ${SCRIPT_DIR}/predict.py \
#     --model_path "${MODEL_PATH}" \
#     --image_dir "${TEST_IMAGES_DIR}" \
#     --output_file "${OUTPUT_CSV}" \
#     --model_type "repvit_m2_3" \
#     --num_categories 3 \
#     --use_gpu 