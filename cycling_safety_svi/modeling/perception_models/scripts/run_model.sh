#!/bin/bash

# General script to train any perception model with a specified YAML config file
# Usage: ./scripts/run_model.sh <config_file_name>

# Validate input
if [ $# -lt 1 ]; then
    echo "Error: Missing config file name"
    echo "Usage: $0 <config_file_name>"
    echo "Example: $0 repvit_config.yaml"
    exit 1
fi

CONFIG_FILE_NAME="$1"

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the parent models directory
MODELS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Get the repository root directory (3 levels up from models dir)
REPO_ROOT="$(cd "$MODELS_DIR/../../.." && pwd)"

# Set up base directories
BASE_OUTPUT_DIR="${MODELS_DIR}/output"
DATA_DIR="${REPO_ROOT}/data"

# Get the machine name and system
MACHINE_NAME=$(hostname)
SYS_OS=$(uname -s)

# Extract model name from config file (remove _config.yaml)
MODEL_NAME=$(echo $CONFIG_FILE_NAME | sed 's/_config.yaml//g')

# Set the output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

# Set paths based on the system OS
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

# Path to the configuration YAML file
CONFIG_FILE="${MODELS_DIR}/configs/${CONFIG_FILE_NAME}"
TMP_CONFIG_FILE="${OUTPUT_DIR}/${CONFIG_FILE_NAME%.yaml}_${TIMESTAMP}.yaml"

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: Config file not found: ${CONFIG_FILE}"
    echo "Available config files:"
    ls -1 "${MODELS_DIR}/configs"
    exit 1
fi

# Create a copy of the config file
cp "${CONFIG_FILE}" "${TMP_CONFIG_FILE}"

# Update output_dir in the config file
sed -i.bak "s|output_dir:.*|output_dir: \"${OUTPUT_DIR}\"|g" "${TMP_CONFIG_FILE}"

# Update data_file in the config if it uses placeholder
sed -i.bak "s|__DATA_FILE__|${DATA_FILE}|g" "${TMP_CONFIG_FILE}"
sed -i.bak "s|data_file:.*|data_file: \"${DATA_FILE}\"|g" "${TMP_CONFIG_FILE}"

# Update img_path in the config if it uses placeholder
sed -i.bak "s|__IMG_PATH__|${IMG_PATH}|g" "${TMP_CONFIG_FILE}"
sed -i.bak "s|img_path:.*|img_path: \"${IMG_PATH}\"|g" "${TMP_CONFIG_FILE}"

echo "Using configuration file: ${TMP_CONFIG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Data file: ${DATA_FILE}"
echo "Image path: ${IMG_PATH}"

# Extract batch_size and num_categories from config
BATCH_SIZE=$(grep -E "^\s*batch_size:" ${TMP_CONFIG_FILE} | awk '{print $2}')
NUM_CATEGORIES=$(grep -E "^\s*num_categories:" ${TMP_CONFIG_FILE} | awk '{print $2}')

# Check if optuna is enabled
USE_OPTUNA=$(grep -E "^\s*use_optuna:" ${TMP_CONFIG_FILE} | awk '{print $2}' | tr '[:upper:]' '[:lower:]')

# Set defaults if not found
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=8
fi
if [ -z "$NUM_CATEGORIES" ]; then
    NUM_CATEGORIES=3
fi

# Run the training script
echo "Running training with batch_size=${BATCH_SIZE}, num_categories=${NUM_CATEGORIES}"
cd ${MODELS_DIR}

# Only pass --no_optuna if optuna is disabled in config
if [ "$USE_OPTUNA" = "false" ] || [ "$USE_OPTUNA" = "no" ]; then
    echo "Optuna hyperparameter optimization is disabled"
    python3 main.py \
        --config "${TMP_CONFIG_FILE}" \
        --num_categories ${NUM_CATEGORIES} \
        --no_optuna
else
    echo "Optuna hyperparameter optimization is enabled"
    python3 main.py \
        --config "${TMP_CONFIG_FILE}" \
        --num_categories ${NUM_CATEGORIES}
fi

echo "Training completed. Results saved to ${OUTPUT_DIR}"

# Optional: Get the trained model path (uncomment and modify as needed)
MODEL_PATH="${OUTPUT_DIR}/$(ls -t ${OUTPUT_DIR}/*PerceptionModel*.pt 2>/dev/null | head -n 1)"
if [ -f "${MODEL_PATH}" ]; then
    echo "Model saved at: ${MODEL_PATH}"
    
    # Uncomment the following to run prediction automatically after training
    # TEST_IMAGES_DIR="${DATA_DIR}/test_images"
    # OUTPUT_CSV="${OUTPUT_DIR}/predictions.csv"
    # echo "Running predictions on test images..."
    # python predict.py \
    #     --model_path "${MODEL_PATH}" \
    #     --image_dir "${TEST_IMAGES_DIR}" \
    #     --output_file "${OUTPUT_CSV}" \
    #     --model_type "${MODEL_NAME}" \
    #     --num_categories ${NUM_CATEGORIES} \
    #     --use_gpu
fi 