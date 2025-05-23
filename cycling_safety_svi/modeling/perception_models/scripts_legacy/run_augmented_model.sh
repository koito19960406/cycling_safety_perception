#!/bin/bash
set -e

# Change to script directory
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Default paths
DEFAULT_DATA_FILE="/home/kiito/cycling_safety_svi/data/raw/perceptionratings.csv"
DEFAULT_IMG_PATH="/srv/shared/bicycle_project_roos/images_scaled"
DEFAULT_OUTPUT_DIR="./output"

# Determine paths for data file, image directory, and output directory
DATA_FILE=${1:-$DEFAULT_DATA_FILE}
IMG_PATH=${2:-$DEFAULT_IMG_PATH}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}

# Check if default data file exists, if not print a warning
if [ ! -f "$DEFAULT_DATA_FILE" ]; then
    echo "Warning: Default data file not found at $DEFAULT_DATA_FILE"
    echo "You may need to specify the data file path manually using --data_file."
fi

# Print information
echo "Starting model training with 3 categories and enhanced augmentation..."
echo "Using data file: $DATA_FILE"
echo "Using image path: $IMG_PATH"
echo "Using output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script with 3 categories and best parameters
python main.py \
    --config configs/three_category_config.yaml \
    --no_optuna \
    --data_file "$DATA_FILE" \
    --img_path "$IMG_PATH" \
    --output_dir "$OUTPUT_DIR"

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Training complete! Check the output directory for results."
else
    echo "Error: Training failed."
    echo "You can try running with explicit paths:"
    echo "python main.py --config configs/three_category_config.yaml --no_optuna --data_file /path/to/perceptionratings.csv --img_path /path/to/images --output_dir ./output"
    exit 1
fi 