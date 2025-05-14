#!/bin/bash

# Train perception model using 3-category configuration with best parameters

# Change to the script directory
cd "$(dirname "$0")"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found in the current directory."
    echo "Current directory: $(pwd)"
    echo "Make sure you're running this script from the perception_models directory."
    exit 1
fi

# Install or update requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create output directory if it doesn't exist
mkdir -p output

# Check if data directory exists
DATA_PATH="../../data/raw/perceptionratings.csv"
if [ ! -f "$DATA_PATH" ]; then
    echo "Warning: Default data file not found at $DATA_PATH"
    echo "You may need to specify the data file path manually using --data_file."
fi

# Run the model with 3-category config and best parameters
echo "Starting model training with 3 categories using best parameters..."
python main.py --config configs/three_category_config.yaml --no_optuna "$@"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    echo "You can try running with explicit paths:"
    echo "python main.py --config configs/three_category_config.yaml --no_optuna --data_file /path/to/perceptionratings.csv --img_path /path/to/images --output_dir ./output"
    exit 1
fi

echo "Training complete! Check the output directory for results." 