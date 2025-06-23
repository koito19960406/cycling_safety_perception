#!/bin/bash

# Script to run Grad-CAM visualizations for cycling safety model
# This will generate attention heatmaps showing what the model focuses on

echo "Running Grad-CAM visualizations for cycling safety model..."

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_gradcam_visualizations.py"

# Default parameters (modify as needed)
IMAGES_DIR="/srv/shared/bicycle_project_roos/images_scaled"
MODEL_PATH="cycling_safety_svi/cycling_safety_subjective_learning_pairwise/models/vgg_syn+ber.pt"
OUTPUT_DIR="/home/kiito/cycling_safety_svi/data/processed"
BACKBONE="vgg"
MODEL_TYPE="rsscnn"
BATCH_SIZE=4
MAX_IMAGES=50  # Limit for testing

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Run the Grad-CAM visualization
python3 "$PYTHON_SCRIPT" \
    --images_dir "$IMAGES_DIR" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --backbone "$BACKBONE" \
    --model_type "$MODEL_TYPE" \
    --batch_size "$BATCH_SIZE" \
    --max_images "$MAX_IMAGES" \
    --device auto

echo "Grad-CAM visualization complete!"
echo "Check the output directory: $OUTPUT_DIR/gradcam_visualizations/"
echo "  - heatmaps/: Raw attention heatmaps"
echo "  - overlays/: Original images with attention overlays (50% alpha blend)"
echo "  - gradcam_results.csv: Summary statistics"