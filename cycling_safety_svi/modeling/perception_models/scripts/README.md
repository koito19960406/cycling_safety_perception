# Perception Model Training Scripts

This directory contains scripts to simplify running different perception model training configurations.

## Available Scripts

- `run_model.sh`: General script that can run any model configuration
- `run_repvit.sh`: Runs RepVit model training
- `run_convnextv2.sh`: Runs ConvNextV2 model training
- `run_augmented.sh`: Runs model with augmentation (three_category_config_v2)
- `run_zscore.sh`: Runs model with z-score standardization enabled (uses three_category_config.yaml as base)
- `run_zscore_config.sh`: Runs model with the dedicated z-score configuration (zscore_config.yaml)

## Usage

### Running a specific model

To run one of the predefined models, simply execute its script:

```bash
# Run RepVit model
./run_repvit.sh

# Run ConvNextV2 model
./run_convnextv2.sh

# Run augmented model
./run_augmented.sh

# Run with z-score standardization
./run_zscore.sh

# Run with dedicated z-score config
./run_zscore_config.sh
```

### Running with a custom configuration

To run with any specific YAML configuration file from the `configs` directory:

```bash
# Syntax: ./run_model.sh <config_file_name>
./run_model.sh repvit_config.yaml
./run_model.sh convnextv2_config.yaml
./run_model.sh three_category_config_v2.yaml
./run_model.sh zscore_config.yaml
```

### Z-Score Standardization

The `--dataset.use_z_score true` parameter can be added to any configuration to enable z-score standardization.
This normalizes each image individually by:
- Calculating the mean and standard deviation of each image
- Transforming pixels to have zero mean and unit variance: `(x - mean) / std`

Z-score standardization may improve model performance on images with varying lighting conditions
or when images come from multiple sources with different characteristics.

## Features

- Automatically sets up paths based on your system (macOS/Linux)
- Creates timestamped output directories
- Creates a copy of the config file with updated paths
- Automatically detects and uses parameters from the config file

## Adding a New Model

1. Create a new config file in `../configs/` (e.g., `new_model_config.yaml`)
2. Create a new script for easy access:

```bash
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
chmod +x "${SCRIPT_DIR}/run_model.sh"
"${SCRIPT_DIR}/run_model.sh" "new_model_config.yaml"
```

3. Make it executable: `chmod +x run_new_model.sh` 