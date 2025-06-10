# Discrete Choice Model with Perception and Segmentation Features

This module implements discrete choice models using Biogeme to analyze how various features influence cycling route choices. It tests different feature combinations as specified in the analysis outline:

- Base model (only traffic lights and travel time)
- Perception variables only (traffic safety, social safety, beauty)
- Segmentation variables only (urban elements)
- Combination of perception and segmentation variables
- Deep image features from the CVDCM model
- Full model with all feature types

## Requirements

- Python 3.7+
- PyTorch
- Biogeme
- pandas
- numpy
- scikit-image
- transformers

Install the required packages:

```bash
pip install biogeme pandas numpy torch torchvision scikit-image transformers
```

## Data Structure

The model expects the following data:

1. **Choice data**: CSV file with choice experiment data including:
   - IMG1, IMG2: Image filenames
   - TL1, TL2: Traffic light counts for each alternative
   - TT1, TT2: Travel times for each alternative
   - CHOICE: Choice value (1 for first alternative, 2 for second)
   - ID: Respondent ID

2. **Perception data** (optional): CSV file with perception ratings:
   - imageid: Image identifier
   - traffic_safety: Traffic safety rating
   - social_safety: Social safety rating 
   - beauty: Beauty perception rating

3. **Segmentation data** (optional): CSV file with segmentation statistics:
   - image_id: Image identifier
   - Various segmentation class percentages (e.g., road, sidewalk, building, etc.)

## Usage

```python
# Example usage
from discrete_choice_model import run_choice_models

results, comparison = run_choice_models(
    data_path='path/to/choice_data.csv',
    img_dir='path/to/images_directory',
    model_path='path/to/pretrained_cvdcm_model',
    perception_data_path='path/to/perception_data.csv',
    segmentation_data_path='path/to/segmentation_data.csv',
    output_dir='results'
)
```

## Command Line Usage

The script can also be run from the command line:

```bash
python discrete_choice_model.py \
    --data_path path/to/choice_data.csv \
    --img_dir path/to/images_directory \
    --model_path path/to/pretrained_cvdcm_model \
    --perception_data path/to/perception_data.csv \
    --segmentation_data path/to/segmentation_data.csv \
    --output_dir results
```

## Model Types

1. **Base Model**: Uses only traffic lights and travel time attributes
2. **Perception Only**: Adds perception variables (traffic safety, social safety, beauty)
3. **Segmentation Only**: Adds segmentation variables (urban elements)
4. **Perception and Segmentation**: Combines both perception and segmentation variables
5. **Image Features**: Uses deep image features from the CVDCM model
6. **Full Model**: Combines all available features

## Output

The script generates:

- HTML files with detailed model results for each model type
- A CSV file comparing all models (log likelihood, rho-square, number of parameters)
- Pickle files with full model results for further analysis

## Notes

- If perception or segmentation data is not provided, the corresponding models will be skipped
- The image feature models use the top 10 feature dimensions by default
- All models estimate coefficients for traffic lights and travel time variables 