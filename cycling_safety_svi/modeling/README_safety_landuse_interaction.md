# Safety-Landuse Interaction Model

This script extends the best choice model with **safety × landuse interaction effects**. It adds interaction terms between safety scores and land use categories (buildenvironment) to understand how safety perceptions vary across different built environments.

## Features

- **Land Use Categories**: Uses 5 categories from main_design.csv:
  - `Wijkontslu` (Shopping centers/intersections)
  - `Industriet` (Industrial areas)
  - `Woongebied` (Residential areas) - **Reference category**
  - `Hoofdweg` (Main roads)
  - `Recreatie` (Recreation areas)

- **Interaction Effects**: Creates safety × landuse interaction terms to test if safety effects differ by built environment
- **Feature Extraction**: Automatically extracts and uses the same segmentation features from the original trained model
- **Train/Test Evaluation**: Fits model on training data and evaluates on test data
- **Complete Analysis**: Provides parameter estimates, significance tests, and interpretation
- **Multiple Outputs**: Saves .pickle, .tex, .html, CSV files, and summary reports with both training and test metrics

## Usage

### Basic Usage
```bash
python safety_landuse_interaction_model.py --model_path path/to/best_model.pickle
```

### Advanced Usage
```bash
python safety_landuse_interaction_model.py \
    --model_path reports/models/choice_20241207_123456/best_model_top_15_with_safety.pickle \
    --main_design_path data/raw/main_design.csv \
    --output_dir reports/models/interaction \
    --include_seg_features 10
```

### Arguments

- `--model_path` (optional): Path to the trained model pickle file from choice_model_benchmark.py (default: uses predefined path)
- `--main_design_path` (optional): Path to main_design.csv with land use data (default: `data/raw/main_design.csv`)
- `--output_dir` (optional): Output directory for results (default: `reports/models/interaction`)

## Model Specification

The interaction model extends the base choice model:

### Base Model:
```
V = β_TL × TL/3 + β_TT × TT/10 + β_SAFETY × safety_score + β_SEG × segmentation_features
```

### Interaction Model:
```
V = β_TL × TL/3 + β_TT × TT/10 + β_SAFETY × safety_score + β_SEG × segmentation_features
    + β_LANDUSE_CAT × landuse_dummy
    + β_SAFETY_LANDUSE_CAT × (safety_score × landuse_dummy)
```

Where:
- **Main Effects**: Safety score has a base effect (β_SAFETY) 
- **Landuse Effects**: Each landuse category has a main effect (β_LANDUSE_CAT)
- **Interaction Effects**: Safety effect varies by landuse (β_SAFETY_LANDUSE_CAT)

**Total Safety Effect** for each category = β_SAFETY + β_SAFETY_LANDUSE_CAT

## Outputs

The script creates a timestamped folder with:

### Model Files:
- `safety_landuse_interaction.pickle` - Biogeme model results
- `safety_landuse_interaction.html` - Formatted model results
- `safety_landuse_interaction.tex` - LaTeX formatted results

### Analysis Files:
- `parameter_estimates.csv` - All parameter estimates with significance
- `interaction_effects_analysis.csv` - Detailed interaction effects analysis
- `train_metrics.csv` - Training dataset performance metrics
- `test_metrics.csv` - Test dataset performance metrics  
- `combined_metrics.csv` - Combined train/test metrics
- `model_summary.json` - Complete model information and metrics
- `interaction_model_report.txt` - Human-readable summary report

### Example Output:
```
SAFETY EFFECTS BY LAND USE:
  Woongebied (Reference):
    Total safety effect: 0.450000 (Reference category)
  
  Industriet:
    Total safety effect: 0.520000
    Interaction coefficient: 0.070000 **
    P-value: 0.012000 **
  
  Hoofdweg:
    Total safety effect: 0.380000
    Interaction coefficient: -0.070000 *
    P-value: 0.035000 *
```

## Interpretation

- **Positive interaction**: Safety matters MORE in that built environment
- **Negative interaction**: Safety matters LESS in that built environment  
- **Significance levels**: *** p<0.001, ** p<0.01, * p<0.05

## Example Workflow

1. **Run choice model benchmark** to get best model:
   ```bash
   python choice_model_benchmark.py
   # Find best model pickle file in reports/models/choice_TIMESTAMP/
   ```

2. **Run interaction analysis**:
   ```bash
   python safety_landuse_interaction_model.py \
       --model_path reports/models/choice_20241207_123456/best_model_top_15_with_safety.pickle
   ```

3. **Check results** in `reports/models/interaction/safety_landuse_TIMESTAMP/`

## Data Requirements

- **Choice data**: `data/raw/cv_dcm.csv` 
- **Safety scores**: `data/processed/predicted_danish/cycling_safety_scores.csv`
- **Land use data**: `data/raw/main_design.csv` (with `alt1_buildenvironment`, `alt2_buildenvironment`)
- **Segmentation data**: `data/processed/segmentation_results/pixel_ratios.csv` (optional)

## Notes

- Uses 'Woongebied' (residential) as reference category
- Automatically handles missing values and unknown categories
- Includes top segmentation features based on variance
- Creates dummy variables for all non-reference categories
- Estimates both main effects and interaction effects simultaneously 