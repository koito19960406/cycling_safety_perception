# Perceived Traffic Safety's Impact on Cycling Route Choice with Street View Images

Replication code for the paper submitted to *Transportation*.

This study integrates computer vision-derived safety perception scores into a stated preference framework to estimate cyclists' willingness-to-pay for perceived safety improvements and analyse heterogeneity across demographic groups.

## Data

The analysis uses data from:

- **Stated choice experiment and street-level images**: Collected by [Terra (2024)](https://doi.org/10.XXX). Contact the original authors for access.
- **Safety perception model**: Pre-trained model by [Costa et al. (2025)](https://doi.org/10.XXX), included as a git submodule (`cycling_safety_svi/cycling_safety_subjective_learning_pairwise/`).
- **Survey responses**: SQLite database (`data/raw/database_2024_10_07_135133.db`).

Place data files in `data/raw/` and `data/processed/` as described in the scripts.

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/koito19960406/cycling_safety_perception.git
cd cycling_safety_perception

# Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync
```

## Replication Steps

Run the scripts in this order:

### Step 1: Apply safety perception model to images

```bash
python cycling_safety_svi/1_apply_safety_model.py
```

Generates perceived safety scores for each street-level image using the pre-trained model by Costa et al. (2025). Output: `data/processed/predicted_danish/cycling_safety_scores.csv`.

### Step 2: Semantic segmentation

Pixel-level segmentation of images using Mask2Former pre-trained on Mapillary Vistas. Output: `data/processed/segmentation_results/pixel_ratios.csv`.

See `cycling_safety_svi/README_cycling_safety_setup.md` for details.

### Step 3: Estimate choice models (Models 1-4)

```bash
python cycling_safety_svi/modeling/choice_model_benchmark.py
```

Estimates four mixed logit models with stepwise feature selection and train-test validation. Includes WTP-space estimation. Output: model results in `reports/models/`.

### Step 4: Estimate demographic interaction models

```bash
python cycling_safety_svi/modeling/safety_demographics_interaction_model.py
```

Estimates 17 interaction models between perceived safety and demographic variables. Output: interaction model results in `reports/models/interaction/`.

### Step 5: Post-modeling analysis and figures

```bash
python cycling_safety_svi/visualization/post_modeling_analysis.py
```

Generates all figures for the paper (utility comparisons, safety score distributions, scatter plot matrices, image grids with Grad-CAM). Output: `reports/figures/post_modeling_analysis/`.

### Step 6: Generate descriptive statistics table

```bash
python cycling_safety_svi/reports/generate_descriptive_statistics.py
```

Generates the sample descriptive statistics table. Output: `reports/models/descriptive_statistics.tex`.

## Project Structure

```
cycling_safety_svi/
├── 1_apply_safety_model.py          # Step 1: Apply pre-trained safety model
├── config.py                        # Project configuration
├── dataset.py                       # Data utilities
├── features.py                      # Feature engineering
├── plots.py                         # Plotting utilities
├── cycling_safety_subjective_learning_pairwise/  # Git submodule (Costa et al. model)
├── modeling/
│   ├── choice_model_benchmark.py    # Step 3: Main choice model estimation
│   ├── mxl_functions.py             # Mixed logit helper functions
│   ├── stepwise_train_test.py       # Train-test validation
│   ├── safety_demographics_interaction_model.py  # Step 4: Demographic interactions
│   ├── compute_lrt_all_models.py    # Likelihood ratio tests
│   ├── extract_all_interaction_params.py  # Parameter extraction
│   └── correlation.py               # Correlation analysis
├── visualization/
│   ├── post_modeling_analysis.py     # Step 5: Generate figures
│   └── generate_gradcam_visualizations.py  # Grad-CAM heatmaps
├── exploration/                      # Data exploration scripts
└── reports/
    └── generate_descriptive_statistics.py  # Step 6: Descriptive stats table
```

## Requirements

Python 3.10+. Key dependencies: biogeme, torch, torchvision, pandas, numpy, matplotlib, seaborn.

See `requirements.txt` for the full list.

## License

See `LICENSE`.
