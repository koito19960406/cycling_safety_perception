# Cycling Safety Perception Analysis Setup

This document explains the complete setup for analyzing cycling safety perception using the pairwise comparison model from [Costa et al. (2025)](https://github.com/mncosta/cycling_safety_subjective_learning_pairwise).

## Summary of What Was Set Up

✅ **Git submodule added**: The cycling safety subjective learning model has been added as a git submodule  
✅ **Model weights location identified**: Place your `.pt` file in `cycling_safety_svi/cycling_safety_subjective_learning_pairwise/models/`  
✅ **Model application script created**: `cycling_safety_svi/apply_safety_model.py`  
✅ **Visualization script created**: `cycling_safety_svi/visualization/visualize_predicted_scores.py`  
✅ **Output directories created**: Results will be saved in `data/processed/predicted_danish/`  
✅ **Visualization output**: Plots will be saved in `reports/figures/predicted_images/`  

## Current Status

- **Model file**: You have placed the VGG-based model at `cycling_safety_svi/cycling_safety_subjective_learning_pairwise/models/vgg_syn+ber.pt`
- **Images to process**: 6,719 images found in `/srv/shared/bicycle_project_roos/images_scaled/`
- **Scripts**: Ready to run model inference and create visualizations

## Usage Instructions

### 1. Apply the Model to Images

Run the model on all images in the directory:

```bash
python cycling_safety_svi/apply_safety_model.py
```

**Options available:**
- `--images_dir`: Directory containing images (default: `/srv/shared/bicycle_project_roos/images_scaled`)
- `--model_path`: Path to model file (default: uses your `vgg_syn+ber.pt`)
- `--backbone`: CNN backbone (`vgg` is set as default for your model)
- `--model_type`: Model type (`rsscnn` for ranking + classification)
- `--batch_size`: Batch size for processing (default: 32)
- `--device`: Device to use (`auto`, `cpu`, or `cuda`)

**Output:**
- CSV file with safety scores: `data/processed/predicted_danish/cycling_safety_scores.csv`
- Columns: `image_name`, `safety_score`

### 2. Create Visualizations

After running the model, create visualizations:

```bash
python cycling_safety_svi/visualization/visualize_predicted_scores.py
```

**Options available:**
- `--results_csv`: Path to results CSV (default: uses the output from step 1)
- `--images_dir`: Directory with original images
- `--output_dir`: Where to save plots (default: `reports/figures/predicted_images/`)
- `--images_per_class`: Number of images per safety class (default: 20)
- `--n_categories`: Number of safety categories (default: 5)

**Output:**
- `safety_categories_grid.png`: Grid showing 20 images per safety category (5 categories)
- `safety_categories_grid_distribution.png`: Histogram and box plots of score distributions
- Console output with detailed statistics

## Expected Results

### Model Output
The model will generate safety scores for each image. These scores represent:
- **Higher scores**: Perceived as safer cycling environments
- **Lower scores**: Perceived as less safe cycling environments

### Visualization Categories
Images will be categorized into 5 classes based on quantiles:
1. **Very Unsafe**: Lowest 20% of scores
2. **Unsafe**: 20-40% of scores  
3. **Neutral**: 40-60% of scores
4. **Safe**: 60-80% of scores
5. **Very Safe**: Highest 20% of scores

### Visualization Output
- **Grid plot**: 5 rows (categories) × 20 columns (images) showing sample images from each safety category
- **Distribution plots**: Histogram of all scores and box plots by category
- **Statistics**: Detailed summary of score distributions

## Technical Details

### Model Architecture
- **Backbone**: VGG19 (based on your model file name)
- **Type**: Ranking + Classification (rsscnn)
- **Input**: 224×224 RGB images with ImageNet normalization
- **Output**: Individual safety scores via self-pairing strategy

### File Structure
```
cycling_safety_svi/
├── apply_safety_model.py                    # Main model application script
├── visualization/
│   └── visualize_predicted_scores.py       # Visualization script
└── cycling_safety_subjective_learning_pairwise/  # Git submodule
    ├── models/
    │   └── vgg_syn+ber.pt                  # Your model weights
    ├── nets/                               # Model architecture
    ├── scripts/                            # Original training/testing scripts
    └── utils/                              # Utilities

data/processed/predicted_danish/             # Output CSV files
reports/figures/predicted_images/            # Output visualization plots
```

## References

**Paper**: M. Costa, M. Marques, C. L. Azevedo, F. W. Siebert and F. Moura, "Which Cycling Environment Appears Safer? Learning Cycling Safety Perceptions From Pairwise Image Comparisons," in IEEE Transactions on Intelligent Transportation Systems, vol. 26, no. 2, pp. 1689-1700, Feb. 2025, doi: 10.1109/TITS.2024.3507639.

**Code Repository**: https://github.com/mncosta/cycling_safety_subjective_learning_pairwise

## Next Steps

1. **Run the model**: `python cycling_safety_svi/apply_safety_model.py`
2. **Create visualizations**: `python cycling_safety_svi/visualization/visualize_predicted_scores.py`
3. **Analyze results**: Review the generated CSV and visualization plots
4. **Optional**: Adjust visualization parameters (number of categories, images per class) as needed

The system is now ready to process all 6,719 images and generate comprehensive safety perception analysis and visualizations. 