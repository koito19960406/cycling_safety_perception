# GOLEM-DC: Joint Causal Discovery and Choice Modeling

## Overview

GOLEM-DC (GOLEM for Discrete Choice) is a novel approach that combines causal structure learning with discrete choice modeling using a single joint optimization objective. Instead of the traditional two-stage approach, GOLEM-DC simultaneously learns:

1. **Causal relationships** between features (adjacency matrix A)
2. **Choice model parameters** (utility function θ)

This joint optimization allows the model to discover how product/alternative attributes causally influence each other while incorporating this causal structure into choice prediction.

## Key Innovation

**Traditional Two-Stage Approach:**
```
1. Learn causal structure: max p(X | A)
2. Use fixed structure for choice: max p(y | X_causal, θ)
```

**GOLEM-DC Joint Approach:**
```
Single objective: max p(y | X_causal(A), θ) - λ₁||A||₁ - λ₂h(A)
Where: X_causal = X(I - A^T)^(-1)
```

## Features

- **Joint optimization** of causal structure and choice parameters
- **Causal transformation** of features before choice modeling
- **Gumbel noise** for proper random utility modeling
- **DAG constraints** to ensure valid causal structures
- **Sparsity regularization** for interpretable causal graphs
- **Comprehensive evaluation** with multiple metrics (accuracy, log-likelihood, AIC, BIC, Pseudo-R²)

## Installation

1. Ensure you have the required dependencies:
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm
```

2. Clone or download the GOLEM-DC implementation files:
- `golem_dc_model.py`: Core model architecture
- `golem_dc_data.py`: Data loading and preprocessing
- `golem_dc_trainer.py`: Training and evaluation utilities
- `run_golem_dc.py`: Main script to run the pipeline

## Usage

### Basic Usage

Run GOLEM-DC with default settings:

```bash
python run_golem_dc.py
```

### Advanced Usage

Customize the model with various arguments:

```bash
python run_golem_dc.py \
    --choice_data data/raw/cv_dcm.csv \
    --safety_scores data/processed/predicted_danish/cycling_safety_scores.csv \
    --segmentation_data data/processed/segmentation_results/pixel_ratios.csv \
    --hidden_dim 128 \
    --lambda_1 0.05 \
    --lambda_2 2.0 \
    --n_epochs 300 \
    --learning_rate 0.0005 \
    --batch_size 128 \
    --verbose
```

### Key Arguments

**Data Arguments:**
- `--choice_data`: Path to choice data CSV (travel time, traffic lights, choices)
- `--safety_scores`: Path to predicted safety scores CSV
- `--segmentation_data`: Path to segmentation pixel ratios CSV
- `--baseline_results`: Path to baseline model results for comparison
- `--seg_features`: List of segmentation features to use

**Model Arguments:**
- `--hidden_dim`: Hidden dimension for utility network (default: 64)
- `--lambda_1`: L1 penalty weight for sparsity (default: 0.01)
- `--lambda_2`: DAG constraint penalty weight (default: 1.0)

**Training Arguments:**
- `--n_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 30)
- `--use_validation`: Use validation set during training

**Other Arguments:**
- `--output_dir`: Output directory for results (default: reports/models)
- `--causal_threshold`: Threshold for displaying causal edges (default: 0.1)
- `--seed`: Random seed (default: 42)
- `--verbose`: Verbose output during training

## Data Format

### Choice Data (CSV)
Required columns:
- `TT1`, `TT2`: Travel time for alternatives 1 and 2
- `TL1`, `TL2`: Number of traffic lights for alternatives 1 and 2
- `IMG1`, `IMG2`: Image filenames for alternatives 1 and 2
- `CHOICE`: Chosen alternative (1 or 2)
- `train`, `test` (optional): Train/test split indicators

### Safety Scores (CSV)
Required columns:
- `image_name`: Image filename
- `safety_score`: Predicted cycling safety score

### Segmentation Data (CSV)
Required columns:
- `filename_key`: Image filename
- Various segmentation class columns (e.g., 'Road', 'Sidewalk', 'Bike Lane', etc.)

## Outputs

The model creates a timestamped directory with the following outputs:

1. **Model Files:**
   - `golem_dc_model.pt`: Trained model state
   - `config.json`: Configuration used for training

2. **Results:**
   - `training_history.csv`: Loss and accuracy over epochs
   - `test_metrics.json`: Test set performance metrics
   - `model_comparison.json`: Comparison with baseline (if provided)
   - `causal_matrix.csv`: Learned causal adjacency matrix

3. **Visualizations:**
   - `training_history.png`: Training curves
   - `causal_structure.png`: Heatmap of causal relationships

4. **Reports:**
   - `summary_report.txt`: Comprehensive summary of results

## Example Results

After training, you'll see output like:

```
Test Set Performance:
   - Accuracy: 0.7234
   - Log-likelihood: -1842.35
   - Average log-likelihood: -0.4123
   - AIC: 4128.70
   - BIC: 4392.15
   - Pseudo R²: 0.3847
   - Number of parameters: 222

Strongest causal relationships:
  safety_score → traffic_lights_norm: 0.412
  seg_bike_lane → safety_score: 0.387
  seg_car → safety_score: -0.321
  ...
```

## Understanding the Causal Structure

The learned causal matrix shows how features influence each other:
- **Positive values**: Direct positive causal effect
- **Negative values**: Direct negative causal effect
- **Near-zero values**: No direct causal relationship

Example interpretations:
- `safety_score → travel_time: -0.25`: Higher safety scores cause perception of lower travel time
- `seg_bike_lane → safety_score: 0.40`: More bike lane coverage increases safety perception
- `seg_car → seg_bicycle: -0.15`: More cars reduce bicycle presence

## Comparison with Baseline

GOLEM-DC typically outperforms traditional discrete choice models by:
- **Better choice prediction** through causal understanding
- **More interpretable** relationships between features
- **Robust to confounding** through explicit causal modeling
- **Policy insights** for infrastructure planning

## Troubleshooting

1. **Memory Issues**: Reduce batch size or number of segmentation features
2. **Convergence Problems**: Adjust learning rate or lambda values
3. **Singular Matrix Errors**: The model has fallback approximations, but consider reducing lambda_2
4. **Poor Performance**: Try different feature combinations or hyperparameters

## Citation

If you use GOLEM-DC in your research, please cite:

```bibtex
@article{golem-dc2024,
  title={GOLEM-DC: Joint Causal Discovery and Discrete Choice Modeling},
  author={Your Name},
  journal={Transportation Research},
  year={2024}
}
```

## Future Extensions

- Multi-alternative choice sets (>2 alternatives)
- Heterogeneous causal structures across population segments
- Time-varying causal relationships
- Integration with deep learning feature extractors

## Contact

For questions or issues, please create an issue in the repository or contact the maintainers. 