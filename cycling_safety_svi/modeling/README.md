# Structural Equation Modeling for Cycling Safety Analysis

This module provides a comprehensive implementation of Structural Equation Models (SEM) for analyzing cycling safety and route preferences based on image segmentation data.

## Overview

The SEM models in this project use a mediation framework to analyze how different features in street images (traffic features, social features, and beauty features) affect perceived safety and route preferences.

## Model Types

We implement several SEM model variants:

1. **Full Model**: Complex model with all paths and cross-paths between variables
2. **Simple Model**: Simplified model with fewer cross-paths
3. **Minimal Model**: Minimal model with essential paths and fewer indicators
4. **Direct-Only Model**: Direct effects only (no mediation)
5. **Mediation-Only Model**: Mediation effects only (no direct effects)
6. **Direct-Mediated Model**: All segmentation variables directly affect perceptions

## OOP Implementation

The code uses an object-oriented architecture:

- `SEMModel`: Base class for all SEM models
- `ModelType`: Enum defining available model types
- `SEMModelRegistry`: Registry for model types and implementation classes
- Specialized model classes for each model type

## Files

- `sem_classes.py`: Core OOP implementation with base classes and registry
- `sem_models.py`: Implementations of different SEM model types
- `sem_utils.py`: Utility functions for data preparation and model comparison
- `run_sem.py`: Command-line interface for running models

## Running the Analysis

You can run the complete SEM analysis using the provided script:

```bash
./run_sem_analysis.sh
```

This will:
1. Organize existing model results
2. Clean model comparison outputs
3. Run all SEM models with improved organization
4. Perform stepwise variable selection for different mediators

## Individual Commands

You can also run individual analysis steps:

```bash
# Run a single model
python -m cycling_safety_svi.modeling.run_sem run-model --model-type minimal

# Run all models and compare
python -m cycling_safety_svi.modeling.run_sem run-all-models

# Perform stepwise selection
python -m cycling_safety_svi.modeling.run_sem stepwise-selection --target-mediator "traffic_safety"

# Organize results
python -m cycling_safety_svi.modeling.run_sem organize-results

# Clean model comparison output
python -m cycling_safety_svi.modeling.run_sem clean-model-comparison
```

## Results

All results are stored in `reports/models/sem/` with the following organization:

- Each model type has its own subdirectory
- Model comparison results are in the root directory
- Stepwise selection results are in the `stepwise/` subdirectory

The key output files include:
- `*_estimates.csv`: Model parameter estimates
- `*_mediation_effects.csv`: Calculated mediation effects
- `*_fit_indices.csv`: Model fit statistics
- `*_plot.png`: Path diagram visualization
- `model_comparison.csv`: Comparison of fit statistics across models
- `model_comparison.png`: Visualization of model comparison

## Dependencies

Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- semopy
- typer
- loguru

Install requirements with:
```bash
pip install -r requirements_sem.txt
```

## References

- Hayes, A. F. (2017). Introduction to mediation, moderation, and conditional process analysis: A regression-based approach. Guilford publications.
- Kline, R. B. (2015). Principles and practice of structural equation modeling. Guilford publications.
- Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction in social psychological research: Conceptual, strategic, and statistical considerations. Journal of personality and social psychology, 51(6), 1173. 