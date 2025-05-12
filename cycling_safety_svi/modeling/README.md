# Structural Equation Modeling for Cycling Safety Analysis

This directory contains scripts for implementing Structural Equation Models (SEM) to analyze the relationship between street features and cycling route utility.

## Overview

The SEM models in this project aim to:

1. Create latent constructs for traffic safety, social safety, and beauty from street segmentation data
2. Use these constructs as mediators between segmentation features and cycling route utility (V_img)
3. Analyze both direct and indirect effects of street features on cycling route preferences

## Files

- `sem_model.py`: Implementation of a mediation-focused SEM model
- `sem_advanced.py`: Advanced implementation with multiple mediation model specifications, exploratory data analysis, and comprehensive model comparison

## Mediation Framework

Our models use a mediation framework where:

1. **Independent Variables (X)**: Segmentation features grouped into traffic-related, social-related, and aesthetics-related clusters
2. **Mediators (M)**: Latent constructs for perceived traffic safety, social safety, and beauty
3. **Dependent Variable (Y)**: Cycling route utility (V_img)

This approach tests whether the effect of street features on route choice is mediated through perceptions of safety and beauty.

## Theory

Structural Equation Modeling (SEM) with mediation allows us to:

- Test whether street features affect cycling route utility directly or indirectly through perceptions
- Estimate the proportion of effects that are mediated
- Compare competing mediation hypotheses (full vs. partial mediation)
- Anchor the latent constructs to observed perception ratings

In our models, the latent constructs are anchored to the observed perception ratings to ensure they reflect participants' subjective perceptions, while still being influenced by objective street features.

## Usage

### Basic Mediation Model

```bash
python -m cycling_safety_svi.modeling.sem_model
```

This runs a mediation model with default parameters and saves results to the models directory.

### Advanced Mediation Analysis

```bash
python -m cycling_safety_svi.modeling.sem_advanced --explore --output-dir=./models/sem_results --figures-dir=./figures/sem_analysis
```

Parameters:
- `--explore`: Run exploratory data analysis (default: True)
- `--output-dir`: Directory to save model results (default: models/sem_models)
- `--figures-dir`: Directory to save EDA figures (default: figures/sem_analysis)

## Model Specifications

The advanced script tests multiple mediation model specifications:

1. **Full Mediation**: Street features affect V_img only through perception mediators
2. **Partial Mediation**: Street features affect V_img both directly and through perception mediators
3. **Simplified Mediation**: Reduced model with fewer cross-paths and indicators
4. **Direct Effects Only**: Baseline model with no mediation for comparison

## Output Analysis

The models produce:

1. **Path Coefficients**: Direct relationships between variables
2. **Mediation Effects**: Indirect effects through each mediator
3. **Proportion Mediated**: Percentage of total effect that occurs through mediators
4. **Model Comparisons**: Statistical comparisons of competing mediation models

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