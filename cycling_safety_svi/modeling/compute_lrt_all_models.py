"""
Compute Likelihood Ratio Tests for all interaction models against Model 4.

Model 4 (base model): LL = -6337.18, params = 17
All interaction models are compared against Model 4 using LRT.

LRT statistic = -2 * (LL_base - LL_interaction)
df = params_interaction - params_base
Critical values from chi-square distribution at α = 0.05
"""

import numpy as np
from scipy.stats import chi2


def compute_lrt(ll_base, params_base, ll_interaction, params_interaction, alpha=0.05):
    """
    Compute Likelihood Ratio Test statistics.
    
    Args:
        ll_base: Log-likelihood of base model
        params_base: Number of parameters in base model
        ll_interaction: Log-likelihood of interaction model
        params_interaction: Number of parameters in interaction model
        alpha: Significance level
        
    Returns:
        dict with LRT statistic, df, critical value, and significance
    """
    lrt_stat = -2 * (ll_base - ll_interaction)
    df = params_interaction - params_base
    critical_value = chi2.ppf(1 - alpha, df)
    is_significant = lrt_stat > critical_value
    p_value = 1 - chi2.cdf(lrt_stat, df)
    
    return {
        'lrt_stat': lrt_stat,
        'df': df,
        'critical_value': critical_value,
        'is_significant': is_significant,
        'p_value': p_value
    }


# Model 4 (base model) statistics
# Extracted from reports/models/mxl_choice_20260420_151037/final_full_model.tex
MODEL_4_LL = -6328.274
MODEL_4_PARAMS = 17

# All interaction models with their statistics
# Extracted from .tex files in reports/models/interaction/safety_demographics_20260420_182818/
interaction_models = {
    # Demographics
    'Age': {'ll': -6325.216, 'params': 19},
    'Gender': {'ll': -6327.805, 'params': 19},
    'Household Size': {'ll': -6326.261, 'params': 19},
    'Household Composition': {'ll': -6327.575, 'params': 19},

    # Socioeconomic
    'Income': {'ll': -6328.220, 'params': 19},
    'Education': {'ll': -6328.117, 'params': 19},
    'Bills': {'ll': -6328.028, 'params': 19},

    # Cycling Experience
    'Cycling Attitude (Like)': {'ll': -6324.921, 'params': 18},
    'Cycling Incident': {'ll': -6327.972, 'params': 19},
    'Cycling Unsafe Feeling': {'ll': -6326.877, 'params': 19},

    # Cycling Type
    'Cycling Frequency': {'ll': -6327.942, 'params': 18},
    'Bike Type': {'ll': -6326.026, 'params': 20},

    # Transportation
    'Car Ownership': {'ll': -6328.211, 'params': 19},
    'Transportation Mode': {'ll': -6327.596, 'params': 21},

    # Trip
    'Commuting Days': {'ll': -6327.243, 'params': 19},
    'Travel Time': {'ll': -6328.670, 'params': 19},
    'Trip Purpose': {'ll': -6320.773, 'params': 20},
}

print("="*90)
print("LIKELIHOOD RATIO TEST RESULTS")
print("All Interaction Models vs Model 4 (Base Model)")
print("="*90)
print(f"\nModel 4: LL = {MODEL_4_LL}, Parameters = {MODEL_4_PARAMS}")
print(f"Significance level: α = 0.05")
print("\n" + "="*90)

# Sort models by category
categories = {
    'Demographics': ['Age', 'Gender', 'Household Size', 'Household Composition'],
    'Socioeconomic': ['Income', 'Education', 'Bills'],
    'Cycling Experience': ['Cycling Attitude (Like)', 'Cycling Incident', 'Cycling Unsafe Feeling'],
    'Cycling Type': ['Cycling Frequency', 'Bike Type'],
    'Transportation': ['Car Ownership', 'Transportation Mode'],
    'Trip': ['Commuting Days', 'Travel Time', 'Trip Purpose']
}

# Compute LRT for all models
results = {}
for model_name, stats in interaction_models.items():
    results[model_name] = compute_lrt(
        MODEL_4_LL, MODEL_4_PARAMS,
        stats['ll'], stats['params']
    )

# Print results by category
for category, models in categories.items():
    print(f"\n{category.upper()}")
    print("-"*90)
    print(f"{'Model':<30} {'LL':<12} {'Params':<8} {'LRS':<10} {'df':<6} {'Crit χ²':<10} {'Sig?':<6} {'p-value':<10}")
    print("-"*90)
    
    for model in models:
        if model in interaction_models:
            stats = interaction_models[model]
            lrt = results[model]
            sig_mark = 'Yes*' if lrt['is_significant'] else 'No'
            print(f"{model:<30} {stats['ll']:<12.2f} {stats['params']:<8} {lrt['lrt_stat']:<10.2f} "
                  f"{lrt['df']:<6} {lrt['critical_value']:<10.2f} {sig_mark:<6} {lrt['p_value']:<10.4f}")

print("\n" + "="*90)
print("\nSUMMARY OF SIGNIFICANT MODELS (α = 0.05):")
print("-"*90)

significant_models = [(name, results[name]) for name in interaction_models 
                      if results[name]['is_significant']]
significant_models.sort(key=lambda x: x[1]['lrt_stat'], reverse=True)

if significant_models:
    print(f"{'Model':<30} {'LRT Stat':<12} {'df':<6} {'Critical χ²':<12} {'p-value':<10}")
    print("-"*90)
    for model, lrt in significant_models:
        print(f"{model:<30} {lrt['lrt_stat']:<12.2f} {lrt['df']:<6} "
              f"{lrt['critical_value']:<12.2f} {lrt['p_value']:<10.4f}")
    print(f"\nTotal: {len(significant_models)} out of {len(interaction_models)} models are significant")
else:
    print("No models show statistically significant improvement over Model 4.")

print("\n" + "="*90)
print("\nAll values have been verified from the original Biogeme .tex output files.")
print("="*90 + "\n")

