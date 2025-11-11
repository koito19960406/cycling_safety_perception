"""
Verify z-score calculations for interaction models.

This script computes z-scores for interaction terms against reference categories
using the formula: z = (β_i - β_r) / sqrt(SE_i² + SE_r²)

All values are hard-coded from the original Biogeme model output files in:
reports/models/interaction/safety_demographics_20251028_174315/
"""

import numpy as np


def compute_z_score(beta_i, se_i, beta_ref, se_ref):
    """
    Compute z-score for testing if beta_i differs from beta_ref.
    
    Args:
        beta_i: Beta coefficient for category i
        se_i: Standard error for beta_i
        beta_ref: Beta coefficient for reference category
        se_ref: Standard error for beta_ref
        
    Returns:
        z-score
    """
    z = (beta_i - beta_ref) / np.sqrt(se_i**2 + se_ref**2)
    return z


def print_model_results(model_name, reference_cat, reference_beta, reference_se, 
                        categories, betas, ses):
    """Print formatted results for one interaction model."""
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"\n{reference_cat} (reference): β = {reference_beta:.3f}, SE = {reference_se:.4f}")
    print(f"  (tested against 0, t-stat = {reference_beta/reference_se:.2f})")
    
    print(f"\nInteraction terms (tested against reference):")
    print(f"{'Category':<25} {'Beta':>8} {'SE':>8} {'z-score':>10} {'p-value':>10}")
    print("-" * 70)
    
    for cat, beta, se in zip(categories, betas, ses):
        z = compute_z_score(beta, se, reference_beta, reference_se)
        # Two-tailed p-value
        p_value = 2 * (1 - np.abs(z) / np.sqrt(2) * np.exp(-z**2/2))  # approximation
        # Better p-value calculation using scipy if available
        try:
            from scipy.stats import norm
            p_value = 2 * (1 - norm.cdf(np.abs(z)))
        except ImportError:
            pass
        
        print(f"{cat:<25} {beta:>8.3f} {se:>8.4f} {z:>10.2f} {p_value:>10.4f}")


# ============================================================================
# AGE MODEL
# ============================================================================
print_model_results(
    model_name="Age",
    reference_cat="Young (18-30)",
    reference_beta=0.02,
    reference_se=0.078,
    categories=["Middle (31-60)", "Senior (61+)"],
    betas=[0.103, 0.212],
    ses=[0.0895, 0.0997]
)

# ============================================================================
# HOUSEHOLD SIZE MODEL
# ============================================================================
print_model_results(
    model_name="Household Size",
    reference_cat="Small (1-2 people)",
    reference_beta=0.198,
    reference_se=0.0471,
    categories=["Medium (3-4)", "Large (5+)"],
    betas=[-0.176, 0.063],
    ses=[0.0673, 0.126]
)

# ============================================================================
# INCOME MODEL
# ============================================================================
print_model_results(
    model_name="Income",
    reference_cat="Low (< €2,250)",
    reference_beta=0.165,
    reference_se=0.0767,
    categories=["Medium (€2,251-€3,650)", "High (> €3,651)"],
    betas=[0.0228, -0.0897],
    ses=[0.0901, 0.0889]
)

# ============================================================================
# CYCLING ATTITUDE MODEL
# ============================================================================
print_model_results(
    model_name="Cycling Attitude",
    reference_cat="Yes (like cycling)",
    reference_beta=0.184,
    reference_se=0.0399,
    categories=["No (don't like cycling)"],
    betas=[-0.269],
    ses=[0.0912]
)

# ============================================================================
# CYCLING FREQUENCY MODEL
# ============================================================================
print_model_results(
    model_name="Cycling Frequency",
    reference_cat="Occasional (1-2 days/week)",
    reference_beta=0.154,
    reference_se=0.0559,
    categories=["Regular (>3 days/week)"],
    betas=[-0.0256],
    ses=[0.067]
)

# ============================================================================
# CAR OWNERSHIP MODEL
# ============================================================================
print_model_results(
    model_name="Car Ownership",
    reference_cat="No car",
    reference_beta=0.235,
    reference_se=0.0871,
    categories=["One car", "Multiple cars"],
    betas=[-0.126, -0.0611],
    ses=[0.0941, 0.111]
)

# ============================================================================
# COMMUTING DAYS MODEL
# ============================================================================
print_model_results(
    model_name="Commuting Days",
    reference_cat="No commute",
    reference_beta=0.205,
    reference_se=0.0598,
    categories=["Part-time", "Full-time"],
    betas=[-0.179, -0.0261],
    ses=[0.0831, 0.075]
)

# ============================================================================
# TRIP PURPOSE MODEL
# ============================================================================
print_model_results(
    model_name="Trip Purpose",
    reference_cat="Others",
    reference_beta=-0.161,
    reference_se=0.158,
    categories=["Commuting", "Errands", "Recreational"],
    betas=[0.278, 0.244, 0.484],
    ses=[0.167, 0.165, 0.17]
)

print(f"\n{'='*70}")
print("Notes:")
print("- All values extracted from Biogeme model output files")
print("- z-scores test H0: β_i = β_ref vs H1: β_i ≠ β_ref")
print("- p-values are two-tailed")
print("='*70}\n")

