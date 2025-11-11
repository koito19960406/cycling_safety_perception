"""
Extract all interaction model parameters and compute z-scores.

This script reads all 17 interaction model .tex files, extracts safety parameters,
computes z-scores, and generates a comprehensive summary.
"""

import re
import os
from pathlib import Path
import numpy as np


def extract_safety_params(tex_file_path):
    """
    Extract safety interaction parameters from a Biogeme .tex file.
    
    Returns: dict with parameter names, values, and standard errors
    """
    with open(tex_file_path, 'r') as f:
        content = f.read()
    
    # Find parameter estimates section
    param_section = re.search(r'%%Parameter estimates.*?\n\\begin{tabular}.*?\n(.*?)\\end{tabular}', 
                              content, re.DOTALL)
    
    if not param_section:
        return None
    
    params_text = param_section.group(1)
    
    # Extract all B_SAFETY parameters
    safety_params = {}
    pattern = r'B_SAFETY_(\w+)\s+&\s+([-\d.]+)\s+&\s+([-\d.]+)\s+&\s+([-\d.]+)\s+&'
    
    for match in re.finditer(pattern, params_text):
        param_name = match.group(1)
        value = float(match.group(2))
        se = float(match.group(3))
        t_stat = float(match.group(4))
        
        safety_params[param_name] = {
            'value': value,
            'se': se,
            't_stat': t_stat
        }
    
    return safety_params


def compute_z_score(beta_i, se_i, beta_ref, se_ref):
    """Compute z-score for testing beta_i vs beta_ref."""
    return (beta_i - beta_ref) / np.sqrt(se_i**2 + se_ref**2)


def identify_reference_category(params):
    """
    Identify reference category (usually has '_base' suffix or smallest code).
    """
    # Look for _base suffix
    for name in params.keys():
        if '_base' in name.lower():
            return name
    
    # If no _base, return first parameter (often the reference)
    return list(params.keys())[0] if params else None


# Base directory
base_dir = Path("/Users/koichiito/Documents/NUS PhD/Academic Matter/cycling_safety_perception/reports/models/interaction/safety_demographics_20251028_174315")

# All interaction models
models = {
    'Age': 'demographic_age/demographics_interaction_demographic_age.tex',
    'Gender': 'demographic_gender/demographics_interaction_demographic_gender.tex',
    'Household Size': 'demographic_household_size/demographics_interaction_demographic_household_size.tex',
    'Household Composition': 'demographic_household_composition/demographics_interaction_demographic_household_composition.tex',
    'Income': 'socioeconomic_income/demographics_interaction_socioeconomic_income.tex',
    'Education': 'socioeconomic_education/demographics_interaction_socioeconomic_education.tex',
    'Bills': 'socioeconomic_bills_EXCLUDE/demographics_interaction_socioeconomic_bills.tex',
    'Cycling Attitude': 'cycling_experience_cyclinglike/demographics_interaction_cycling_experience_cyclinglike.tex',
    'Cycling Incident': 'cycling_experience_cyclingincident/demographics_interaction_cycling_experience_cyclingincident.tex',
    'Cycling Unsafe': 'cycling_experience_cyclingunsafe/demographics_interaction_cycling_experience_cyclingunsafe.tex',
    'Cycling Frequency': 'cycling_type_cycler/demographics_interaction_cycling_type_cycler.tex',
    'Bike Type': 'cycling_type_biketype/demographics_interaction_cycling_type_biketype.tex',
    'Car Ownership': 'transportation_car/demographics_interaction_transportation_car.tex',
    'Transportation Mode': 'transportation_transportation/demographics_interaction_transportation_transportation.tex',
    'Commuting Days': 'trip_commutingdays/demographics_interaction_trip_commutingdays.tex',
    'Travel Time': 'trip_traveltime/demographics_interaction_trip_traveltime.tex',
    'Trip Purpose': 'trip_trippurpose/demographics_interaction_trip_trippurpose.tex',
}

print("="*100)
print("EXTRACTING ALL INTERACTION MODEL PARAMETERS")
print("="*100)

all_results = {}

for model_name, file_path in models.items():
    full_path = base_dir / file_path
    
    if not full_path.exists():
        print(f"\n❌ {model_name}: File not found at {file_path}")
        continue
    
    params = extract_safety_params(full_path)
    
    if not params:
        print(f"\n❌ {model_name}: Could not extract parameters")
        continue
    
    # Identify reference category
    ref_cat = identify_reference_category(params)
    
    if not ref_cat:
        print(f"\n❌ {model_name}: Could not identify reference category")
        continue
    
    ref_value = params[ref_cat]['value']
    ref_se = params[ref_cat]['se']
    ref_t = params[ref_cat]['t_stat']
    
    print(f"\n{'='*100}")
    print(f"Model: {model_name}")
    print(f"{'='*100}")
    print(f"\nReference: {ref_cat}")
    print(f"  β = {ref_value:.3f}, SE = {ref_se:.4f}, t-stat = {ref_t:.2f}")
    
    print(f"\nInteraction terms (tested vs reference):")
    print(f"{'Category':<30} {'Beta':>8} {'SE':>8} {'z-score':>10}")
    print("-"*100)
    
    model_results = {
        'reference': {
            'name': ref_cat,
            'beta': ref_value,
            'se': ref_se,
            't_stat': ref_t
        },
        'interactions': {}
    }
    
    for cat_name, cat_data in params.items():
        if cat_name != ref_cat:
            z_score = compute_z_score(cat_data['value'], cat_data['se'], 
                                     ref_value, ref_se)
            
            print(f"{cat_name:<30} {cat_data['value']:>8.3f} {cat_data['se']:>8.4f} {z_score:>10.2f}")
            
            model_results['interactions'][cat_name] = {
                'beta': cat_data['value'],
                'se': cat_data['se'],
                't_stat': cat_data['t_stat'],
                'z_score': z_score
            }
    
    all_results[model_name] = model_results

print("\n" + "="*100)
print("\nExtraction complete!")
print(f"Successfully processed {len(all_results)} out of {len(models)} models")
print("="*100 + "\n")

# Save results to file for use in table generation
import json
output_file = Path("/Users/koichiito/Documents/NUS PhD/Academic Matter/cycling_safety_perception/cycling_safety_svi/modeling/all_interaction_params.json")
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Results saved to: {output_file}")

