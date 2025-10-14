"""
Safety-Demographics Interaction Model (Mixed Logit Version - Method 1)

This script extends a trained choice model with safety * demographics interaction effects.
It uses a mixed logit (MXL) formulation with the Method 1 approach where both base and 
interaction effects are random parameters sharing a common sigma (heterogeneity structure).

Key difference from original:
- Base safety coefficient is random per demographic group
- Interaction effects are implemented as separate random parameters
- All groups share common sigma (can be relaxed if needed)
- This captures demographic-specific heterogeneity in both mean AND variance

Usage:
    python safety_demographics_interaction_model_method1.py --model_path /path/to/final_model.pickle
"""

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import sqlite3
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
import logging
from biogeme.expressions import Beta, Variable, log, exp, bioDraws
from pathlib import Path
from datetime import datetime
import json
from mxl_functions import (
    estimate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results
)


class SafetyDemographicsInteractionModel:
    """Extends a trained choice model with safety * demographics interaction effects using MXL Method 1."""
    
    def __init__(self, model_path, model_group, demographic_variables, output_dir):
        """Initialize the demographics interaction model"""
        self.model_path = Path(model_path)
        self.model_group = model_group
        self.demographic_variables = demographic_variables
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Safety-Demographics Interaction Model (MXL Method 1) - Group: {self.model_group}")
        print(f"Base model path: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        
        logging.getLogger('biogeme').setLevel(logging.WARNING)
        
        # Mappings with sanitized labels for biogeme compatibility
        self.demographic_mappings = {
            'age': {
                1: 'age_18_30', 2: 'age_31_45', 3: 'age_46_60', 
                4: 'age_61_75', 5: 'age_76_plus'
            },
            'gender': {
                1: 'Male', 2: 'Female', 3: 'Other', 4: 'Prefer_not_to_say'
            },
            'household_composition': {
                1: 'Live_alone', 2: 'Couple_no_children', 3: 'Couple_with_children',
                4: 'One_adult_with_children', 5: 'Two_plus_adults_not_couple', 6: 'Other_hh'
            },
            'household_size': {
                1: 'hh_1', 2: 'hh_2', 3: 'hh_3', 
                4: 'hh_4', 5: 'hh_5', 6: 'hh_6_plus'
            },
            'education': {
                1: 'No_education', 2: 'Primary_education', 3: 'Lower_vocational',
                4: 'Lower_secondary', 5: 'Intermediate_vocational', 6: 'MULO_or_MMS',
                7: 'HAVO', 8: 'HBS_VWO_etc', 9: 'HBO',
                10: 'University', 11: 'MSc', 12: 'PhD',
                13: 'Other_edu', 14: 'Prefer_not_to_say_edu'
            },
            'income': {
                1: 'inc_lt_1250', 2: 'inc_1251_1700', 3: 'inc_1701_2250',
                4: 'inc_2251_3650', 5: 'inc_3651_7000', 6: 'inc_gt_7001',
                7: 'Unknown_inc', 8: 'Prefer_not_to_say_inc'
            },
            'bills': {
                1: 'bills_very_easy', 2: 'bills_easy', 3: 'bills_reasonable', 
                4: 'bills_difficult', 5: 'bills_very_difficult', 6: 'Unknown_bills'
            },
            'transportation': {
                1: 'Walk', 2: 'Bike', 3: 'Public_transport', 4: 'Car_transport', 5: 'Other_transport'
            },
            'car': {
                1: 'No_cars', 2: 'car_1', 3: 'car_2', 4: 'car_3_plus'
            },
            'traveltime': {
                1: 'No_commute', 2: 'tt_lt_10_min', 3: 'tt_10_20_min',
                4: 'tt_20_30_min', 5: 'tt_30_40_min'
            },
            'commutingdays': {
                1: 'cd_no_commute', 2: 'cd_1_day_week', 3: 'cd_2_days_week',
                4: 'cd_3_days_week', 5: 'cd_4_days_week', 6: 'cd_5_plus_days_week'
            },
            'cycler': {
                1: 'Do_not_cycle', 2: 'cycle_lt_1_week', 3: 'cycle_1_day_week',
                4: 'cycle_2_days_week', 5: 'cycle_3_days_week', 6: 'cycle_4_days_week',
                7: 'cycle_5_plus_days_week'
            },
            'cyclingincident': {
                1: 'incid_yes_severe', 2: 'incid_yes_mild', 3: 'incid_no'
            },
            'cyclinglike': {1: 'like_yes', 2: 'like_no'},
            'cyclingunsafe': {
                1: 'unsafe_yes_sometimes', 2: 'unsafe_yes_evening_night', 3: 'unsafe_no'
            },
            'biketype': {
                1: 'Regular_bike', 2: 'Racing_bike', 3: 'E_bike', 
                4: 'Fatbike', 5: 'Other_bike'
            },
            'trippurpose': {
                1: 'purpose_commuting', 2: 'purpose_errands', 3: 'purpose_recreational', 4: 'purpose_other'
            }
        }
        
        self.feature_name_mapping = {
            'B_LANE_MARKING___GENERAL': 'Lane Marking - General',
            'B_TRAFFIC_SIGN_(FRONT)': 'Traffic Sign (Front)',
            'B_UTILITY_POLE': 'Utility Pole'
        }
        
        self.num_draws = 1000
        self.individual_id = 'RID'
        self.min_obs_per_individual = 15
        
    def load_trained_model_data(self):
        """Load the trained model and extract its data and structure"""
        print("\nLoading base model to extract segmentation features...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_results = pickle.load(f)
        
        self.original_model_features = self._extract_model_features(model_results)
        print(self.original_model_features)
        print(f"✓ Extracted {len(self.original_model_features)} segmentation features from base model.")

    def _extract_model_features(self, model_results):
        """Extract segmentation features used in the original trained model"""
        param_names = list(model_results.betaNames)
        
        segmentation_features = []
        for param_name in param_names:
            if param_name.startswith('B_') and param_name not in ['B_TT', 'B_TL', 'B_SAFETY_SCORE'] and not param_name.startswith('sigma_'):
                # Map back to original feature name
                feature_name = self.feature_name_mapping.get(param_name, param_name.replace('B_', ''))
                segmentation_features.append(feature_name)
        return segmentation_features
        
    def load_and_prepare_data(self, 
                             cv_dcm_path='data/raw/cv_dcm.csv',
                             database_path='data/raw/database_2024_10_07_135133.db',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv'):
        """Load and prepare all datasets with enhanced demographics and segmentation"""
        print("\nLoading and preparing all datasets...")
        self.choice_data = pd.read_csv(cv_dcm_path)
        
        print("\nApplying data cleaning steps...")
        self.choice_data = apply_data_cleaning(
            self.choice_data, 
            individual_id=self.individual_id,
            min_obs=self.min_obs_per_individual,
            fix_problematic_rid=True
        )

        self._load_enhanced_demographics(database_path)
        self.safety_scores = pd.read_csv(safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        
        if self.original_model_features:
            seg_chunks = [chunk for chunk in pd.read_csv(segmentation_path, chunksize=1000)]
            self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
            self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
            # replace " " with "_" in column names
            self.segmentation_data.columns = [col.replace(' ', '_').replace('___', ' - ') for col in self.segmentation_data.columns]
        else:
            self.segmentation_data = None
            print("No segmentation features in base model, skipping segmentation data.")

        self._merge_all_datasets()
        self._process_demographics()
        # save self.merged_data in a csv file
        merged_path = self.output_dir.parent / f'merged_data_{self.model_group}.csv'
        self.merged_data.to_csv(merged_path, index=False)
        print(f"Final dataset prepared with {len(self.merged_data)} observations.")
        
    def _load_enhanced_demographics(self, database_path):
        """Load enhanced demographics from database including new variables."""
        print("Loading enhanced demographics from database...")
        conn = sqlite3.connect(database_path)
        
        demographic_cols = list(self.demographic_mappings.keys())
        # The user requested to exclude 'work' from the analysis
        if 'work' in demographic_cols:
            demographic_cols.remove('work')

        query = f"SELECT respondent_id, set_id, {', '.join(demographic_cols)} FROM Response WHERE age IS NOT NULL AND gender IS NOT NULL"
        
        try:
            self.demographics = pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Database query failed, likely missing columns: {e}")
            print("Attempting to load without new demographic columns...")
            
            new_cols = ['household_composition', 'household_size', 'bills', 'transportation', 'commutingdays', 'trippurpose', 'traveltime']
            demographic_cols = [c for c in demographic_cols if c not in new_cols]
            
            query = f"SELECT respondent_id, set_id, {', '.join(demographic_cols)} FROM Response WHERE age IS NOT NULL AND gender IS NOT NULL"
            self.demographics = pd.read_sql_query(query, conn)
        
        conn.close()
        
        # Handle specific data cleaning tasks
        if 'traveltime' in self.demographics.columns:
            self.demographics['traveltime'] = self.demographics['traveltime'].replace(6, 5)

        # drop any rows with NaN in demographics
        self.demographics.dropna(subset=demographic_cols, inplace=True)
        # For set_id == 63, there are two rows with the same RID. Set the second one's set_id to 63999
        mask = self.demographics['set_id'] == 63
        idx = self.demographics[mask].index
        if len(idx) > 1:
            self.demographics.at[idx[1], 'set_id'] = 63999
    
    def _merge_all_datasets(self):
        """Merge all datasets into a single dataframe for modeling."""
        merged_data = self.choice_data.merge(self.demographics, left_on='RID', right_on='set_id', how='inner')
        
        safety_dict = dict(zip(self.safety_scores['image_name'], self.safety_scores['safety_score']))
        merged_data['SAFETY_SCORE1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['SAFETY_SCORE2'] = merged_data['IMG2'].map(safety_dict)
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['SAFETY_SCORE1'].fillna(mean_safety, inplace=True)
        merged_data['SAFETY_SCORE2'].fillna(mean_safety, inplace=True)
        
        if self.segmentation_data is not None:
            segmentation_dict = {row['filename_key'] + '.jpg': row.drop('filename_key').to_dict() 
                                 for _, row in self.segmentation_data.iterrows() if pd.notna(row['filename_key'])}
            for feature in self.original_model_features:
                feature_col = feature.replace(' ', '_').replace('___', ' - ')
                merged_data[f"{feature_col}_1"] = merged_data['IMG1'].map(lambda x: segmentation_dict.get(x, {}).get(feature_col, 0))
                merged_data[f"{feature_col}_2"] = merged_data['IMG2'].map(lambda x: segmentation_dict.get(x, {}).get(feature_col, 0))
        
        self.merged_data = merged_data
        
    def _process_demographics(self):
        """Create categorical columns from numeric codes."""
        for col, mapping in self.demographic_mappings.items():
            if mapping:
                self.merged_data[f'{col}_cat'] = self.merged_data[col].map(mapping)
        
        self.merged_data = self.merged_data[
            (self.merged_data['age_cat'] != 'other') & (self.merged_data['gender_cat'] != 'other')
        ].copy()
        
        # Drop rows with any remaining NaNs in demographic data that will be used for dummies
        demographic_cat_cols = [f'{col}_cat' for col in self.demographic_mappings.keys() if f'{col}_cat' in self.merged_data.columns]
        self.merged_data.dropna(subset=demographic_cat_cols, inplace=True)

    def identify_demographic_categories(self, data):
        """
        Identify all demographic categories present in the data for the current demographic variable.
        Returns reference category and list of non-reference categories.
        """
        demographic_categories = {}
        
        for demo in self.demographic_variables:
            cat_col = f'{demo}_cat'
            if cat_col not in data.columns:
                continue
                
            # Get the numeric keys corresponding to the categories present in the data
            reverse_mapping = {v: k for k, v in self.demographic_mappings[demo].items()}
            present_cats = data[cat_col].dropna().unique()
            data_keys = [reverse_mapping[cat] for cat in present_cats if cat in reverse_mapping]
            
            if data_keys:
                ref_key = min(data_keys)
                ref_cat = self.demographic_mappings[demo][ref_key]
                
                # Get all non-reference categories
                non_ref_cats = [cat for cat in present_cats if cat != ref_cat]
                
                demographic_categories[demo] = {
                    'reference': ref_cat,
                    'categories': non_ref_cats,
                    'all_categories': present_cats
                }
                
                print(f"\n{demo}: Reference = {ref_cat}, Other categories = {non_ref_cats}")
        
        return demographic_categories

    def estimate_interaction_model(self):
        """
        Estimate the MXL safety * demographics interaction model using Method 1.
        
        Method 1 Approach:
        - Base safety parameter is random for reference group: B_SAFETY_ref_rnd
        - Each demographic group gets its own random parameter: B_SAFETY_group_rnd
        - All groups share a common sigma: sigma_SAFETY_common
        - Utility: V = ... + B_SAFETY_group_rnd * SAFETY_SCORE
        """
        print("\nEstimating Safety * Demographics Interaction Model (MXL Method 1)...")
        print("Method 1: Both base and interaction effects are random parameters with shared sigma")
        
        model_data = self.merged_data.copy()
        
        # Identify demographic categories
        demographic_info = self.identify_demographic_categories(model_data)
        
        # Prepare panel data (no dummy creation needed for Method 1)
        features = self.original_model_features + ['SAFETY_SCORE']
        
        # Add demographic column to features so it's available in biogeme data
        for demo in self.demographic_variables:
            if demo in model_data.columns:
                features.append(demo)
        
        _, biodata_wide, obs_per_ind = prepare_panel_data(
            model_data, 
            self.individual_id, 
            'CHOICE', 
            features
        )
        
        # ============================================================
        # METHOD 1: RANDOM PARAMETERS FOR EACH DEMOGRAPHIC GROUP
        # ============================================================
        
        # 1. Define standard random parameters (TT, TL)
        B_TT = Beta('B_TT', -1, None, None, 0)
        B_TL = Beta('B_TL', -1, None, None, 0)
        sigma_TT = Beta('sigma_TT', 0.1, None, None, 0)
        sigma_TL = Beta('sigma_TL', 0.1, None, None, 0)
        
        B_TT_rnd = -exp(B_TT + sigma_TT * bioDraws('B_TT_rnd', 'NORMAL_HALTON2'))
        B_TL_rnd = -exp(B_TL + sigma_TL * bioDraws('B_TL_rnd', 'NORMAL_HALTON2'))
        
        # 2. Define COMMON sigma for all safety parameters
        sigma_SAFETY_common = Beta('sigma_SAFETY_common', 0.1, None, None, 0)
        
        # 3. Define random safety parameters for each demographic group
        safety_random_params = {}
        
        for demo in self.demographic_variables:
            if demo not in demographic_info:
                continue
                
            ref_cat = demographic_info[demo]['reference']
            categories = demographic_info[demo]['categories']
            
            # Create base parameter for reference group
            ref_param_name = f'B_SAFETY_{ref_cat}_base'
            B_ref = Beta(ref_param_name, 1.0, None, None, 0)
            draw_ref = bioDraws(f'{ref_param_name}_rnd', 'NORMAL_HALTON2')
            B_ref_rnd = exp(B_ref + sigma_SAFETY_common * draw_ref)
            safety_random_params[ref_cat] = B_ref_rnd
            
            # Create parameters for each non-reference category
            for cat in categories:
                cat_param_name = f'B_SAFETY_{cat}'
                B_cat = Beta(cat_param_name, 1.0, None, None, 0)
                # CRITICAL: Use same draw name as reference to share random component
                draw_cat = bioDraws(f'{cat_param_name}_rnd', 'NORMAL_HALTON2')
                B_cat_rnd = exp(B_cat + sigma_SAFETY_common * draw_cat)
                safety_random_params[cat] = B_cat_rnd
        
        # 4. Define fixed parameters for segmentation features
        fixed_params = {}
        for f in self.original_model_features:
            param_name = f"B_{f.replace(' - ', '___').replace(' ', '_')}"
            fixed_params[f] = Beta(param_name, 0, None, None, 0)
        
        # 5. Get the demographic variable name for the current model group
        demo_var = self.demographic_variables[0]  # We only have one variable per model group
        demo_col = Variable(demo_var)
        
        # 6. Create utility function
        V = []
        for q in range(obs_per_ind):
            V1, V2 = 0, 0
            
            # Add TT and TL random parameters
            v1_tt, v2_tt = f"TT1_{q}", f"TT2_{q}"
            v1_tl, v2_tl = f"TL1_{q}", f"TL2_{q}"
            
            if v1_tt in biodata_wide.variables:
                V1 += B_TT_rnd * Variable(v1_tt) / 10
                V2 += B_TT_rnd * Variable(v2_tt) / 10
            
            if v1_tl in biodata_wide.variables:
                V1 += B_TL_rnd * Variable(v1_tl) / 3
                V2 += B_TL_rnd * Variable(v2_tl) / 3
            
            # Add fixed segmentation parameters
            for feature, param in fixed_params.items():
                feature_col = feature.replace(' ', '_').replace('___', ' - ')
                v1_name = f"{feature_col}_1_{q}"
                v2_name = f"{feature_col}_2_{q}"
                
                if v1_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name)
                    V2 += param * Variable(v2_name)
            
            # Add safety random parameters based on demographic category
            # This is the key difference: we select the appropriate random parameter
            # based on the individual's demographic category
            v1_safety = f"SAFETY_SCORE1_{q}"
            v2_safety = f"SAFETY_SCORE2_{q}"
            
            if v1_safety in biodata_wide.variables:
                # Build conditional expression for demographic-specific safety parameter
                # Structure: (B_base + (demo==cat2)*B_cat2 + (demo==cat3)*B_cat3 + ...) * SAFETY_SCORE
                
                # Get reference category
                ref_cat = demographic_info[demo_var]['reference']
                non_ref_cats = demographic_info[demo_var]['categories']
                
                # Start with reference category as base (no conditional)
                safety_param_expr = safety_random_params[ref_cat]
                
                # Add conditional terms for non-reference categories
                for cat in non_ref_cats:
                    # Find the numeric code for this category
                    cat_code = None
                    for code, label in self.demographic_mappings[demo_var].items():
                        if label == cat:
                            cat_code = code
                            break
                    
                    if cat_code is not None:
                        # Add conditional interaction: (demo == cat_code) * B_SAFETY_cat_rnd
                        safety_param_expr += (demo_col == cat_code) * safety_random_params[cat]
                
                # Multiply the entire parameter expression by safety scores once
                V1 += safety_param_expr * Variable(v1_safety)
                V2 += safety_param_expr * Variable(v2_safety)
            
            V.append({1: V1, 2: V2})
        
        # 7. Estimate model
        model_name = f'demographics_interaction_{self.model_group}'
        AV = {1: 1, 2: 1}
        
        results = estimate_mxl(
            V, 
            AV, 
            'CHOICE', 
            obs_per_ind, 
            self.num_draws, 
            biodata_wide, 
            model_name, 
            self.output_dir
        )
        
        self.results = (results, obs_per_ind)
        print_mxl_results(results.data)
        
        return results

    def generate_results_table(self):
        """Generates and saves a LaTeX table for the interaction model."""
        if not hasattr(self, 'results'):
            print("Model not estimated yet. Cannot generate table.")
            return

        print("\nGenerating results table...")
        train_res, obs_per_ind = self.results
        
        train_metrics = extract_mxl_metrics(train_res, obs_per_ind, train_res.data.numberOfObservations)
        params = train_res.get_estimated_parameters()
        all_param_names = sorted(list(params.index))

        def format_p(p):
            return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        table_title = f"Safety-Demographics Interaction Model (MXL Method 1) - {self.model_group.replace('_', ' ').title()}"
        table_label = f"tab:demographics_interaction_{self.model_group}"
        
        lines = [
            "\\begin{table}[htbp]", "\\centering", f"\\caption{{{table_title}}}",
            f"\\label{{{table_label}}}", "\\resizebox{0.8\\textwidth}{!}{%",
            "\\begin{tabular}{lc}", "\\toprule",
            "& \\textbf{Coefficient (t-stat)} \\\\", "\\midrule",
            "\\multicolumn{2}{l}{\\textit{Goodness of fit}} \\\\", "\\hline",
            f"Sample size & {train_metrics['n_observations']} \\\\",
            f"Log-Likelihood & {train_metrics['log_likelihood']:.2f} \\\\",
            f"Rho-squared & {train_metrics['pseudo_r2']:.4f} \\\\",
            "\\hline", "\\multicolumn{2}{l}{\\textit{Parameters}} \\\\", "\\hline"
        ]

        for param in all_param_names:
            p_name_latex = param.replace('_', '\\_')
            val = params.loc[param, 'Value']
            t = params.loc[param, 'Rob. t-test']
            p = params.loc[param, 'Rob. p-value']
            stars = format_p(p)
            val_str = f"{val:.3f}{stars} ({t:.2f})"
            lines.append(f"{p_name_latex} & {val_str} \\\\")
        
        lines.extend([
            "\\hline", "\\bottomrule",
            "\\multicolumn{2}{l}{\\textsuperscript{***}p<0.001, \\textsuperscript{**}p<0.01, \\textsuperscript{*}p<0.05}",
            "\\end{tabular}}", "\\end{table}"
        ])

        latex_content = "\n".join(lines)
        table_path = self.output_dir / f'demographics_interaction_model_{self.model_group}.tex'
        with open(table_path, 'w') as f: 
            f.write(latex_content)
        print(f"LaTeX table saved to {table_path}")
    
    def run_analysis(self):
        """Run the complete safety * demographics interaction analysis"""
        self.load_trained_model_data()
        self.load_and_prepare_data()
        self.estimate_interaction_model()
        self.generate_results_table()
        print(f"\n✓ Analysis complete. Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run Safety-Demographics Interaction Model (MXL Method 1)')
    parser.add_argument('--model_path', type=str, 
                        default='reports/models/mxl_choice_20250725_122947/final_full_model.pickle',
                        help='Path to the trained base model pickle file')
    args = parser.parse_args()

    # Define model groups - one demographic variable per group
    model_groups = {
        "demographic_age": ["age"],
        "demographic_gender": ["gender"],
        "demographic_household_composition": ["household_composition"],
        "demographic_household_size": ["household_size"],
        "socioeconomic_education": ["education"],
        "socioeconomic_income": ["income"],
        "socioeconomic_bills": ["bills"],
        "cycling_experience_cyclingincident": ["cyclingincident"],
        "cycling_experience_cyclinglike": ["cyclinglike"],
        "cycling_experience_cyclingunsafe": ["cyclingunsafe"],
        "cycling_type_cycler": ["cycler"],
        "cycling_type_biketype": ["biketype"],
        "transportation_car": ["car"],
        "transportation_transportation": ["transportation"],
        "trip_trippurpose": ["trippurpose"],
        "trip_traveltime": ["traveltime"],
        "trip_commutingdays": ["commutingdays"]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path('reports/models/interaction') / f"safety_demographics_{timestamp}"

    for group_name, variables in model_groups.items():
        print(f"\n{'='*80}")
        print(f"Running analysis for group: {group_name}")
        print(f"{'='*80}")
        
        group_output_dir = base_output_dir / group_name

        interaction_model = SafetyDemographicsInteractionModel(
            model_path=args.model_path,
            model_group=group_name,
            demographic_variables=variables,
            output_dir=group_output_dir
        )
        
        try:
            interaction_model.run_analysis()
        except Exception as e:
            print(f"ERROR in {group_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"All analyses complete!")
    print(f"Results saved to: {base_output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()