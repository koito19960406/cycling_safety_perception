"""
Safety-Demographics Interaction Model (Mixed Logit Version)

This script extends a trained choice model with safety * demographics interaction effects.
It uses a mixed logit (MXL) formulation, where TT, TL, and SAFETY_SCORE are random parameters.
The script takes a trained model as input to ensure the same segmentation features are used.
It adds trippurpose and traveltime as demographic variables.

Usage:
    python safety_demographics_interaction_model.py --model_path /path/to/final_model.pickle
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
    estimate_mxl, simulate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results
)


class SafetyDemographicsInteractionModel:
    """Extends a trained choice model with safety * demographics interaction effects using MXL."""
    
    def __init__(self, model_path, model_group, demographic_variables, output_dir):
        """Initialize the demographics interaction model"""
        self.model_path = Path(model_path)
        self.model_group = model_group
        self.demographic_variables = demographic_variables
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Safety-Demographics Interaction Model (MXL) - Group: {self.model_group}")
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
        else:
            self.segmentation_data = None
            print("No segmentation features in base model, skipping segmentation data.")
        
        self._merge_all_datasets()
        self._process_demographics()
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

        # The user requested to use the provided labels, so we comment out dynamic mapping generation
        # for col in self.demographics.columns:
        #     if col in self.demographic_mappings and col not in ['age', 'gender']:
        #         counts = self.demographics[col].value_counts().sort_index()
        #         print(f"\n{col} distribution:\n{counts}")
        #         unique_vals = sorted([x for x in self.demographics[col].unique() if pd.notna(x)])
        #         self.demographic_mappings[col] = {val: f'{col}_{int(val)}' for val in unique_vals}
        #         print(f"Created mapping for {col}: {self.demographic_mappings[col]}")

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
        merged_data['SAFETY_SCORE_1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['SAFETY_SCORE_2'] = merged_data['IMG2'].map(safety_dict)
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['SAFETY_SCORE_1'].fillna(mean_safety, inplace=True)
        merged_data['SAFETY_SCORE_2'].fillna(mean_safety, inplace=True)
        
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

    def create_demographic_dummy_variables(self, data):
        """Create dummy variables for demographic categories and safety interactions."""
        print(f"Creating dummy variables and interaction terms for group: {self.model_group}...")
        
        interaction_features = []
        main_effect_features = []
        
        def create_dummies(feature, ref_cat):
            cat_col = f'{feature}_cat'
            if cat_col not in data.columns: return []
            
            cats = [c for c in data[cat_col].unique() if pd.notna(c) and c != ref_cat]
            for cat in cats:
                # Labels from demographic_mappings are already sanitized
                dummy_name = cat
                
                data[f'{dummy_name}_1'] = (data[cat_col] == cat).astype(int)
                data[f'{dummy_name}_2'] = data[f'{dummy_name}_1']
                main_effect_features.append(dummy_name)
                
                interact_name = f"safety_{dummy_name}"
                data[f'{interact_name}_1'] = data['SAFETY_SCORE_1'] * data[f'{dummy_name}_1']
                data[f'{interact_name}_2'] = data['SAFETY_SCORE_2'] * data[f'{dummy_name}_2']
                interaction_features.append(interact_name)
            return cats

        for demo in self.demographic_variables:
            # The logic to determine the reference category is now centralized here.
            # It finds the lowest numeric key present in the actual data for that demographic.
            if f'{demo}_cat' in data.columns and data[f'{demo}_cat'].notna().any():
                
                # Get the numeric keys corresponding to the categories present in the data
                reverse_mapping = {v: k for k, v in self.demographic_mappings[demo].items()}
                present_cats = data[f'{demo}_cat'].dropna().unique()
                data_keys = [reverse_mapping[cat] for cat in present_cats if cat in reverse_mapping]
                
                if data_keys:
                    ref_key = min(data_keys)
                    ref_cat = self.demographic_mappings[demo][ref_key]
                    create_dummies(demo, ref_cat)
                else:
                    print(f"Warning: Could not find a valid reference category for '{demo}'. Skipping dummy creation.")

        return data, interaction_features, main_effect_features

    def estimate_interaction_model(self):
        """Estimate the MXL safety * demographics interaction model."""
        print("\nEstimating Safety * Demographics Interaction Model (MXL)...")
        
        train_data = self.merged_data[self.merged_data['train'] == 1].copy()
        test_data = self.merged_data[self.merged_data['test'] == 1].copy()
        
        train_data, interaction_features, main_effect_features = self.create_demographic_dummy_variables(train_data)
        
        # The test data needs to have the same columns as the training data.
        # We create dummies on test, then align columns.
        test_data, _, _ = self.create_demographic_dummy_variables(test_data)
        
        # Align columns - crucial for consistent model evaluation
        train_cols = set(train_data.columns)
        test_cols = set(test_data.columns)
        
        missing_in_test = train_cols - test_cols
        for c in missing_in_test:
            test_data[c] = 0
            
        extra_in_test = test_cols - train_cols
        test_data = test_data.drop(columns=list(extra_in_test))
        
        # Ensure order is the same
        test_data = test_data[train_data.columns]

        features = self.original_model_features + interaction_features + main_effect_features + ['SAFETY_SCORE']
        
        # Prepare panel data
        _, biodata_wide, obs_per_ind = prepare_panel_data(train_data, self.individual_id, 'CHOICE', features)
        
        # Define parameters and utility
        random_params_config = {
            'TT': {'mean_init': -0.2, 'sigma_init': 0.1}, 'TL': {'mean_init': -0.3, 'sigma_init': 0.1},
            'SAFETY_SCORE': {'mean_init': 1.0, 'sigma_init': 0.1}
        }
        random_params = {p: (Beta(f'B_{p}', c['mean_init'], None,None,0) + Beta(f'sigma_{p}', c['sigma_init'], None,None,0) * bioDraws(f'{p}_rnd', 'NORMAL_HALTON2')) 
                         for p, c in random_params_config.items()}
        
        fixed_features = self.original_model_features + interaction_features + main_effect_features
        fixed_params = {f: Beta(f"B_{f.replace(' - ', '___').replace(' ', '_')}", 0, None,None,0) for f in fixed_features}

        V = []
        for q in range(obs_per_ind):
            V1, V2 = 0, 0
            # Random
            for name, param in random_params.items():
                scale = 10 if name == 'TT' else (3 if name == 'TL' else 1)
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name) / scale
                    V2 += param * Variable(v2_name) / scale
            # Fixed
            for name, param in fixed_params.items():
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name)
                    V2 += param * Variable(v2_name)
            V.append({1: V1, 2: V2})

        # Estimate and simulate
        model_name = f'demographics_interaction_{self.model_group}'
        results = estimate_mxl(V, {1:1, 2:1}, 'CHOICE', obs_per_ind, self.num_draws, biodata_wide, model_name, self.output_dir)
        _, test_biodata_wide, _ = prepare_panel_data(test_data, self.individual_id, 'CHOICE', features)
        test_sim_results = simulate_mxl(V, {1:1,2:1}, 'CHOICE', obs_per_ind, self.num_draws, test_biodata_wide, results.get_beta_values(), model_name)
        
        self.results = (results, test_sim_results, obs_per_ind)
        print_mxl_results(results)

    def generate_results_table(self):
        """Generates and saves a LaTeX table for the interaction model."""
        if not hasattr(self, 'results'):
            print("Model not estimated yet. Cannot generate table.")
            return

        print("\nGenerating results table...")
        train_res, test_res, obs_per_ind = self.results
        
        train_metrics = extract_mxl_metrics(train_res, obs_per_ind, train_res.data.numberOfObservations)
        params = train_res.get_estimated_parameters()
        all_param_names = sorted(list(params.index))

        def format_p(p):
            return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        table_title = f"Safety-Demographics Interaction Model (MXL) - {self.model_group.replace('_', ' ').title()}"
        table_label = f"tab:demographics_interaction_{self.model_group}"
        
        lines = [
            "\\begin{table}[htbp]", "\\centering", f"\\caption{{{table_title}}}",
            f"\\label{{{table_label}}}", "\\resizebox{0.8\\textwidth}{!}{%",
            "\\begin{tabular}{lc}", "\\toprule",
            "& \\textbf{Coefficient (t-stat)} \\\\", "\\midrule",
            "\\multicolumn{2}{l}{\\textit{Goodness of fit}} \\\\", "\\hline",
            f"Sample size (Train) & {train_metrics['n_observations']} \\\\",
            f"Sample size (Test) & {test_res['n_observations']} \\\\",
            f"Log-Likelihood (Train) & {train_metrics['log_likelihood']:.2f} \\\\",
            f"Log-Likelihood (Test) & {test_res['log_likelihood']:.2f} \\\\",
            f"Rho-squared (Train) & {train_metrics['pseudo_r2']:.4f} \\\\",
            f"Rho-squared (Test) & {test_res['pseudo_r2']:.4f} \\\\",
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
        with open(table_path, 'w') as f: f.write(latex_content)
        print(f"LaTeX table saved to {table_path}")
    
    def run_analysis(self):
        """Run the complete safety * demographics interaction analysis"""
        self.load_trained_model_data()
        self.load_and_prepare_data()
        self.estimate_interaction_model()
        # self.generate_results_table()
        print(f"\n✓ Analysis complete. Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run Safety-Demographics Interaction Model (MXL)')
    parser.add_argument('--model_path', type=str, 
                        default='reports/models/mxl_choice_20250709_183213/final_full_model.pickle',
                        help='Path to the trained base model pickle file')
    args = parser.parse_args()

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
        print(f"\n----- Running analysis for group: {group_name} -----")
        
        group_output_dir = base_output_dir / group_name

        interaction_model = SafetyDemographicsInteractionModel(
            model_path=args.model_path,
            model_group=group_name,
            demographic_variables=variables,
            output_dir=group_output_dir
        )
        interaction_model.run_analysis()

if __name__ == "__main__":
    main() 