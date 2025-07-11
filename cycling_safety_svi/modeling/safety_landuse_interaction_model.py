"""
Safety-Landuse Interaction Model (Mixed Logit Version)

This script extends a trained choice model with safety * landuse interaction effects
using a mixed logit (MXL) formulation.

Usage:
    python safety_landuse_interaction_model.py --model_path path/to/final_model.pickle
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta, Variable, log, exp, bioDraws

from mxl_functions import (
    estimate_mxl, simulate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results
)


class SafetyLanduseInteractionModel:
    """Extends a trained choice model with safety * landuse interaction effects using MXL."""
    
    def __init__(self, model_path, main_design_path='data/raw/main_design.csv', output_dir='reports/models/interaction'):
        self.model_path = Path(model_path)
        self.main_design_path = Path(main_design_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"safety_landuse_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Safety-Landuse Interaction Model (MXL)")
        print(f"Base model path: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        
        logging.getLogger('biogeme').setLevel(logging.WARNING)

        self.landuse_categories = ['Wijkontslu', 'Industriet', 'Woongebied', 'Hoofdweg', 'Recreatie']
        self.feature_name_mapping = {
            'B_LANE_MARKING___GENERAL': 'Lane Marking - General',
            'B_TRAFFIC_SIGN_(FRONT)': 'Traffic Sign (Front)',
            'B_UTILITY_POLE': 'Utility Pole'
        }
        
        self.num_draws = 1000
        self.individual_id = 'RID'
        self.min_obs_per_individual = 15
        
    def load_trained_model_data(self):
        """Load the trained model and extract its segmentation features."""
        print("\nLoading base model to extract segmentation features...")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_results = pickle.load(f)
        
        self.original_model_features = self._extract_model_features(model_results)
        print(f"✓ Extracted {len(self.original_model_features)} segmentation features from base model.")

    def _extract_model_features(self, model_results):
        """Extract segmentation features used in the original trained model."""
        param_names = list(model_results.betaNames)
        
        segmentation_features = []
        for param_name in param_names:
            if param_name.startswith('B_') and param_name not in ['B_TT', 'B_TL', 'B_SAFETY_SCORE'] and not param_name.startswith('sigma_'):
                feature_name = self.feature_name_mapping.get(param_name, param_name.replace('B_', ''))
                segmentation_features.append(feature_name)
        return segmentation_features

    def load_and_prepare_data(self, cv_dcm_path='data/raw/cv_dcm.csv',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv'):
        """Load and prepare all datasets, including land use data."""
        print("\nLoading and preparing all datasets...")
        self.choice_data = pd.read_csv(cv_dcm_path)

        print("\nApplying data cleaning steps...")
        self.choice_data = apply_data_cleaning(
            self.choice_data,
            individual_id=self.individual_id,
            min_obs=self.min_obs_per_individual,
            fix_problematic_rid=True
        )

        self.main_design = pd.read_csv(self.main_design_path)
        self.safety_scores = pd.read_csv(safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        
        if self.original_model_features:
            seg_chunks = [chunk for chunk in pd.read_csv(segmentation_path, chunksize=1000)]
            self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
            self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
        else:
            self.segmentation_data = None
        
        self._merge_datasets_with_landuse()
        print(f"Final dataset prepared with {len(self.merged_data)} observations.")

    def _merge_datasets_with_landuse(self):
        """Merge all datasets including land use information."""
        merged_data = self.choice_data.copy()
        
        safety_dict = dict(zip(self.safety_scores['image_name'], self.safety_scores['safety_score']))
        merged_data['safety_score_1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['safety_score_2'] = merged_data['IMG2'].map(safety_dict)
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['safety_score_1'].fillna(mean_safety, inplace=True)
        merged_data['safety_score_2'].fillna(mean_safety, inplace=True)

        img_to_task = {}
        for _, row in self.main_design.iterrows():
            if pd.notna(row['alt1_imageid']):
                img_to_task[row['alt1_imageid'] + '.jpg'] = row['alt1_buildenvironment']
            if pd.notna(row['alt2_imageid']):
                img_to_task[row['alt2_imageid'] + '.jpg'] = row['alt2_buildenvironment']
        
        merged_data['landuse_1'] = merged_data['IMG1'].map(img_to_task)
        merged_data['landuse_2'] = merged_data['IMG2'].map(img_to_task)
        # Fill NA landuse with mode, but only after mapping
        if not merged_data['landuse_1'].empty:
            most_common = merged_data['landuse_1'].mode()[0]
            merged_data['landuse_1'].fillna(most_common, inplace=True)
            merged_data['landuse_2'].fillna(most_common, inplace=True)

        if self.segmentation_data is not None:
            segmentation_dict = {row['filename_key'] + '.jpg': row.drop('filename_key').to_dict() 
                                 for _, row in self.segmentation_data.iterrows() if pd.notna(row['filename_key'])}
            for feature in self.original_model_features:
                feature_col = feature.replace(' ', '_').replace('___', ' - ')
                merged_data[f"{feature_col}_1"] = merged_data['IMG1'].map(lambda x: segmentation_dict.get(x, {}).get(feature_col, 0))
                merged_data[f"{feature_col}_2"] = merged_data['IMG2'].map(lambda x: segmentation_dict.get(x, {}).get(feature_col, 0))

        self.merged_data = merged_data
    
    def create_landuse_dummy_variables(self, data):
        """Create dummy variables for land use categories and safety interactions."""
        print("Creating dummy variables and interaction terms...")
        reference_category = 'Woongebied'
        interaction_features = []
        main_effect_features = []

        for category in self.landuse_categories:
            if category != reference_category:
                main_effect_name = f'landuse_{category}'
                main_effect_features.append(main_effect_name)
                data[f'{main_effect_name}_1'] = (data['landuse_1'] == category).astype(int)
                data[f'{main_effect_name}_2'] = (data['landuse_2'] == category).astype(int)

                interact_name = f"safety_landuse_{category}"
                interaction_features.append(interact_name)
                data[f'{interact_name}_1'] = data['safety_score_1'] * data[f'{main_effect_name}_1']
                data[f'{interact_name}_2'] = data['safety_score_2'] * data[f'{main_effect_name}_2']
                
        return data, main_effect_features, interaction_features

    def estimate_interaction_model(self):
        """Estimate the MXL safety * landuse interaction model."""
        print("\nEstimating Safety * Landuse Interaction Model (MXL)...")
        
        train_data = self.merged_data[self.merged_data['train'] == 1].copy()
        test_data = self.merged_data[self.merged_data['test'] == 1].copy()
        
        train_data, main_effects, interaction_effects = self.create_landuse_dummy_variables(train_data)
        test_data, _, _ = self.create_landuse_dummy_variables(test_data)

        # Rename safety columns for consistency with panel data preparation
        train_data.rename(columns={'safety_score_1': 'SAFETY_SCORE_1', 'safety_score_2': 'SAFETY_SCORE_2'}, inplace=True)
        test_data.rename(columns={'safety_score_1': 'SAFETY_SCORE_1', 'safety_score_2': 'SAFETY_SCORE_2'}, inplace=True)

        base_features = ['TT', 'TL', 'SAFETY_SCORE']
        seg_features = [f.replace(' ', '_').replace('___', ' - ') for f in self.original_model_features]
        all_features = base_features + seg_features + main_effects + interaction_effects
        
        _, biodata_wide, obs_per_ind = prepare_panel_data(train_data, self.individual_id, 'CHOICE', all_features)
        
        random_params_config = {
            'TT': {'mean_init': -0.2, 'sigma_init': 0.1}, 
            'TL': {'mean_init': -0.3, 'sigma_init': 0.1},
            'SAFETY_SCORE': {'mean_init': 1.0, 'sigma_init': 0.1}
        }
        random_params = {
            p: (Beta(f'B_{p}', c['mean_init'], None, None, 0) + 
                Beta(f'sigma_{p}', c['sigma_init'], None, None, 0) * bioDraws(f'{p}_rnd', 'NORMAL_HALTON2')) 
            for p, c in random_params_config.items()
        }
        
        fixed_features = seg_features + main_effects + interaction_effects
        fixed_params = {f: Beta(f"B_{f.replace(' - ', '___').replace(' ', '_')}", 0, None, None, 0) for f in fixed_features}

        V = []
        for q in range(obs_per_ind):
            V1, V2 = 0, 0
            # Random parameters (TT, TL, SAFETY_SCORE)
            for name, param in random_params.items():
                scale = 10 if name == 'TT' else (3 if name == 'TL' else 1)
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables and v2_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name) / scale
                    V2 += param * Variable(v2_name) / scale
            
            # Fixed parameters (segmentation, landuse main effects, interactions)
            for name, param in fixed_params.items():
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables and v2_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name)
                    V2 += param * Variable(v2_name)
            V.append({1: V1, 2: V2})

        results = estimate_mxl(V, {1:1, 2:1}, 'CHOICE', obs_per_ind, self.num_draws, biodata_wide, 'landuse_interaction', self.output_dir)
        
        _, test_biodata_wide, _ = prepare_panel_data(test_data, self.individual_id, 'CHOICE', all_features)
        
        test_sim_results = simulate_mxl(V, {1:1,2:1}, 'CHOICE', obs_per_ind, self.num_draws, test_biodata_wide, results.get_beta_values(), 'landuse_interaction')
        
        self.results = (results, test_sim_results, obs_per_ind)
        print_mxl_results(results)

    def generate_results_table(self):
        """Generates and saves a LaTeX table for the interaction model."""
        if not hasattr(self, 'results'): return
        print("\nGenerating results table...")
        train_res, test_res, obs_per_ind = self.results
        
        train_metrics = extract_mxl_metrics(train_res, obs_per_ind, train_res.data.numberOfObservations)
        params = train_res.get_estimated_parameters()
        all_param_names = sorted(list(params.index))

        def format_p(p):
            return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        lines = [
            "\\begin{table}[htbp]",
            "    \\centering",
            "    \\caption{Safety-Landuse Interaction Model (MXL)}",
            "    \\label{tab:landuse_interaction}",
            "    \\resizebox{0.8\\textwidth}{!}{%",
            "    \\begin{tabular}{lc}",
            "        \\toprule",
            "        & \\textbf{Coefficient (t-stat)} \\\\",
            "        \\midrule",
            "        \\multicolumn{2}{l}{\\textit{Goodness of fit}} \\\\",
            "        \\hline",
            f"        Sample size (Train) & {train_metrics['n_observations']} \\\\",
            f"        Sample size (Test) & {test_res['n_observations']} \\\\",
            f"        Log-Likelihood (Train) & {train_metrics['log_likelihood']:.2f} \\\\",
            f"        Log-Likelihood (Test) & {test_res['log_likelihood']:.2f} \\\\",
            f"        Rho-squared (Train) & {train_metrics['pseudo_r2']:.4f} \\\\",
            f"        Rho-squared (Test) & {test_res['pseudo_r2']:.4f} \\\\",
            "        \\hline",
            "        \\multicolumn{2}{l}{\\textit{Parameters}} \\\\",
            "        \\hline"
        ]

        for param in all_param_names:
            p_name_latex = param.replace('_', '\\_')
            val = params.loc[param, 'Value']
            t = params.loc[param, 'Rob. t-test']
            p = params.loc[param, 'Rob. p-value']
            stars = format_p(p)
            val_str = f"{val:.3f}{stars} ({t:.2f})"
            lines.append(f"        {p_name_latex} & {val_str} \\\\")
        
        lines.extend([
            "        \\hline",
            "        \\bottomrule",
            "        \\multicolumn{2}{l}{\\textsuperscript{***}p<0.001, \\textsuperscript{**}p<0.01, \\textsuperscript{*}p<0.05}",
            "    \\end{tabular}}",
            "\\end{table}"
        ])

        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'landuse_interaction_model.tex'
        with open(table_path, 'w') as f: f.write(latex_content)
        print(f"LaTeX table saved to {table_path}")

    def run_analysis(self):
        """Run the complete safety * landuse interaction analysis."""
        self.load_trained_model_data()
        self.load_and_prepare_data()
        self.estimate_interaction_model()
        self.generate_results_table()
        print(f"\n✓ Analysis complete. Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run Safety-Landuse Interaction Model (MXL)')
    parser.add_argument('--model_path', type=str,
                        default='reports/models/mxl_choice_20250709_183213/final_full_model.pickle',
                        help='Path to the trained base model pickle file')
    args = parser.parse_args()
    
    interaction_model = SafetyLanduseInteractionModel(model_path=args.model_path)
    interaction_model.run_analysis()

if __name__ == "__main__":
    main() 