"""
Choice Model Benchmarking Script (Mixed Logit)

This script implements and benchmarks different mixed logit discrete choice models:
1. Base model: only TL1, TT1, TL2, TT2 (with random parameters)
2. Base + safety scores: TL1, TT1, TL2, TT2, SAFETY_SCORE (with random parameters)
3. Base + segmentation: TL1, TT1, TL2, TT2, pixel ratios (TL/TT random, segmentation fixed)
4. Base + safety + segmentation: TL1, TT1, TL2, TT2, SAFETY_SCORE, pixel ratios

Uses mixed logit models with random parameters for TT, TL, and SAFETY_SCORE.
Compares models using Log-Likelihood, BIC, AIC, and Pseudo R-squared metrics.
"""

import os
import pandas as pd
import numpy as np
import scipy.stats
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import logging
from biogeme.expressions import Beta, Variable, log, exp, bioDraws
from pathlib import Path

from datetime import datetime

# Import MXL functions
from mxl_functions import (
    estimate_mxl, simulate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results
)


class ChoiceModelBenchmark:
    """Benchmarks different mixed logit discrete choice models with various feature combinations"""
    
    def __init__(self, base_output_dir='reports/models'):
        """Initializes the benchmark environment, setting up output directories and logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_folder = f"mxl_choice_{timestamp}"
        self.output_dir = Path(base_output_dir) / timestamped_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Mixed Logit Model Results will be saved to: {self.output_dir}")
        
        logging.getLogger('biogeme').setLevel(logging.WARNING)
        
        self.num_draws = 1000
        self.individual_id = 'RID'
        self.min_obs_per_individual = 15
        
    def load_and_prepare_data(self, 
                             cv_dcm_path='data/raw/cv_dcm.csv',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv',
                             original_results_path='data/raw/df_choice_with_Vimg.csv'):
        """Load and prepare all datasets for mixed logit modeling with data cleaning"""
        
        print("Loading datasets...")
        
        # Load main choice data
        self.choice_data = pd.read_csv(cv_dcm_path)
        print(f"Loaded choice data: {len(self.choice_data)} observations")
        
        # Apply data cleaning steps from reference implementation
        print("\nApplying data cleaning steps...")
        self.choice_data = apply_data_cleaning(
            self.choice_data, 
            individual_id=self.individual_id,
            min_obs=self.min_obs_per_individual,
            fix_problematic_rid=True
        )
        
        # Reset index after cleaning to ensure contiguous indices
        self.choice_data = self.choice_data.reset_index(drop=True)
        print(f"Index reset after cleaning. New shape: {self.choice_data.shape}")
        
        # Print train/test split after cleaning
        train_count = self.choice_data[self.choice_data['train'] == 1].shape[0]
        test_count = self.choice_data[self.choice_data['test'] == 1].shape[0]
        print(f"After cleaning - Training observations: {train_count}, Test observations: {test_count}")

        # Load safety scores
        self.safety_scores = pd.read_csv(safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        print(f"Loaded safety scores: {len(self.safety_scores)} images")
        
        # Load segmentation data
        print("Loading segmentation data...")
        # Read in chunks due to large file size
        seg_chunks = []
        for chunk in pd.read_csv(segmentation_path, chunksize=1000):
            seg_chunks.append(chunk)
        self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
        self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
        print(f"Loaded segmentation data: {len(self.segmentation_data)} images")
        
        # Load original model results for comparison
        self.original_results = pd.read_csv(original_results_path)
        print(f"Loaded original results: {len(self.original_results)} observations")
        
        # Prepare merged dataset
        self._merge_datasets()
        
    def _merge_datasets(self):
        """Merge all datasets into a single dataframe for modeling"""
        
        print("Merging datasets...")
        
        # Start with choice data
        merged_data = self.choice_data.copy()
        
        # Add safety scores for both alternatives
        safety_dict = dict(zip(self.safety_scores['image_name'], self.safety_scores['safety_score']))
        
        merged_data['safety_score_1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['safety_score_2'] = merged_data['IMG2'].map(safety_dict)
        
        # Prepare segmentation data more efficiently
        print("Processing segmentation data...")
        
        # Create a more efficient lookup dictionary
        segmentation_dict = {}
        for _, row in self.segmentation_data.iterrows():
            img_name = row['filename_key'] + '.jpg'
            if pd.isna(row['filename_key']):
                continue
            # Store all segmentation features for this image
            features = row.drop('filename_key').to_dict()
            segmentation_dict[img_name] = features
        
        # Get all segmentation feature names (excluding filename_key)
        seg_feature_names = [col for col in self.segmentation_data.columns if col != 'filename_key']
        
        # Prepare containers for new columns
        new_columns_data = {}
        
        # Initialize all segmentation columns
        for feature in seg_feature_names:
            new_columns_data[f"{feature}_1"] = [None] * len(merged_data)
            new_columns_data[f"{feature}_2"] = [None] * len(merged_data)
        
        # Fill segmentation features efficiently
        for idx, row in merged_data.iterrows():
            # Process IMG1
            img1 = row['IMG1']
            if img1 in segmentation_dict:
                for feature, value in segmentation_dict[img1].items():
                    new_columns_data[f"{feature}_1"][idx] = value
            
            # Process IMG2
            img2 = row['IMG2']
            if img2 in segmentation_dict:
                for feature, value in segmentation_dict[img2].items():
                    new_columns_data[f"{feature}_2"][idx] = value
        
        # Add all new columns at once using pd.concat
        new_columns_df = pd.DataFrame(new_columns_data, index=merged_data.index)
        merged_data = pd.concat([merged_data, new_columns_df], axis=1)
        
        # Fill missing values with 0 for segmentation features
        seg_cols = [col for col in merged_data.columns if col.endswith('_1') or col.endswith('_2')]
        for col in seg_cols:
            if col not in ['TL1', 'TT1', 'TL2', 'TT2', 'safety_score_1', 'safety_score_2']:
                merged_data[col] = merged_data[col].fillna(0)
        
        # Fill missing safety scores with mean
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['safety_score_1'] = merged_data['safety_score_1'].fillna(mean_safety)
        merged_data['safety_score_2'] = merged_data['safety_score_2'].fillna(mean_safety)
        
        self.merged_data = merged_data
        print(f"Merged dataset created: {len(merged_data)} observations, {len(merged_data.columns)} features")
        
        # Get list of available segmentation features
        self.seg_features = []
        for col in merged_data.columns:
            if col.endswith('_1') and not col.startswith(('TL', 'TT', 'safety_score')):
                feature_name = col[:-2]  # Remove '_1' suffix
                self.seg_features.append(feature_name)

        print(f"Available segmentation features: {len(self.seg_features)}")
        
        # Filter and scale segmentation features
        self._filter_and_scale_segmentation_features()
        
    def _filter_and_scale_segmentation_features(self, variance_threshold=1e-6, scale_features=False):
        """
        Filter out segmentation features with very low variance and optionally apply z-score scaling
        
        Args:
            variance_threshold: Minimum variance threshold for feature inclusion (applied after scaling if enabled).
                              Default 0.01 means keeping features with at least 1% of expected variance.
            scale_features: Whether to apply z-score scaling to features before filtering. Default False.
        """
        print("Filtering segmentation features...")
        
        # Get segmentation columns
        seg_cols_1 = [f"{feature}_1" for feature in self.seg_features if f"{feature}_1" in self.merged_data.columns]
        seg_cols_2 = [f"{feature}_2" for feature in self.seg_features if f"{feature}_2" in self.merged_data.columns]
        
        all_seg_cols = seg_cols_1 + seg_cols_2
        
        if not all_seg_cols:
            print("No segmentation features found to process")
            return
        
        original_feature_count = len(self.seg_features)
        
        # STEP 1: Apply z-score scaling if enabled
        scaled_features = []
        if scale_features:
            print("Applying z-score scaling to segmentation features...")
            
            for feature in self.seg_features:
                col1 = f"{feature}_1"
                col2 = f"{feature}_2"
                
                if col1 in self.merged_data.columns and col2 in self.merged_data.columns:
                    # Combine data from both alternatives for fitting the scaler
                    combined_values = pd.concat([
                        self.merged_data[col1].dropna(),
                        self.merged_data[col2].dropna()
                    ])
                    
                    # Check if there's any variation in the raw data
                    if combined_values.std() == 0:
                        print(f"Skipping feature '{feature}' - no variation in raw data")
                        continue
                    
                    # Fit scaler on combined data
                    scaler = StandardScaler()
                    scaler.fit(combined_values.values.reshape(-1, 1))
                    
                    # Transform both alternatives
                    # Handle NaN values by preserving them
                    mask1 = ~self.merged_data[col1].isna()
                    mask2 = ~self.merged_data[col2].isna()
                    
                    if mask1.any():
                        self.merged_data.loc[mask1, col1] = scaler.transform(
                            self.merged_data.loc[mask1, col1].values.reshape(-1, 1)
                        ).flatten()
                    
                    if mask2.any():
                        self.merged_data.loc[mask2, col2] = scaler.transform(
                            self.merged_data.loc[mask2, col2].values.reshape(-1, 1)
                        ).flatten()
                    
                    scaled_features.append(feature)
            
            print(f"Z-score scaling applied to {len(scaled_features)} segmentation features")
        else:
            scaled_features = self.seg_features
        
        # STEP 2: Calculate variances and filter
        print("Calculating variances for filtering...")
        
        variances = {}
        features_to_keep = []
        
        for feature in scaled_features:
            col1 = f"{feature}_1"
            col2 = f"{feature}_2"
            
            if col1 in self.merged_data.columns and col2 in self.merged_data.columns:
                # Calculate combined variance across both alternatives
                combined_values = pd.concat([
                    self.merged_data[col1].dropna(),
                    self.merged_data[col2].dropna()
                ])
                
                feature_variance = combined_values.var()
                variances[feature] = feature_variance
                
                # Keep feature if variance is above threshold
                if feature_variance > variance_threshold:
                    features_to_keep.append(feature)
                else:
                    print(f"Removing feature '{feature}' due to low variance: {feature_variance:.2e}")
        
        print(f"Features before variance filtering: {len(scaled_features)}")
        print(f"Features after variance filtering: {len(features_to_keep)}")
        
        # Update the list of available segmentation features
        self.seg_features = features_to_keep
        
        if not self.seg_features:
            print("Warning: No segmentation features remaining after variance filtering.")
        
        # Store feature statistics for reference
        self.feature_stats = {
            'original_feature_count': original_feature_count,
            'scaled_feature_count': len(scaled_features),
            'filtered_feature_count': len(features_to_keep),
            'removed_features_no_variation': [f for f in self.seg_features if f not in scaled_features],
            'removed_features_low_variance': [f for f in scaled_features if f not in features_to_keep],
            'feature_variances': variances,
            'variance_threshold': variance_threshold,
            'scaling_applied': scale_features,
            'processing_order': 'z-score_scaling_then_variance_filtering' if scale_features else 'variance_filtering_only'
        }
        
        print(f"Final segmentation features available: {len(self.seg_features)}")
        
    def _sanitize_name_for_beta(self, feature_name):
        """Creates a Biogeme-compatible name for a beta parameter."""
        s_name = feature_name.replace(' - ', '___').replace(' ', '_')
        return f"B_{s_name}"

    def run_backward_elimination(self, significance_level=0.05):
        """
        Performs backward elimination feature selection using MNL to find significant variables.
        """
        print("\nStarting backward elimination for feature selection...")
        
        # Create a mapping from beta names back to original feature names
        beta_to_feature_map = {
            self._sanitize_name_for_beta(f): f for f in self.seg_features
        }
        beta_to_feature_map['B_SAFETY_SCORE'] = 'SAFETY_SCORE'
        
        train_data = self.merged_data[self.merged_data['train'] == 1].copy()
        
        # Start with all available segmentation features + safety
        features_to_consider = self.seg_features + ['SAFETY_SCORE']
        
        while True:
            print(f"Testing model with {len(features_to_consider)} features...")
            
            # Prepare data for the current set of features
            attributes = ['TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
            current_seg_features = [f for f in features_to_consider if f in self.seg_features]
            
            for feature in current_seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
            
            if 'SAFETY_SCORE' in features_to_consider:
                attributes.extend(['safety_score_1', 'safety_score_2'])
                
            model_data = train_data[attributes].copy().dropna()
            
            database = db.Database('backward_elimination', model_data)
            
            # Define variables
            V1_comps, V2_comps = [], []
            variables = {col: Variable(col) for col in model_data.columns if col != 'CHOICE'}
            
            # Base variables
            B_TT = Beta('B_TT', -0.2, None, None, 0)
            B_TL = Beta('B_TL', -0.3, None, None, 0)
            V1_comps.extend([B_TT * variables['TT1'] / 10, B_TL * variables['TL1'] / 3])
            V2_comps.extend([B_TT * variables['TT2'] / 10, B_TL * variables['TL2'] / 3])

            # Segmentation and safety variables
            betas = {'TT': B_TT, 'TL': B_TL}
            for feature in features_to_consider:
                beta_name = (
                    'B_SAFETY_SCORE' if feature == 'SAFETY_SCORE' 
                    else self._sanitize_name_for_beta(feature)
                )
                beta = Beta(beta_name, 0, None, None, 0)
                betas[feature] = beta
                
                v1_name = 'safety_score_1' if feature == 'SAFETY_SCORE' else f"{feature}_1"
                v2_name = 'safety_score_2' if feature == 'SAFETY_SCORE' else f"{feature}_2"
                
                if v1_name in variables and v2_name in variables:
                    V1_comps.append(beta * variables[v1_name])
                    V2_comps.append(beta * variables[v2_name])

            # Estimate MNL model
            V = {1: sum(V1_comps), 2: sum(V2_comps)}
            prob = models.logit(V, {1: 1, 2: 1}, Variable('CHOICE'))
            biogeme = bio.BIOGEME(database, log(prob))
            biogeme.modelName = f"elimination_{len(features_to_consider)}"
            results = biogeme.estimate(verbose=False)
            
            # Check for significance using robust p-values
            params_df = results.get_estimated_parameters()
            p_values = params_df['Rob. p-value']

            # Exclude base parameters from elimination
            p_values_to_check = p_values.drop(['B_TT', 'B_TL'], errors='ignore')

            if p_values_to_check.empty:
                print("No more features to eliminate.")
                break

            max_p_value = p_values_to_check.max()
            
            if max_p_value > significance_level:
                feature_to_remove_beta = p_values_to_check.idxmax()
                feature_to_remove = beta_to_feature_map[feature_to_remove_beta]
                print(f"Removing feature '{feature_to_remove}' with p-value: {max_p_value:.4f}")
                features_to_consider.remove(feature_to_remove)
            else:
                print("All remaining features are significant.")
                break
        
        self.final_significant_features = features_to_consider
        print(f"Final significant features: {self.final_significant_features}")
        return self.final_significant_features
        
    def estimate_all_models(self):
        """
        Estimates all benchmark models:
        1. Base Model (TT, TL random)
        2. Base + Safety Model (TT, TL, SAFETY_SCORE random)
        3. Full Model (Base + Safety + Significant Segmentation features)
        4. Segmentation-Only Model (Base + Significant Segmentation features)
        """
        print("\nEstimating all benchmark models...")

        # 1. Base Model
        print("Estimating Base Model...")
        self.base_model_results = self._estimate_final_mxl(
            features=[], model_name='base_model'
        )
        print_mxl_results(self.base_model_results[0])

        # 2. Base + Safety Model
        print("\nEstimating Base + Safety Model...")
        self.base_safety_model_results = self._estimate_final_mxl(
            features=['SAFETY_SCORE'], model_name='base_safety_model'
        )
        print_mxl_results(self.base_safety_model_results[0])

        if not hasattr(self, 'final_significant_features'):
            print("Backward elimination must be run first to estimate segmentation models.")
            return

        # 3. Full Model (with safety if significant)
        print("\nEstimating Full Final Model...")
        full_model_features = self.final_significant_features
        self.full_model_results = self._estimate_final_mxl(
            full_model_features, 'final_full_model'
        )
        print_mxl_results(self.full_model_results[0])

        # 4. Segmentation-Only Model (safety removed)
        print("\nEstimating Segmentation-Only Final Model...")
        seg_only_features = [f for f in full_model_features if f != 'SAFETY_SCORE']
        self.seg_only_model_results = self._estimate_final_mxl(
            seg_only_features, 'final_seg_only_model'
        )
        print_mxl_results(self.seg_only_model_results[0])

    def _estimate_final_mxl(self, features, model_name):
        """Helper to estimate a single final MXL model."""
        train_data = self.merged_data[self.merged_data['train'] == 1].copy()
        
        attributes = [self.individual_id, 'TL1', 'TT1', 'TL2', 'TT2', 'CHOICE', 'train', 'test']
        seg_features = [f for f in features if f not in ['TT', 'TL', 'SAFETY_SCORE']]
        
        for feature in seg_features:
            attributes.extend([f"{feature}_1", f"{feature}_2"])
        
        if 'SAFETY_SCORE' in features:
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        model_data = self.merged_data[attributes].copy().dropna()
        model_data = model_data.rename(columns={
            'safety_score_1': 'SAFETY_SCORE1',
            'safety_score_2': 'SAFETY_SCORE2'
        })

        _, biodata_wide, obs_per_ind = prepare_panel_data(
            model_data[model_data['train'] == 1], self.individual_id, 'CHOICE'
        )

        # Define random parameters
        random_params_config = {
            'TT': {'mean_init': -0.2, 'sigma_init': 0.1, 'dist': 'normal'},
            'TL': {'mean_init': -0.3, 'sigma_init': 0.1, 'dist': 'normal'}
        }
        if 'SAFETY_SCORE' in features:
            random_params_config['SAFETY_SCORE'] = {'mean_init': 1.0, 'sigma_init': 0.1, 'dist': 'normal'}
        
        random_params = {}
        for param, config in random_params_config.items():
            mean = Beta(f'B_{param}', config['mean_init'], None, None, 0)
            sigma = Beta(f'sigma_{param}', config['sigma_init'], None, None, 0)
            draws = bioDraws(f'{param}_rnd', 'NORMAL_HALTON2')
            random_params[param] = mean + sigma * draws
            
        # Define fixed parameters
        fixed_params = {
            f: Beta(self._sanitize_name_for_beta(f), 0, None, None, 0) 
            for f in seg_features
        }

        # Define utility functions
        V = []
        for q in range(obs_per_ind):
            V1, V2 = 0, 0
            # Random params
            for name, param in random_params.items():
                v1_name, v2_name = f"{name}1_{q}", f"{name}2_{q}"
                if v1_name in biodata_wide.variables:
                    scale = 10 if name == 'TT' else (3 if name == 'TL' else 1)
                    V1 += param * Variable(v1_name) / scale
                    V2 += param * Variable(v2_name) / scale
            # Fixed params
            for name, param in fixed_params.items():
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name)
                    V2 += param * Variable(v2_name)
            V.append({1: V1, 2: V2})

        results = estimate_mxl(V, {1:1, 2:1}, 'CHOICE', obs_per_ind, self.num_draws, biodata_wide, model_name, self.output_dir)

        # Simulate on test data
        _, test_biodata_wide, _ = prepare_panel_data(
            model_data[model_data['test'] == 1], self.individual_id, 'CHOICE'
        )
        test_sim_results = simulate_mxl(
            V, {1: 1, 2: 1}, 'CHOICE', obs_per_ind, self.num_draws,
            test_biodata_wide, results.get_beta_values(), model_name
        )

        return results, test_sim_results, obs_per_ind

    def generate_results_table(self):
        """Generates and saves a LaTeX table comparing the final models."""
        print("\nGenerating final results table...")

        models_to_report = {}
        if hasattr(self, 'base_model_results'):
            models_to_report['Base'] = self.base_model_results
        if hasattr(self, 'base_safety_model_results'):
            models_to_report['Base+Safety'] = self.base_safety_model_results
        if hasattr(self, 'seg_only_model_results'):
             models_to_report['Base+Seg'] = self.seg_only_model_results
        if hasattr(self, 'full_model_results'):
            models_to_report['Base+Seg+Safety'] = self.full_model_results

        if not models_to_report:
            print("No models have been estimated. Cannot generate table.")
            return

        model_metrics = {}
        model_params = {}
        all_param_names = set()

        for name, (train_res, test_res, obs_per_ind) in models_to_report.items():
            model_metrics[name] = {
                'train': extract_mxl_metrics(train_res, obs_per_ind, train_res.data.numberOfObservations),
                'test': test_res
            }
            params = train_res.get_estimated_parameters()
            model_params[name] = params
            all_param_names.update(params.index)

        all_param_names = sorted(list(all_param_names))

        # Helper for formatting
        def format_p_value(p):
            if p < 0.001: return '***'
            if p < 0.01: return '**'
            if p < 0.05: return '*'
            return ''

        # Create a map from sanitized beta names back to original feature names for pretty printing
        beta_to_feature_map = {
            self._sanitize_name_for_beta(f): f for f in self.seg_features
        }
        for p in ['TT', 'TL', 'SAFETY_SCORE']:
            beta_to_feature_map[f'B_{p}'] = f'B_{p}'
            beta_to_feature_map[f'sigma_{p}'] = f'sigma_{p}'
        
        header = " & ".join([f"\\textbf{{{name}}}" for name in models_to_report.keys()])
        
        tabular_spec = 'l' + 'c' * len(models_to_report)
        
        lines = [
            "\\begin{table}[htbp]",
            "    \\centering",
            "    \\caption{Final Model Comparison}",
            "    \\label{tab:final_model_comparison}",
            "    \\resizebox{\\textwidth}{!}{%",
            f"    \\begin{{tabular}}{{{tabular_spec}}}",
            "    \\toprule",
            f"    & {header} \\\\",
            "    \\midrule",
            f"    \\multicolumn{{{len(models_to_report) + 1}}}{{l}}{{\\textit{{Goodness of fit}}}} \\\\",
            "    \\hline",
        ]
        
        # Goodness of fit rows
        stats_to_report = {
            'Sample size (Train)': ('train', 'n_observations', 'd'),
            'Sample size (Test)': ('test', 'n_observations', 'd'),
            'Log-Likelihood (Train)': ('train', 'log_likelihood', '.2f'),
            'Log-Likelihood (Test)': ('test', 'log_likelihood', '.2f'),
            'Rho-squared (Train)': ('train', 'pseudo_r2', '.4f'),
            'Rho-squared (Test)': ('test', 'pseudo_r2', '.4f')
        }

        for display_name, (dataset, key, fmt) in stats_to_report.items():
            values = []
            for name in models_to_report.keys():
                metric = model_metrics[name][dataset][key]
                if fmt == 'd':
                    values.append(f"{metric:d}")
                else:
                    values.append(f"{metric:{fmt}}")
            lines.append(f"    {display_name} & {' & '.join(values)} \\\\")

        lines.extend([
            "    \\hline",
            f"    \\multicolumn{{{len(models_to_report) + 1}}}{{l}}{{\\textit{{Parameters}}}} \\\\",
            "    \\hline"
        ])

        # Parameter values
        for param in all_param_names:
            clean_name = beta_to_feature_map.get(param, param)
            p_name_latex = clean_name.replace('_', '\\_')
            
            values = []
            for name in models_to_report.keys():
                params = model_params[name]
                if param in params.index:
                    val = params.loc[param, 'Value']
                    t = params.loc[param, 'Rob. t-test']
                    p = params.loc[param, 'Rob. p-value']
                    stars = format_p_value(p)
                    val_str = f"{val:.3f}{stars} ({t:.2f})"
                else:
                    val_str = "--"
                values.append(val_str)

            lines.append(f"    {p_name_latex} & {' & '.join(values)} \\\\")
        
        lines.extend([
            "    \\hline",
            "    \\bottomrule",
            f"    \\multicolumn{{{len(models_to_report) + 1}}}{{l}}{{\\textsuperscript{{***}}$p<0.001$, \\textsuperscript{{**}}$p<0.01$, \\textsuperscript{{*}}$p<0.05$}}\\\\",
            "    \\end{tabular}",
            "    }",
            "\\end{table}"
        ])

        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'final_model_comparison.tex'
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX table saved to {table_path}")
    
def main():
    """Main function to run the choice model benchmark"""
    
    print("=== Choice Model Benchmarking ===")
    
    benchmark = ChoiceModelBenchmark()
    benchmark.load_and_prepare_data()
    
    # Run backward elimination to find significant features
    benchmark.run_backward_elimination()
    
    # Estimate final MXL models with the selected features
    benchmark.estimate_all_models()
    
    # Generate and save the final results table
    benchmark.generate_results_table()
    
    print(f"\nResults saved to: {benchmark.output_dir}")
    print("✓ Benchmarking completed successfully!")


if __name__ == "__main__":
    main() 