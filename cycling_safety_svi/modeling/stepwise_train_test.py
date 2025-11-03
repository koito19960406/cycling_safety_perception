"""
Stepwise Train/Test Validation for Choice Models

This script implements stepwise modeling with train/test validation:
- Splits data 80:20 by individuals (RIDs)
- Runs backward elimination on training data
- Trains models on training data
- Evaluates models on test data with fixed parameters
- Compares train vs test performance metrics
"""

import os
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import logging
from biogeme.expressions import Beta, Variable, log, exp, bioDraws
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from mxl_functions import (
    estimate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results, simulate_mxl
)


class StepwiseTrainTestValidation:
    """Stepwise modeling with train/test validation"""
    
    def __init__(self, base_output_dir='reports/models', train_ratio=0.8, random_state=42):
        """
        Initialize the validation environment
        
        Args:
            base_output_dir: Base directory for output
            train_ratio: Proportion of data for training (default 0.8)
            random_state: Random seed for reproducible splits
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_folder = f"stepwise_train_test_{timestamp}"
        self.output_dir = Path(base_output_dir) / timestamped_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
        
        logging.getLogger('biogeme').setLevel(logging.WARNING)
        
        self.num_draws = 1000
        self.individual_id = 'RID'
        self.min_obs_per_individual = 15
        self.train_ratio = train_ratio
        self.random_state = random_state
        
    def load_and_prepare_data(self, 
                             cv_dcm_path='data/raw/cv_dcm.csv',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv',
                             original_results_path='data/raw/df_choice_with_Vimg.csv'):
        """Load and prepare all datasets for mixed logit modeling"""
        
        print("Loading datasets...")
        
        self.choice_data = pd.read_csv(cv_dcm_path)
        print(f"Loaded choice data: {len(self.choice_data)} observations")
        
        print("\nApplying data cleaning steps...")
        self.choice_data = apply_data_cleaning(
            self.choice_data, 
            individual_id=self.individual_id,
            min_obs=self.min_obs_per_individual,
            fix_problematic_rid=True
        )
        
        self.choice_data = self.choice_data.reset_index(drop=True)
        print(f"Index reset after cleaning. New shape: {self.choice_data.shape}")
        
        self.safety_scores = pd.read_csv(safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        print(f"Loaded safety scores: {len(self.safety_scores)} images")
        
        print("Loading segmentation data...")
        seg_chunks = []
        for chunk in pd.read_csv(segmentation_path, chunksize=1000):
            seg_chunks.append(chunk)
        self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
        self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
        print(f"Loaded segmentation data: {len(self.segmentation_data)} images")
        
        self.original_results = pd.read_csv(original_results_path)
        print(f"Loaded original results: {len(self.original_results)} observations")
        
        self._merge_datasets()
        
    def _merge_datasets(self):
        """Merge all datasets into a single dataframe"""
        
        print("Merging datasets...")
        
        merged_data = self.choice_data.copy()
        
        safety_dict = dict(zip(self.safety_scores['image_name'], self.safety_scores['safety_score']))
        merged_data['safety_score_1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['safety_score_2'] = merged_data['IMG2'].map(safety_dict)
        
        print("Processing segmentation data...")
        
        segmentation_dict = {}
        for _, row in self.segmentation_data.iterrows():
            img_name = row['filename_key'] + '.jpg'
            if pd.isna(row['filename_key']):
                continue
            features = row.drop('filename_key').to_dict()
            segmentation_dict[img_name] = features
        
        seg_feature_names = [col for col in self.segmentation_data.columns if col != 'filename_key']
        
        new_columns_data = {}
        for feature in seg_feature_names:
            new_columns_data[f"{feature}_1"] = [None] * len(merged_data)
            new_columns_data[f"{feature}_2"] = [None] * len(merged_data)
        
        for idx, row in merged_data.iterrows():
            img1 = row['IMG1']
            if img1 in segmentation_dict:
                for feature, value in segmentation_dict[img1].items():
                    new_columns_data[f"{feature}_1"][idx] = value
            
            img2 = row['IMG2']
            if img2 in segmentation_dict:
                for feature, value in segmentation_dict[img2].items():
                    new_columns_data[f"{feature}_2"][idx] = value
        
        new_columns_df = pd.DataFrame(new_columns_data, index=merged_data.index)
        merged_data = pd.concat([merged_data, new_columns_df], axis=1)
        
        seg_cols = [col for col in merged_data.columns if col.endswith('_1') or col.endswith('_2')]
        for col in seg_cols:
            if col not in ['TL1', 'TT1', 'TL2', 'TT2', 'safety_score_1', 'safety_score_2']:
                merged_data[col] = merged_data[col].fillna(0)
        
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['safety_score_1'] = merged_data['safety_score_1'].fillna(mean_safety)
        merged_data['safety_score_2'] = merged_data['safety_score_2'].fillna(mean_safety)
        
        self.merged_data = merged_data
        print(f"Merged dataset created: {len(merged_data)} observations, {len(merged_data.columns)} features")
        
        self.seg_features = []
        for col in merged_data.columns:
            if col.endswith('_1') and not col.startswith(('TL', 'TT', 'safety_score')):
                feature_name = col[:-2]
                self.seg_features.append(feature_name)

        print(f"Available segmentation features: {len(self.seg_features)}")
        
        self._filter_and_scale_segmentation_features()
        
    def _filter_and_scale_segmentation_features(self, variance_threshold=1e-6, scale_features=False):
        """Filter out segmentation features with very low variance"""
        print("Filtering segmentation features...")
        
        seg_cols_1 = [f"{feature}_1" for feature in self.seg_features if f"{feature}_1" in self.merged_data.columns]
        seg_cols_2 = [f"{feature}_2" for feature in self.seg_features if f"{feature}_2" in self.merged_data.columns]
        all_seg_cols = seg_cols_1 + seg_cols_2
        
        if not all_seg_cols:
            print("No segmentation features found to process")
            return
        
        original_feature_count = len(self.seg_features)
        
        scaled_features = []
        if scale_features:
            print("Applying z-score scaling to segmentation features...")
            
            for feature in self.seg_features:
                col1 = f"{feature}_1"
                col2 = f"{feature}_2"
                
                if col1 in self.merged_data.columns and col2 in self.merged_data.columns:
                    combined_values = pd.concat([
                        self.merged_data[col1].dropna(),
                        self.merged_data[col2].dropna()
                    ])
                    
                    if combined_values.std() == 0:
                        print(f"Skipping feature '{feature}' - no variation in raw data")
                        continue
                    
                    scaler = StandardScaler()
                    scaler.fit(combined_values.values.reshape(-1, 1))
                    
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
        
        print("Calculating variances for filtering...")
        
        variances = {}
        features_to_keep = []
        
        for feature in scaled_features:
            col1 = f"{feature}_1"
            col2 = f"{feature}_2"
            
            if col1 in self.merged_data.columns and col2 in self.merged_data.columns:
                combined_values = pd.concat([
                    self.merged_data[col1].dropna(),
                    self.merged_data[col2].dropna()
                ])
                
                feature_variance = combined_values.var()
                variances[feature] = feature_variance
                
                if feature_variance > variance_threshold:
                    features_to_keep.append(feature)
                else:
                    print(f"Removing feature '{feature}' due to low variance: {feature_variance:.2e}")
        
        print(f"Features before variance filtering: {len(scaled_features)}")
        print(f"Features after variance filtering: {len(features_to_keep)}")
        
        self.seg_features = features_to_keep
        
        if not self.seg_features:
            print("Warning: No segmentation features remaining after variance filtering.")
        
        print(f"Final segmentation features available: {len(self.seg_features)}")
        
    def split_train_test(self):
        """
        Split data into train and test sets by individuals (RIDs)
        Maintains panel structure by splitting at individual level
        """
        print("\nSplitting data into train/test sets...")
        
        unique_rids = self.merged_data[self.individual_id].unique()
        n_rids = len(unique_rids)
        n_train_rids = int(n_rids * self.train_ratio)
        
        np.random.seed(self.random_state)
        shuffled_rids = np.random.permutation(unique_rids)
        
        train_rids = shuffled_rids[:n_train_rids]
        test_rids = shuffled_rids[n_train_rids:]
        
        self.train_data = self.merged_data[self.merged_data[self.individual_id].isin(train_rids)].copy()
        self.test_data = self.merged_data[self.merged_data[self.individual_id].isin(test_rids)].copy()
        
        print(f"Train set: {len(self.train_data)} observations from {len(train_rids)} individuals")
        print(f"Test set: {len(self.test_data)} observations from {len(test_rids)} individuals")
        print(f"Train ratio: {len(train_rids)/n_rids:.2%}, Test ratio: {len(test_rids)/n_rids:.2%}")
        
    def _sanitize_name_for_beta(self, feature_name):
        """Creates a Biogeme-compatible name for a beta parameter."""
        s_name = feature_name.replace(' - ', '___').replace(' ', '_')
        return f"B_{s_name}"

    def run_backward_elimination(self, significance_level=0.05):
        """
        Performs backward elimination feature selection using MNL on training data.
        """
        print("\nStarting backward elimination for feature selection (on training data)...")
        
        beta_to_feature_map = {
            self._sanitize_name_for_beta(f): f for f in self.seg_features
        }
        beta_to_feature_map['B_SAFETY_SCORE'] = 'SAFETY_SCORE'
        
        model_data_full = self.train_data.copy()
        
        features_to_consider = self.seg_features + ['SAFETY_SCORE']
        
        while True:
            print(f"Testing model with {len(features_to_consider)} features...")
            
            attributes = ['TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
            current_seg_features = [f for f in features_to_consider if f in self.seg_features]
            
            for feature in current_seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
            
            if 'SAFETY_SCORE' in features_to_consider:
                attributes.extend(['safety_score_1', 'safety_score_2'])
                
            model_data = model_data_full[attributes].copy().dropna()
            
            database = db.Database('backward_elimination', model_data)
            
            V1_comps, V2_comps = [], []
            variables = {col: Variable(col) for col in model_data.columns if col != 'CHOICE'}
            
            B_TT = Beta('B_TT', -0.2, None, None, 0)
            B_TL = Beta('B_TL', -0.3, None, None, 0)
            V1_comps.extend([B_TT * variables['TT1'] / 10, B_TL * variables['TL1'] / 3])
            V2_comps.extend([B_TT * variables['TT2'] / 10, B_TL * variables['TL2'] / 3])

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

            V = {1: sum(V1_comps), 2: sum(V2_comps)}
            prob = models.logit(V, {1: 1, 2: 1}, Variable('CHOICE'))
            biogeme = bio.BIOGEME(database, log(prob))
            biogeme.modelName = f"elimination_{len(features_to_consider)}"
            biogeme.generate_pickle = False
            biogeme.generate_html = False
            biogeme.save_iterations = False
            results = biogeme.estimate(verbose=False)
            
            params_df = results.get_estimated_parameters()
            p_values = params_df['Rob. p-value']

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
        
    def _estimate_final_mxl(self, features, model_name, train_data):
        """Helper to estimate a single final MXL model on training data."""
        
        attributes = [self.individual_id, 'TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
        seg_features = [f for f in features if f not in ['TT', 'TL', 'SAFETY_SCORE']]
        
        for feature in seg_features:
            attributes.extend([f"{feature}_1", f"{feature}_2"])
        
        if 'SAFETY_SCORE' in features:
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        model_data = train_data[attributes].copy().dropna()
        model_data = model_data.rename(columns={
            'safety_score_1': 'SAFETY_SCORE1',
            'safety_score_2': 'SAFETY_SCORE2'
        })

        _, biodata_wide, obs_per_ind = prepare_panel_data(
            model_data, self.individual_id, 'CHOICE'
        )

        random_params_config = {
            'TT': {'mean_init': -1, 'sigma_init': 0.1, 'dist': 'lognormal'},
            'TL': {'mean_init': -1, 'sigma_init': 0.1, 'dist': 'lognormal'}
        }
        if 'SAFETY_SCORE' in features:
            random_params_config['SAFETY_SCORE'] = {'mean_init': 1.0, 'sigma_init': 0.1, 'dist': 'normal'}
        
        random_params = {}
        for param, config in random_params_config.items():
            mean = Beta(f'B_{param}', config['mean_init'], None, None, 0)
            sigma = Beta(f'sigma_{param}', config['sigma_init'], None, None, 0)
            draws = bioDraws(f'{param}_rnd', 'NORMAL_HALTON2')
            if config['dist'] == 'lognormal':
                random_params[param] = -exp(mean + sigma * draws)
            else:
                random_params[param] = mean + sigma * draws
            
        fixed_params = {
            f: Beta(self._sanitize_name_for_beta(f), 0, None, None, 0) 
            for f in seg_features
        }

        V = []
        for q in range(obs_per_ind):
            V1, V2 = 0, 0
            for name, param in random_params.items():
                v1_name, v2_name = f"{name}1_{q}", f"{name}2_{q}"
                if v1_name in biodata_wide.variables:
                    scale = 10 if name == 'TT' else (3 if name == 'TL' else 1)
                    V1 += param * Variable(v1_name) / scale
                    V2 += param * Variable(v2_name) / scale
            for name, param in fixed_params.items():
                v1_name, v2_name = f"{name}_1_{q}", f"{name}_2_{q}"
                if v1_name in biodata_wide.variables:
                    V1 += param * Variable(v1_name)
                    V2 += param * Variable(v2_name)
            V.append({1: V1, 2: V2})

        results = estimate_mxl(V, {1:1, 2:1}, 'CHOICE', obs_per_ind, self.num_draws, biodata_wide, model_name, self.output_dir)

        return results, obs_per_ind, V
    
    def _evaluate_on_test_data(self, V, obs_per_ind, train_results, model_name, features):
        """Evaluate trained model on test data with fixed parameters"""
        
        attributes = [self.individual_id, 'TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
        seg_features = [f for f in features if f not in ['TT', 'TL', 'SAFETY_SCORE']]
        
        for feature in seg_features:
            attributes.extend([f"{feature}_1", f"{feature}_2"])
        
        if 'SAFETY_SCORE' in features:
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        test_model_data = self.test_data[attributes].copy().dropna()
        test_model_data = test_model_data.rename(columns={
            'safety_score_1': 'SAFETY_SCORE1',
            'safety_score_2': 'SAFETY_SCORE2'
        })

        _, test_biodata_wide, _ = prepare_panel_data(
            test_model_data, self.individual_id, 'CHOICE'
        )
        
        # Handle different result object structures
        betas = train_results.get_beta_values()
            
        sim_results = simulate_mxl(
            V, {1: 1, 2: 1}, 'CHOICE', obs_per_ind, self.num_draws,
            test_biodata_wide, betas, model_name
        )
        
        return sim_results, test_model_data
        
    def estimate_stepwise_model(self):
        """
        Estimates the final stepwise model (with significant features) on training data 
        and evaluates on test data
        """
        print("\nEstimating stepwise model on training data...")

        if not hasattr(self, 'final_significant_features'):
            print("Backward elimination must be run first.")
            return

        model_name = 'stepwise_model'
        features = self.final_significant_features
        
        print(f"\nEstimating {model_name} with {len(features)} features...")
        print(f"Features: {features}")
        
        train_results, obs_per_ind, V = self._estimate_final_mxl(
            features, model_name, self.train_data
        )
        print_mxl_results(train_results)
        
        print(f"\nEvaluating {model_name} on test data...")
        test_results, test_model_data = self._evaluate_on_test_data(
            V, obs_per_ind, train_results, f"{model_name}_test", features
        )
        
        # Handle different result object structures for getting number of observations
        if hasattr(train_results, 'data'):
            n_train_obs = train_results.data.numberOfObservations
        else:
            n_train_obs = train_results.numberOfObservations
        
        n_train_individuals = n_train_obs // obs_per_ind
        train_metrics = extract_mxl_metrics(
            train_results.data, obs_per_ind, n_train_individuals
        )
        
        test_ll = test_results['LL']
        test_rho2 = test_results['rho_square']
        n_test_individuals = len(test_model_data[self.individual_id].unique())
        n_test_obs = len(test_model_data)
        
        self.model_results = {
            model_name: {
                'train_results': train_results,
                'test_results': test_results,
                'train_metrics': train_metrics,
                'test_ll': test_ll,
                'test_rho2': test_rho2,
                'test_n_individuals': n_test_individuals,
                'test_n_observations': n_test_obs,
                'obs_per_ind': obs_per_ind,
                'V': V,
                'features': features
            }
        }
        
        print(f"\n{'='*60}")
        print(f"TRAIN vs TEST PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"\nTraining Performance:")
        print(f"  Log-likelihood: {train_metrics['log_likelihood']:.4f}")
        print(f"  Rho-square: {train_metrics['pseudo_r2']:.4f}")
        print(f"  Observations: {train_metrics['n_observations']}")
        print(f"  Individuals: {n_train_individuals}")
        
        print(f"\nTest Performance:")
        print(f"  Log-likelihood: {test_ll:.4f}")
        print(f"  Rho-square: {test_rho2:.4f}")
        print(f"  Observations: {n_test_obs}")
        print(f"  Individuals: {n_test_individuals}")
    
    def generate_results_table(self):
        """Generates and saves LaTeX tables comparing train vs test performance"""
        print("\nGenerating results table...")

        if not hasattr(self, 'model_results'):
            print("No models have been estimated. Cannot generate table.")
            return

        lines = [
            "\\begin{table}[htbp]",
            "    \\centering",
            "    \\caption{Train vs Test Model Performance Comparison}",
            "    \\label{tab:train_test_comparison}",
            "    \\resizebox{\\textwidth}{!}{%",
            "    \\begin{tabular}{lcccccc}",
            "    \\toprule",
            "    Model & \\multicolumn{3}{c}{Training} & \\multicolumn{3}{c}{Test} \\\\",
            "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
            "    & LL & $\\rho^2$ & N & LL & $\\rho^2$ & N \\\\",
            "    \\midrule",
        ]
        
        for model_name, results in self.model_results.items():
            train_ll = results['train_metrics']['log_likelihood']
            train_rho2 = results['train_metrics']['pseudo_r2']
            train_n = results['train_metrics']['n_observations']
            
            test_ll = results['test_ll']
            test_rho2 = results['test_rho2']
            test_n = results['test_n_observations']
            
            n_features = len(results.get('features', []))
            model_display = f"Stepwise Model ({n_features} features)"
            lines.append(
                f"    {model_display} & {train_ll:.2f} & {train_rho2:.4f} & {train_n} & "
                f"{test_ll:.2f} & {test_rho2:.4f} & {test_n} \\\\"
            )
        
        lines.extend([
            "    \\bottomrule",
            "    \\end{tabular}",
            "    }",
            "\\end{table}"
        ])

        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'train_test_comparison.tex'
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        print(f"LaTeX table saved to {table_path}")


def main():
    """Main function to run stepwise train/test validation"""
    
    print("=== Stepwise Train/Test Validation ===")
    
    validator = StepwiseTrainTestValidation(train_ratio=0.8, random_state=42)
    validator.load_and_prepare_data()
    
    validator.split_train_test()
    
    validator.run_backward_elimination()
    
    validator.estimate_stepwise_model()
    
    validator.generate_results_table()
    
    print(f"\nResults saved to: {validator.output_dir}")
    print("✓ Validation completed successfully!")


if __name__ == "__main__":
    main()

