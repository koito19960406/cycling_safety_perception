"""
Choice Model Benchmarking Script (Mixed Logit & Multinomial Logit)

This script implements and benchmarks different discrete choice models:

Mixed Logit Models (MXL):
1. Base model: only TL1, TT1, TL2, TT2 (with random parameters)
2. Base + safety scores: TL1, TT1, TL2, TT2, SAFETY_SCORE (with random parameters)
3. Base + segmentation: TL1, TT1, TL2, TT2, pixel ratios (TL/TT random, segmentation fixed)
4. Base + safety + segmentation: TL1, TT1, TL2, TT2, SAFETY_SCORE, pixel ratios

WTP Space Models (both MXL and MNL):
- MXL WTP models with log-normal distributed willingness-to-pay parameters
- MNL WTP models with fixed willingness-to-pay parameters for comparison
- Estimates WTP for safety vs travel time and safety vs traffic lights

Uses mixed logit models with random parameters for TT, TL, and SAFETY_SCORE.
Compares models using Log-Likelihood, BIC, AIC, and Pseudo R-squared metrics.
Generates LaTeX tables comparing MXL and MNL WTP estimates.
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
from sklearn.preprocessing import StandardScaler

from datetime import datetime

# Import MXL functions
from mxl_functions import (
    estimate_mxl, prepare_panel_data, apply_data_cleaning,
    extract_mxl_metrics, print_mxl_results, estimate_wtp_mnl
)


class ChoiceModelBenchmark:
    """Benchmarks different mixed logit discrete choice models with various feature combinations"""
    
    def __init__(self, base_output_dir='reports/models', checkpoint_dir=None):
        """Initializes the benchmark environment, setting up output directories and logging."""
        if checkpoint_dir:
            # Use existing checkpoint directory
            self.output_dir = Path(checkpoint_dir)
            if not self.output_dir.exists():
                raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")
            print(f"Using checkpoint directory: {self.output_dir}")
            self.use_checkpoint = True
        else:
            # Create new timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_folder = f"mxl_choice_{timestamp}"
            self.output_dir = Path(base_output_dir) / timestamped_folder
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Mixed Logit Model Results will be saved to: {self.output_dir}")
            self.use_checkpoint = False
        
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
        
    def _model_exists(self, model_name):
        """Check if a model's results already exist in the output directory."""
        if not self.use_checkpoint:
            return False
            
        # Check for pickle file (main indicator that model was estimated)
        pickle_path = self.output_dir / f"{model_name}.pickle"
        return pickle_path.exists()
    
    def _load_existing_model(self, model_name):
        """Load existing model results from checkpoint directory."""
        import pickle
        
        pickle_path = self.output_dir / f"{model_name}.pickle"
        if not pickle_path.exists():
            return None
            
        try:
            with open(pickle_path, 'rb') as f:
                results = pickle.load(f)
            print(f"✓ Loaded existing model results for {model_name}")
            return results
        except Exception as e:
            print(f"Warning: Could not load {model_name}: {e}")
            return None
        
    def _sanitize_name_for_beta(self, feature_name):
        """Creates a Biogeme-compatible name for a beta parameter."""
        s_name = feature_name.replace(' - ', '___').replace(' ', '_')
        return f"B_{s_name}"

    def run_backward_elimination(self, significance_level=0.05):
        """
        Performs backward elimination feature selection using MNL to find significant variables.
        """
        print("\nStarting backward elimination for feature selection...")
        
        # Check if final_significant_features already exists from checkpoint
        if self.use_checkpoint:
            features_pickle_path = self.output_dir / 'final_significant_features.pickle'
            if features_pickle_path.exists():
                try:
                    import pickle
                    with open(features_pickle_path, 'rb') as f:
                        self.final_significant_features = pickle.load(f)
                    print(f"✓ Loaded existing significant features from checkpoint: {self.final_significant_features}")
                    return self.final_significant_features
                except Exception as e:
                    print(f"Warning: Could not load significant features from checkpoint: {e}")
        
        # Create a mapping from beta names back to original feature names
        beta_to_feature_map = {
            self._sanitize_name_for_beta(f): f for f in self.seg_features
        }
        beta_to_feature_map['B_SAFETY_SCORE'] = 'SAFETY_SCORE'
        
        # Use the full dataset for backward elimination
        model_data_full = self.merged_data.copy()
        
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
                
            model_data = model_data_full[attributes].copy().dropna()
            
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
            biogeme.generate_pickle = False
            biogeme.generate_html = False
            biogeme.save_iterations = False
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
        
        # Save final_significant_features to pickle for future use
        features_pickle_path = self.output_dir / 'final_significant_features.pickle'
        try:
            import pickle
            with open(features_pickle_path, 'wb') as f:
                pickle.dump(self.final_significant_features, f)
            print(f"✓ Saved significant features to {features_pickle_path}")
        except Exception as e:
            print(f"Warning: Could not save significant features: {e}")
        
        return self.final_significant_features
        
    def estimate_all_models(self):
        """
        Estimates all benchmark models:
        1. Base Model (TT, TL random)
        2. Base + Safety Model (TT, TL, SAFETY_SCORE random)
        3. Full Model (Base + Safety + Significant Segmentation features)
        4. Segmentation-Only Model (Base + Significant Segmentation features)
        5. WTP Space Models for computing willingness-to-pay metrics
        """
        print("\nEstimating all benchmark models...")

        # 1. Base Model
        if self._model_exists('base_model'):
            print("✓ Base Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('base_model')
            if existing_results:
                self.base_model_results = (existing_results, None)  # Approximate structure
            else:
                print("Estimating Base Model...")
                self.base_model_results = self._estimate_final_mxl(
                    features=[], model_name='base_model'
                )
        else:
            print("Estimating Base Model...")
            self.base_model_results = self._estimate_final_mxl(
                features=[], model_name='base_model'
            )
        
        if hasattr(self, 'base_model_results') and self.base_model_results[0]:
            print_mxl_results(self.base_model_results[0])

        # 2. Base + Safety Model
        if self._model_exists('base_safety_model'):
            print("✓ Base + Safety Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('base_safety_model')
            if existing_results:
                self.base_safety_model_results = (existing_results, None)
            else:
                print("Estimating Base + Safety Model...")
                self.base_safety_model_results = self._estimate_final_mxl(
                    features=['SAFETY_SCORE'], model_name='base_safety_model'
                )
        else:
            print("Estimating Base + Safety Model...")
            self.base_safety_model_results = self._estimate_final_mxl(
                features=['SAFETY_SCORE'], model_name='base_safety_model'
            )
        
        if hasattr(self, 'base_safety_model_results') and self.base_safety_model_results[0]:
            print_mxl_results(self.base_safety_model_results[0])

        if not hasattr(self, 'final_significant_features'):
            print("Backward elimination must be run first to estimate segmentation models.")
            return

        # 3. Full Model (with safety if significant)
        if self._model_exists('final_full_model'):
            print("✓ Full Final Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('final_full_model')
            if existing_results:
                self.full_model_results = (existing_results, None)
            else:
                print("Estimating Full Final Model...")
                full_model_features = self.final_significant_features
                self.full_model_results = self._estimate_final_mxl(
                    full_model_features, 'final_full_model'
                )
        else:
            print("Estimating Full Final Model...")
            full_model_features = self.final_significant_features
            self.full_model_results = self._estimate_final_mxl(
                full_model_features, 'final_full_model'
            )
        
        if hasattr(self, 'full_model_results') and self.full_model_results[0]:
            print_mxl_results(self.full_model_results[0])

        # 4. Segmentation-Only Model (safety removed)
        if self._model_exists('final_seg_only_model'):
            print("✓ Segmentation-Only Final Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('final_seg_only_model')
            if existing_results:
                self.seg_only_model_results = (existing_results, None)
            else:
                print("Estimating Segmentation-Only Final Model...")
                seg_only_features = [f for f in self.final_significant_features if f != 'SAFETY_SCORE']
                self.seg_only_model_results = self._estimate_final_mxl(
                    seg_only_features, 'final_seg_only_model'
                )
        else:
            print("Estimating Segmentation-Only Final Model...")
            seg_only_features = [f for f in self.final_significant_features if f != 'SAFETY_SCORE']
            self.seg_only_model_results = self._estimate_final_mxl(
                seg_only_features, 'final_seg_only_model'
            )
        
        if hasattr(self, 'seg_only_model_results') and self.seg_only_model_results[0]:
            print_mxl_results(self.seg_only_model_results[0])
        
        # 5. WTP Space Models
        print("\nEstimating WTP Space Models...")
        self._estimate_wtp_models()

    def _estimate_final_mxl(self, features, model_name):
        """Helper to estimate a single final MXL model."""
        
        attributes = [self.individual_id, 'TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
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
            model_data, self.individual_id, 'CHOICE'
        )

        # Define random parameters
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

        return results.data, obs_per_ind

    def _estimate_wtp_models(self):
        """
        Estimate models in WTP space to compute willingness-to-pay for safety scores.
        Following the approach from Lab_02A_with_answers.ipynb where tc (travel cost) 
        is replaced with TT (travel time) and Time is replaced with Safety score.
        """
        print("\nEstimating models in WTP space...")
        
        if not hasattr(self, 'final_significant_features'):
            print("Backward elimination must be run first.")
            return
            
        # Check if SAFETY_SCORE is in significant features
        if 'SAFETY_SCORE' not in self.final_significant_features:
            print("SAFETY_SCORE not significant, skipping WTP space estimation.")
            return
            
        # 1. MXL WTP Space Model: Safety vs Travel Time
        if self._model_exists('wtp_mxl_safety_vs_tt'):
            print("✓ MXL WTP Safety vs Travel Time Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('wtp_mxl_safety_vs_tt')
            if existing_results:
                self.wtp_mxl_safety_tt_results = (existing_results, None, 'TT', 'SAFETY_SCORE')
            else:
                print("Estimating MXL WTP Space Model: Safety vs Travel Time...")
                self.wtp_mxl_safety_tt_results = self._estimate_wtp_mxl(
                    wtp_attribute='SAFETY_SCORE', 
                    cost_attribute='TT',
                    model_name='wtp_mxl_safety_vs_tt'
                )
        else:
            print("Estimating MXL WTP Space Model: Safety vs Travel Time...")
            self.wtp_mxl_safety_tt_results = self._estimate_wtp_mxl(
                wtp_attribute='SAFETY_SCORE', 
                cost_attribute='TT',
                model_name='wtp_mxl_safety_vs_tt'
            )
        
        # 2. MXL WTP Space Model: Safety vs Traffic Lights
        if self._model_exists('wtp_mxl_safety_vs_tl'):
            print("✓ MXL WTP Safety vs Traffic Lights Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('wtp_mxl_safety_vs_tl')
            if existing_results:
                self.wtp_mxl_safety_tl_results = (existing_results, None, 'TL', 'SAFETY_SCORE')
            else:
                print("Estimating MXL WTP Space Model: Safety vs Traffic Lights...")
                self.wtp_mxl_safety_tl_results = self._estimate_wtp_mxl(
                    wtp_attribute='SAFETY_SCORE',
                    cost_attribute='TL', 
                    model_name='wtp_mxl_safety_vs_tl'
                )
        else:
            print("Estimating MXL WTP Space Model: Safety vs Traffic Lights...")
            self.wtp_mxl_safety_tl_results = self._estimate_wtp_mxl(
                wtp_attribute='SAFETY_SCORE',
                cost_attribute='TL', 
                model_name='wtp_mxl_safety_vs_tl'
            )
            
        # 3. MNL WTP Space Model: Safety vs Travel Time
        if self._model_exists('wtp_mnl_safety_vs_tt'):
            print("✓ MNL WTP Safety vs Travel Time Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('wtp_mnl_safety_vs_tt')
            if existing_results:
                self.wtp_mnl_safety_tt_results = (existing_results, 'TT', 'SAFETY_SCORE')
            else:
                print("Estimating MNL WTP Space Model: Safety vs Travel Time...")
                self.wtp_mnl_safety_tt_results = self._estimate_wtp_mnl(
                    wtp_attribute='SAFETY_SCORE', 
                    cost_attribute='TT',
                    model_name='wtp_mnl_safety_vs_tt'
                )
        else:
            print("Estimating MNL WTP Space Model: Safety vs Travel Time...")
            self.wtp_mnl_safety_tt_results = self._estimate_wtp_mnl(
                wtp_attribute='SAFETY_SCORE', 
                cost_attribute='TT',
                model_name='wtp_mnl_safety_vs_tt'
            )
        
        # 4. MNL WTP Space Model: Safety vs Traffic Lights
        if self._model_exists('wtp_mnl_safety_vs_tl'):
            print("✓ MNL WTP Safety vs Traffic Lights Model already exists, loading from checkpoint...")
            existing_results = self._load_existing_model('wtp_mnl_safety_vs_tl')
            if existing_results:
                self.wtp_mnl_safety_tl_results = (existing_results, 'TL', 'SAFETY_SCORE')
            else:
                print("Estimating MNL WTP Space Model: Safety vs Traffic Lights...")
                self.wtp_mnl_safety_tl_results = self._estimate_wtp_mnl(
                    wtp_attribute='SAFETY_SCORE',
                    cost_attribute='TL', 
                    model_name='wtp_mnl_safety_vs_tl'
                )
        else:
            print("Estimating MNL WTP Space Model: Safety vs Traffic Lights...")
            self.wtp_mnl_safety_tl_results = self._estimate_wtp_mnl(
                wtp_attribute='SAFETY_SCORE',
                cost_attribute='TL', 
                model_name='wtp_mnl_safety_vs_tl'
            )
        
        # Compute and store WTP metrics
        self._compute_wtp_metrics()

    def _estimate_wtp_mxl(self, wtp_attribute, cost_attribute, model_name):
        """
        Estimate a separate mixed logit model in WTP space.
        Following the reference Lab_02A approach where WTP models are completely separate
        from utility space models, with their own parameter definitions.
        
        Includes all significant segmentation features from backward elimination.
        
        Args:
            wtp_attribute: The attribute we want to compute WTP for (e.g., 'SAFETY_SCORE')
            cost_attribute: The cost attribute (e.g., 'TT' or 'TL')  
            model_name: Name of the model
        """
        from mxl_functions import estimate_wtp_mxl, prepare_panel_data
        from biogeme.expressions import Beta, Variable, bioDraws, exp
        
        # Prepare data - include cost, WTP, and segmentation attributes
        attributes = [self.individual_id, 'CHOICE']
        
        # Add cost attribute columns
        if cost_attribute == 'TT':
            attributes.extend(['TT1', 'TT2'])
            cost_scale = 10
        else:  # TL
            attributes.extend(['TL1', 'TL2'])
            cost_scale = 3
            
        # Add WTP attribute columns    
        if wtp_attribute == 'SAFETY_SCORE':
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        # Add significant segmentation features
        if hasattr(self, 'final_significant_features'):
            seg_features = [f for f in self.final_significant_features if f not in ['SAFETY_SCORE', 'TT', 'TL']]
            for feature in seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
            print(f"Including {len(seg_features)} segmentation features in WTP model: {seg_features}")
        else:
            seg_features = []
            print("Warning: No final_significant_features found, WTP model will only include safety and cost")
            
        model_data = self.merged_data[attributes].copy().dropna()
        
        # Rename safety score columns to match expected format
        if wtp_attribute == 'SAFETY_SCORE':
            model_data = model_data.rename(columns={
                'safety_score_1': 'SAFETY_SCORE1',
                'safety_score_2': 'SAFETY_SCORE2'
            })

        _, biodata_wide, obs_per_ind = prepare_panel_data(
            model_data, self.individual_id, 'CHOICE'
        )

        # Define WTP space parameters following Lab_02A reference implementation
        # Parameters for log-normal WTP distribution
        mu = Beta('mu', -1, None, None, 0)  # Starting value from reference
        sigma = Beta('sigma', 1, None, None, 0)  # Starting value from reference
        
        # Cost parameter (fixed, negative) - this is like B_tc in the reference
        B_cost = Beta('B_cost', -0.1, None, None, 0)  # Starting value from reference
            
        # Random WTP parameter (log-normal distribution)
        # This is like vtt_rnd in the reference: vtt_rnd = exp(mu + sigma * bioDraws('vtt_rnd', 'NORMAL_HALTON2'))
        wtp_rnd = -exp(mu + sigma * bioDraws('wtp_rnd', 'NORMAL_HALTON2'))

        # Define fixed parameters for segmentation features
        fixed_params = {}
        for feature in seg_features:
            beta_name = self._sanitize_name_for_beta(feature)
            fixed_params[feature] = Beta(beta_name, 0, None, None, 0)

        # Define utility functions in WTP space following Lab_02A approach:
        # V_L = B_tc * (CostL + vtt_rnd * TimeL) + segmentation_terms
        # V_R = B_tc * (CostR + vtt_rnd * TimeR) + segmentation_terms
        V = []
        for q in range(obs_per_ind):
            # Get variable names for this observation
            cost1_name = f"{cost_attribute}1_{q}"
            cost2_name = f"{cost_attribute}2_{q}"
            wtp1_name = f"{wtp_attribute}1_{q}"
            wtp2_name = f"{wtp_attribute}2_{q}"
            
            V1 = V2 = 0
            
            # Build WTP space utility: B_cost * (Cost + WTP * WTP_attribute)
            if (cost1_name in biodata_wide.variables and 
                wtp1_name in biodata_wide.variables):
                
                # Following Lab_02A: V = B_cost * (Cost + WTP_rnd * WTP_attribute)
                V1 = B_cost * (Variable(cost1_name) / cost_scale + wtp_rnd * Variable(wtp1_name))
                V2 = B_cost * (Variable(cost2_name) / cost_scale + wtp_rnd * Variable(wtp2_name))
                
                # Add segmentation features (outside the WTP factorization)
                for feature in seg_features:
                    var1_name = f"{feature}_1_{q}"
                    var2_name = f"{feature}_2_{q}"
                    if var1_name in biodata_wide.variables and var2_name in biodata_wide.variables:
                        V1 += fixed_params[feature] * Variable(var1_name)
                        V2 += fixed_params[feature] * Variable(var2_name)
            
            V.append({1: V1, 2: V2})

        # Estimate the model using the WTP-specific estimation function
        results = estimate_wtp_mxl(V, {1: 1, 2: 1}, 'CHOICE', obs_per_ind, 
                                  self.num_draws, biodata_wide, model_name, self.output_dir)
        
        return results.data, obs_per_ind, cost_attribute, wtp_attribute

    def _estimate_wtp_mnl(self, wtp_attribute, cost_attribute, model_name):
        """
        Estimate a multinomial logit model in WTP space (no panel structure).
        This provides a simpler, non-random parameter comparison to the MXL WTP models.
        
        Args:
            wtp_attribute: The attribute we want to compute WTP for (e.g., 'SAFETY_SCORE')
            cost_attribute: The cost attribute (e.g., 'TT' or 'TL')  
            model_name: Name of the model
        """
        from biogeme.expressions import Beta, Variable
        
        # Prepare data - include cost, WTP, and segmentation attributes
        attributes = ['CHOICE']
        
        # Add cost attribute columns
        if cost_attribute == 'TT':
            attributes.extend(['TT1', 'TT2'])
            cost_scale = 10
        else:  # TL
            attributes.extend(['TL1', 'TL2'])
            cost_scale = 3
            
        # Add WTP attribute columns    
        if wtp_attribute == 'SAFETY_SCORE':
            attributes.extend(['safety_score_1', 'safety_score_2'])
        
        # Add significant segmentation features
        if hasattr(self, 'final_significant_features'):
            seg_features = [f for f in self.final_significant_features if f not in ['SAFETY_SCORE', 'TT', 'TL']]
            for feature in seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
            print(f"Including {len(seg_features)} segmentation features in MNL WTP model: {seg_features}")
        else:
            seg_features = []
            print("Warning: No final_significant_features found, MNL WTP model will only include safety and cost")
            
        model_data = self.merged_data[attributes].copy().dropna()
        
        # Rename safety score columns to match expected format
        if wtp_attribute == 'SAFETY_SCORE':
            model_data = model_data.rename(columns={
                'safety_score_1': 'SAFETY_SCORE1',
                'safety_score_2': 'SAFETY_SCORE2'
            })

        database = db.Database('mnl_wtp', model_data)

        # Define WTP space parameters for MNL model
        # WTP parameter (fixed, not random like in MXL)
        B_wtp = Beta('B_wtp', 1.0, None, None, 0)  # Fixed WTP parameter
        
        # Cost parameter (fixed, negative)
        B_cost = Beta('B_cost', -0.1, None, None, 0)

        # Define fixed parameters for segmentation features
        fixed_params = {}
        for feature in seg_features:
            beta_name = self._sanitize_name_for_beta(feature)
            fixed_params[feature] = Beta(beta_name, 0, None, None, 0)

        # Define variables
        variables = {col: Variable(col) for col in model_data.columns if col != 'CHOICE'}

        # Define utility functions in WTP space for MNL:
        # V = B_cost * (Cost + B_wtp * WTP_attribute) + segmentation_terms
        cost1_name = f"{cost_attribute}1"
        cost2_name = f"{cost_attribute}2"
        wtp1_name = f"{wtp_attribute}1"
        wtp2_name = f"{wtp_attribute}2"
        
        V1 = V2 = 0
        
        # Build WTP space utility: B_cost * (Cost + B_wtp * WTP_attribute)
        if (cost1_name in variables and wtp1_name in variables):
            V1 = B_cost * (variables[cost1_name] / cost_scale + B_wtp * variables[wtp1_name])
            V2 = B_cost * (variables[cost2_name] / cost_scale + B_wtp * variables[wtp2_name])
            
            # Add segmentation features (outside the WTP factorization)
            for feature in seg_features:
                var1_name = f"{feature}_1"
                var2_name = f"{feature}_2"
                if var1_name in variables and var2_name in variables:
                    V1 += fixed_params[feature] * variables[var1_name]
                    V2 += fixed_params[feature] * variables[var2_name]
        
        V = {1: V1, 2: V2}

        # Estimate the model using the MNL estimation function
        results = estimate_wtp_mnl(V, {1: 1, 2: 1}, 'CHOICE', database, model_name, self.output_dir)
        
        return results.data, cost_attribute, wtp_attribute

    def _compute_wtp_metrics(self):
        """
        Compute WTP metrics from the separately estimated WTP space models.
        These are extracted from dedicated WTP space models, not computed as ratios
        from utility space models. Now handles both MXL and MNL models.
        
        IMPORTANT: Cost scaling adjustment
        - In utility functions, we use scaled costs: TT/10 and TL/3
        - WTP parameters represent willingness to trade safety for SCALED cost units
        - To get WTP in original units (minutes/lights), we multiply by cost_scale
        - This applies to both point estimates and standard errors
        """
        print("\nComputing WTP metrics from WTP space models...")
        
        self.wtp_metrics = {}
        
        # MXL Safety vs Travel Time WTP (from separate MXL WTP space model)
        if hasattr(self, 'wtp_mxl_safety_tt_results'):
            results, obs_per_ind, cost_attr, wtp_attr = self.wtp_mxl_safety_tt_results
            
            # Get the cost scale used in the model
            cost_scale = 10 if cost_attr == 'TT' else 3
            
            # Extract mu and sigma from betas
            mu_beta = None
            sigma_beta = None
            for beta in results.betas:
                if beta.name == 'mu':
                    mu_beta = beta
                elif beta.name == 'sigma':
                    sigma_beta = beta
            
            if mu_beta is not None and sigma_beta is not None:
                # For log-normal distribution, mean = exp(mu + sigma^2/2)
                # This gives WTP in scaled units, so multiply by cost_scale for original units
                mu = float(mu_beta.value)
                sigma = float(sigma_beta.value)
                mean_wtp_scaled = np.exp(mu + sigma**2/2)
                mean_wtp = mean_wtp_scaled * cost_scale  # Convert to original units
                
                self.wtp_metrics['mxl_safety_vs_tt'] = {
                    'mean_wtp_minutes_per_unit': mean_wtp,
                    'mean_wtp_scaled': mean_wtp_scaled,
                    'cost_scale': cost_scale,
                    'mu': mu,
                    'sigma': sigma,
                    'log_likelihood': results.logLike,
                    'model_type': 'MXL'
                }
                print(f"MXL WTP for Safety vs Travel Time: {mean_wtp:.3f} minutes per safety unit (scaled: {mean_wtp_scaled:.3f})")
            else:
                print("Warning: Could not find mu and sigma parameters in MXL WTP model results")
        
        # MXL Safety vs Traffic Lights WTP (from separate MXL WTP space model)
        if hasattr(self, 'wtp_mxl_safety_tl_results'):
            results, obs_per_ind, cost_attr, wtp_attr = self.wtp_mxl_safety_tl_results
            
            # Get the cost scale used in the model
            cost_scale = 10 if cost_attr == 'TT' else 3
            
            # Extract mu and sigma from betas
            mu_beta = None
            sigma_beta = None
            for beta in results.betas:
                if beta.name == 'mu':
                    mu_beta = beta
                elif beta.name == 'sigma':
                    sigma_beta = beta
            
            if mu_beta is not None and sigma_beta is not None:
                # For log-normal distribution, mean = exp(mu + sigma^2/2)
                # This gives WTP in scaled units, so multiply by cost_scale for original units
                mu = float(mu_beta.value)
                sigma = float(sigma_beta.value)
                mean_wtp_scaled = np.exp(mu + sigma**2/2)
                mean_wtp = mean_wtp_scaled * cost_scale  # Convert to original units
                
                self.wtp_metrics['mxl_safety_vs_tl'] = {
                    'mean_wtp_lights_per_unit': mean_wtp,
                    'mean_wtp_scaled': mean_wtp_scaled,
                    'cost_scale': cost_scale,
                    'mu': mu,
                    'sigma': sigma,
                    'log_likelihood': results.logLike,
                    'model_type': 'MXL'
                }
                print(f"MXL WTP for Safety vs Traffic Lights: {mean_wtp:.3f} traffic lights per safety unit (scaled: {mean_wtp_scaled:.3f})")
            else:
                print("Warning: Could not find mu and sigma parameters in MXL WTP model results")
                
        # MNL Safety vs Travel Time WTP (from separate MNL WTP space model)
        if hasattr(self, 'wtp_mnl_safety_tt_results'):
            results, cost_attr, wtp_attr = self.wtp_mnl_safety_tt_results
            
            # Get the cost scale used in the model
            cost_scale = 10 if cost_attr == 'TT' else 3
            
            # Extract B_wtp from betas (direct WTP estimate in MNL)
            wtp_beta = None
            for beta in results.betas:
                if beta.name == 'B_wtp':
                    wtp_beta = beta
                    break
            
            if wtp_beta is not None:
                # For MNL, B_wtp gives WTP in scaled units, so multiply by cost_scale for original units
                mean_wtp_scaled = float(wtp_beta.value)
                mean_wtp = mean_wtp_scaled * cost_scale  # Convert to original units
                
                # Scale standard error as well if available
                std_err_scaled = float(wtp_beta.stdErr) if hasattr(wtp_beta, 'stdErr') else None
                std_err = std_err_scaled * cost_scale if std_err_scaled is not None else None
                
                self.wtp_metrics['mnl_safety_vs_tt'] = {
                    'mean_wtp_minutes_per_unit': mean_wtp,
                    'wtp_estimate': mean_wtp,
                    'wtp_estimate_scaled': mean_wtp_scaled,
                    'cost_scale': cost_scale,
                    'std_error': std_err,
                    'std_error_scaled': std_err_scaled,
                    't_stat': float(wtp_beta.robust_tTest) if hasattr(wtp_beta, 'robust_tTest') else None,
                    'p_value': float(wtp_beta.robust_pValue) if hasattr(wtp_beta, 'robust_pValue') else None,
                    'log_likelihood': results.logLike,
                    'model_type': 'MNL'
                }
                print(f"MNL WTP for Safety vs Travel Time: {mean_wtp:.3f} minutes per safety unit (scaled: {mean_wtp_scaled:.3f})")
            else:
                print("Warning: Could not find B_wtp parameter in MNL WTP model results")
                
        # MNL Safety vs Traffic Lights WTP (from separate MNL WTP space model)
        if hasattr(self, 'wtp_mnl_safety_tl_results'):
            results, cost_attr, wtp_attr = self.wtp_mnl_safety_tl_results
            
            # Get the cost scale used in the model
            cost_scale = 10 if cost_attr == 'TT' else 3
            
            # Extract B_wtp from betas (direct WTP estimate in MNL)
            wtp_beta = None
            for beta in results.betas:
                if beta.name == 'B_wtp':
                    wtp_beta = beta
                    break
            
            if wtp_beta is not None:
                # For MNL, B_wtp gives WTP in scaled units, so multiply by cost_scale for original units
                mean_wtp_scaled = float(wtp_beta.value)
                mean_wtp = mean_wtp_scaled * cost_scale  # Convert to original units
                
                # Scale standard error as well if available
                std_err_scaled = float(wtp_beta.stdErr) if hasattr(wtp_beta, 'stdErr') else None
                std_err = std_err_scaled * cost_scale if std_err_scaled is not None else None
                
                self.wtp_metrics['mnl_safety_vs_tl'] = {
                    'mean_wtp_lights_per_unit': mean_wtp,
                    'wtp_estimate': mean_wtp,
                    'wtp_estimate_scaled': mean_wtp_scaled,
                    'cost_scale': cost_scale,
                    'std_error': std_err,
                    'std_error_scaled': std_err_scaled,
                    't_stat': float(wtp_beta.robust_tTest) if hasattr(wtp_beta, 'robust_tTest') else None,
                    'p_value': float(wtp_beta.robust_pValue) if hasattr(wtp_beta, 'robust_pValue') else None,
                    'log_likelihood': results.logLike,
                    'model_type': 'MNL'
                }
                print(f"MNL WTP for Safety vs Traffic Lights: {mean_wtp:.3f} traffic lights per safety unit (scaled: {mean_wtp_scaled:.3f})")
            else:
                print("Warning: Could not find B_wtp parameter in MNL WTP model results")

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

        for name, (train_res, obs_per_ind) in models_to_report.items():
            if not obs_per_ind:
                obs_per_ind = self.min_obs_per_individual
                
            model_metrics[name] = {
                'train': extract_mxl_metrics(train_res, obs_per_ind, train_res.numberOfObservations)
            }
            # Use train_res.betas instead of get_estimated_parameters()
            betas = train_res.betas
            model_params[name] = betas
            all_param_names.update([beta.name for beta in betas])

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
            'Sample size': ('train', 'n_observations', 'd'),
            'Log-Likelihood': ('train', 'log_likelihood', '.2f'),
            'Rho-squared': ('train', 'pseudo_r2', '.4f'),
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
                betas = model_params[name]
                # Find the beta object with matching name
                beta_obj = None
                for beta in betas:
                    if beta.name == param:
                        beta_obj = beta
                        break
                
                if beta_obj is not None:
                    val = float(beta_obj.value)
                    t = float(beta_obj.robust_tTest)
                    p = float(beta_obj.robust_pValue)
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
        
        # Generate WTP results table if available
        if hasattr(self, 'wtp_metrics'):
            self._generate_wtp_table()
    
    def _generate_wtp_table(self):
        """Generate a LaTeX table with WTP results comparing MXL and MNL models."""
        print("\nGenerating WTP results table...")
        
        lines = [
            "\\begin{table}[htbp]",
            "    \\centering",
            "    \\caption{Willingness-to-Pay Results for Safety Perception: Mixed Logit vs Multinomial Logit}",
            "    \\label{tab:wtp_results}",
            "    \\begin{tabular}{lcccc}",
            "    \\toprule",
            "    & \\multicolumn{2}{c}{\\textbf{Mixed Logit (MXL)}} & \\multicolumn{2}{c}{\\textbf{Multinomial Logit (MNL)}} \\\\",
            "    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
            "    Trade-off & Mean WTP & 95\\% CI & Mean WTP & 95\\% CI \\\\",
            "    \\midrule",
        ]
        
        # Safety vs Travel Time
        mxl_tt_content = "--"
        mnl_tt_content = "--"
        
        if 'mxl_safety_vs_tt' in self.wtp_metrics:
            wtp_tt = self.wtp_metrics['mxl_safety_vs_tt']['mean_wtp_minutes_per_unit']
            mu = self.wtp_metrics['mxl_safety_vs_tt']['mu']
            sigma = self.wtp_metrics['mxl_safety_vs_tt']['sigma']
            cost_scale = self.wtp_metrics['mxl_safety_vs_tt']['cost_scale']
            
            # Approximate 95% CI for log-normal mean (using delta method approximation)
            # Apply scaling to convert from scaled units to original units
            ci_lower_scaled = np.exp(mu + sigma**2/2) * 0.8  # Approximate in scaled units
            ci_upper_scaled = np.exp(mu + sigma**2/2) * 1.2  # Approximate in scaled units
            ci_lower = ci_lower_scaled * cost_scale  # Convert to original units
            ci_upper = ci_upper_scaled * cost_scale  # Convert to original units
            
            mxl_tt_content = f"{wtp_tt:.2f} min & [{ci_lower:.2f}, {ci_upper:.2f}]"
        
        if 'mnl_safety_vs_tt' in self.wtp_metrics:
            wtp_tt = self.wtp_metrics['mnl_safety_vs_tt']['mean_wtp_minutes_per_unit']
            std_err = self.wtp_metrics['mnl_safety_vs_tt'].get('std_error', None)
            
            if std_err is not None:
                # 95% CI using normal approximation: ±1.96 * std_err
                ci_lower = wtp_tt - 1.96 * std_err
                ci_upper = wtp_tt + 1.96 * std_err
                mnl_tt_content = f"{wtp_tt:.2f} min & [{ci_lower:.2f}, {ci_upper:.2f}]"
            else:
                mnl_tt_content = f"{wtp_tt:.2f} min & [--]"
        
        lines.append(f"    Safety vs Travel Time & {mxl_tt_content} & {mnl_tt_content} \\\\")
        
        # Safety vs Traffic Lights
        mxl_tl_content = "--"
        mnl_tl_content = "--"
        
        if 'mxl_safety_vs_tl' in self.wtp_metrics:
            wtp_tl = self.wtp_metrics['mxl_safety_vs_tl']['mean_wtp_lights_per_unit']
            mu = self.wtp_metrics['mxl_safety_vs_tl']['mu']
            sigma = self.wtp_metrics['mxl_safety_vs_tl']['sigma']
            cost_scale = self.wtp_metrics['mxl_safety_vs_tl']['cost_scale']
            
            # Approximate 95% CI for log-normal mean
            # Apply scaling to convert from scaled units to original units
            ci_lower_scaled = np.exp(mu + sigma**2/2) * 0.8  # Approximate in scaled units
            ci_upper_scaled = np.exp(mu + sigma**2/2) * 1.2  # Approximate in scaled units
            ci_lower = ci_lower_scaled * cost_scale  # Convert to original units
            ci_upper = ci_upper_scaled * cost_scale  # Convert to original units
            
            mxl_tl_content = f"{wtp_tl:.2f} lights & [{ci_lower:.2f}, {ci_upper:.2f}]"
        
        if 'mnl_safety_vs_tl' in self.wtp_metrics:
            wtp_tl = self.wtp_metrics['mnl_safety_vs_tl']['mean_wtp_lights_per_unit']
            std_err = self.wtp_metrics['mnl_safety_vs_tl'].get('std_error', None)
            
            if std_err is not None:
                # 95% CI using normal approximation: ±1.96 * std_err
                ci_lower = wtp_tl - 1.96 * std_err
                ci_upper = wtp_tl + 1.96 * std_err
                mnl_tl_content = f"{wtp_tl:.2f} lights & [{ci_lower:.2f}, {ci_upper:.2f}]"
            else:
                mnl_tl_content = f"{wtp_tl:.2f} lights & [--]"
        
        lines.append(f"    Safety vs Traffic Lights & {mxl_tl_content} & {mnl_tl_content} \\\\")
        
        lines.extend([
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\\\[0.5em]",
            "    \\parbox{\\textwidth}{\\footnotesize",
            "    Note: WTP values indicate how much additional travel time (minutes) or",
            "    traffic lights cyclists are willing to accept for a one-unit increase in",
            "    perceived safety score. MXL confidence intervals are approximate using",
            "    log-normal distribution properties. MNL confidence intervals use normal",
            "    approximation with robust standard errors where available.}",
            "\\end{table}"
        ])

        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'wtp_results.tex'
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        print(f"WTP table saved to {table_path}")
        
        # Also generate individual tables for each model type for detailed analysis
        self._generate_detailed_wtp_tables()
    
    def _generate_detailed_wtp_tables(self):
        """Generate separate detailed tables for MXL and MNL WTP models."""
        
        # Generate MXL WTP table
        if any(key.startswith('mxl_') for key in self.wtp_metrics.keys()):
            self._generate_mxl_wtp_table()
        
        # Generate MNL WTP table  
        if any(key.startswith('mnl_') for key in self.wtp_metrics.keys()):
            self._generate_mnl_wtp_table()
    
    def _generate_mxl_wtp_table(self):
        """Generate detailed table for MXL WTP results."""
        lines = [
            "\\begin{table}[htbp]",
            "    \\centering", 
            "    \\caption{Mixed Logit WTP Space Model Results}",
            "    \\label{tab:wtp_mxl_detailed}",
            "    \\begin{tabular}{lcccc}",
            "    \\toprule",
            "    Trade-off & Mean WTP & $\\mu$ & $\\sigma$ & Log-Likelihood \\\\",
            "    \\midrule",
        ]
        
        if 'mxl_safety_vs_tt' in self.wtp_metrics:
            metrics = self.wtp_metrics['mxl_safety_vs_tt']
            wtp = metrics['mean_wtp_minutes_per_unit']
            mu = metrics['mu']
            sigma = metrics['sigma']
            ll = metrics['log_likelihood']
            lines.append(f"    Safety vs Travel Time & {wtp:.3f} min & {mu:.3f} & {sigma:.3f} & {ll:.2f} \\\\")
        
        if 'mxl_safety_vs_tl' in self.wtp_metrics:
            metrics = self.wtp_metrics['mxl_safety_vs_tl'] 
            wtp = metrics['mean_wtp_lights_per_unit']
            mu = metrics['mu']
            sigma = metrics['sigma']
            ll = metrics['log_likelihood']
            lines.append(f"    Safety vs Traffic Lights & {wtp:.3f} lights & {mu:.3f} & {sigma:.3f} & {ll:.2f} \\\\")
        
        lines.extend([
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\\\[0.5em]",
            "    \\parbox{\\textwidth}{\\footnotesize",
            "    Note: Mixed logit WTP space models with log-normal WTP distribution.",
            "    Mean WTP = exp($\\mu$ + $\\sigma^2$/2). Includes significant segmentation features.}",
            "\\end{table}"
        ])
        
        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'wtp_mxl_detailed.tex'
        with open(table_path, 'w') as f:
            f.write(latex_content)
        print(f"Detailed MXL WTP table saved to {table_path}")
    
    def _generate_mnl_wtp_table(self):
        """Generate detailed table for MNL WTP results."""
        lines = [
            "\\begin{table}[htbp]",
            "    \\centering",
            "    \\caption{Multinomial Logit WTP Space Model Results}",
            "    \\label{tab:wtp_mnl_detailed}",
            "    \\begin{tabular}{lcccc}",
            "    \\toprule",
            "    Trade-off & WTP Estimate & Std. Error & t-statistic & p-value \\\\",
            "    \\midrule",
        ]
        
        if 'mnl_safety_vs_tt' in self.wtp_metrics:
            metrics = self.wtp_metrics['mnl_safety_vs_tt']
            wtp = metrics['wtp_estimate']
            std_err = metrics.get('std_error', '--')
            t_stat = metrics.get('t_stat', '--')
            p_val = metrics.get('p_value', '--')
            
            std_err_str = f"{std_err:.3f}" if std_err != '--' else '--'
            t_stat_str = f"{t_stat:.3f}" if t_stat != '--' else '--'
            p_val_str = f"{p_val:.3f}" if p_val != '--' else '--'
            
            lines.append(f"    Safety vs Travel Time & {wtp:.3f} min & {std_err_str} & {t_stat_str} & {p_val_str} \\\\")
        
        if 'mnl_safety_vs_tl' in self.wtp_metrics:
            metrics = self.wtp_metrics['mnl_safety_vs_tl']
            wtp = metrics['wtp_estimate']
            std_err = metrics.get('std_error', '--')
            t_stat = metrics.get('t_stat', '--')
            p_val = metrics.get('p_value', '--')
            
            std_err_str = f"{std_err:.3f}" if std_err != '--' else '--'
            t_stat_str = f"{t_stat:.3f}" if t_stat != '--' else '--'
            p_val_str = f"{p_val:.3f}" if p_val != '--' else '--'
            
            lines.append(f"    Safety vs Traffic Lights & {wtp:.3f} lights & {std_err_str} & {t_stat_str} & {p_val_str} \\\\")
        
        lines.extend([
            "    \\bottomrule",
            "    \\end{tabular}",
            "    \\\\[0.5em]",
            "    \\parbox{\\textwidth}{\\footnotesize",
            "    Note: Multinomial logit WTP space models with fixed WTP parameters.",
            "    Includes significant segmentation features. Statistics use robust standard errors.}",
            "\\end{table}"
        ])
        
        latex_content = "\n".join(lines)
        table_path = self.output_dir / 'wtp_mnl_detailed.tex'
        with open(table_path, 'w') as f:
            f.write(latex_content)
        print(f"Detailed MNL WTP table saved to {table_path}")
    
def main(checkpoint_dir=None):
    """
    Main function to run the choice model benchmark
    
    Args:
        checkpoint_dir: Optional path to existing model results directory for checkpoint loading
                       e.g., 'reports/models/mxl_choice_20250725_122947'
    """
    
    print("=== Choice Model Benchmarking ===")
    
    benchmark = ChoiceModelBenchmark(checkpoint_dir=checkpoint_dir)
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
    main(checkpoint_dir='reports/models/mxl_choice_20250725_122947') 
    # main()