"""
Choice Model Benchmarking Script

This script implements and benchmarks different discrete choice models:
1. Base model: only TL1, TT1, TL2, TT2
2. Base + safety scores: TL1, TT1, TL2, TT2, safety scores
3. Base + segmentation: TL1, TT1, TL2, TT2, pixel ratios
4. Base + safety + segmentation: TL1, TT1, TL2, TT2, safety scores, pixel ratios

Compares models using Log-Likelihood, BIC, AIC, and Pseudo R-squared metrics.
"""

import os
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import logging
from biogeme.expressions import Beta, Variable, log, exp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class ChoiceModelBenchmark:
    """Benchmarks different discrete choice models with various feature combinations"""
    
    def __init__(self, base_output_dir='reports/models'):
        # Create timestamped folder name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_folder = f"choice_{timestamp}"
        
        # Create the timestamped output directory
        self.output_dir = Path(base_output_dir) / timestamped_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.output_dir}")
        
        # Configure Biogeme logging to be less verbose
        logger = logging.getLogger('biogeme')
        logger.setLevel(logging.WARNING)
        
        self.results = {}
        
    def load_and_prepare_data(self, 
                             cv_dcm_path='data/raw/cv_dcm.csv',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv',
                             original_results_path='data/raw/df_choice_with_Vimg.csv'):
        """Load and prepare all datasets for modeling"""
        
        print("Loading datasets...")
        
        # Load main choice data
        self.choice_data = pd.read_csv(cv_dcm_path)
        print(f"Loaded choice data: {len(self.choice_data)} observations")
        # print the number of rows with "train" and "test" columns
        train_count = self.choice_data[self.choice_data['train'] == 1].shape[0]
        test_count = self.choice_data[self.choice_data['test'] == 1].shape[0]
        print(f"Training observations: {train_count}, Test observations: {test_count}")

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
            print("Warning: No segmentation features remaining after variance filtering")
            return
        
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
        
    def estimate_mnl(self, V1, V2, Choice, database, name, save_auxiliary=True):
        """
        Standardized estimation function that matches the working MNL.ipynb approach
        
        Args:
            save_auxiliary: If False, don't save .pickle, .html, .tex files (for intermediate models)
        """
        # Create a dictionary to list the utility functions with the numbering of alternatives
        V = {1: V1, 2: V2}
            
        # Create a dictionary called av to describe the availability conditions of each alternative
        av = {1: 1, 2: 1} 

        # Define the choice model: The function models.logit() computes the MNL choice probabilities
        prob = models.logit(V, av, Choice)

        # Define the log-likelihood   
        LL = log(prob)
       
        # Create the Biogeme object
        biogeme = bio.BIOGEME(database, LL)
        
        # Set the model name
        biogeme.modelName = name

        # Configure to save results in our timestamped folder
        biogeme.generate_pickle = save_auxiliary
        biogeme.generate_html = save_auxiliary
        biogeme.save_iterations = save_auxiliary
        
        # Change working directory to our output folder for Biogeme files
        original_cwd = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Calculate the null log-likelihood
            biogeme.calculate_null_loglikelihood(av)

            # Start the estimation
            results = biogeme.estimate()
            
            # Save additional results manually only if requested
            if save_auxiliary:
                try:
                    results.write_latex()
                except AttributeError:
                    results.writeLaTeX()
                
                try:
                    results.write_pickle()
                except AttributeError:
                    results.writePickle()
                    
                try:
                    results.write_html()
                except AttributeError:
                    results.writeHTML()
        finally:
            # Always return to original directory
            os.chdir(original_cwd)
         
        return results
        
    def run_base_model(self, data_subset=None):
        """Run base model with only TL1, TT1, TL2, TT2"""
        
        print("Running base model...")
        
        if data_subset is None:
            data_subset = self.merged_data
            
        # Select only required columns and ensure they are numeric - SAME ORDER AS MNL.ipynb
        attributes = ['TL1', 'TT1', 'TL2', 'TT2', 'CHOICE']
        model_data = data_subset[attributes].copy()
        model_data = model_data.dropna()
        
        # Ensure proper data types
        model_data = model_data.astype(float)
        
        print(f"Base model data shape: {model_data.shape}")
        print(f"Choice values: {model_data['CHOICE'].unique()}")
        
        # Create Biogeme database - SAME AS MNL.ipynb
        database = db.Database('base_model', model_data)
        
        # Create the variables for biogeme - SAME AS MNL.ipynb
        TL1, TT1, TL2, TT2, CHOICE = database.variables['TL1'], database.variables['TT1'], database.variables['TL2'], database.variables['TT2'], database.variables['CHOICE']

        # Define parameters - SAME INITIAL VALUES AS MNL.ipynb
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        
        # Utility functions - SAME AS MNL.ipynb
        V1 = B_TL * TL1 / 3 + B_TT * TT1 / 10
        V2 = B_TL * TL2 / 3 + B_TT * TT2 / 10
        
        # Use the standardized estimation function
        results = self.estimate_mnl(V1, V2, CHOICE, database, 'base_model')
        
        return results
    
    def run_base_plus_safety_model(self, data_subset=None):
        """Run base model + safety scores"""
        
        print("Running base + safety model...")
        
        if data_subset is None:
            data_subset = self.merged_data
            
        # Select required columns - maintain same order as base model
        attributes = ['TL1', 'TT1', 'TL2', 'TT2', 'safety_score_1', 'safety_score_2', 'CHOICE']
        model_data = data_subset[attributes].copy()
        model_data = model_data.dropna()
        
        if len(model_data) == 0:
            raise ValueError("No data available after filtering for safety scores")
        
        # Ensure proper data types
        model_data = model_data.astype(float)
        
        print(f"Safety model data shape: {model_data.shape}")
        
        database = db.Database('base_safety_model', model_data)
        
        # Create the variables for biogeme
        TL1, TT1, TL2, TT2, safety_score_1, safety_score_2, CHOICE = database.variables['TL1'], database.variables['TT1'], database.variables['TL2'], database.variables['TT2'], database.variables['safety_score_1'], database.variables['safety_score_2'], database.variables['CHOICE']
        
        # Define parameters
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        B_SAFETY = Beta('B_SAFETY', 0, None, None, 0)
        
        # Define utility functions
        V1 = B_TL * TL1 / 3 + B_TT * TT1 / 10 + B_SAFETY * safety_score_1
        V2 = B_TL * TL2 / 3 + B_TT * TT2 / 10 + B_SAFETY * safety_score_2
        
        # Use the standardized estimation function
        results = self.estimate_mnl(V1, V2, CHOICE, database, 'base_safety_model')
        
        return results
    
    def run_base_plus_segmentation_model(self, data_subset=None, max_features=20):
        """Run base model + segmentation pixel ratios"""
        
        print("Running base + segmentation model...")
        
        if data_subset is None:
            data_subset = self.merged_data
            
        # Select segmentation features (already filtered for variance and scaled)
        # Use the available features up to max_features limit
        seg_features_to_use = self.seg_features[:max_features]
        
        print(f"Using top {len(seg_features_to_use)} segmentation features: {seg_features_to_use}")
        
        # Prepare model data with selected features
        attributes = ['TL1', 'TT1', 'TL2', 'TT2']
        for feature in seg_features_to_use:
            attributes.extend([f"{feature}_1", f"{feature}_2"])
        attributes.append('CHOICE')
        
        model_data = data_subset[attributes].copy()
        model_data = model_data.dropna()
        
        # Ensure proper data types
        model_data = model_data.astype(float)
        
        print(f"Segmentation model data shape: {model_data.shape}")
        
        database = db.Database('base_segmentation_model', model_data)
        
        # Create variables
        TL1, TT1, TL2, TT2, CHOICE = database.variables['TL1'], database.variables['TT1'], database.variables['TL2'], database.variables['TT2'], database.variables['CHOICE']
        
        # Define parameters
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        
        # Add segmentation variables and parameters
        seg_terms_1 = [B_TL * TL1 / 3, B_TT * TT1 / 10]
        seg_terms_2 = [B_TL * TL2 / 3, B_TT * TT2 / 10]
        
        for feature in seg_features_to_use:
            try:
                var1 = database.variables[f"{feature}_1"]
                var2 = database.variables[f"{feature}_2"]
                beta = Beta(f"B_{feature.replace(' ', '_').replace('-', '_')}", 0, None, None, 0)
                
                seg_terms_1.append(beta * var1)
                seg_terms_2.append(beta * var2)
            except Exception as e:
                print(f"Warning: Could not add feature {feature}: {e}")
                continue
        
        # Define utility functions
        V1 = sum(seg_terms_1)
        V2 = sum(seg_terms_2)
        
        # Use the standardized estimation function
        results = self.estimate_mnl(V1, V2, CHOICE, database, 'base_segmentation_model')
        
        return results
    
    def run_full_model(self, data_subset=None, max_seg_features=20):
        """Run base model + safety scores + segmentation pixel ratios"""
        
        print("Running full model (base + safety + segmentation)...")
        
        if data_subset is None:
            data_subset = self.merged_data
            
        # Select segmentation features (already filtered for variance and scaled)
        # Use the available features up to max_seg_features limit
        seg_features_to_use = self.seg_features[:max_seg_features]
        
        print(f"Using top {len(seg_features_to_use)} segmentation features: {seg_features_to_use}")
        
        # Prepare model data
        attributes = ['TL1', 'TT1', 'TL2', 'TT2', 'safety_score_1', 'safety_score_2']
        for feature in seg_features_to_use:
            attributes.extend([f"{feature}_1", f"{feature}_2"])
        attributes.append('CHOICE')
        
        model_data = data_subset[attributes].copy()
        model_data = model_data.dropna()
        
        if len(model_data) == 0:
            raise ValueError("No data available after filtering")
        
        # Ensure proper data types
        model_data = model_data.astype(float)
        
        print(f"Full model data shape: {model_data.shape}")
        
        database = db.Database('full_model', model_data)
        
        # Create variables
        TL1, TT1, TL2, TT2, safety_score_1, safety_score_2, CHOICE = database.variables['TL1'], database.variables['TT1'], database.variables['TL2'], database.variables['TT2'], database.variables['safety_score_1'], database.variables['safety_score_2'], database.variables['CHOICE']
        
        # Define parameters
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        B_SAFETY = Beta('B_SAFETY', 0, None, None, 0)
        
        # Add segmentation variables and parameters
        V1_components = [B_TL * TL1 / 3, B_TT * TT1 / 10, B_SAFETY * safety_score_1]
        V2_components = [B_TL * TL2 / 3, B_TT * TT2 / 10, B_SAFETY * safety_score_2]
        
        for feature in seg_features_to_use:
            try:
                var1 = database.variables[f"{feature}_1"]
                var2 = database.variables[f"{feature}_2"]
                beta = Beta(f"B_{feature.replace(' ', '_').replace('-', '_')}", 0, None, None, 0)
                
                V1_components.append(beta * var1)
                V2_components.append(beta * var2)
            except Exception as e:
                print(f"Warning: Could not add feature {feature}: {e}")
                continue
        
        # Define utility functions
        V1 = sum(V1_components)
        V2 = sum(V2_components)
        
        # Use the standardized estimation function
        results = self.estimate_mnl(V1, V2, CHOICE, database, 'full_model')
        
        return results
    
    def run_top_n_feature_selection(self, data_subset=None, n_features=10, include_safety=True):
        """
        Select top N segmentation features based on variance ranking (to avoid overfitting)
        
        Args:
            data_subset: Data to use for selection
            n_features: Number of segmentation features to include
            include_safety: Whether to include safety_score
        
        Returns:
            Model results and feature list
        """
        print(f"Running top-{n_features} feature selection (include_safety={include_safety})...")
        
        if data_subset is None:
            data_subset = self.merged_data
        
        # Calculate variance for all available features (only once if not already done)
        if not hasattr(self, 'feature_variances'):
            print("Calculating feature variances...")
            self.feature_variances = self._calculate_feature_variances(data_subset)
            
            # Rank segmentation features by variance (highest first)
            seg_variances = [(name, var) for name, var in self.feature_variances.items() 
                           if name in self.seg_features]
            self.ranked_seg_features = sorted(seg_variances, key=lambda x: x[1], reverse=True)
            print(f"Top 10 segmentation features by variance: {self.ranked_seg_features[:10]}")
        
        # Always start with base features
        selected_features = ['TL', 'TT']
        
        # Add safety_score if requested and has reasonable variance
        if include_safety:
            safety_variance = self.feature_variances.get('safety_score', 0)
            if safety_variance > 0.01:  # Include if variance > threshold
                selected_features.append('safety_score')
                print(f"Including safety_score (variance: {safety_variance:.6f})")
            else:
                print(f"Excluding safety_score (low variance: {safety_variance:.6f})")
        
        # Add top N segmentation features
        added_features = 0
        for feature_name, variance in self.ranked_seg_features:
            if added_features < n_features:
                selected_features.append(feature_name)
                added_features += 1
        
        print(f"Selected features ({len(selected_features)} total): {selected_features}")
        
        # Build the model with selected features
        results = self._build_stepwise_model_with_features(data_subset, selected_features)
        
        if results is None:
            raise ValueError(f"Failed to build top-{n_features} model")
        
        print(f"Top-{n_features} feature selection completed! LL: {results.data.logLike:.6f}")
        
        return results, selected_features
    
    def _calculate_feature_variances(self, data_subset):
        """
        Calculate variance for all available features (safety score + segmentation features)
        
        Args:
            data_subset: Data to calculate variances on
            
        Returns:
            Dictionary mapping feature names to their variances
        """
        variances = {}
        
        # Calculate safety score variance
        if 'safety_score_1' in data_subset.columns and 'safety_score_2' in data_subset.columns:
            # Combine data from both alternatives
            safety_values = pd.concat([
                data_subset['safety_score_1'].dropna(),
                data_subset['safety_score_2'].dropna()
            ])
            variances['safety_score'] = safety_values.var()
        
        # Calculate segmentation feature variances
        for feature in self.seg_features:
            col1 = f"{feature}_1"
            col2 = f"{feature}_2"
            
            if col1 in data_subset.columns and col2 in data_subset.columns:
                # Combine data from both alternatives
                combined_values = pd.concat([
                    data_subset[col1].dropna(),
                    data_subset[col2].dropna()
                ])
                
                if len(combined_values) > 0:
                    feature_variance = combined_values.var()
                    variances[feature] = feature_variance
                else:
                    variances[feature] = 0.0
        
        return variances
    
    def run_stepwise_feature_selection_no_safety(self, data_subset=None):
        """
        Take the stepwise_best model and remove safety_score if it was selected
        
        Args:
            data_subset: Data to use for model estimation
        
        Returns:
            Model results without safety feature and modified feature list
        """
        print("Running step-wise model without safety score...")
        
        if data_subset is None:
            data_subset = self.merged_data
        
        # Check if stepwise selection has been run
        if not hasattr(self, 'stepwise_history'):
            raise ValueError("Must run stepwise feature selection first before running no-safety version")
        
        # Get the features from stepwise_best model
        stepwise_features = self.stepwise_history[-1]['features'].copy()
        
        # Remove safety_score if it was selected
        features_no_safety = [f for f in stepwise_features if f != 'safety_score']
        
        print(f"Original stepwise features: {stepwise_features}")
        print(f"Features without safety: {features_no_safety}")
        
        if 'safety_score' not in stepwise_features:
            print("Safety score was not selected in stepwise model, features are identical")
        else:
            print("Removing safety_score from selected features")
        
        # Prepare model data with features (excluding safety)
        attributes = ['TL1', 'TT1', 'TL2', 'TT2']
        
        # Add segmentation features (excluding safety)
        for feature in features_no_safety:
            if feature not in ['TL', 'TT'] and feature in self.seg_features:
                attributes.extend([f"{feature}_1", f"{feature}_2"])
        
        attributes.append('CHOICE')
        
        model_data = data_subset[attributes].copy().dropna().astype(float)
        
        print(f"Model data shape (no safety): {model_data.shape}")
        
        # Create database
        database = db.Database('stepwise_no_safety_model', model_data)
        
        # Create variables
        var_dict = {}
        for col in attributes[:-1]:  # Exclude CHOICE
            var_dict[col] = database.variables[col]
        choice_var = database.variables['CHOICE']
        
        # Create parameters and utility functions
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        
        V1_components = [B_TL * var_dict['TL1'] / 3, B_TT * var_dict['TT1'] / 10]
        V2_components = [B_TL * var_dict['TL2'] / 3, B_TT * var_dict['TT2'] / 10]
        
        # Add segmentation features (NO SAFETY SCORE)
        for feature in features_no_safety:
            if feature not in ['TL', 'TT']:
                col1 = f"{feature}_1"
                col2 = f"{feature}_2"
                if col1 in var_dict and col2 in var_dict:
                    beta = Beta(f"B_{feature.replace(' ', '_').replace('-', '_')}", 0, None, None, 0)
                    V1_components.append(beta * var_dict[col1])
                    V2_components.append(beta * var_dict[col2])
        
        # Build utility functions
        V1 = sum(V1_components)
        V2 = sum(V2_components)
        
        # Estimate final model
        results = self.estimate_mnl(V1, V2, choice_var, database, 'stepwise_no_safety_model')
        
        print(f"Step-wise model (no safety) completed!")
        print(f"Final model features: {features_no_safety}")
        print(f"Final log-likelihood: {results.data.logLike:.6f}")
        
        # Create a simplified history for consistency with other methods
        simplified_history = [{
            'step': 0,
            'added_feature': 'stepwise_best_minus_safety',
            'features': features_no_safety,
            'log_likelihood': results.data.logLike,
            'improvement': 0.0
        }]
        
        # Store selection history for analysis
        self.stepwise_no_safety_history = simplified_history
        
        return results, features_no_safety, simplified_history
    
    def _build_stepwise_model_with_features(self, data_subset, selected_features):
        """
        Build a stepwise model using a specific set of selected features
        
        Args:
            data_subset: Data to use for model estimation
            selected_features: List of features to include in the model
        
        Returns:
            Biogeme model results
        """
        try:
            # Prepare model data with selected features
            attributes = ['TL1', 'TT1', 'TL2', 'TT2']
            
            # Add safety score if selected
            if 'safety_score' in selected_features:
                attributes.extend(['safety_score_1', 'safety_score_2'])
            
            # Add segmentation features (excluding safety and base features)
            for feature in selected_features:
                if feature not in ['TL', 'TT', 'safety_score'] and feature in self.seg_features:
                    attributes.extend([f"{feature}_1", f"{feature}_2"])
            
            attributes.append('CHOICE')
            
            model_data = data_subset[attributes].copy().dropna().astype(float)
            
            if len(model_data) == 0:
                print(f"No data available for features: {selected_features}")
                return None
            
            # Create database
            database = db.Database('stepwise_model', model_data)
            
            # Create variables
            var_dict = {}
            for col in attributes[:-1]:  # Exclude CHOICE
                var_dict[col] = database.variables[col]
            choice_var = database.variables['CHOICE']
            
            # Create parameters and utility functions
            B_TL = Beta('B_TL', 0, None, None, 0)
            B_TT = Beta('B_TT', 0, None, None, 0)
            
            V1_components = [B_TL * var_dict['TL1'] / 3, B_TT * var_dict['TT1'] / 10]
            V2_components = [B_TL * var_dict['TL2'] / 3, B_TT * var_dict['TT2'] / 10]
            
            # Add safety terms if selected
            if 'safety_score' in selected_features:
                B_SAFETY = Beta('B_SAFETY', 0, None, None, 0)
                V1_components.append(B_SAFETY * var_dict['safety_score_1'])
                V2_components.append(B_SAFETY * var_dict['safety_score_2'])
            
            # Add segmentation features
            for feature in selected_features:
                if feature not in ['TL', 'TT', 'safety_score']:
                    col1 = f"{feature}_1"
                    col2 = f"{feature}_2"
                    if col1 in var_dict and col2 in var_dict:
                        beta = Beta(f"B_{feature.replace(' ', '_').replace('-', '_')}", 0, None, None, 0)
                        V1_components.append(beta * var_dict[col1])
                        V2_components.append(beta * var_dict[col2])
            
            # Build utility functions
            V1 = sum(V1_components)
            V2 = sum(V2_components)
            
            # Estimate model (don't save auxiliary files for intermediate models)
            results = self.estimate_mnl(V1, V2, choice_var, database, 'stepwise_model', save_auxiliary=False)
            
            return results
            
        except Exception as e:
            print(f"Error building stepwise model with features {selected_features}: {e}")
            return None
    
    def compute_utilities_for_all_images(self, biogeme_results, model_name):
        """
        Compute utility values for all images using the estimated model parameters
        
        Args:
            biogeme_results: Trained Biogeme model results
            model_name: Name of the model (for column naming)
            
        Returns:
            DataFrame with image names and their utilities
        """
        print(f"Computing utilities for model: {model_name}")
        
        # Get the estimated parameters
        estimated_params = biogeme_results.get_estimated_parameters()
        param_dict = {}
        for param_name, param_data in estimated_params.iterrows():
            param_dict[param_name] = param_data['Value']
        
        # Get all unique images from both IMG1 and IMG2
        all_images = set(self.merged_data['IMG1'].dropna().unique()) | set(self.merged_data['IMG2'].dropna().unique())
        
        utilities = []
        
        for image in all_images:
            try:
                # Find a row where this image appears to get its features
                img_row = None
                alt_num = None
                
                # Check if image appears as IMG1
                img1_rows = self.merged_data[self.merged_data['IMG1'] == image]
                if len(img1_rows) > 0:
                    img_row = img1_rows.iloc[0]
                    alt_num = 1
                else:
                    # Check if image appears as IMG2
                    img2_rows = self.merged_data[self.merged_data['IMG2'] == image]
                    if len(img2_rows) > 0:
                        img_row = img2_rows.iloc[0]
                        alt_num = 2
                
                if img_row is None:
                    print(f"Warning: Could not find data for image {image}")
                    continue
                
                # Calculate utility based on model parameters
                utility = 0
                
                # Add base terms (TL and TT)
                if alt_num == 1:
                    utility += param_dict.get('B_TL', 0) * img_row['TL1'] / 3
                    utility += param_dict.get('B_TT', 0) * img_row['TT1'] / 10
                else:
                    utility += param_dict.get('B_TL', 0) * img_row['TL2'] / 3
                    utility += param_dict.get('B_TT', 0) * img_row['TT2'] / 10
                
                # Add safety terms if present
                if 'B_SAFETY' in param_dict:
                    safety_col = f'safety_score_{alt_num}'
                    if safety_col in img_row and not pd.isna(img_row[safety_col]):
                        utility += param_dict['B_SAFETY'] * img_row[safety_col]
                
                # Add segmentation terms if present
                for param_name in param_dict.keys():
                    if param_name.startswith('B_') and param_name not in ['B_TL', 'B_TT', 'B_SAFETY']:
                        # Extract feature name from parameter name
                        feature_name = param_name[2:]  # Remove 'B_' prefix
                        feature_col = f"{feature_name}_{alt_num}"
                        
                        if feature_col in img_row and not pd.isna(img_row[feature_col]):
                            utility += param_dict[param_name] * img_row[feature_col]
                
                utilities.append({
                    'image_name': image,
                    f'V{model_name}': utility
                })
                
            except Exception as e:
                print(f"Warning: Could not compute utility for image {image}: {e}")
                continue
        
        utilities_df = pd.DataFrame(utilities)
        print(f"Computed utilities for {len(utilities_df)} images")
        
        return utilities_df
    
    def save_utility_comparison(self, stepwise_results, stepwise_no_safety_results):
        """
        Save utility comparison in the format of df_choice_with_Vimg.csv
        """
        print("Computing and saving utility comparison...")
        
        # Compute utilities for both models
        utilities_stepwise = self.compute_utilities_for_all_images(stepwise_results, 'stepwise_best')
        utilities_no_safety = self.compute_utilities_for_all_images(stepwise_no_safety_results, 'stepwise_wo_safety')
        
        # Merge utilities
        utilities_combined = utilities_stepwise.merge(utilities_no_safety, on='image_name', how='outer')
        
        # Create output directory if it doesn't exist
        output_dir = Path('data/processed/model_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original choice data to get the proper format
        choice_data = self.merged_data.copy()
        
        # Add utilities for IMG1 and IMG2
        img1_utils_stepwise = choice_data['IMG1'].map(dict(zip(utilities_combined['image_name'], utilities_combined['Vstepwise_best'])))
        img2_utils_stepwise = choice_data['IMG2'].map(dict(zip(utilities_combined['image_name'], utilities_combined['Vstepwise_best'])))
        
        img1_utils_no_safety = choice_data['IMG1'].map(dict(zip(utilities_combined['image_name'], utilities_combined['Vstepwise_wo_safety'])))
        img2_utils_no_safety = choice_data['IMG2'].map(dict(zip(utilities_combined['image_name'], utilities_combined['Vstepwise_wo_safety'])))
        
        # Create the output dataframe in the same format as df_choice_with_Vimg.csv
        output_df = choice_data[['IMG1', 'IMG2', 'CHOICE', 'train', 'test']].copy()
        output_df['V1_stepwise_best'] = img1_utils_stepwise
        output_df['V2_stepwise_best'] = img2_utils_stepwise
        output_df['V1_stepwise_wo_safety'] = img1_utils_no_safety
        output_df['V2_stepwise_wo_safety'] = img2_utils_no_safety
        
        # Calculate choice probabilities for both models
        # Stepwise best model probabilities
        exp_v1_stepwise = np.exp(output_df['V1_stepwise_best'].fillna(0))
        exp_v2_stepwise = np.exp(output_df['V2_stepwise_best'].fillna(0))
        output_df['prob1_stepwise_best'] = exp_v1_stepwise / (exp_v1_stepwise + exp_v2_stepwise)
        output_df['prob2_stepwise_best'] = exp_v2_stepwise / (exp_v1_stepwise + exp_v2_stepwise)
        
        # Stepwise wo safety model probabilities
        exp_v1_no_safety = np.exp(output_df['V1_stepwise_wo_safety'].fillna(0))
        exp_v2_no_safety = np.exp(output_df['V2_stepwise_wo_safety'].fillna(0))
        output_df['prob1_stepwise_wo_safety'] = exp_v1_no_safety / (exp_v1_no_safety + exp_v2_no_safety)
        output_df['prob2_stepwise_wo_safety'] = exp_v2_no_safety / (exp_v1_no_safety + exp_v2_no_safety)
        
        # Add predicted choice based on highest probability
        output_df['predicted_choice_stepwise_best'] = np.where(output_df['prob1_stepwise_best'] > output_df['prob2_stepwise_best'], 1, 2)
        output_df['predicted_choice_stepwise_wo_safety'] = np.where(output_df['prob1_stepwise_wo_safety'] > output_df['prob2_stepwise_wo_safety'], 1, 2)
        
        # Calculate prediction accuracy
        output_df['correct_stepwise_best'] = (output_df['CHOICE'] == output_df['predicted_choice_stepwise_best']).astype(int)
        output_df['correct_stepwise_wo_safety'] = (output_df['CHOICE'] == output_df['predicted_choice_stepwise_wo_safety']).astype(int)
        
        # Save the comparison file
        output_path = output_dir / 'df_choice_with_Vimg_comparison.csv'
        output_df.to_csv(output_path, index=False)
        print(f"Utility comparison saved to: {output_path}")
        
        # Print some summary statistics
        print("\nSummary Statistics:")
        print(f"Stepwise best model accuracy: {output_df['correct_stepwise_best'].mean():.3f}")
        print(f"Stepwise wo safety model accuracy: {output_df['correct_stepwise_wo_safety'].mean():.3f}")
        
        return output_df
    
    def extract_original_model_metrics(self):
        """Extract metrics from original model results"""
        
        # Calculate log-likelihood from original results
        original_ll = self.original_results['LLn'].sum()
        
        # Count parameters - this is estimated based on the original model structure
        # The original model likely had parameters for TL, TT, and image features
        estimated_params = 10  # Rough estimate
        
        n_obs = len(self.original_results)
        
        # Calculate AIC and BIC
        aic = 2 * estimated_params - 2 * original_ll
        bic = np.log(n_obs) * estimated_params - 2 * original_ll
        
        # Calculate pseudo R-squared (McFadden's)
        # Assuming null log-likelihood (equal choice probability)
        null_ll = n_obs * np.log(0.5)
        pseudo_r2 = 1 - (original_ll / null_ll)
        
        return {
            'log_likelihood': original_ll,
            'n_parameters': estimated_params,
            'n_observations': n_obs,
            'AIC': aic,
            'BIC': bic,
            'pseudo_r2': pseudo_r2
        }
    
    def evaluate_model_on_data(self, biogeme_results, data_subset, model_name):
        """
        Evaluate a trained model on a given dataset by recreating the model with the same structure
        but different data and using the estimated parameters from the training.
        
        Args:
            biogeme_results: Trained Biogeme model results
            data_subset: Data to evaluate on
            model_name: Name of the model for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get the estimated parameters
            estimated_params = biogeme_results.get_estimated_parameters()
            
            # For simplicity, we'll calculate log-likelihood based on model type
            # This is a simplified evaluation - ideally we'd rebuild the exact model
            
            n_obs = len(data_subset)
            n_params = len(biogeme_results.data.betaValues)
            
            # Extract parameter values
            param_dict = {}
            for param_name, param_data in estimated_params.iterrows():
                param_dict[param_name] = param_data['Value']
            
            # Calculate utilities and probabilities for each choice situation
            log_like_total = 0
            
            for idx, row in data_subset.iterrows():
                try:
                    # Base model calculation (TL and TT terms)
                    V1 = param_dict.get('B_TL', 0) * row['TL1'] / 3 + param_dict.get('B_TT', 0) * row['TT1'] / 10
                    V2 = param_dict.get('B_TL', 0) * row['TL2'] / 3 + param_dict.get('B_TT', 0) * row['TT2'] / 10
                    
                    # Add safety terms if present
                    if 'B_SAFETY' in param_dict and 'safety_score_1' in row and 'safety_score_2' in row:
                        if not pd.isna(row['safety_score_1']) and not pd.isna(row['safety_score_2']):
                            V1 += param_dict['B_SAFETY'] * row['safety_score_1']
                            V2 += param_dict['B_SAFETY'] * row['safety_score_2']
                    
                    # Add segmentation terms if present
                    for param_name in param_dict.keys():
                        if param_name.startswith('B_') and param_name not in ['B_TL', 'B_TT', 'B_SAFETY']:
                            # Extract feature name from parameter name
                            feature_name = param_name[2:]  # Remove 'B_' prefix
                            col1 = f"{feature_name}_1"
                            col2 = f"{feature_name}_2"
                            
                            if col1 in row and col2 in row:
                                if not pd.isna(row[col1]) and not pd.isna(row[col2]):
                                    V1 += param_dict[param_name] * row[col1]
                                    V2 += param_dict[param_name] * row[col2]
                    
                    # Calculate choice probabilities using logit formula
                    exp_V1 = np.exp(V1)
                    exp_V2 = np.exp(V2)
                    
                    prob_1 = exp_V1 / (exp_V1 + exp_V2)
                    prob_2 = exp_V2 / (exp_V1 + exp_V2)
                    
                    # Calculate log-likelihood contribution
                    chosen_alt = int(row['CHOICE'])
                    if chosen_alt == 1:
                        log_like_total += np.log(max(prob_1, 1e-10))  # Avoid log(0)
                    elif chosen_alt == 2:
                        log_like_total += np.log(max(prob_2, 1e-10))  # Avoid log(0)
                        
                except Exception as e:
                    # Skip this observation if there's an error
                    continue
            
            # Calculate final metrics
            aic = 2 * n_params - 2 * log_like_total
            bic = np.log(n_obs) * n_params - 2 * log_like_total
            
            # Calculate pseudo R-squared
            null_ll = n_obs * np.log(0.5)  # Assuming equal choice probability
            pseudo_r2 = 1 - (log_like_total / null_ll) if null_ll != 0 else 0
            
            return {
                'log_likelihood': log_like_total,
                'n_parameters': n_params,
                'n_observations': n_obs,
                'AIC': aic,
                'BIC': bic,
                'pseudo_r2': pseudo_r2
            }
            
        except Exception as e:
            print(f"Warning: Could not evaluate {model_name} on dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_models(self):
        """Run all model variants and collect results"""
        
        print("Starting model benchmarking...")
        
        # Split data into train and test
        train_data = self.merged_data[self.merged_data['train'] == 1].copy()
        test_data = self.merged_data[self.merged_data['test'] == 1].copy()
        
        print(f"Using {len(train_data)} training observations")
        print(f"Using {len(test_data)} test observations")
        print(f"Total available segmentation features: {len(self.seg_features)}")
        
        # Initialize results dictionaries for train and test
        self.train_results = {}
        self.test_results = {}
        
        try:
            # Run base model
            base_results = self.run_base_model(train_data)
            self.train_results['base'] = self._extract_metrics(base_results)
            self.test_results['base'] = self.evaluate_model_on_data(base_results, test_data, 'base')
            print("✓ Base model completed")
        except Exception as e:
            print(f"✗ Base model failed: {e}")
            self.train_results['base'] = None
            self.test_results['base'] = None
            
        try:
            # Run base + safety model
            safety_results = self.run_base_plus_safety_model(train_data)
            self.train_results['base_safety'] = self._extract_metrics(safety_results)
            self.test_results['base_safety'] = self.evaluate_model_on_data(safety_results, test_data, 'base_safety')
            print("✓ Base + safety model completed")
        except Exception as e:
            print(f"✗ Base + safety model failed: {e}")
            self.train_results['base_safety'] = None
            self.test_results['base_safety'] = None
            
        # Run top-N feature selection models
        max_features = min(50, len(self.seg_features))
        top_n_models = {}  # Store for creating no-safety versions
        
        for n_features in range(1, max_features + 1):
            # With safety
            try:
                print(f"Running top-{n_features} features (with safety)...")
                results_with_safety, features_with_safety = self.run_top_n_feature_selection(
                    train_data, n_features=n_features, include_safety=True
                )
                
                model_name = f'top_{n_features}_with_safety'
                self.train_results[model_name] = self._extract_metrics(results_with_safety)
                self.test_results[model_name] = self.evaluate_model_on_data(results_with_safety, test_data, model_name)
                print(f"✓ Top-{n_features} with safety completed")
                
                # Store for creating no-safety versions
                top_n_models[n_features] = {
                    'results_with_safety': results_with_safety,
                    'features_with_safety': features_with_safety
                }
                
            except Exception as e:
                print(f"✗ Top-{n_features} with safety failed: {e}")
                model_name = f'top_{n_features}_with_safety'
                self.train_results[model_name] = None
                self.test_results[model_name] = None
            
            # Without safety
            try:
                print(f"Running top-{n_features} features (without safety)...")
                results_no_safety, features_no_safety = self.run_top_n_feature_selection(
                    train_data, n_features=n_features, include_safety=False
                )
                
                model_name = f'top_{n_features}_no_safety'
                self.train_results[model_name] = self._extract_metrics(results_no_safety)
                self.test_results[model_name] = self.evaluate_model_on_data(results_no_safety, test_data, model_name)
                print(f"✓ Top-{n_features} without safety completed")
                
                # Store for creating no-safety versions
                if n_features in top_n_models:
                    top_n_models[n_features]['results_no_safety'] = results_no_safety
                    top_n_models[n_features]['features_no_safety'] = features_no_safety
                
            except Exception as e:
                print(f"✗ Top-{n_features} without safety failed: {e}")
                model_name = f'top_{n_features}_no_safety'
                self.train_results[model_name] = None
                self.test_results[model_name] = None
        
        print("✓ All top-N models completed")
        
        # Store both for backward compatibility
        self.results = self.train_results
        
        # Find the best model on test data and create utility comparison
        print("Finding best model on test data for utility comparison...")
        best_test_model = self._find_best_test_model()
        
        if best_test_model:
            print(f"Best test model: {best_test_model['name']} (Test LL: {best_test_model['test_ll']:.6f})")
            self._create_utility_comparison_from_best_model(best_test_model)
        
        print("All models completed!")

    def _find_best_test_model(self):
        """Find the model with best log-likelihood on test data"""
        best_model = None
        best_ll = float('-inf')
        
        for model_name, metrics in self.test_results.items():
            if metrics is not None:
                test_ll = metrics['log_likelihood']
                if test_ll > best_ll:
                    best_ll = test_ll
                    best_model = {
                        'name': model_name,
                        'test_ll': test_ll,
                        'train_metrics': self.train_results[model_name],
                        'test_metrics': metrics
                    }
        
        return best_model

    def _create_utility_comparison_from_best_model(self, best_model):
        """Create utility comparison using the best test model and its no-safety version"""
        try:
            model_name = best_model['name']
            print(f"Creating utility comparison from best model: {model_name}")
            
            # Find corresponding model features and create no-safety version
            if model_name.startswith('top_') and '_with_safety' in model_name:
                # Extract n_features from model name
                n_features = int(model_name.split('_')[1])
                
                # Get the with-safety features and create no-safety version
                best_results_with_safety, best_features_with_safety = self.run_top_n_feature_selection(
                    self.merged_data, n_features=n_features, include_safety=True
                )
                
                # Save auxiliary files for the best model
                print(f"Saving auxiliary files for best model with {n_features} features...")
                self._save_best_model_files(best_results_with_safety, f'best_model_top_{n_features}_with_safety')
                
                # Create no-safety version by removing safety_score
                best_features_no_safety = [f for f in best_features_with_safety if f != 'safety_score']
                best_results_no_safety = self._build_stepwise_model_with_features(
                    self.merged_data, best_features_no_safety
                )
                
                if best_results_no_safety:
                    # Save auxiliary files for the best model without safety score
                    print(f"Saving auxiliary files for best model without safety score...")
                    self._save_best_model_files(best_results_no_safety, f'best_model_top_{n_features}_no_safety')
                    
                    self.save_utility_comparison(best_results_with_safety, best_results_no_safety)
                    print("✓ Utility comparison completed")
                else:
                    print("✗ Failed to create no-safety version of best model")
            else:
                print(f"Cannot create utility comparison for model type: {model_name}")
                
        except Exception as e:
            print(f"✗ Utility comparison failed: {e}")
     
    def _save_best_model_files(self, biogeme_results, model_name):
        """Save .tex, .html, and .pickle files for the best model"""
        try:
            import os
            
            # Change to output directory
            original_cwd = os.getcwd()
            os.chdir(self.output_dir)
            
            # Set model name for Biogeme
            biogeme_results.data.modelName = model_name
            
            try:
                # Save LaTeX file
                try:
                    biogeme_results.write_latex()
                except AttributeError:
                    biogeme_results.writeLaTeX()
                
                # Save pickle file
                try:
                    biogeme_results.write_pickle()
                except AttributeError:
                    biogeme_results.writePickle()
                
                # Save HTML file
                try:
                    biogeme_results.write_html()
                except AttributeError:
                    biogeme_results.writeHTML()
                
                print(f"✓ Saved auxiliary files (.tex, .html, .pickle) for {model_name} in {self.output_dir}")
                
            finally:
                # Always return to original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"Warning: Could not save auxiliary files for {model_name}: {e}")
    
    def _extract_metrics(self, biogeme_results):
        """Extract key metrics from Biogeme results"""
        
        # Try different attribute names for compatibility with different biogeme versions
        try:
            bic = biogeme_results.data.bayesianInformationCriterion
        except AttributeError:
            try:
                bic = biogeme_results.data.BIC
            except AttributeError:
                # Calculate BIC manually if not available
                n_obs = biogeme_results.data.numberOfObservations
                n_params = len(biogeme_results.data.betaValues)
                log_like = biogeme_results.data.logLike
                bic = np.log(n_obs) * n_params - 2 * log_like
        
        try:
            aic = biogeme_results.data.akaike
        except AttributeError:
            try:
                aic = biogeme_results.data.AIC
            except AttributeError:
                # Calculate AIC manually if not available
                n_params = len(biogeme_results.data.betaValues)
                log_like = biogeme_results.data.logLike
                aic = 2 * n_params - 2 * log_like
        
        return {
            'log_likelihood': biogeme_results.data.logLike,
            'n_parameters': len(biogeme_results.data.betaValues),
            'n_observations': biogeme_results.data.numberOfObservations,
            'AIC': aic,
            'BIC': bic,
            'pseudo_r2': biogeme_results.data.rhoSquare
        }
    
    def create_comparison_table(self):
        """Create comparison table of all models for both train and test"""
        
        train_data = []
        test_data = []
        
        # Create train comparison data
        for model_name, metrics in self.train_results.items():
            if metrics is not None:
                train_data.append({
                    'Model': model_name,
                    'Dataset': 'Train',
                    'Log-Likelihood': metrics['log_likelihood'],
                    'N Parameters': metrics['n_parameters'],
                    'N Observations': metrics['n_observations'],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'Pseudo R²': metrics['pseudo_r2']
                })
        
        # Create test comparison data
        for model_name, metrics in self.test_results.items():
            if metrics is not None:
                test_data.append({
                    'Model': model_name,
                    'Dataset': 'Test',
                    'Log-Likelihood': metrics['log_likelihood'],
                    'N Parameters': metrics['n_parameters'],
                    'N Observations': metrics['n_observations'],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'Pseudo R²': metrics['pseudo_r2']
                })
        
        # Combine train and test data
        comparison_data = train_data + test_data
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Dataset then by AIC (lower is better)
        comparison_df = comparison_df.sort_values(['Dataset', 'Log-Likelihood'])
        
        return comparison_df
    
    def create_separate_comparison_tables(self):
        """Create separate comparison tables for train and test"""
        
        # Train table
        train_data = []
        for model_name, metrics in self.train_results.items():
            if metrics is not None:
                train_data.append({
                    'Model': model_name,
                    'Log-Likelihood': metrics['log_likelihood'],
                    'N Parameters': metrics['n_parameters'],
                    'N Observations': metrics['n_observations'],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'Pseudo R²': metrics['pseudo_r2']
                })
        
        train_df = pd.DataFrame(train_data).sort_values('Log-Likelihood', ascending=False)
        
        # Test table
        test_data = []
        for model_name, metrics in self.test_results.items():
            if metrics is not None:
                test_data.append({
                    'Model': model_name,
                    'Log-Likelihood': metrics['log_likelihood'],
                    'N Parameters': metrics['n_parameters'],
                    'N Observations': metrics['n_observations'],
                    'AIC': metrics['AIC'],
                    'BIC': metrics['BIC'],
                    'Pseudo R²': metrics['pseudo_r2']
                })
        
        test_df = pd.DataFrame(test_data).sort_values('Log-Likelihood', ascending=False)
        
        return train_df, test_df
    
    def save_results(self):
        """Save all results to files"""
        
        # Create comparison tables
        combined_df = self.create_comparison_table()
        train_df, test_df = self.create_separate_comparison_tables()
        
        # Save combined results to CSV
        csv_path = self.output_dir / 'model_comparison_combined.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"Combined results saved to {csv_path}")
        
        # Save separate train and test results
        train_csv_path = self.output_dir / 'model_comparison_train.csv'
        train_df.to_csv(train_csv_path, index=False)
        print(f"Training results saved to {train_csv_path}")
        
        test_csv_path = self.output_dir / 'model_comparison_test.csv'
        test_df.to_csv(test_csv_path, index=False)
        print(f"Test results saved to {test_csv_path}")
        
        # Create LaTeX tables
        # Combined table
        latex_combined = combined_df.to_latex(
            index=False,
            float_format='%.4f',
            caption='Comparison of Discrete Choice Models (Train and Test)',
            label='tab:model_comparison_combined'
        )
        
        latex_combined_path = self.output_dir / 'model_comparison_combined.tex'
        with open(latex_combined_path, 'w') as f:
            f.write(latex_combined)
        print(f"Combined LaTeX table saved to {latex_combined_path}")
        
        # Train table
        latex_train = train_df.to_latex(
            index=False,
            float_format='%.4f',
            caption='Comparison of Discrete Choice Models (Training Set)',
            label='tab:model_comparison_train'
        )
        
        latex_train_path = self.output_dir / 'model_comparison_train.tex'
        with open(latex_train_path, 'w') as f:
            f.write(latex_train)
        print(f"Training LaTeX table saved to {latex_train_path}")
        
        # Test table
        latex_test = test_df.to_latex(
            index=False,
            float_format='%.4f',
            caption='Comparison of Discrete Choice Models (Test Set)',
            label='tab:model_comparison_test'
        )
        
        latex_test_path = self.output_dir / 'model_comparison_test.tex'
        with open(latex_test_path, 'w') as f:
            f.write(latex_test)
        print(f"Test LaTeX table saved to {latex_test_path}")
        
        # Create visualizations
        self._create_visualization(train_df, test_df)
        
        # Save model configuration and summary
        self._save_model_summary()
        
        return combined_df, train_df, test_df
    
    def _save_model_summary(self):
        """Save a summary of all models and their configurations"""
        
        summary_info = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'output_directory': str(self.output_dir),
            'data_info': {
                'total_observations': len(self.merged_data),
                'train_observations': len(self.merged_data[self.merged_data['train'] == 1]),
                'test_observations': len(self.merged_data[self.merged_data['test'] == 1]),
                'available_segmentation_features': len(self.seg_features),
                'feature_processing': getattr(self, 'feature_stats', {})
            },
            'models_run': list(self.train_results.keys()) if hasattr(self, 'train_results') else [],
            'segmentation_features_available': self.seg_features[:20]  # First 20 features
        }
        
        # Save as JSON
        import json
        summary_path = self.output_dir / 'model_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        print(f"Model summary saved to {summary_path}")
        
        # Save as text file for easy reading
        summary_text_path = self.output_dir / 'model_summary.txt'
        with open(summary_text_path, 'w') as f:
            f.write("CHOICE MODEL BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {summary_info['timestamp']}\n")
            f.write(f"Output Directory: {summary_info['output_directory']}\n\n")
            
            f.write("DATA INFORMATION:\n")
            f.write(f"  Total observations: {summary_info['data_info']['total_observations']}\n")
            f.write(f"  Training observations: {summary_info['data_info']['train_observations']}\n")
            f.write(f"  Test observations: {summary_info['data_info']['test_observations']}\n")
            f.write(f"  Available segmentation features: {summary_info['data_info']['available_segmentation_features']}\n")
            
            # Add feature processing information
            if 'feature_processing' in summary_info['data_info'] and summary_info['data_info']['feature_processing']:
                fp = summary_info['data_info']['feature_processing']
                f.write(f"  Original features (before processing): {fp.get('original_feature_count', 'N/A')}\n")
                f.write(f"  Features after z-score scaling: {fp.get('scaled_feature_count', 'N/A')}\n")
                f.write(f"  Features after variance filtering: {fp.get('filtered_feature_count', 'N/A')}\n")
                f.write(f"  Variance threshold used: {fp.get('variance_threshold', 'N/A')}\n")
                f.write(f"  Processing order: {fp.get('processing_order', 'N/A')}\n")
                if fp.get('removed_features_no_variation'):
                    f.write(f"  Removed features (no variation): {len(fp['removed_features_no_variation'])}\n")
                if fp.get('removed_features_low_variance'):
                    f.write(f"  Removed features (low variance after scaling): {len(fp['removed_features_low_variance'])}\n")
            f.write("\n")
            
            f.write("MODELS RUN:\n")
            base_models = [m for m in summary_info['models_run'] if not any(x in m for x in ['_10feat', '_20feat', '_30feat', '_40feat', '_50feat']) or 'feat' not in m]
            seg_models = [m for m in summary_info['models_run'] if 'base_segmentation_' in m and 'feat' in m]
            full_models = [m for m in summary_info['models_run'] if 'full_' in m and 'feat' in m]
            
            f.write("  Base Models:\n")
            for model in base_models:
                f.write(f"    - {model}\n")
            
            if seg_models:
                f.write("  Segmentation Models (with varying feature counts):\n")
                for model in sorted(seg_models):
                    f.write(f"    - {model}\n")
            
            if full_models:
                f.write("  Full Models (with varying feature counts):\n")
                for model in sorted(full_models):
                    f.write(f"    - {model}\n")
            
            f.write(f"\nSEGMENTATION FEATURES (first 20):\n")
            for i, feature in enumerate(summary_info['segmentation_features_available'], 1):
                f.write(f"  {i:2d}. {feature}\n")
        
        print(f"Model summary text saved to {summary_text_path}")
    
    def _save_stepwise_history(self, selection_history, selected_features, suffix=''):
        """Save step-wise feature selection history and analysis"""
        
        # Save selection history as JSON
        import json
        stepwise_path = self.output_dir / f'stepwise_selection_history{suffix}.json'
        history_data = {
            'selection_history': selection_history,
            'final_selected_features': selected_features,
            'total_steps': len(selection_history) - 1,  # Exclude step 0 (base model)
            'final_log_likelihood': selection_history[-1]['log_likelihood'] if selection_history else None
        }
        
        with open(stepwise_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Step-wise selection history saved to {stepwise_path}")
        
        # Save as readable text file
        stepwise_text_path = self.output_dir / f'stepwise_selection_summary{suffix}.txt'
        with open(stepwise_text_path, 'w') as f:
            f.write("STEP-WISE FEATURE SELECTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total steps completed: {len(selection_history) - 1}\n")
            f.write(f"Final selected features: {len(selected_features)}\n")
            if selection_history:
                f.write(f"Final log-likelihood: {selection_history[-1]['log_likelihood']:.6f}\n")
                total_improvement = selection_history[-1]['log_likelihood'] - selection_history[0]['log_likelihood']
                f.write(f"Total LL improvement: {total_improvement:.6f}\n")
            f.write("\n")
            
            f.write("SELECTION STEPS:\n")
            for step_info in selection_history:
                step_num = step_info['step']
                added_feature = step_info['added_feature']
                ll = step_info['log_likelihood']
                improvement = step_info['improvement']
                
                if step_num == 0:
                    f.write(f"Step {step_num}: {added_feature} (LL: {ll:.6f})\n")
                else:
                    f.write(f"Step {step_num}: Added '{added_feature}' (LL: {ll:.6f}, improvement: +{improvement:.6f})\n")
            
            f.write(f"\nFINAL SELECTED FEATURES:\n")
            for i, feature in enumerate(selected_features, 1):
                f.write(f"  {i:2d}. {feature}\n")
        
        print(f"Step-wise selection summary saved to {stepwise_text_path}")
        
        # Create visualization of selection process
        self._create_stepwise_visualization(selection_history, suffix)
    
    def _create_stepwise_visualization(self, selection_history, suffix=''):
        """Create visualization of step-wise selection process"""
        
        if len(selection_history) < 2:
            return
        
        import matplotlib.pyplot as plt
        
        steps = [h['step'] for h in selection_history]
        log_likelihoods = [h['log_likelihood'] for h in selection_history]
        improvements = [h['improvement'] for h in selection_history]
        features = [h['added_feature'] for h in selection_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Log-likelihood progression
        ax1.plot(steps, log_likelihoods, 'o-', linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('Selection Step')
        ax1.set_ylabel('Log-Likelihood')
        ax1.set_title('Step-wise Feature Selection: Log-Likelihood Progression')
        ax1.grid(True, alpha=0.3)
        
        # Annotate points with feature names
        for i, (step, ll, feature) in enumerate(zip(steps, log_likelihoods, features)):
            if i > 0:  # Skip base model
                ax1.annotate(feature, (step, ll), xytext=(5, 5), textcoords='offset points', 
                           fontsize=9, ha='left', rotation=45)
        
        # Plot 2: Log-likelihood improvements by step
        if len(improvements) > 1:
            improvement_steps = steps[1:]  # Skip step 0
            improvement_values = improvements[1:]  # Skip step 0
            improvement_features = features[1:]  # Skip base model
            
            bars = ax2.bar(improvement_steps, improvement_values, color='lightgreen', alpha=0.7)
            ax2.set_xlabel('Selection Step')
            ax2.set_ylabel('Log-Likelihood Improvement')
            ax2.set_title('Step-wise Feature Selection: LL Improvement per Step')
            ax2.grid(True, alpha=0.3)
            
            # Add feature names on bars
            for bar, feature in zip(bars, improvement_features):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        feature, ha='center', va='bottom', rotation=45, fontsize=9)
        
        plt.tight_layout()
        
        # Save visualization
        stepwise_viz_path = self.output_dir / f'stepwise_selection_process{suffix}.png'
        plt.savefig(stepwise_viz_path, dpi=300, bbox_inches='tight')
        print(f"Step-wise selection visualization saved to {stepwise_viz_path}")
        plt.close()
    
    def _create_visualization(self, train_df, test_df):
        """Create visualization of model comparison for train and test sets"""
        
        # Create feature count analysis visualization
        self._create_feature_count_analysis(train_df, test_df)
        
        # Create overall comparison (limit to top 10 models by AIC)
        train_top = train_df.head(10)
        test_top = test_df.head(10)
        
        # Set up the figure with subplots for train and test
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        fig.suptitle('Top 10 Models: Discrete Choice Model Comparison (Train vs Test)', fontsize=16)
        
        # Define metrics to plot
        metrics = ['Log-Likelihood', 'AIC', 'BIC', 'Pseudo R²']
        metric_labels = ['Log-Likelihood\n(Higher is Better)', 'AIC\n(Lower is Better)', 
                        'BIC\n(Lower is Better)', 'Pseudo R²\n(Higher is Better)']
        
        # Plot training results
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[0, i]
            bars = ax.bar(range(len(train_top)), train_top[metric], color='skyblue', alpha=0.7)
            ax.set_title(f'Training Set - {label}')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(train_top)))
            ax.set_xticklabels(train_top['Model'], rotation=45, ha='right')
        
        # Plot test results
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[1, i]
            bars = ax.bar(range(len(test_top)), test_top[metric], color='lightcoral', alpha=0.7)
            ax.set_title(f'Test Set - {label}')
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(test_top)))
            ax.set_xticklabels(test_top['Model'], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.output_dir / 'model_comparison_top10.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"Top 10 models visualization saved to {viz_path}")
        plt.close()
        
        # Create a side-by-side comparison for key metrics (top models only)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Top 10 Model Performance: Train vs Test', fontsize=16)
        
        # AIC comparison
        ax1 = axes[0]
        x_pos = np.arange(len(train_top))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, train_top['AIC'], width, label='Train', color='skyblue', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, test_top['AIC'], width, label='Test', color='lightcoral', alpha=0.7)
        
        ax1.set_title('AIC Comparison (Lower is Better)')
        ax1.set_ylabel('AIC')
        ax1.set_xlabel('Model')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(train_top['Model'], rotation=45, ha='right')
        ax1.legend()
        
        # Pseudo R² comparison
        ax2 = axes[1]
        bars3 = ax2.bar(x_pos - width/2, train_top['Pseudo R²'], width, label='Train', color='skyblue', alpha=0.7)
        bars4 = ax2.bar(x_pos + width/2, test_top['Pseudo R²'], width, label='Test', color='lightcoral', alpha=0.7)
        
        ax2.set_title('Pseudo R² Comparison (Higher is Better)')
        ax2.set_ylabel('Pseudo R²')
        ax2.set_xlabel('Model')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(test_top['Model'], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save side-by-side comparison
        viz_comparison_path = self.output_dir / 'model_comparison_top10_side_by_side.png'
        plt.savefig(viz_comparison_path, dpi=300, bbox_inches='tight')
        print(f"Top 10 models side-by-side comparison saved to {viz_comparison_path}")
        plt.close()
    
    def _create_feature_count_analysis(self, train_df, test_df):
        """Create specific analysis of feature count effects"""
        
        # Extract feature counts for segmentation and full models
        seg_models = []
        full_models = []
        
        for _, row in train_df.iterrows():
            model_name = row['Model']
            if 'base_segmentation_' in model_name and 'feat' in model_name:
                feat_count = int(model_name.split('_')[2].replace('feat', ''))
                seg_models.append({
                    'feature_count': feat_count,
                    'train_aic': row['AIC'],
                    'train_pseudo_r2': row['Pseudo R²'],
                    'train_ll': row['Log-Likelihood']
                })
            elif 'full_' in model_name and 'feat' in model_name:
                feat_count = int(model_name.split('_')[1].replace('feat', ''))
                full_models.append({
                    'feature_count': feat_count,
                    'train_aic': row['AIC'],
                    'train_pseudo_r2': row['Pseudo R²'],
                    'train_ll': row['Log-Likelihood']
                })
        
        # Add test metrics
        for _, row in test_df.iterrows():
            model_name = row['Model']
            if 'base_segmentation_' in model_name and 'feat' in model_name:
                feat_count = int(model_name.split('_')[2].replace('feat', ''))
                for seg_model in seg_models:
                    if seg_model['feature_count'] == feat_count:
                        seg_model['test_aic'] = row['AIC']
                        seg_model['test_pseudo_r2'] = row['Pseudo R²']
                        seg_model['test_ll'] = row['Log-Likelihood']
            elif 'full_' in model_name and 'feat' in model_name:
                feat_count = int(model_name.split('_')[1].replace('feat', ''))
                for full_model in full_models:
                    if full_model['feature_count'] == feat_count:
                        full_model['test_aic'] = row['AIC']
                        full_model['test_pseudo_r2'] = row['Pseudo R²']
                        full_model['test_ll'] = row['Log-Likelihood']
        
        # Sort by feature count
        seg_models.sort(key=lambda x: x['feature_count'])
        full_models.sort(key=lambda x: x['feature_count'])
        
        # Create feature count analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Count Analysis: Effect on Model Performance', fontsize=16)
        
        # Segmentation models
        if seg_models:
            feature_counts = [m['feature_count'] for m in seg_models]
            
            # AIC plot
            axes[0, 0].plot(feature_counts, [m['train_aic'] for m in seg_models], 'o-', label='Train', color='skyblue')
            axes[0, 0].plot(feature_counts, [m['test_aic'] for m in seg_models], 's-', label='Test', color='lightcoral')
            axes[0, 0].set_title('Segmentation Models - AIC')
            axes[0, 0].set_xlabel('Number of Features')
            axes[0, 0].set_ylabel('AIC (Lower is Better)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Pseudo R² plot
            axes[0, 1].plot(feature_counts, [m['train_pseudo_r2'] for m in seg_models], 'o-', label='Train', color='skyblue')
            axes[0, 1].plot(feature_counts, [m['test_pseudo_r2'] for m in seg_models], 's-', label='Test', color='lightcoral')
            axes[0, 1].set_title('Segmentation Models - Pseudo R²')
            axes[0, 1].set_xlabel('Number of Features')
            axes[0, 1].set_ylabel('Pseudo R² (Higher is Better)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Log-Likelihood plot
            axes[0, 2].plot(feature_counts, [m['train_ll'] for m in seg_models], 'o-', label='Train', color='skyblue')
            axes[0, 2].plot(feature_counts, [m['test_ll'] for m in seg_models], 's-', label='Test', color='lightcoral')
            axes[0, 2].set_title('Segmentation Models - Log-Likelihood')
            axes[0, 2].set_xlabel('Number of Features')
            axes[0, 2].set_ylabel('Log-Likelihood (Higher is Better)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Full models
        if full_models:
            feature_counts = [m['feature_count'] for m in full_models]
            
            # AIC plot
            axes[1, 0].plot(feature_counts, [m['train_aic'] for m in full_models], 'o-', label='Train', color='skyblue')
            axes[1, 0].plot(feature_counts, [m['test_aic'] for m in full_models], 's-', label='Test', color='lightcoral')
            axes[1, 0].set_title('Full Models - AIC')
            axes[1, 0].set_xlabel('Number of Features')
            axes[1, 0].set_ylabel('AIC (Lower is Better)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Pseudo R² plot
            axes[1, 1].plot(feature_counts, [m['train_pseudo_r2'] for m in full_models], 'o-', label='Train', color='skyblue')
            axes[1, 1].plot(feature_counts, [m['test_pseudo_r2'] for m in full_models], 's-', label='Test', color='lightcoral')
            axes[1, 1].set_title('Full Models - Pseudo R²')
            axes[1, 1].set_xlabel('Number of Features')
            axes[1, 1].set_ylabel('Pseudo R² (Higher is Better)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Log-Likelihood plot
            axes[1, 2].plot(feature_counts, [m['train_ll'] for m in full_models], 'o-', label='Train', color='skyblue')
            axes[1, 2].plot(feature_counts, [m['test_ll'] for m in full_models], 's-', label='Test', color='lightcoral')
            axes[1, 2].set_title('Full Models - Log-Likelihood')
            axes[1, 2].set_xlabel('Number of Features')
            axes[1, 2].set_ylabel('Log-Likelihood (Higher is Better)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save feature count analysis
        feature_analysis_path = self.output_dir / 'feature_count_analysis.png'
        plt.savefig(feature_analysis_path, dpi=300, bbox_inches='tight')
        print(f"Feature count analysis saved to {feature_analysis_path}")
        plt.close()


def main():
    """Main function to run the choice model benchmark"""
    
    print("=== Choice Model Benchmarking ===")
    
    # Initialize benchmark
    benchmark = ChoiceModelBenchmark()
    
    # Load and prepare data
    benchmark.load_and_prepare_data()
    
    # Run all models
    benchmark.run_all_models()
    
    # Save results
    combined_df, train_df, test_df = benchmark.save_results()
    
    print("\n=== TRAINING SET RESULTS ===")
    print(train_df.to_string(index=False))
    
    print("\n=== TEST SET RESULTS ===")
    print(test_df.to_string(index=False))
    
    print(f"\nResults saved to: {benchmark.output_dir}")
    print("✓ Benchmarking completed successfully!")


if __name__ == "__main__":
    main() 