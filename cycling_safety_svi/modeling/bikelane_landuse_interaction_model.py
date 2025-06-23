"""
Bike Lane-Landuse Interaction Model

This script extends the best choice model with bike lane * landuse interaction effects.
It takes a trained model (pickle file) and adds interaction terms between bike lane presence/quality 
and land use categories (buildenvironment), while excluding safety from the models.

The script automatically extracts segmentation features from the original model and maps
the parameter names to the actual segmentation feature names using a built-in dictionary.

Usage:
    python bikelane_landuse_interaction_model.py --model_path path/to/best_model.pickle
    python bikelane_landuse_interaction_model.py --model_path path/to/best_model.pickle --main_design_path path/to/main_design.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import pickle
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.results as res
import logging
from biogeme.expressions import Beta, Variable, log, exp
from pathlib import Path
from datetime import datetime
import json


class BikelaneLanduseInteractionModel:
    """Extends a trained choice model with bike lane * landuse interaction effects"""
    
    def __init__(self, model_path='reports/models/choice_20250606_162138/best_model_top_35_with_safety.pickle', 
                 main_design_path='data/raw/main_design.csv', 
                 output_dir='reports/models/bikelane_interaction'):
        """
        Initialize the interaction model
        
        Args:
            model_path: Path to the trained model pickle file
            main_design_path: Path to main_design.csv with land use data
            output_dir: Output directory for results
        """
        self.model_path = Path(model_path)
        self.main_design_path = Path(main_design_path)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"bikelane_landuse_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Bike Lane-Landuse Interaction Model")
        print(f"Model path: {self.model_path}")
        print(f"Main design path: {self.main_design_path}")
        print(f"Output directory: {self.output_dir}")
        
        # Configure logging
        logger = logging.getLogger('biogeme')
        logger.setLevel(logging.WARNING)
        
        # Land use categories found in the data
        self.landuse_categories = ['Wijkontslu', 'Industriet', 'Woongebied', 'Hoofdweg', 'Recreatie']
        
        # Manual mapping from pickle feature names to actual segmentation feature names
        self.feature_name_mapping = {
            'barrier': 'Barrier',
            'bicycle': 'Bicycle',
            'bike lane': 'Bike Lane',
            'billboard': 'Billboard',
            'boat': 'Boat',
            'bridge': 'Bridge',
            'building': 'Building',
            'bus': 'Bus',
            'car': 'Car',
            'curb': 'Curb',
            'fence': 'Fence',
            'guard rail': 'Guard Rail',
            'lane marking   crosswalk': 'Lane Marking - Crosswalk',
            'lane marking   general': 'Lane Marking - General',
            'on rails': 'On Rails',
            'other vehicle': 'Other Vehicle',
            'parking': 'Parking',
            'pedestrian area': 'Pedestrian Area',
            'person': 'Person',
            'pole': 'Pole',
            'rail track': 'Rail Track',
            'road': 'Road',
            'sand': 'Sand',
            'sidewalk': 'Sidewalk',
            'sky': 'Sky',
            'snow': 'Snow',
            'terrain': 'Terrain',
            'traffic sign (front)': 'Traffic Sign (Front)',
            'trash can': 'Trash Can',
            'truck': 'Truck',
            'tunnel': 'Tunnel',
            'utility pole': 'Utility Pole',
            'vegetation': 'Vegetation',
            'wall': 'Wall',
            'water': 'Water'
        }
        
    def load_trained_model_data(self):
        """Load the trained model and extract its data and structure"""
        
        print("Loading trained model...")
        
        try:
            # Try loading using Biogeme's bioResults method first
            try:
                model_results = res.bioResults(pickleFile=str(self.model_path))
                print(f"✓ Loaded trained model using bioResults from {self.model_path}")
            except:
                # Fallback to pickle.load
                with open(self.model_path, 'rb') as f:
                    model_results = pickle.load(f)
                print(f"✓ Loaded trained model using pickle.load from {self.model_path}")
            
            # Extract the features used in the original model (excluding safety)
            self.original_model_features = self._extract_model_features(model_results)
            print(f"✓ Extracted {len(self.original_model_features)} features from original model")
            
            return model_results
        except Exception as e:
            raise ValueError(f"Could not load model from {self.model_path}: {e}")
    
    def _extract_model_features(self, model_results):
        """Extract the features that were used in the original trained model, excluding safety"""
        
        try:
            # Get parameter names from the model results
            try:
                estimated_params = model_results.get_estimated_parameters()
                param_names = estimated_params.index.tolist()
            except AttributeError:
                try:
                    param_names = list(model_results.data.betaNames.keys()) if hasattr(model_results.data, 'betaNames') else []
                except:
                    param_names = list(model_results.data.betaValues.keys()) if hasattr(model_results.data, 'betaValues') else []
            
            print(f"Found parameter names: {param_names}")
            
            # Extract segmentation features from parameter names, excluding safety
            segmentation_features = []
            for param_name in param_names:
                if param_name.startswith('B_') and param_name not in ['B_TL', 'B_TT', 'B_SAFETY']:
                    # Extract feature name from parameter name (remove 'B_' prefix)
                    feature_name = param_name[2:]
                    # Convert back from parameter naming convention
                    original_feature_name = feature_name.replace('_', ' ').lower()
                    
                    # Try to match with actual segmentation feature names
                    if original_feature_name not in segmentation_features:
                        segmentation_features.append(original_feature_name)
            
            print(f"Extracted segmentation features from original model (excluding safety): {segmentation_features}")
            return segmentation_features
            
        except Exception as e:
            print(f"Warning: Could not extract features from original model: {e}")
            print("Will proceed without segmentation features.")
            return []
    
    def load_and_prepare_data(self, cv_dcm_path='data/raw/cv_dcm.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv'):
        """Load and prepare all datasets, including land use data (excluding safety scores)"""
        
        print("Loading and preparing datasets...")
        
        # First load the trained model to extract its features
        self.trained_model_results = self.load_trained_model_data()
        
        # Load main choice data
        self.choice_data = pd.read_csv(cv_dcm_path)
        print(f"Loaded choice data: {len(self.choice_data)} observations")
        
        # Load main design data for land use information
        self.main_design = pd.read_csv(self.main_design_path)
        print(f"Loaded main design data: {len(self.main_design)} tasks")
        
        # Load segmentation data (we need this for bike lane)
        print("Loading segmentation data...")
        try:
            seg_chunks = []
            for chunk in pd.read_csv(segmentation_path, chunksize=1000):
                seg_chunks.append(chunk)
            self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
            self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
            print(f"Loaded segmentation data: {len(self.segmentation_data)} images")
        except Exception as e:
            print(f"Warning: Could not load segmentation data: {e}")
            self.segmentation_data = None
        
        # Merge all datasets
        self._merge_datasets_with_landuse()
        
    def _merge_datasets_with_landuse(self):
        """Merge all datasets including land use information (no safety scores)"""
        
        print("Merging datasets with land use information...")
        
        # Start with choice data
        merged_data = self.choice_data.copy()
        
        # Merge with main design data to get land use information
        # Create image to task mapping from main design
        img_to_task_1 = {}
        img_to_task_2 = {}
        
        for _, row in self.main_design.iterrows():
            # Extract image filename from alt1_imageid and alt2_imageid
            if pd.notna(row['alt1_imageid']):
                img_name_1 = row['alt1_imageid'] + '.jpg'
                img_to_task_1[img_name_1] = {
                    'task_id': row['task_id'],
                    'buildenvironment': row['alt1_buildenvironment'],
                    'wegtype': row['alt1_wegtype'],
                    'speedcar': row['alt1_speedcar']
                }
            
            if pd.notna(row['alt2_imageid']):
                img_name_2 = row['alt2_imageid'] + '.jpg'
                img_to_task_2[img_name_2] = {
                    'task_id': row['task_id'],
                    'buildenvironment': row['alt2_buildenvironment'],
                    'wegtype': row['alt2_wegtype'],
                    'speedcar': row['alt2_speedcar']
                }
        
        # Add land use information to merged data
        merged_data['landuse_1'] = merged_data['IMG1'].map(lambda x: img_to_task_1.get(x, {}).get('buildenvironment', 'Unknown'))
        merged_data['landuse_2'] = merged_data['IMG2'].map(lambda x: img_to_task_2.get(x, {}).get('buildenvironment', 'Unknown'))
        
        # Add segmentation features (including bike lane)
        if self.segmentation_data is not None:
            segmentation_dict = {}
            for _, row in self.segmentation_data.iterrows():
                img_name = row['filename_key'] + '.jpg'
                if pd.isna(row['filename_key']):
                    continue
                features = row.drop('filename_key').to_dict()
                segmentation_dict[img_name] = features
            
            # Get segmentation feature names
            seg_feature_names = [col for col in self.segmentation_data.columns if col != 'filename_key']
            
            # Add segmentation features for both alternatives
            for feature in seg_feature_names:
                merged_data[f"{feature}_1"] = merged_data['IMG1'].map(lambda x: segmentation_dict.get(x, {}).get(feature, 0))
                merged_data[f"{feature}_2"] = merged_data['IMG2'].map(lambda x: segmentation_dict.get(x, {}).get(feature, 0))
        
        # Handle unknown land use by assigning to most common category
        most_common_landuse = merged_data['landuse_1'].mode().iloc[0] if not merged_data['landuse_1'].mode().empty else 'Woongebied'
        merged_data['landuse_1'] = merged_data['landuse_1'].replace('Unknown', most_common_landuse)
        merged_data['landuse_2'] = merged_data['landuse_2'].replace('Unknown', most_common_landuse)
        
        self.merged_data = merged_data
        print(f"Merged dataset created: {len(merged_data)} observations, {len(merged_data.columns)} features")
        
        # Print land use distribution
        print("\nLand use distribution (Alternative 1):")
        print(merged_data['landuse_1'].value_counts())
        print("\nLand use distribution (Alternative 2):")
        print(merged_data['landuse_2'].value_counts())
        
        # Print bike lane statistics if available
        if 'Bike Lane_1' in merged_data.columns:
            print(f"\nBike Lane statistics:")
            print(f"  Alternative 1: mean={merged_data['Bike Lane_1'].mean():.4f}, std={merged_data['Bike Lane_1'].std():.4f}")
            print(f"  Alternative 2: mean={merged_data['Bike Lane_2'].mean():.4f}, std={merged_data['Bike Lane_2'].std():.4f}")
    
    def create_bikelane_landuse_dummy_variables(self, data_subset):
        """Create dummy variables for land use categories and bike lane interaction terms"""
        
        print("Creating bike lane * land use interaction variables...")
        
        # Use 'Woongebied' as reference category (most common residential area)
        reference_category = 'Recreatie'
        
        # Ensure bike lane data exists
        if 'Bike Lane_1' not in data_subset.columns or 'Bike Lane_2' not in data_subset.columns:
            print("Warning: Bike Lane data not found. Creating zero bike lane columns.")
            data_subset['Bike Lane_1'] = 0
            data_subset['Bike Lane_2'] = 0
        
        for category in self.landuse_categories:
            if category != reference_category:
                # Create dummy variables for both alternatives
                data_subset[f'landuse_{category}_1'] = (data_subset['landuse_1'] == category).astype(int)
                data_subset[f'landuse_{category}_2'] = (data_subset['landuse_2'] == category).astype(int)
                
                # Create interaction terms with bike lane
                data_subset[f'bikelane_landuse_{category}_1'] = data_subset['Bike Lane_1'] * data_subset[f'landuse_{category}_1']
                data_subset[f'bikelane_landuse_{category}_2'] = data_subset['Bike Lane_2'] * data_subset[f'landuse_{category}_2']
        
        print(f"Created dummy variables for land use categories (reference: {reference_category})")
        print(f"Created bike lane * landuse interaction terms")
        
        return data_subset, reference_category
    
    def estimate_interaction_model(self, train_data, test_data=None):
        """Estimate the bike lane * landuse interaction model (excluding safety)"""
        
        print("Estimating Bike Lane * Landuse Interaction Model")
        print("=" * 50)
        
        # Prepare the modeling data with landuse and bike lane variables
        train_model_data, reference_category = self.create_bikelane_landuse_dummy_variables(train_data)
        
        # Required columns for the base model (excluding safety)
        required_cols = ['CHOICE', 'TL1', 'TT1', 'TL2', 'TT2', 'Bike Lane_1', 'Bike Lane_2']
        
        # Land use dummy variable columns
        landuse_cols = []
        for category in self.landuse_categories:
            if category != reference_category:
                landuse_cols.extend([f'landuse_{category}_1', f'landuse_{category}_2'])
        
        # Interaction term columns
        interaction_cols = []
        for category in self.landuse_categories:
            if category != reference_category:
                interaction_cols.extend([f'bikelane_landuse_{category}_1', f'bikelane_landuse_{category}_2'])
        
        # Initialize segmentation columns (excluding bike lane since it's already in the main model)
        seg_cols = []
        seg_features_used = []
        
        # Add other segmentation features from original model if available
        if hasattr(self, 'original_model_features') and len(self.original_model_features) > 0 and self.segmentation_data is not None:
            print(f"Using segmentation features from original model: {self.original_model_features}")
            
            # Map original feature names to column names and check availability (excluding bike lane)
            for feature in self.original_model_features:
                # Skip bike lane since it's already included as main effect
                if feature.lower() in ['bike lane']:
                    continue
                    
                # Use manual mapping first
                mapped_feature = self.feature_name_mapping.get(feature, feature)
                
                # Try different possible column name formats
                possible_names = [
                    mapped_feature,
                    mapped_feature.replace(' ', '_'),
                    mapped_feature.replace(' ', '').lower(),
                    mapped_feature.replace('_', ' '),
                    mapped_feature.replace(' ', '_').title(),
                    mapped_feature.title().replace(' ', '_'),
                    mapped_feature.replace('-', '_'),
                    mapped_feature.replace(' - ', '_'),
                    feature,
                    feature.replace(' ', '_'),
                    feature.replace(' ', '').lower(),
                    feature.replace('_', ' '),
                    feature.replace('   ', ' ').replace('  ', ' ').strip()
                ]
                
                feature_found = False
                for possible_name in possible_names:
                    col1 = f'{possible_name}_1'
                    col2 = f'{possible_name}_2'
                    if col1 in train_model_data.columns and col2 in train_model_data.columns:
                        seg_cols.extend([col1, col2])
                        seg_features_used.append(possible_name)
                        feature_found = True
                        print(f"✓ Mapped '{feature}' -> '{possible_name}'")
                        break
                
                if not feature_found:
                    print(f"Warning: Could not find segmentation feature '{feature}' (mapped to '{mapped_feature}') in data")
            
            print(f"Successfully included {len(seg_features_used)} other segmentation features: {seg_features_used}")
        else:
            print("No additional segmentation features to include")
        
        # Combine all required columns
        all_cols = required_cols + landuse_cols + interaction_cols + seg_cols
        train_model_data = train_model_data[all_cols].copy().dropna()
        
        print(f"Training data shape after feature selection: {train_model_data.shape}")
        print(f"Features included: {len(all_cols) - 1}")  # Exclude CHOICE
        
        # Create Biogeme database for training
        train_database = db.Database('bikelane_landuse_interaction_train', train_model_data)
        
        # Create variables
        TL1, TT1, TL2, TT2 = train_database.variables['TL1'], train_database.variables['TT1'], train_database.variables['TL2'], train_database.variables['TT2']
        bikelane1, bikelane2 = train_database.variables['Bike Lane_1'], train_database.variables['Bike Lane_2']
        CHOICE = train_database.variables['CHOICE']
        
        # Define base parameters (excluding safety)
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        B_BIKELANE = Beta('B_BIKELANE', 0, None, None, 0)
        
        # Start with base utility
        V1_components = [B_TL * TL1 / 3, B_TT * TT1 / 10, B_BIKELANE * bikelane1]
        V2_components = [B_TL * TL2 / 3, B_TT * TT2 / 10, B_BIKELANE * bikelane2]
        
        # Add land use main effects
        landuse_params = {}
        for category in self.landuse_categories:
            if category != reference_category:
                param_name = f"B_LANDUSE_{category.upper()}"
                landuse_params[category] = Beta(param_name, 0, None, None, 0)
                
                landuse_var1 = train_database.variables[f'landuse_{category}_1']
                landuse_var2 = train_database.variables[f'landuse_{category}_2']
                
                V1_components.append(landuse_params[category] * landuse_var1)
                V2_components.append(landuse_params[category] * landuse_var2)
        
        # Add bike lane * landuse interaction effects
        interaction_params = {}
        for category in self.landuse_categories:
            if category != reference_category:
                param_name = f"B_BIKELANE_LANDUSE_{category.upper()}"
                interaction_params[category] = Beta(param_name, 0, None, None, 0)
                
                interaction_var1 = train_database.variables[f'bikelane_landuse_{category}_1']
                interaction_var2 = train_database.variables[f'bikelane_landuse_{category}_2']
                
                V1_components.append(interaction_params[category] * interaction_var1)
                V2_components.append(interaction_params[category] * interaction_var2)
        
        # Add other segmentation features
        seg_params = {}
        for feature in seg_features_used:
            param_name = f"B_{feature.replace(' ', '_').replace('-', '_').upper()}"
            seg_params[feature] = Beta(param_name, 0, None, None, 0)
            
            seg_var1 = train_database.variables[f'{feature}_1']
            seg_var2 = train_database.variables[f'{feature}_2']
            
            V1_components.append(seg_params[feature] * seg_var1)
            V2_components.append(seg_params[feature] * seg_var2)
        
        # Define utility functions
        V1 = sum(V1_components)
        V2 = sum(V2_components)
        
        # Estimate the model on training data
        train_results = self._estimate_mnl(V1, V2, CHOICE, train_database, 'bikelane_landuse_interaction')
        
        # Evaluate on test data if provided
        test_results = None
        if test_data is not None:
            test_results = self._evaluate_on_test_data(train_results, test_data, reference_category, seg_features_used, all_cols)
        
        # Save model information
        model_info = {
            'reference_landuse_category': reference_category,
            'landuse_categories': self.landuse_categories,
            'interaction_terms': list(interaction_params.keys()),
            'segmentation_features': seg_features_used,
            'model_components': {
                'base_effects': ['TL', 'TT', 'BIKELANE'],
                'landuse_main_effects': list(landuse_params.keys()),
                'bikelane_landuse_interactions': list(interaction_params.keys()),
                'segmentation_effects': list(seg_params.keys()) if seg_params else []
            }
        }
        
        self.model_info = model_info
        return train_results, test_results
    
    def _estimate_mnl(self, V1, V2, Choice, database, name):
        """Estimate MNL model using Biogeme"""
        
        # Create utility dictionary
        V = {1: V1, 2: V2}
        
        # Availability conditions
        av = {1: 1, 2: 1}
        
        # Define choice model
        prob = models.logit(V, av, Choice)
        LL = log(prob)
        
        # Create Biogeme object
        biogeme = bio.BIOGEME(database, LL)
        biogeme.modelName = name
        
        # Configure to save results
        biogeme.generate_pickle = True
        biogeme.generate_html = True
        biogeme.save_iterations = True
        
        # Change to output directory
        original_cwd = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Calculate null log-likelihood
            biogeme.calculate_null_loglikelihood(av)
            
            # Estimate model
            results = biogeme.estimate()
            
            # Save additional files
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
            os.chdir(original_cwd)
        
        return results
    
    def _evaluate_on_test_data(self, train_results, test_data, reference_category, seg_features_used, all_cols):
        """Evaluate the trained model on test data"""
        
        print("Evaluating model on test data...")
        
        # Prepare test data with same features
        test_data_copy = test_data.copy()
        test_model_data, _ = self.create_bikelane_landuse_dummy_variables(test_data_copy)
        test_model_data = test_model_data[all_cols].copy().dropna()
        
        print(f"Test data shape after feature selection: {test_model_data.shape}")
        
        # Get estimated parameters from training
        try:
            estimated_params = train_results.get_estimated_parameters()
            param_dict = {}
            for param_name, param_data in estimated_params.iterrows():
                param_dict[param_name] = param_data['Value']
        except AttributeError:
            param_dict = train_results.data.betaValues if hasattr(train_results.data, 'betaValues') else {}
        
        # Calculate log-likelihood on test data
        log_likelihood_test = 0
        n_obs_test = len(test_model_data)
        
        for idx, row in test_model_data.iterrows():
            # Calculate utilities for both alternatives
            V1 = param_dict.get('B_TL', 0) * row['TL1'] / 3 + param_dict.get('B_TT', 0) * row['TT1'] / 10
            V2 = param_dict.get('B_TL', 0) * row['TL2'] / 3 + param_dict.get('B_TT', 0) * row['TT2'] / 10
            
            # Add bike lane terms
            V1 += param_dict.get('B_BIKELANE', 0) * row['Bike Lane_1']
            V2 += param_dict.get('B_BIKELANE', 0) * row['Bike Lane_2']
            
            # Add land use main effects
            for category in self.landuse_categories:
                if category != reference_category:
                    param_name = f"B_LANDUSE_{category.upper()}"
                    if param_name in param_dict:
                        V1 += param_dict[param_name] * row[f'landuse_{category}_1']
                        V2 += param_dict[param_name] * row[f'landuse_{category}_2']
            
            # Add interaction effects
            for category in self.landuse_categories:
                if category != reference_category:
                    param_name = f"B_BIKELANE_LANDUSE_{category.upper()}"
                    if param_name in param_dict:
                        V1 += param_dict[param_name] * row[f'bikelane_landuse_{category}_1']
                        V2 += param_dict[param_name] * row[f'bikelane_landuse_{category}_2']
            
            # Add segmentation effects
            for feature in seg_features_used:
                param_name = f"B_{feature.replace(' ', '_').replace('-', '_').upper()}"
                if param_name in param_dict:
                    V1 += param_dict[param_name] * row[f'{feature}_1']
                    V2 += param_dict[param_name] * row[f'{feature}_2']
            
            # Calculate choice probabilities
            exp_V1 = np.exp(V1)
            exp_V2 = np.exp(V2)
            prob_1 = exp_V1 / (exp_V1 + exp_V2)
            prob_2 = exp_V2 / (exp_V1 + exp_V2)
            
            # Add to log-likelihood
            chosen_alt = int(row['CHOICE'])
            if chosen_alt == 1:
                log_likelihood_test += np.log(max(prob_1, 1e-10))
            elif chosen_alt == 2:
                log_likelihood_test += np.log(max(prob_2, 1e-10))
        
        # Calculate test metrics
        n_params = len(train_results.data.betaValues)
        aic_test = 2 * n_params - 2 * log_likelihood_test
        bic_test = np.log(n_obs_test) * n_params - 2 * log_likelihood_test
        
        # Calculate pseudo R-squared for test data
        null_ll_test = n_obs_test * np.log(0.5)
        pseudo_r2_test = 1 - (log_likelihood_test / null_ll_test) if null_ll_test != 0 else 0
        
        test_metrics = {
            'log_likelihood': log_likelihood_test,
            'n_parameters': n_params,
            'n_observations': n_obs_test,
            'AIC': aic_test,
            'BIC': bic_test,
            'pseudo_r2': pseudo_r2_test
        }
        
        print(f"✓ Test evaluation completed. Test LL: {log_likelihood_test:.6f}")
        return test_metrics
    
    def analyze_interaction_effects(self, results):
        """Analyze and interpret the bike lane * landuse interaction effects"""
        
        print("Analyzing bike lane * landuse interaction effects...")
        
        # Get estimated parameters
        try:
            estimated_params = results.get_estimated_parameters()
        except AttributeError:
            try:
                param_dict = {}
                if hasattr(results.data, 'betaValues'):
                    for param_name, value in results.data.betaValues.items():
                        param_dict[param_name] = {'Value': value}
                        if hasattr(results.data, 'stErr') and param_name in results.data.stErr:
                            param_dict[param_name]['Std err'] = results.data.stErr[param_name]
                        if hasattr(results.data, 'pValues') and param_name in results.data.pValues:
                            param_dict[param_name]['Robust P-value'] = results.data.pValues[param_name]
                        if hasattr(results.data, 'tStats') and param_name in results.data.tStats:
                            param_dict[param_name]['Robust T-stat'] = results.data.tStats[param_name]
                
                estimated_params = pd.DataFrame(param_dict).T
            except Exception as e:
                print(f"Error creating parameter dataframe: {e}")
                return {}
        
        # Extract interaction effects
        interaction_effects = {}
        main_bikelane_effect = estimated_params.loc['B_BIKELANE', 'Value'] if 'B_BIKELANE' in estimated_params.index else 0
        
        for category in self.landuse_categories:
            if category != self.model_info['reference_landuse_category']:
                interaction_param = f"B_BIKELANE_LANDUSE_{category.upper()}"
                if interaction_param in estimated_params.index:
                    interaction_coef = estimated_params.loc[interaction_param, 'Value']
                    total_bikelane_effect = main_bikelane_effect + interaction_coef
                    
                    interaction_effects[category] = {
                        'interaction_coefficient': interaction_coef,
                        'main_bikelane_effect': main_bikelane_effect,
                        'total_bikelane_effect': total_bikelane_effect,
                        'p_value': estimated_params.loc[interaction_param, 'Robust P-value'] if 'Robust P-value' in estimated_params.columns else None,
                        't_stat': estimated_params.loc[interaction_param, 'Robust T-stat'] if 'Robust T-stat' in estimated_params.columns else None
                    }
        
        # Reference category effect
        interaction_effects[self.model_info['reference_landuse_category']] = {
            'interaction_coefficient': 0,  # Reference category
            'main_bikelane_effect': main_bikelane_effect,
            'total_bikelane_effect': main_bikelane_effect,
            'p_value': None,
            't_stat': None
        }
        
        return interaction_effects
    
    def save_results(self, train_results, test_results, interaction_effects):
        """Save all results and analysis"""
        
        print("Saving results...")
        
        # Save training metrics
        train_metrics = {
            'log_likelihood': train_results.data.logLike,
            'n_parameters': len(train_results.data.betaValues),
            'n_observations': train_results.data.numberOfObservations,
            'AIC': getattr(train_results.data, 'akaike', 2 * len(train_results.data.betaValues) - 2 * train_results.data.logLike),
            'BIC': getattr(train_results.data, 'bayesianInformationCriterion', 
                          np.log(train_results.data.numberOfObservations) * len(train_results.data.betaValues) - 2 * train_results.data.logLike),
            'pseudo_r2': train_results.data.rhoSquare
        }
        
        # Save parameter estimates
        try:
            param_estimates = train_results.get_estimated_parameters()
            param_estimates.to_csv(self.output_dir / 'parameter_estimates.csv')
        except AttributeError:
            try:
                param_dict = {}
                if hasattr(train_results.data, 'betaValues'):
                    for param_name, value in train_results.data.betaValues.items():
                        param_dict[param_name] = {'Value': value}
                        if hasattr(train_results.data, 'stErr') and param_name in train_results.data.stErr:
                            param_dict[param_name]['Std err'] = train_results.data.stErr[param_name]
                
                param_estimates = pd.DataFrame(param_dict).T
                param_estimates.to_csv(self.output_dir / 'parameter_estimates.csv')
            except Exception as e:
                print(f"Warning: Could not save parameter estimates: {e}")
        
        # Save interaction effects analysis
        interaction_df = pd.DataFrame(interaction_effects).T
        interaction_df.to_csv(self.output_dir / 'interaction_effects_analysis.csv')
        
        # Save model information and metrics
        model_summary = {
            'model_info': self.model_info,
            'train_metrics': train_metrics,
            'test_metrics': test_results if test_results is not None else None,
            'interaction_effects': interaction_effects,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original_model_path': str(self.model_path),
            'main_design_path': str(self.main_design_path)
        }
        
        with open(self.output_dir / 'model_summary.json', 'w') as f:
            json.dump(model_summary, f, indent=2, default=str)
        
        # Save separate CSV files for train and test metrics
        train_metrics_df = pd.DataFrame([train_metrics])
        train_metrics_df['dataset'] = 'train'
        train_metrics_df.to_csv(self.output_dir / 'train_metrics.csv', index=False)
        
        if test_results is not None:
            test_metrics_df = pd.DataFrame([test_results])
            test_metrics_df['dataset'] = 'test'
            test_metrics_df.to_csv(self.output_dir / 'test_metrics.csv', index=False)
            
            # Combined metrics file
            combined_metrics_df = pd.concat([train_metrics_df, test_metrics_df], ignore_index=True)
            combined_metrics_df.to_csv(self.output_dir / 'combined_metrics.csv', index=False)
        
        # Create summary report
        self._create_summary_report(train_results, interaction_effects, train_metrics, test_results)
        
        print(f"✓ Results saved to {self.output_dir}")
        return train_metrics, test_results
    
    def _create_summary_report(self, train_results, interaction_effects, train_metrics, test_metrics=None):
        """Create a human-readable summary report"""
        
        report_path = self.output_dir / 'bikelane_interaction_model_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("BIKE LANE * LANDUSE INTERACTION MODEL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model estimated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original model: {self.model_path}\n")
            f.write(f"Main design data: {self.main_design_path}\n\n")
            
            f.write("TRAINING PERFORMANCE:\n")
            f.write(f"  Log-likelihood: {train_metrics['log_likelihood']:.6f}\n")
            f.write(f"  Number of parameters: {train_metrics['n_parameters']}\n")
            f.write(f"  Number of observations: {train_metrics['n_observations']}\n")
            f.write(f"  AIC: {train_metrics['AIC']:.6f}\n")
            f.write(f"  BIC: {train_metrics['BIC']:.6f}\n")
            f.write(f"  Pseudo R²: {train_metrics['pseudo_r2']:.6f}\n\n")
            
            if test_metrics is not None:
                f.write("TEST PERFORMANCE:\n")
                f.write(f"  Log-likelihood: {test_metrics['log_likelihood']:.6f}\n")
                f.write(f"  Number of parameters: {test_metrics['n_parameters']}\n")
                f.write(f"  Number of observations: {test_metrics['n_observations']}\n")
                f.write(f"  AIC: {test_metrics['AIC']:.6f}\n")
                f.write(f"  BIC: {test_metrics['BIC']:.6f}\n")
                f.write(f"  Pseudo R²: {test_metrics['pseudo_r2']:.6f}\n\n")
            
            f.write("LAND USE CATEGORIES:\n")
            f.write(f"  Reference category: {self.model_info['reference_landuse_category']}\n")
            f.write(f"  Other categories: {', '.join([c for c in self.landuse_categories if c != self.model_info['reference_landuse_category']])}\n\n")
            
            f.write("BIKE LANE EFFECTS BY LAND USE:\n")
            for category, effects in interaction_effects.items():
                f.write(f"  {category}:\n")
                f.write(f"    Total bike lane effect: {effects['total_bikelane_effect']:.6f}\n")
                if effects['interaction_coefficient'] != 0:
                    f.write(f"    Interaction coefficient: {effects['interaction_coefficient']:.6f}\n")
                    if effects['p_value'] is not None:
                        significance = "***" if effects['p_value'] < 0.001 else "**" if effects['p_value'] < 0.01 else "*" if effects['p_value'] < 0.05 else ""
                        f.write(f"    P-value: {effects['p_value']:.6f} {significance}\n")
                else:
                    f.write(f"    (Reference category)\n")
                f.write("\n")
            
            f.write("MODEL COMPONENTS:\n")
            f.write(f"  Base effects: {', '.join(self.model_info['model_components']['base_effects'])}\n")
            f.write(f"  Landuse main effects: {', '.join(self.model_info['model_components']['landuse_main_effects'])}\n")
            f.write(f"  Bike Lane*Landuse interactions: {', '.join(self.model_info['model_components']['bikelane_landuse_interactions'])}\n")
            if self.model_info['model_components']['segmentation_effects']:
                f.write(f"  Other segmentation effects: {', '.join(self.model_info['model_components']['segmentation_effects'])}\n")
        
        print(f"✓ Summary report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete bike lane * landuse interaction analysis"""
        
        print("Starting bike lane * landuse interaction analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Split into train and test data
        if 'train' in self.merged_data.columns and 'test' in self.merged_data.columns:
            train_data = self.merged_data[self.merged_data['train'] == 1].copy()
            test_data = self.merged_data[self.merged_data['test'] == 1].copy()
            print(f"Using training data: {len(train_data)} observations")
            print(f"Using test data: {len(test_data)} observations")
        else:
            print("Warning: No train/test split found, using 80/20 split")
            train_data = self.merged_data.sample(frac=0.8, random_state=42)
            test_data = self.merged_data.drop(train_data.index)
            print(f"Created training data: {len(train_data)} observations")
            print(f"Created test data: {len(test_data)} observations")
        
        # Estimate interaction model on train, evaluate on test
        train_results, test_metrics = self.estimate_interaction_model(train_data, test_data)
        
        # Analyze interaction effects
        interaction_effects = self.analyze_interaction_effects(train_results)
        
        # Save results
        train_metrics, test_metrics = self.save_results(train_results, test_metrics, interaction_effects)
        
        print("✓ Bike lane * landuse interaction analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        
        return train_results, test_metrics, interaction_effects, train_metrics


def main():
    """Main function to run bike lane * landuse interaction analysis"""
    
    parser = argparse.ArgumentParser(description='Add bike lane * landuse interaction effects to choice model')
    parser.add_argument('--model_path', type=str, default='reports/models/choice_20250606_162138/best_model_top_35_with_safety.pickle',
                       help='Path to the trained model pickle file')
    parser.add_argument('--main_design_path', type=str, default='data/raw/main_design.csv',
                       help='Path to main_design.csv with land use data')
    parser.add_argument('--output_dir', type=str, default='reports/models/bikelane_interaction',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not Path(args.main_design_path).exists():
        raise FileNotFoundError(f"Main design file not found: {args.main_design_path}")
    
    # Initialize and run analysis
    interaction_model = BikelaneLanduseInteractionModel(
        model_path=args.model_path,
        main_design_path=args.main_design_path,
        output_dir=args.output_dir
    )
    
    train_results, test_metrics, interaction_effects, train_metrics = interaction_model.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY OF BIKE LANE * LANDUSE INTERACTION EFFECTS")
    print("="*60)
    
    for category, effects in interaction_effects.items():
        print(f"{category}: Total bike lane effect = {effects['total_bikelane_effect']:.6f}")
        if effects['interaction_coefficient'] != 0:
            significance = ""
            if effects['p_value'] is not None:
                if effects['p_value'] < 0.001:
                    significance = " ***"
                elif effects['p_value'] < 0.01:
                    significance = " **"
                elif effects['p_value'] < 0.05:
                    significance = " *"
            print(f"         Interaction coef = {effects['interaction_coefficient']:.6f}{significance}")
    
    print(f"\nTraining fit: LL = {train_metrics['log_likelihood']:.6f}, Pseudo R² = {train_metrics['pseudo_r2']:.6f}")
    if test_metrics is not None:
        print(f"Test fit: LL = {test_metrics['log_likelihood']:.6f}, Pseudo R² = {test_metrics['pseudo_r2']:.6f}")


if __name__ == "__main__":
    main() 