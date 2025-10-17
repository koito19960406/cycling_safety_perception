"""
Compute Image Utilities from Final Full Model (Model 4)

This script loads the final_full_model from the choice model benchmark and computes
the utility value for each unique image using the estimated parameters.

The utilities are computed using the model structure:
V = B_TT * TT + B_TL * TL + B_SAFETY_SCORE * safety_score + sum(B_seg_i * seg_feature_i)

Note: Since we're computing utilities for images (not choice situations), we use
fixed representative values for TT and TL, and the actual values for safety scores
and segmentation features.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
# Import biogeme to enable unpickling of model results
import biogeme.biogeme  # noqa: F401
import biogeme.expressions  # noqa: F401


class Model4UtilityComputer:
    """Computes utilities for images using the final full model (Model 4)"""
    
    def __init__(self, 
                 model_dir='reports/models/mxl_choice_20250725_122947',
                 safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                 segmentation_path='data/processed/segmentation_results/pixel_ratios.csv'):
        """
        Initialize the utility computer
        
        Args:
            model_dir: Directory containing the model pickle files
            safety_scores_path: Path to safety scores CSV
            segmentation_path: Path to segmentation features CSV
        """
        self.model_dir = Path(model_dir)
        self.safety_scores_path = safety_scores_path
        self.segmentation_path = segmentation_path
        
        # Load data
        self.load_model()
        self.load_data()
        
    def load_model(self):
        """Load the final full model and extract parameters"""
        print("Loading final full model...")
        
        # Load model results
        model_pickle_path = self.model_dir / 'final_full_model.pickle'
        if not model_pickle_path.exists():
            raise FileNotFoundError(f"Model pickle not found: {model_pickle_path}")
        
        with open(model_pickle_path, 'rb') as f:
            self.model_results = pickle.load(f)
        
        # Load significant features
        features_pickle_path = self.model_dir / 'final_significant_features.pickle'
        if features_pickle_path.exists():
            with open(features_pickle_path, 'rb') as f:
                self.significant_features = pickle.load(f)
            print(f"Loaded {len(self.significant_features)} significant features")
        else:
            print("Warning: Significant features file not found, will use all features from model")
            self.significant_features = []
        
        # Extract parameter estimates
        self.extract_parameters()
        
    def extract_parameters(self):
        """Extract parameter values from model results"""
        print("Extracting parameter estimates...")
        
        self.params = {}
        
        # Get all beta parameters from model results
        for beta in self.model_results.betas:
            param_name = beta.name
            param_value = float(beta.value)
            
            # Only use mean parameters (not sigma parameters for random parameters)
            if not param_name.startswith('sigma_'):
                self.params[param_name] = param_value
                print(f"  {param_name}: {param_value:.6f}")
        
        print(f"Total parameters extracted: {len(self.params)}")
        
    def load_data(self):
        """Load safety scores and segmentation features"""
        print("\nLoading data...")
        
        # Load safety scores
        self.safety_scores = pd.read_csv(self.safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        print(f"Loaded safety scores for {len(self.safety_scores)} images")
        
        # Load segmentation data
        print("Loading segmentation data...")
        seg_chunks = []
        for chunk in pd.read_csv(self.segmentation_path, chunksize=1000):
            seg_chunks.append(chunk)
        self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
        self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
        
        # Add .jpg extension to match with safety scores
        self.segmentation_data['image_name'] = self.segmentation_data['filename_key'] + '.jpg'
        
        print(f"Loaded segmentation data for {len(self.segmentation_data)} images")
        
    def compute_utilities(self, tt_value=0.0, tl_value=0.0):
        """
        Compute utilities for all images
        
        Args:
            tt_value: Fixed value for travel time (default 0 to focus on image attributes)
            tl_value: Fixed value for traffic lights (default 0 to focus on image attributes)
            
        Returns:
            DataFrame with image_name and utility columns
        """
        print("\nComputing utilities for all images...")
        
        # Merge safety scores with segmentation data
        merged_data = self.safety_scores.merge(
            self.segmentation_data,
            on='image_name',
            how='inner'
        )
        
        print(f"Merged data: {len(merged_data)} images")
        
        # Initialize utilities array
        utilities = np.zeros(len(merged_data))
        
        # Add base utility components (TT and TL) - using mean values from parameters
        # Note: We use 0 for TT and TL since we want utilities to reflect image attributes only
        if 'B_TT' in self.params:
            utilities += self.params['B_TT'] * tt_value / 10  # Scale by 10 as in model
        
        if 'B_TL' in self.params:
            utilities += self.params['B_TL'] * tl_value / 3  # Scale by 3 as in model
        
        # Add safety score contribution
        if 'B_SAFETY_SCORE' in self.params:
            utilities += self.params['B_SAFETY_SCORE'] * merged_data['safety_score'].values
            print(f"Added safety score contribution (B_SAFETY_SCORE = {self.params['B_SAFETY_SCORE']:.4f})")
        
        # Add segmentation feature contributions
        seg_feature_count = 0
        for param_name, param_value in self.params.items():
            # Skip non-segmentation parameters
            if param_name in ['B_TT', 'B_TL', 'B_SAFETY_SCORE']:
                continue
            
            # Extract feature name from parameter name (e.g., B_Road -> Road)
            if param_name.startswith('B_'):
                feature_name = param_name[2:].replace('___', ' - ').replace('_', ' ')
                
                # Try to find matching column in segmentation data
                matching_cols = [col for col in merged_data.columns if col == feature_name]
                
                if matching_cols:
                    feature_col = matching_cols[0]
                    utilities += param_value * merged_data[feature_col].values
                    seg_feature_count += 1
        
        print(f"Added {seg_feature_count} segmentation feature contributions")
        
        # Create output dataframe
        result_df = pd.DataFrame({
            'image_name': merged_data['image_name'],
            'utility_model4': utilities,
            'safety_score': merged_data['safety_score']
        })
        
        # Sort by utility for easier inspection
        result_df = result_df.sort_values('utility_model4', ascending=False).reset_index(drop=True)
        
        return result_df
    
    def save_utilities(self, output_path='data/processed/model_results/image_utilities_model4.csv'):
        """Compute and save utilities to CSV"""
        
        # Compute utilities
        utilities_df = self.compute_utilities()
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        utilities_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Utilities saved to: {output_path}")
        print(f"  Total images: {len(utilities_df)}")
        print(f"  Utility range: [{utilities_df['utility_model4'].min():.3f}, {utilities_df['utility_model4'].max():.3f}]")
        print(f"  Mean utility: {utilities_df['utility_model4'].mean():.3f}")
        print(f"  Std utility: {utilities_df['utility_model4'].std():.3f}")
        
        return utilities_df


def main():
    """Main function to compute utilities from Model 4"""
    
    print("=== Computing Image Utilities from Model 4 (Final Full Model) ===\n")
    
    # Initialize computer
    computer = Model4UtilityComputer()
    
    # Compute and save utilities
    utilities_df = computer.save_utilities()
    
    # Display top 10 and bottom 10 images by utility
    print("\nTop 10 images by utility:")
    print(utilities_df.head(10).to_string(index=False))
    
    print("\nBottom 10 images by utility:")
    print(utilities_df.tail(10).to_string(index=False))
    
    print("\n✓ Utility computation completed successfully!")
    

if __name__ == "__main__":
    main()

