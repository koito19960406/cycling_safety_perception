#!/usr/bin/env python
"""
Example script for running the discrete choice models with different feature combinations.
"""

import os
import argparse
import pandas as pd
from discrete_choice_model import run_choice_models


def main():
    parser = argparse.ArgumentParser(description='Run discrete choice models example')
    parser.add_argument('--data_path', type=str, 
                        default='data/processed/choice_data.csv',
                        help='Path to choice data CSV')
    parser.add_argument('--img_dir', type=str, 
                        default='/srv/shared/bicycle_project_roos/images_scaled',
                        help='Directory containing images')
    parser.add_argument('--model_path', type=str, 
                        default='models/cvdcm_best.pth',
                        help='Path to pre-trained CVDCM model')
    parser.add_argument('--perception_data', type=str, 
                        default='data/processed/perception_data_aggregated.csv',
                        help='Path to perception data CSV')
    parser.add_argument('--segmentation_data', type=str, 
                        default='data/processed/segmentation_statistics.csv',
                        help='Path to segmentation data CSV')
    parser.add_argument('--output_dir', type=str, 
                        default='results/discrete_choice',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Running discrete choice models with different feature combinations...")
    results, comparison_df = run_choice_models(
        args.data_path, 
        args.img_dir,
        args.model_path,
        args.perception_data,
        args.segmentation_data,
        args.output_dir
    )
    
    # Print model comparison
    print("\nModel comparison:")
    print(comparison_df)
    
    # Print best model based on rho-square
    best_model = comparison_df['rho_square'].idxmax()
    best_rho = comparison_df.loc[best_model, 'rho_square']
    print(f"\nBest model: {best_model} (rho-square: {best_rho:.4f})")
    
    # Check if perception variables improve the model
    if 'base' in comparison_df.index and 'perception_only' in comparison_df.index:
        diff = comparison_df.loc['perception_only', 'rho_square'] - comparison_df.loc['base', 'rho_square']
        print(f"Perception variables impact: {diff:.4f} (rho-square)")
    
    # Check if segmentation variables improve the model
    if 'base' in comparison_df.index and 'segmentation_only' in comparison_df.index:
        diff = comparison_df.loc['segmentation_only', 'rho_square'] - comparison_df.loc['base', 'rho_square']
        print(f"Segmentation variables impact: {diff:.4f} (rho-square)")
    
    # Check combined effect
    if 'base' in comparison_df.index and 'full_model' in comparison_df.index:
        diff = comparison_df.loc['full_model', 'rho_square'] - comparison_df.loc['base', 'rho_square']
        print(f"Full model impact: {diff:.4f} (rho-square)")
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main() 