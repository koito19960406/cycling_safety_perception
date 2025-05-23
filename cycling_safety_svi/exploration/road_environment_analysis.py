#!/usr/bin/env python3
"""
Road Environment Analysis

This script analyzes correlations between road types, land use, speed limits
and safety perception ratings from the cycling safety study.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
import argparse
import logging
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PERCEPTION_DATA = "data/raw/perceptionratings.csv"
DEFAULT_SECOND_DESIGN = "data/raw/second_design.csv"
DEFAULT_OUTPUT_DIR = "reports/figures/road_analysis"

def extract_road_features(path_str):
    """
    Extract road type, land use, and speed limit from the image path
    
    Args:
        path_str: String with the image path
        
    Returns:
        tuple: (road_type, land_use, speed_limit)
    """
    # Use regex to extract components
    pattern = r'.*\\(.*?)\\(.*?)\\(.*?)\\'
    match = re.search(pattern, path_str)
    
    if match:
        road_type = match.group(1).strip()
        land_use = match.group(2).strip()
        speed_limit = match.group(3).strip()
        return road_type, land_use, speed_limit
    
    # Fallback if pattern doesn't match
    parts = path_str.split('\\')
    if len(parts) >= 4:
        return parts[1].strip(), parts[2].strip(), parts[3].strip()
    
    return "Unknown", "Unknown", "Unknown"

def load_and_prepare_data(perception_file, design_file):
    """
    Load and prepare the data for analysis
    
    Args:
        perception_file: Path to perception ratings CSV
        design_file: Path to second design CSV
        
    Returns:
        DataFrame: Combined data with road features and ratings
    """
    logger.info(f"Loading perception data from {perception_file}")
    perception_df = pd.read_csv(perception_file)
    
    logger.info(f"Loading design data from {design_file}")
    design_df = pd.read_csv(design_file)
    
    # Extract road features from the imagepath column
    logger.info("Extracting road features from image paths")
    design_df[['road_type', 'land_use', 'speed_limit']] = design_df.apply(
        lambda row: pd.Series(extract_road_features(row['imagepath'])), 
        axis=1
    )
    
    # Merge the datasets based on imageid
    logger.info("Merging datasets based on imageid")
    merged_df = pd.merge(
        perception_df,
        design_df[['imageid', 'road_type', 'land_use', 'speed_limit']],
        on='imageid', 
        how='inner'
    )
    
    logger.info(f"Merged data has {len(merged_df)} rows")
    
    return merged_df

def analyze_road_types(df, output_dir=None):
    """
    Analyze the relationship between road types and safety ratings
    
    Args:
        df: DataFrame with road features and ratings
        output_dir: Directory to save output figures
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Analyzing road types and safety ratings")
    
    # Calculate average ratings by road type
    road_type_ratings = df.groupby('road_type').agg({
        'traffic_safety': ['mean', 'std', 'count'],
        'social_safety': ['mean', 'std', 'count'],
        'beautiful': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    road_type_ratings.columns = ['_'.join(col).strip('_') for col in road_type_ratings.columns.values]
    
    # Sort by traffic safety rating (highest to lowest)
    road_type_ratings = road_type_ratings.sort_values('traffic_safety_mean', ascending=False)
    
    logger.info("Road type analysis results:")
    logger.info("\n" + str(road_type_ratings))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Create bar positions
    road_types = road_type_ratings['road_type']
    x = np.arange(len(road_types))
    width = 0.25
    
    # Plot bars for each rating type
    plt.bar(x - width, road_type_ratings['traffic_safety_mean'], width, 
            label='Traffic Safety', color='blue', alpha=0.7)
    plt.bar(x, road_type_ratings['social_safety_mean'], width, 
            label='Social Safety', color='green', alpha=0.7)
    plt.bar(x + width, road_type_ratings['beautiful_mean'], width, 
            label='Beauty', color='orange', alpha=0.7)
    
    # Add error bars
    plt.errorbar(x - width, road_type_ratings['traffic_safety_mean'], 
                yerr=road_type_ratings['traffic_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x, road_type_ratings['social_safety_mean'], 
                yerr=road_type_ratings['social_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x + width, road_type_ratings['beautiful_mean'], 
                yerr=road_type_ratings['beautiful_std'], fmt='none', color='black', capsize=5)
    
    # Customize the plot
    plt.xlabel('Road Type', fontsize=14)
    plt.ylabel('Average Rating (1-5)', fontsize=14)
    plt.title('Average Ratings by Road Type', fontsize=16)
    plt.xticks(x, road_types, rotation=45, ha='right')
    plt.legend()
    plt.ylim(1, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, "ratings_by_road_type.png")
        plt.savefig(output_file, dpi=120)
        logger.info(f"Saved figure to {output_file}")
    
    # Create a table of results
    if output_dir:
        output_csv = os.path.join(output_dir, "road_type_analysis.csv")
        road_type_ratings.to_csv(output_csv, index=False)
        logger.info(f"Saved analysis results to {output_csv}")
    
    return road_type_ratings

def analyze_land_use(df, output_dir=None):
    """
    Analyze the relationship between land use and safety ratings
    
    Args:
        df: DataFrame with road features and ratings
        output_dir: Directory to save output figures
    """
    logger.info("Analyzing land use and safety ratings")
    
    # Calculate average ratings by land use
    land_use_ratings = df.groupby('land_use').agg({
        'traffic_safety': ['mean', 'std', 'count'],
        'social_safety': ['mean', 'std', 'count'],
        'beautiful': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    land_use_ratings.columns = ['_'.join(col).strip('_') for col in land_use_ratings.columns.values]
    
    # Sort by traffic safety rating (highest to lowest)
    land_use_ratings = land_use_ratings.sort_values('traffic_safety_mean', ascending=False)
    
    logger.info("Land use analysis results:")
    logger.info("\n" + str(land_use_ratings))
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    # Create bar positions
    land_uses = land_use_ratings['land_use']
    x = np.arange(len(land_uses))
    width = 0.25
    
    # Plot bars for each rating type
    plt.bar(x - width, land_use_ratings['traffic_safety_mean'], width, 
            label='Traffic Safety', color='blue', alpha=0.7)
    plt.bar(x, land_use_ratings['social_safety_mean'], width, 
            label='Social Safety', color='green', alpha=0.7)
    plt.bar(x + width, land_use_ratings['beautiful_mean'], width, 
            label='Beauty', color='orange', alpha=0.7)
    
    # Add error bars
    plt.errorbar(x - width, land_use_ratings['traffic_safety_mean'], 
                yerr=land_use_ratings['traffic_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x, land_use_ratings['social_safety_mean'], 
                yerr=land_use_ratings['social_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x + width, land_use_ratings['beautiful_mean'], 
                yerr=land_use_ratings['beautiful_std'], fmt='none', color='black', capsize=5)
    
    # Customize the plot
    plt.xlabel('Land Use', fontsize=14)
    plt.ylabel('Average Rating (1-5)', fontsize=14)
    plt.title('Average Ratings by Land Use', fontsize=16)
    plt.xticks(x, land_uses, rotation=45, ha='right')
    plt.legend()
    plt.ylim(1, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, "ratings_by_land_use.png")
        plt.savefig(output_file, dpi=120)
        logger.info(f"Saved figure to {output_file}")
    
    # Create a table of results
    if output_dir:
        output_csv = os.path.join(output_dir, "land_use_analysis.csv")
        land_use_ratings.to_csv(output_csv, index=False)
        logger.info(f"Saved analysis results to {output_csv}")
    
    return land_use_ratings

def analyze_speed_limit(df, output_dir=None):
    """
    Analyze the relationship between speed limit and safety ratings
    
    Args:
        df: DataFrame with road features and ratings
        output_dir: Directory to save output figures
    """
    logger.info("Analyzing speed limit and safety ratings")
    
    # Create a copy and filter out NVT speed limits for this analysis
    speed_df = df.copy()
    speed_df = speed_df[speed_df['speed_limit'] != 'NVT']
    
    # Calculate average ratings by speed limit
    speed_limit_ratings = speed_df.groupby('speed_limit').agg({
        'traffic_safety': ['mean', 'std', 'count'],
        'social_safety': ['mean', 'std', 'count'],
        'beautiful': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    speed_limit_ratings.columns = ['_'.join(col).strip('_') for col in speed_limit_ratings.columns.values]
    
    # Sort by speed limit
    speed_limit_ratings = speed_limit_ratings.sort_values('speed_limit')
    
    logger.info("Speed limit analysis results:")
    logger.info("\n" + str(speed_limit_ratings))
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    
    # Create bar positions
    speed_limits = speed_limit_ratings['speed_limit']
    x = np.arange(len(speed_limits))
    width = 0.25
    
    # Plot bars for each rating type
    plt.bar(x - width, speed_limit_ratings['traffic_safety_mean'], width, 
            label='Traffic Safety', color='blue', alpha=0.7)
    plt.bar(x, speed_limit_ratings['social_safety_mean'], width, 
            label='Social Safety', color='green', alpha=0.7)
    plt.bar(x + width, speed_limit_ratings['beautiful_mean'], width, 
            label='Beauty', color='orange', alpha=0.7)
    
    # Add error bars
    plt.errorbar(x - width, speed_limit_ratings['traffic_safety_mean'], 
                yerr=speed_limit_ratings['traffic_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x, speed_limit_ratings['social_safety_mean'], 
                yerr=speed_limit_ratings['social_safety_std'], fmt='none', color='black', capsize=5)
    plt.errorbar(x + width, speed_limit_ratings['beautiful_mean'], 
                yerr=speed_limit_ratings['beautiful_std'], fmt='none', color='black', capsize=5)
    
    # Customize the plot
    plt.xlabel('Speed Limit (km/h)', fontsize=14)
    plt.ylabel('Average Rating (1-5)', fontsize=14)
    plt.title('Average Ratings by Speed Limit', fontsize=16)
    plt.xticks(x, speed_limits)
    plt.legend()
    plt.ylim(1, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, "ratings_by_speed_limit.png")
        plt.savefig(output_file, dpi=120)
        logger.info(f"Saved figure to {output_file}")
    
    # Create a table of results
    if output_dir:
        output_csv = os.path.join(output_dir, "speed_limit_analysis.csv")
        speed_limit_ratings.to_csv(output_csv, index=False)
        logger.info(f"Saved analysis results to {output_csv}")
    
    return speed_limit_ratings

def analyze_combined_factors(df, output_dir=None):
    """
    Analyze the combined effect of road type and land use on safety ratings
    
    Args:
        df: DataFrame with road features and ratings
        output_dir: Directory to save output figures
    """
    logger.info("Analyzing combined effect of road type and land use")
    
    # Create a combined feature
    df['road_land'] = df['road_type'] + ' / ' + df['land_use']
    
    # Calculate average traffic safety ratings by the combined feature
    combined_ratings = df.groupby(['road_type', 'land_use']).agg({
        'traffic_safety': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    combined_ratings.columns = ['_'.join(col).strip('_') for col in combined_ratings.columns.values]
    
    # Sort by traffic safety rating (highest to lowest)
    combined_ratings = combined_ratings.sort_values('traffic_safety_mean', ascending=False)
    
    logger.info("Combined factors analysis results:")
    logger.info("\n" + str(combined_ratings))
    
    # Create a heatmap of traffic safety by road type and land use
    # Pivot the data for the heatmap
    heatmap_data = df.pivot_table(
        values='traffic_safety', 
        index='road_type', 
        columns='land_use', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', vmin=1, vmax=5, 
                linewidths=.5, fmt='.2f', cbar_kws={'label': 'Average Traffic Safety Rating'})
    plt.title('Average Traffic Safety Rating by Road Type and Land Use', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, "traffic_safety_heatmap.png")
        plt.savefig(output_file, dpi=120)
        logger.info(f"Saved figure to {output_file}")
    
    # Create a separate heatmap for social safety
    social_heatmap_data = df.pivot_table(
        values='social_safety', 
        index='road_type', 
        columns='land_use', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(social_heatmap_data, annot=True, cmap='viridis', vmin=1, vmax=5, 
                linewidths=.5, fmt='.2f', cbar_kws={'label': 'Average Social Safety Rating'})
    plt.title('Average Social Safety Rating by Road Type and Land Use', fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, "social_safety_heatmap.png")
        plt.savefig(output_file, dpi=120)
        logger.info(f"Saved figure to {output_file}")
    
    # Create a table of results
    if output_dir:
        output_csv = os.path.join(output_dir, "combined_factors_analysis.csv")
        combined_ratings.to_csv(output_csv, index=False)
        logger.info(f"Saved analysis results to {output_csv}")
    
    return combined_ratings

def main():
    parser = argparse.ArgumentParser(description='Road Environment Analysis')
    parser.add_argument('--perception', type=str, default=DEFAULT_PERCEPTION_DATA,
                        help=f'Path to perception ratings CSV file (default: {DEFAULT_PERCEPTION_DATA})')
    parser.add_argument('--design', type=str, default=DEFAULT_SECOND_DESIGN,
                        help=f'Path to second design CSV file (default: {DEFAULT_SECOND_DESIGN})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output figures (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare the data
    data = load_and_prepare_data(args.perception, args.design)
    
    # Run the analyses
    road_type_results = analyze_road_types(data, args.output_dir)
    land_use_results = analyze_land_use(data, args.output_dir)
    speed_limit_results = analyze_speed_limit(data, args.output_dir)
    combined_results = analyze_combined_factors(data, args.output_dir)
    
    logger.info("Analysis complete. Results saved to output directory.")

if __name__ == "__main__":
    main() 