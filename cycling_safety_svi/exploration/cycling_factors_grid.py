#!/usr/bin/env python3
"""
Cycling Factors Image Grid

This script creates grid visualizations of traffic safety images grouped by
various cycling-related demographic factors from the database.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sqlite3
from pathlib import Path
import argparse
import logging
from matplotlib.gridspec import GridSpec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_PERCEPTION_DATA = "data/raw/perceptionratings.csv"
DEFAULT_RESPONSE_DATA = "data/raw/database_2024_10_07_135133.db"
DEFAULT_IMG_PATH = "/srv/shared/bicycle_project_roos/images_scaled"
DEFAULT_OUTPUT_DIR = "reports/figures/cycling_factors"

# Mapping dictionaries for demographic variables
CYCLER_MAPPING = {
    1: "Never",
    2: "Less than once a month",
    3: "Once a month",
    4: "2-3 times a month",
    5: "Once a week",
    6: "2-3 times a week",
    7: "4 or more times a week"
}

CYCLING_LIKE_MAPPING = {
    1: "Extremely Like",
    2: "Like",
    3: "Neutral",
    4: "Dislike",
    5: "Extremely Dislike"
}

CYCLING_UNSAFE_MAPPING = {
    1: "Very Safe",
    2: "Safe",
    3: "Neutral",
    4: "Unsafe",
    5: "Very Unsafe"
}

BIKE_TYPE_MAPPING = {
    1: "Regular bike",
    2: "E-bike",
    3: "Racing bike",
    4: "Other"
}

def load_data(perception_file, response_file):
    """
    Load perception ratings and response data and merge them based on set_id
    
    Args:
        perception_file (str): Path to perception ratings CSV file
        response_file (str): Path to response database file
        
    Returns:
        DataFrame: Merged data with demographic information and perception ratings
    """
    logger.info(f"Loading perception data from {perception_file}")
    perception_df = pd.read_csv(perception_file)
    
    logger.info(f"Loading response data from {response_file}")
    conn = sqlite3.connect(response_file)
    
    # Get the response data with cycling-related demographic information
    response_df = pd.read_sql_query(
        """
        SELECT 
            respondent_id, set_id, cycler, cyclinglike, cyclingunsafe, 
            biketype, cyclingincident, cyclinghurry, cyclingroute
        FROM Response 
        WHERE cycler IS NOT NULL
        """, 
        conn
    )
    conn.close()
    
    # Merge the datasets based on set_id
    logger.info("Merging datasets based on set_id")
    # First get unique respondents to avoid duplicating perception data
    unique_respondents = response_df.drop_duplicates(subset=['set_id'])
    
    # Merge perception data with respondent demographics
    merged_df = pd.merge(
        perception_df, 
        unique_respondents, 
        on='set_id', 
        how='inner'
    )
    
    logger.info(f"Merged data has {len(merged_df)} rows")
    
    return merged_df

def prepare_data(merged_df):
    """
    Prepare data for visualization by adding mapped category labels
    
    Args:
        merged_df: DataFrame with merged perception and demographic data
        
    Returns:
        DataFrame: Data prepared for visualization
    """
    # Add mapped category labels
    df = merged_df.copy()
    
    # Map cycling frequency
    df['cycler_category'] = df['cycler'].map(CYCLER_MAPPING)
    
    # Map cycling enjoyment
    df['cycling_like_category'] = df['cyclinglike'].map(CYCLING_LIKE_MAPPING)
    
    # Map safety perception
    df['cycling_unsafe_category'] = df['cyclingunsafe'].map(CYCLING_UNSAFE_MAPPING)
    
    # Map bike type
    df['bike_type_category'] = df['biketype'].map(BIKE_TYPE_MAPPING)
    
    # Simplified cycling frequency categories for visualization
    df['cycling_frequency'] = pd.cut(
        df['cycler'], 
        bins=[0, 2, 5, 7], 
        labels=['Rarely', 'Sometimes', 'Frequently']
    )
    
    return df

def get_images_by_cycling_factor(df, factor_column, img_path, samples_per_category=4):
    """
    Get images grouped by a cycling-related factor and traffic safety rating
    
    Args:
        df: DataFrame with image and demographic data
        factor_column: Column name for the cycling factor to group by
        img_path: Path to image directory
        samples_per_category: Number of samples to select per category
        
    Returns:
        dict: Nested dictionary with factor category and traffic safety rating as keys 
              and image paths as values
    """
    result = {}
    
    # Get all unique categories for this factor
    categories = sorted(df[factor_column].dropna().unique())
    
    # For each category
    for category in categories:
        result[category] = {}
        
        # For each traffic safety rating
        for rating in range(1, 6):
            # Filter data for this specific combination
            filtered_df = df[(df[factor_column] == category) & 
                            (df['traffic_safety'] == rating)]
            
            image_ids = filtered_df['imageid'].unique()
            
            # If we have fewer images than requested, use all of them
            if len(image_ids) <= samples_per_category:
                selected = image_ids
            else:
                selected = random.sample(list(image_ids), samples_per_category)
            
            # Create full paths to images
            image_paths = [os.path.join(img_path, f"{img_id}.jpg") for img_id in selected]
            # Filter out non-existent images
            image_paths = [p for p in image_paths if os.path.exists(p)]
            
            result[category][rating] = image_paths
            logger.info(f"Selected {len(image_paths)} images for {factor_column}={category}, traffic safety rating={rating}")
    
    return result

def plot_cycling_factor_grid(images_by_factor, factor_name, factor_column, output_dir=None):
    """
    Plot a grid of images for a specific cycling factor, organized by category and safety rating
    
    Args:
        images_by_factor: Nested dictionary with factor category and safety rating as keys
        factor_name: Display name for the cycling factor
        factor_column: Column name of the factor used for reference
        output_dir: Directory to save the output images
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all categories for this factor
    categories = sorted(images_by_factor.keys())
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(22, 24))
    
    # Create a gridspec with rows for each category and columns for each rating
    gs = GridSpec(len(categories) + 1, 6, figure=fig)
    
    # Add the main title
    fig.suptitle(f"Traffic Safety Perception by {factor_name}", fontsize=24, y=0.98)
    
    # Add rating headers (columns)
    for j, rating in enumerate(range(1, 6)):
        header_ax = fig.add_subplot(gs[0, j+1])
        header_ax.text(0.5, 0.5, f"Rating {rating}", 
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=16, fontweight='bold')
        header_ax.axis('off')
    
    # For each category
    for i, category in enumerate(categories):
        # Add category header (row)
        header_ax = fig.add_subplot(gs[i+1, 0])
        header_ax.text(0.5, 0.5, f"{category}", 
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=14, fontweight='bold',
                      rotation=0, wrap=True)
        header_ax.axis('off')
        
        # For each safety rating
        for j, rating in enumerate(range(1, 6)):
            # Create a subplot for this category-rating combination
            ax = fig.add_subplot(gs[i+1, j+1])
            ax.axis('off')
            
            # Add a border to visually separate cells
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)
            
            # Get images for this combination
            if rating in images_by_factor[category]:
                image_paths = images_by_factor[category][rating]
                
                if len(image_paths) > 0:
                    # Create a grid layout within this cell
                    n_images = min(len(image_paths), 4)  # Display at most 4 images per cell
                    grid_cols = min(2, n_images)  # 2 columns max
                    grid_rows = (n_images + grid_cols - 1) // grid_cols  # Ceiling division
                    
                    for k, img_path in enumerate(image_paths[:4]):
                        row = k // grid_cols
                        col = k % grid_cols
                        
                        # Calculate the subplot position within the cell
                        ax_pos = ax.get_position()
                        x_start = ax_pos.x0 + col * (ax_pos.width / grid_cols)
                        y_start = ax_pos.y0 + (grid_rows - 1 - row) * (ax_pos.height / grid_rows)
                        width = ax_pos.width / grid_cols
                        height = ax_pos.height / grid_rows
                        
                        # Create a new axis at the calculated position
                        inner_ax = fig.add_axes([x_start, y_start, width, height])
                        
                        try:
                            img = plt.imread(img_path)
                            inner_ax.imshow(img)
                            inner_ax.axis('off')
                            
                            # Add image ID as small caption
                            img_id = os.path.basename(img_path).replace('.jpg', '')
                            inner_ax.set_title(img_id, fontsize=7)
                        except Exception as e:
                            logger.error(f"Error loading image {img_path}: {e}")
            else:
                # If no images for this combination, display a message
                ax.text(0.5, 0.5, "No Data", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=10)
    
    # Add a legend explaining the ratings
    fig.text(0.5, 0.02, "Traffic Safety Ratings: 1 = Very Unsafe, 5 = Very Safe", 
             horizontalalignment='center', fontsize=14)
    
    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])
    
    # Save the figure
    if output_dir:
        output_file = os.path.join(output_dir, f"traffic_safety_by_{factor_column}.png")
        plt.savefig(output_file, dpi=120, bbox_inches='tight')
        logger.info(f"Saved figure to {output_file}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Cycling Factors Image Grid')
    parser.add_argument('--perception', type=str, default=DEFAULT_PERCEPTION_DATA,
                        help=f'Path to perception ratings CSV file (default: {DEFAULT_PERCEPTION_DATA})')
    parser.add_argument('--response', type=str, default=DEFAULT_RESPONSE_DATA,
                        help=f'Path to response database file (default: {DEFAULT_RESPONSE_DATA})')
    parser.add_argument('--img_path', type=str, default=DEFAULT_IMG_PATH,
                        help=f'Path to image directory (default: {DEFAULT_IMG_PATH})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--samples', type=int, default=4,
                        help='Number of samples per demographic category (default: 4)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and merge data
    merged_df = load_data(args.perception, args.response)
    
    # Prepare data for visualization
    prepared_data = prepare_data(merged_df)
    
    # Create image grids for different cycling factors
    
    # 1. Cycling frequency (simplified categories)
    cycling_freq_images = get_images_by_cycling_factor(
        prepared_data, 
        'cycling_frequency', 
        args.img_path, 
        args.samples
    )
    plot_cycling_factor_grid(
        cycling_freq_images, 
        "Cycling Frequency", 
        "cycling_frequency", 
        args.output_dir
    )
    
    # 2. Cycling enjoyment
    cycling_like_images = get_images_by_cycling_factor(
        prepared_data, 
        'cycling_like_category', 
        args.img_path, 
        args.samples
    )
    plot_cycling_factor_grid(
        cycling_like_images, 
        "Cycling Enjoyment", 
        "cycling_like", 
        args.output_dir
    )
    
    # 3. Perceived cycling safety
    cycling_safety_images = get_images_by_cycling_factor(
        prepared_data, 
        'cycling_unsafe_category', 
        args.img_path, 
        args.samples
    )
    plot_cycling_factor_grid(
        cycling_safety_images, 
        "Perceived Cycling Safety", 
        "cycling_unsafe", 
        args.output_dir
    )
    
    # 4. Bike type
    bike_type_images = get_images_by_cycling_factor(
        prepared_data, 
        'bike_type_category', 
        args.img_path, 
        args.samples
    )
    plot_cycling_factor_grid(
        bike_type_images, 
        "Bicycle Type", 
        "bike_type", 
        args.output_dir
    )
    
    logger.info("All image grids created successfully")

if __name__ == "__main__":
    main() 