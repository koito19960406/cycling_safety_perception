#!/usr/bin/env python3
"""
Age Group Traffic Safety Image Viewer

This script creates a grid visualization of traffic safety images
organized by age groups (columns) and safety ratings (rows).
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
import matplotlib.patches as patches

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
DEFAULT_OUTPUT_DIR = "reports/figures/age_safety"

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
    
    # Get the response data with demographic information
    response_df = pd.read_sql_query(
        "SELECT respondent_id, set_id, age FROM Response", 
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

def map_age_group(age_value):
    """Map numeric age value to age group label"""
    age_mapping = {
        1: "18-30",
        2: "31-45", 
        3: "46-60",
        4: "61-75",
        5: "75+"
    }
    return age_mapping.get(age_value, "Unknown")

def prepare_data(merged_df):
    """
    Prepare data for visualization by adding age group labels
    
    Args:
        merged_df: DataFrame with merged perception and demographic data
        
    Returns:
        DataFrame: Data prepared for visualization
    """
    # Add age group labels
    df = merged_df.copy()
    df['age_group'] = df['age'].apply(map_age_group)
    
    return df

def get_images_by_rating_and_age(df, img_path, samples_per_category=4):
    """
    Get images grouped by traffic safety rating and age group
    
    Args:
        df: DataFrame with image and demographic data
        img_path: Path to image directory
        samples_per_category: Number of samples to select per category
        
    Returns:
        dict: Nested dictionary with rating and age group as keys and image paths as values
    """
    result = {}
    
    # Group by rating and age group
    for rating in range(1, 6):  # Ratings from 1 to 5
        result[rating] = {}
        
        for age_group in sorted(['18-30', '31-45', '46-60', '61-75', '75+']):
            # Filter data for this specific combination
            filtered_df = df[(df['traffic_safety'] == rating) & 
                             (df['age_group'] == age_group)]
            
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
            
            result[rating][age_group] = image_paths
            logger.info(f"Selected {len(image_paths)} images for rating {rating}, age {age_group}")
    
    return result

def plot_age_grid(images_by_age, output_dir=None):
    """
    Plot a grid of images organized by safety rating (rows) and age groups (columns)
    
    Args:
        images_by_age: Nested dictionary with rating and age groups as keys and image paths as values
        output_dir: Directory to save the output images
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Age groups as columns
    age_groups = sorted(['18-30', '31-45', '46-60', '61-75', '75+'])
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(25, 30))
    
    # Define the grid structure with extra space for headers
    # We need 5 rows (ratings) and 5 columns (age groups)
    # Plus a row for column headers and a column for row headers
    
    # Create a gridspec with 6 rows and 6 columns
    # Row 0 will contain the age group headers
    # Column 0 will contain the rating headers
    gs = GridSpec(7, 6, figure=fig, height_ratios=[0.3, 1, 1, 1, 1, 1, 0.1])
    
    # Add the main title
    fig.suptitle("Traffic Safety Ratings by Age Group", fontsize=24, y=0.98)
    
    # Add age group headers (columns)
    for j, age_group in enumerate(age_groups):
        header_ax = fig.add_subplot(gs[0, j+1])
        header_ax.text(0.5, 0.5, f"Age {age_group}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='bold')
        header_ax.axis('off')
    
    # Add rating headers (rows)
    for i, rating in enumerate(range(1, 6)):
        header_ax = fig.add_subplot(gs[i+1, 0])
        header_ax.text(0.5, 0.5, f"Rating {rating}", 
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16, fontweight='bold',
                    rotation=90)
        header_ax.axis('off')
    
    # Add a legend explaining the ratings
    legend_text = "Ratings: 1 = Very Unsafe, 5 = Very Safe"
    legend_ax = fig.add_subplot(gs[6, 1:5])
    legend_ax.text(0.5, 0.5, legend_text,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=14)
    legend_ax.axis('off')
    
    # Plot each rating-age group combination
    for i, rating in enumerate(range(1, 6)):
        for j, age_group in enumerate(age_groups):
            # Create a subplot for this rating-age group combination
            ax = fig.add_subplot(gs[i+1, j+1])
            ax.set_title(f"", fontsize=12)  # Empty title since we have headers
            ax.axis('off')
            
            # Add a border to visually separate cells
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1)
            
            # Get images for this combination
            if age_group in images_by_age[rating]:
                image_paths = images_by_age[rating][age_group]
                
                if len(image_paths) > 0:
                    # Create a grid layout within this cell
                    n_images = len(image_paths)
                    grid_cols = min(2, n_images)  # 2 images per row maximum
                    grid_rows = (n_images + grid_cols - 1) // grid_cols  # Ceiling division
                    
                    for k, img_path in enumerate(image_paths):
                        if k < 4:  # Limit to 4 images per cell for readability
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
                                
                                # Add image ID as small text
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
                       fontsize=12)
    
    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])  # Add some margin
    
    # Save the figure if output directory is specified
    if output_dir:
        output_file = os.path.join(output_dir, f"traffic_safety_by_age_groups.png")
        fig.savefig(output_file, dpi=120, bbox_inches='tight')
        logger.info(f"Saved figure to {output_file}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Age Group Traffic Safety Image Viewer')
    parser.add_argument('--perception', type=str, default=DEFAULT_PERCEPTION_DATA,
                        help=f'Path to perception ratings CSV file (default: {DEFAULT_PERCEPTION_DATA})')
    parser.add_argument('--response', type=str, default=DEFAULT_RESPONSE_DATA,
                        help=f'Path to response database file (default: {DEFAULT_RESPONSE_DATA})')
    parser.add_argument('--img_path', type=str, default=DEFAULT_IMG_PATH,
                        help=f'Path to image directory (default: {DEFAULT_IMG_PATH})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--samples', type=int, default=4,
                        help='Number of samples per age category (default: 4)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and merge data
    merged_df = load_data(args.perception, args.response)
    
    # Prepare data for visualization
    prepared_data = prepare_data(merged_df)
    
    # Get images by age groups
    images_by_age = get_images_by_rating_and_age(
        prepared_data, 
        args.img_path, 
        args.samples
    )
    
    # Plot age group grid
    plot_age_grid(images_by_age, args.output_dir)
    logger.info(f"Created grid visualization for traffic safety ratings by age group")

if __name__ == "__main__":
    main() 