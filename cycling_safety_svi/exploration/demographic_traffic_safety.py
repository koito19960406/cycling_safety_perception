#!/usr/bin/env python3
"""
Demographic Traffic Safety Image Viewer

This script links perception ratings with demographic data using set_id as the join key,
then displays images grouped by safety ratings for different demographic categories.
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
DEFAULT_OUTPUT_DIR = "reports/figures/demographic_safety"

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
        "SELECT respondent_id, set_id, age, gender FROM Response", 
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

def map_gender(gender_value):
    """Map numeric gender value to gender label"""
    gender_mapping = {
        1: "Male",
        2: "Female",
        3: "Other"
    }
    return gender_mapping.get(gender_value, "Unknown")

def prepare_data(merged_df):
    """
    Prepare data for visualization by adding age group and gender labels
    
    Args:
        merged_df: DataFrame with merged perception and demographic data
        
    Returns:
        DataFrame: Data prepared for visualization
    """
    # Add age group and gender labels
    df = merged_df.copy()
    df['age_group'] = df['age'].apply(map_age_group)
    df['gender_label'] = df['gender'].apply(map_gender)
    
    return df

def get_images_by_rating_and_demographics(df, img_path, samples_per_category=4):
    """
    Get images grouped by traffic safety rating, age group, and gender
    
    Args:
        df: DataFrame with image and demographic data
        img_path: Path to image directory
        samples_per_category: Number of samples to select per category
        
    Returns:
        dict: Nested dictionary with rating, age group, and gender as keys and image paths as values
    """
    result = {}
    
    # Group by rating, age group, and gender
    for rating in range(1, 6):  # Ratings from 1 to 5
        result[rating] = {}
        
        for age_group in df['age_group'].unique():
            result[rating][age_group] = {}
            
            for gender in df['gender_label'].unique():
                # Filter data for this specific combination
                filtered_df = df[(df['traffic_safety'] == rating) & 
                                 (df['age_group'] == age_group) & 
                                 (df['gender_label'] == gender)]
                
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
                
                result[rating][age_group][gender] = image_paths
                logger.info(f"Selected {len(image_paths)} images for rating {rating}, age {age_group}, gender {gender}")
    
    return result

def plot_demographic_grid(images_by_demographics, output_dir=None):
    """
    Plot a grid of images organized by safety rating (low to high) and gender side by side
    
    Args:
        images_by_demographics: Nested dictionary with demographic data and image paths
        output_dir: Directory to save the output images
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Focus on Male and Female only for clarity
    genders = ['Male', 'Female']
    
    # Create a figure for all ratings
    fig = plt.figure(figsize=(20, 30))
    fig.suptitle(f"Traffic Safety Ratings by Gender", fontsize=20)
    
    # We want ratings vertically (low to high) and genders horizontally
    n_ratings = 5  # 1-5
    n_genders = len(genders)
    
    # Create the main grid: ratings as rows, genders as columns
    gs = GridSpec(n_ratings, n_genders, figure=fig, height_ratios=[1, 1, 1, 1, 1])
    
    # For each rating and gender combination
    for rating in range(1, 6):
        for j, gender in enumerate(genders):
            # Create a subplot for this rating-gender combination
            ax = fig.add_subplot(gs[rating-1, j])
            ax.set_title(f"Rating {rating}, {gender}", fontsize=14)
            ax.axis('off')
            
            # Create a border around the subplot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            
            # Find all images for this rating-gender combination across all age groups
            all_images = []
            for age_group in images_by_demographics[rating]:
                if gender in images_by_demographics[rating][age_group]:
                    all_images.extend(images_by_demographics[rating][age_group][gender])
            
            # Randomly select a subset if we have too many
            max_images = 8  # Show more images per rating-gender combination
            if len(all_images) > max_images:
                all_images = random.sample(all_images, max_images)
            
            if len(all_images) > 0:
                # Create a grid layout within this cell
                n_images = len(all_images)
                grid_cols = 4  # 4 images per row
                grid_rows = (n_images + grid_cols - 1) // grid_cols  # Ceiling division
                
                for k, img_path in enumerate(all_images):
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
                        
                        # Add image ID as title
                        img_id = os.path.basename(img_path).replace('.jpg', '')
                        inner_ax.set_title(img_id, fontsize=8)
                    except Exception as e:
                        logger.error(f"Error loading image {img_path}: {e}")
            else:
                # If no images for this combination, display a message
                ax.text(0.5, 0.5, "No Data", 
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes,
                       fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for the suptitle
    
    # Save the figure if output directory is specified
    if output_dir:
        output_file = os.path.join(output_dir, f"traffic_safety_by_rating_and_gender.png")
        fig.savefig(output_file, dpi=120, bbox_inches='tight')
        logger.info(f"Saved figure to {output_file}")
    
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Demographic Traffic Safety Image Viewer')
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
    
    # Get images by demographic categories
    images_by_demographics = get_images_by_rating_and_demographics(
        prepared_data, 
        args.img_path, 
        args.samples
    )
    
    # Plot a single consolidated grid with ratings as rows and genders as columns
    plot_demographic_grid(images_by_demographics, args.output_dir)
    logger.info(f"Created consolidated grid for traffic safety ratings by gender")

if __name__ == "__main__":
    main() 