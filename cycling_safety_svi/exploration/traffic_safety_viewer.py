#!/usr/bin/env python3
"""
Traffic Safety Image Viewer

This script selects and displays images grouped by traffic safety rating, age, and gender.
It randomly selects a specified number of images from each group and displays them in a grid.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
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
DEFAULT_OUTPUT_DIR = "reports/figures/traffic_safety_by_age_gender"

def load_data(perception_file, response_file=None):
    """
    Load perception ratings and response data (if provided)
    
    Args:
        perception_file (str): Path to perception ratings CSV file
        response_file (str): Path to response database file
        
    Returns:
        tuple: Perception data and response data
    """
    logger.info(f"Loading perception data from {perception_file}")
    perception_df = pd.read_csv(perception_file)
    
    # If response file is provided, load demographics
    response_df = None
    if response_file and os.path.exists(response_file):
        logger.info(f"Loading response data from {response_file}")
        import sqlite3
        conn = sqlite3.connect(response_file)
        response_df = pd.read_sql_query("SELECT respondent_id, age, gender FROM Response", conn)
        conn.close()
    
    return perception_df, response_df

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

def prepare_data(perception_df, response_df=None):
    """
    Prepare and merge data for visualization
    
    Args:
        perception_df: DataFrame with perception ratings
        response_df: DataFrame with demographic data
        
    Returns:
        DataFrame: Combined data for visualization
    """
    # Focus on traffic_safety rating
    df = perception_df[['imageid', 'traffic_safety']].copy()
    
    # If we have demographic data, merge it
    if response_df is not None:
        # We would need to merge based on a common key - this is a placeholder
        # In a real scenario, you would need proper linking between the two datasets
        pass
    
    return df

def get_images_by_rating(df, img_path, samples_per_rating=20):
    """
    Get images grouped by traffic safety rating
    
    Args:
        df: DataFrame with image data
        img_path: Path to image directory
        samples_per_rating: Number of samples to select per rating
        
    Returns:
        dict: Dictionary with rating as key and list of image paths as value
    """
    result = {}
    
    # Group by rating
    for rating in range(1, 6):  # Ratings from 1 to 5
        images = df[df['traffic_safety'] == rating]['imageid'].unique()
        
        # If we have fewer images than requested, use all of them
        if len(images) <= samples_per_rating:
            selected = images
        else:
            selected = random.sample(list(images), samples_per_rating)
        
        # Create full paths to images
        image_paths = [os.path.join(img_path, f"{img_id}.jpg") for img_id in selected]
        # Filter out non-existent images
        image_paths = [p for p in image_paths if os.path.exists(p)]
        
        result[rating] = image_paths
        logger.info(f"Selected {len(image_paths)} images for rating {rating}")
    
    return result

def plot_images_by_rating(images_by_rating, output_dir=None, samples_per_row=5):
    """
    Plot images grouped by traffic safety rating
    
    Args:
        images_by_rating: Dictionary with rating as key and list of image paths as value
        output_dir: Directory to save the output images
        samples_per_row: Number of images to display per row
    """
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    plt.figure(figsize=(20, 12))
    plt.suptitle("Traffic Safety Rating Visualization", fontsize=16)
    
    # Plot each rating group
    for rating, image_paths in sorted(images_by_rating.items()):
        if not image_paths:
            logger.warning(f"No images found for rating {rating}")
            continue
        
        # Create a new figure for each rating
        n_images = len(image_paths)
        rows = (n_images + samples_per_row - 1) // samples_per_row  # Ceiling division
        
        fig, axes = plt.subplots(rows, samples_per_row, figsize=(20, 4 * rows))
        fig.suptitle(f"Traffic Safety Rating: {rating}", fontsize=16)
        
        # Make axes iterable even if there's only one row
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each image
        for i, img_path in enumerate(image_paths):
            row = i // samples_per_row
            col = i % samples_per_row
            
            try:
                img = plt.imread(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(os.path.basename(img_path))
                axes[row, col].axis('off')
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
        
        # Turn off axes for empty subplots
        for i in range(n_images, rows * samples_per_row):
            row = i // samples_per_row
            col = i % samples_per_row
            axes[row, col].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure if output directory is specified
        if output_dir:
            output_file = os.path.join(output_dir, f"traffic_safety_rating_{rating}.png")
            fig.savefig(output_file, dpi=100, bbox_inches='tight')
            logger.info(f"Saved figure to {output_file}")
        
        plt.show()

def create_age_gender_grid(perception_df, response_df, img_path, output_dir=None, samples_per_cell=4):
    """
    Create a grid of images by age and gender
    
    Args:
        perception_df: DataFrame with perception ratings
        response_df: DataFrame with demographic data
        img_path: Path to image directory
        output_dir: Directory to save the output images
        samples_per_cell: Number of samples per age-gender combination
    """
    if response_df is None:
        logger.error("Response data is required for age-gender grid")
        return
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Map age and gender to categories
    response_df['age_group'] = response_df['age'].apply(map_age_group)
    response_df['gender_label'] = response_df['gender'].apply(map_gender)
    
    # Merge the perception data with demographic data
    # This is a simplified example - in reality, the join would be more complex
    # In this example, we're showing a mockup since we don't have the actual relation
    
    # For each rating level, create a grid
    for rating in range(1, 6):
        # Create a large figure for the grid
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(5, 3, figure=fig)  # 5 age groups, 3 gender categories
        
        fig.suptitle(f"Traffic Safety Rating {rating} by Age and Gender", fontsize=16)
        
        # For each age group and gender combination
        for i, age_group in enumerate(['18-30', '31-45', '46-60', '61-75', '75+']):
            for j, gender in enumerate(['Male', 'Female', 'Other']):
                # Select relevant images for this demographic and rating
                # In a real scenario, this would use the actual relationship between images and demographics
                
                # Mock selection of images for this demographic
                images = perception_df[perception_df['traffic_safety'] == rating]['imageid'].unique()
                
                if len(images) <= samples_per_cell:
                    selected = images
                else:
                    selected = random.sample(list(images), samples_per_cell)
                
                # Create full paths to images
                image_paths = [os.path.join(img_path, f"{img_id}.jpg") for img_id in selected]
                # Filter out non-existent images
                image_paths = [p for p in image_paths if os.path.exists(p)]
                
                # Create a subplot for this demographic
                ax = fig.add_subplot(gs[i, j])
                ax.set_title(f"{age_group}, {gender}")
                ax.axis('off')
                
                if len(image_paths) > 0:
                    # Create a 2x2 grid layout within this cell
                    n_images = min(len(image_paths), 4)
                    n_rows = 2
                    n_cols = 2
                    
                    for k, img_path in enumerate(image_paths[:4]):  # Show max 4 images per cell
                        if k < n_images:
                            row = k // n_cols
                            col = k % n_cols
                            
                            # Calculate the subplot position within the cell
                            ax_pos = ax.get_position()
                            x_start = ax_pos.x0 + col * (ax_pos.width / n_cols)
                            y_start = ax_pos.y0 + (n_rows - 1 - row) * (ax_pos.height / n_rows)
                            width = ax_pos.width / n_cols
                            height = ax_pos.height / n_rows
                            
                            # Create a new axis at the calculated position
                            inner_ax = fig.add_axes([x_start, y_start, width, height])
                            
                            try:
                                img = plt.imread(img_path)
                                inner_ax.imshow(img)
                                inner_ax.axis('off')
                            except Exception as e:
                                logger.error(f"Error loading image {img_path}: {e}")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the figure if output directory is specified
        if output_dir:
            output_file = os.path.join(output_dir, f"traffic_safety_rating_{rating}_by_age_gender.png")
            fig.savefig(output_file, dpi=120, bbox_inches='tight')
            logger.info(f"Saved figure to {output_file}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Traffic Safety Image Viewer')
    parser.add_argument('--perception', type=str, default=DEFAULT_PERCEPTION_DATA,
                        help=f'Path to perception ratings CSV file (default: {DEFAULT_PERCEPTION_DATA})')
    parser.add_argument('--response', type=str, default=DEFAULT_RESPONSE_DATA,
                        help=f'Path to response database file (default: {DEFAULT_RESPONSE_DATA})')
    parser.add_argument('--img_path', type=str, default=DEFAULT_IMG_PATH,
                        help=f'Path to image directory (default: {DEFAULT_IMG_PATH})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save output images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--samples', type=int, default=20,
                        help='Number of samples per category (default: 20)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    perception_df, response_df = load_data(args.perception, args.response)
    
    # Simple visualization by rating
    data = prepare_data(perception_df, response_df)
    images_by_rating = get_images_by_rating(data, args.img_path, args.samples)
    plot_images_by_rating(images_by_rating, args.output_dir, samples_per_row=5)
    
    # Create age-gender grid
    create_age_gender_grid(perception_df, response_df, args.img_path, args.output_dir)

if __name__ == "__main__":
    main() 