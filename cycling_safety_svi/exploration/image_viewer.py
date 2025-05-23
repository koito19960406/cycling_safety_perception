"""
Image Viewer for Cycling Safety SVI Project

This script randomly selects and displays 20 images per category for traffic safety,
social safety, and beauty ratings, arranging them in rows by category.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from PIL import Image
import argparse
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_FILE = "data/raw/perceptionratings.csv"
DEFAULT_IMG_PATH = "/srv/shared/bicycle_project_roos/images_scaled"
DEFAULT_OUTPUT_DIR = "reports/figures/sample_images"

def load_perception_data(data_file, num_categories=3):
    """
    Load and categorize perception data from CSV file
    
    Args:
        data_file: Path to CSV file with perception ratings
        num_categories: Number of ordinal categories (3 or 5)
        
    Returns:
        DataFrame with categorized perception ratings
    """
    logger.info(f"Loading perception data from {data_file}")
    
    # Load annotations
    data = pd.read_csv(data_file)
    
    # Categorize perception ratings
    perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
    
    for var in perception_vars:
        if num_categories == 3:
            # 3 categories: Low (1-2), Medium (3), High (4-5)
            bins = [0, 2.5, 3.5, 5.1]
            labels = [0, 1, 2]  # 0-indexed (low, medium, high)
            label_map = {0: 'low', 1: 'medium', 2: 'high'}
        elif num_categories == 5:
            # 5 categories: 1, 2, 3, 4, 5
            bins = [0, 1.5, 2.5, 3.5, 4.5, 5.1]
            labels = [0, 1, 2, 3, 4]  # 0-indexed
            label_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
        else:
            raise ValueError(f"Invalid num_categories: {num_categories}. Must be 3 or 5")
        
        # Create categorical variable
        data[f'{var}_cat'] = pd.cut(
            data[var], 
            bins=bins, 
            labels=labels
        ).astype(int)
        
        # Add string label for display purposes
        data[f'{var}_label'] = data[f'{var}_cat'].map(label_map)
    
    return data

def sample_images_by_category(data, variable, samples_per_category=20, seed=42):
    """
    Sample images for each category of a perception variable
    
    Args:
        data: DataFrame with categorized perception ratings
        variable: Perception variable to sample ('traffic_safety', 'social_safety', 'beautiful')
        samples_per_category: Number of samples to select per category
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping category to list of image IDs
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get unique categories
    categories = sorted(data[f'{variable}_cat'].unique())
    
    # Sample images for each category
    sampled_images = {}
    
    for category in categories:
        # Get all images in this category
        category_images = data[data[f'{variable}_cat'] == category]['imageid'].tolist()
        
        # Sample images (or take all if there are fewer than requested)
        if len(category_images) <= samples_per_category:
            sampled = category_images
        else:
            sampled = random.sample(category_images, samples_per_category)
        
        # Map the numeric category to its label
        category_label = data[data[f'{variable}_cat'] == category][f'{variable}_label'].iloc[0]
        sampled_images[category_label] = sampled
    
    return sampled_images

def plot_sampled_images(sampled_images, img_path, variable, output_dir=None, show_plot=True):
    """
    Plot sampled images for each category
    
    Args:
        sampled_images: Dictionary mapping category to list of image IDs
        img_path: Path to directory containing images
        variable: Perception variable ('traffic_safety', 'social_safety', 'beautiful')
        output_dir: Directory to save plot (if None, plot is not saved)
        show_plot: Whether to display the plot
    """
    img_path = Path(img_path)
    
    # Get categories and number of samples per category
    categories = list(sampled_images.keys())
    n_categories = len(categories)
    n_samples = len(next(iter(sampled_images.values())))
    
    # Create figure and axes
    fig, axes = plt.subplots(n_categories, n_samples, figsize=(n_samples * 2, n_categories * 2))
    
    # Set title
    title_map = {
        'traffic_safety': 'Traffic Safety',
        'social_safety': 'Social Safety',
        'beautiful': 'Beauty'
    }
    fig.suptitle(f'{title_map.get(variable, variable)} Ratings - 20 Random Images per Category', fontsize=16)
    
    # Plot images
    for i, category in enumerate(categories):
        logger.info(f"Processing {category} images for {variable}")
        
        for j, image_id in enumerate(tqdm(sampled_images[category])):
            image_path = img_path / f"{image_id}.jpg"
            
            try:
                # Load and convert image
                image = Image.open(image_path).convert('RGB')
                
                # Plot image
                if n_categories == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.imshow(image)
                ax.axis('off')
                
                # Set title for first column only
                if j == 0:
                    ax.set_title(f"{category}", fontsize=12, y=-0.15)
                
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
                # Display error message on plot
                if n_categories == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                ax.text(0.5, 0.5, f"Image not found", ha='center', va='center')
                ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    
    # Save plot if output directory is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{variable}_samples.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def main():
    """Main function to parse arguments and run the script"""
    parser = argparse.ArgumentParser(description='View sample images by perception category')
    
    parser.add_argument('--data-file', type=str, default=DEFAULT_DATA_FILE,
                        help='Path to perception ratings CSV file')
    parser.add_argument('--img-path', type=str, default=DEFAULT_IMG_PATH,
                        help='Path to directory containing images')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save plots')
    parser.add_argument('--variable', type=str, choices=['traffic_safety', 'social_safety', 'beautiful', 'all'],
                        default='all', help='Perception variable to visualize')
    parser.add_argument('--num-categories', type=int, choices=[3, 5], default=5,
                        help='Number of categories to use (3 or 5)')
    parser.add_argument('--samples-per-category', type=int, default=20,
                        help='Number of samples to select per category')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (only save them)')
    
    args = parser.parse_args()
    
    # Load perception data
    data = load_perception_data(args.data_file, args.num_categories)
    
    # Determine variables to visualize
    if args.variable == 'all':
        variables = ['traffic_safety', 'social_safety', 'beautiful']
    else:
        variables = [args.variable]
    
    # Process each variable
    for variable in variables:
        logger.info(f"Processing {variable} variable")
        
        # Sample images for this variable
        sampled_images = sample_images_by_category(
            data, 
            variable, 
            samples_per_category=args.samples_per_category,
            seed=args.seed
        )
        
        # Plot sampled images
        plot_sampled_images(
            sampled_images, 
            args.img_path, 
            variable, 
            output_dir=args.output_dir,
            show_plot=not args.no_show
        )

if __name__ == "__main__":
    main() 