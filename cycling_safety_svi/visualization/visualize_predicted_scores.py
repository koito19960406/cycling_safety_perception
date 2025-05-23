#!/usr/bin/env python3
"""
Visualization script for cycling safety prediction results.
Categorizes images into 5 safety classes and displays sample images.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def load_results(csv_path):
    """Load the prediction results CSV."""
    return pd.read_csv(csv_path)


def categorize_scores(df, n_categories=5):
    """Categorize safety scores into n_categories bins."""
    df = df.copy()
    
    # Create quantile-based categories
    labels = [f'Class_{i+1}' for i in range(n_categories)]
    df['category'] = pd.qcut(df['safety_score'], 
                             q=n_categories, 
                             labels=labels,
                             duplicates='drop')
    
    return df


def get_category_labels(df):
    """Get descriptive labels for categories based on score ranges."""
    categories = df.groupby('category')['safety_score'].agg(['min', 'max'])
    
    labels = {}
    for cat in categories.index:
        min_score = categories.loc[cat, 'min']
        max_score = categories.loc[cat, 'max']
        
        if 'Class_1' in cat:
            label = f"Very Unsafe\n({min_score:.2f} - {max_score:.2f})"
        elif 'Class_2' in cat:
            label = f"Unsafe\n({min_score:.2f} - {max_score:.2f})"
        elif 'Class_3' in cat:
            label = f"Neutral\n({min_score:.2f} - {max_score:.2f})"
        elif 'Class_4' in cat:
            label = f"Safe\n({min_score:.2f} - {max_score:.2f})"
        elif 'Class_5' in cat:
            label = f"Very Safe\n({min_score:.2f} - {max_score:.2f})"
        else:
            label = f"{cat}\n({min_score:.2f} - {max_score:.2f})"
        
        labels[cat] = label
    
    return labels


def create_visualization(df, images_dir, output_path, 
                         images_per_class=20, figsize=(20, 15)):
    """Create visualization showing sample images for each safety category."""
    
    # Get category labels
    category_labels = get_category_labels(df)
    categories = sorted(df['category'].unique())
    
    # Create figure with reduced height and tighter spacing
    fig, axes = plt.subplots(len(categories), images_per_class, 
                             figsize=figsize)
    
    if len(categories) == 1:
        axes = axes.reshape(1, -1)
    
    title_text = ('Cycling Safety Perception Predictions\n'
                  'Images categorized by predicted safety scores')
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    for row, category in enumerate(categories):
        # Get images for this category
        category_df = df[df['category'] == category].copy()
        
        # Sample images if we have more than needed
        if len(category_df) > images_per_class:
            category_df = category_df.sample(n=images_per_class, 
                                             random_state=42)
        
        # Set category label as ylabel for perfect alignment
        category_label = category_labels.get(category, category)
        ylabel_ax = axes[row, 0]
        ylabel_ax.set_ylabel(category_label, fontsize=12, 
                             fontweight='bold', rotation=0, 
                             labelpad=60, verticalalignment='center')
        
        for col in range(images_per_class):
            ax = axes[row, col]
            
            if col < len(category_df):
                # Load and display image
                image_name = category_df.iloc[col]['image_name']
                score = category_df.iloc[col]['safety_score']
                
                image_path = os.path.join(images_dir, image_name)
                
                try:
                    img = Image.open(image_path)
                    ax.imshow(img)
                    ax.set_title(f'{score:.2f}', fontsize=8)
                except Exception as e:
                    # If image can't be loaded, show placeholder
                    text = f'Image not found\n{image_name}'
                    ax.text(0.5, 0.5, text, 
                            ha='center', va='center', 
                            transform=ax.transAxes, fontsize=6)
                    print(f"Warning: Could not load {image_path}: {e}")
            else:
                # Empty subplot if not enough images
                ax.text(0.5, 0.5, 'No image', 
                        ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    
    # Very tight layout with minimal spacing
    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.1)
    plt.subplots_adjust(left=0.15, top=0.92, bottom=0.01, hspace=0.02)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Also save a summary statistics plot
    dist_path = output_path.replace('.png', '_distribution.png')
    create_score_distribution_plot(df, dist_path)
    
    plt.close()


def create_score_distribution_plot(df, output_path):
    """Create a histogram showing the distribution of safety scores."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram
    ax1.hist(df['safety_score'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Safety Score')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Distribution of Safety Scores')
    ax1.grid(True, alpha=0.3)
    
    # Box plot by category
    categories = sorted(df['category'].unique())
    category_data = [df[df['category'] == cat]['safety_score'].values 
                     for cat in categories]
    
    ax2.boxplot(category_data, labels=categories)
    ax2.set_xlabel('Safety Category')
    ax2.set_ylabel('Safety Score')
    ax2.set_title('Safety Score Distribution by Category')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Score distribution plot saved to: {output_path}")
    
    plt.close()


def print_summary_statistics(df):
    """Print summary statistics for the predictions."""
    print("\n" + "="*60)
    print("CYCLING SAFETY PREDICTION SUMMARY")
    print("="*60)
    
    score_min = df['safety_score'].min()
    score_max = df['safety_score'].max()
    score_mean = df['safety_score'].mean()
    score_median = df['safety_score'].median()
    score_std = df['safety_score'].std()
    
    print(f"Total images processed: {len(df)}")
    print(f"Safety score range: {score_min:.3f} to {score_max:.3f}")
    print(f"Mean safety score: {score_mean:.3f}")
    print(f"Median safety score: {score_median:.3f}")
    print(f"Standard deviation: {score_std:.3f}")
    
    print("\nCategory distribution:")
    category_counts = df['category'].value_counts().sort_index()
    for cat, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {cat}: {count} images ({percentage:.1f}%)")
    
    print("\nScore statistics by category:")
    for cat in sorted(df['category'].unique()):
        cat_data = df[df['category'] == cat]['safety_score']
        cat_mean = cat_data.mean()
        cat_std = cat_data.std()
        cat_min = cat_data.min()
        cat_max = cat_data.max()
        print(f"  {cat}: mean={cat_mean:.3f}, std={cat_std:.3f}, "
              f"range=[{cat_min:.3f}, {cat_max:.3f}]")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize cycling safety prediction results')
    parser.add_argument(
        '--results_csv',
        default='data/processed/predicted_danish/cycling_safety_scores.csv',
        help='Path to the results CSV file')
    parser.add_argument(
        '--images_dir',
        default='/srv/shared/bicycle_project_roos/images_scaled',
        help='Directory containing the original images')
    parser.add_argument(
        '--output_dir',
        default='reports/figures/predicted_images',
        help='Directory to save visualization plots')
    parser.add_argument(
        '--images_per_class',
        type=int,
        default=20,
        help='Number of images to show per safety class')
    parser.add_argument(
        '--n_categories',
        type=int,
        default=5,
        help='Number of safety categories to create')
    
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results_csv):
        print(f"Error: Results file not found at {args.results_csv}")
        print("Please run the apply_safety_model.py script first.")
        return 1
    
    print(f"Loading results from: {args.results_csv}")
    df = load_results(args.results_csv)
    
    # Categorize scores
    df = categorize_scores(df, n_categories=args.n_categories)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization
    output_path = os.path.join(args.output_dir, 'safety_categories_grid.png')
    print("\nCreating visualization...")
    
    create_visualization(df, args.images_dir, output_path, 
                         images_per_class=args.images_per_class)
    
    print("\nVisualization complete!")
    
    return 0


if __name__ == '__main__':
    exit(main()) 