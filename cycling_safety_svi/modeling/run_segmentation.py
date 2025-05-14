#!/usr/bin/env python
"""
Run segmentation on cycling safety images.

This is a utility script to run the segmentation module with proper defaults
for the cycling safety project based on the analysis outline.
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the segmentation module
try:
    from cycling_safety_svi.modeling.segmentation import run_segmentation, process_and_combine_results
except ImportError:
    logger.error("Could not import segmentation module. Make sure it's in the Python path.")
    sys.exit(1)

# Default paths
DEFAULT_INPUT_DIR = "/srv/shared/bicycle_project_roos/images_scaled"
DEFAULT_OUTPUT_DIR = "data/processed/segmented_images"
DEFAULT_SUMMARY_DIR = "data/processed/segmentation_results"
DEFAULT_RESULTS_FILE = "data/processed/segmentation_features.csv"

# Categories relevant for cycling safety
CYCLING_CATEGORIES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", 
    "traffic light", "traffic sign", "vegetation", "terrain", 
    "sky", "person", "rider", "car", "truck", "bus", 
    "train", "motorcycle", "bicycle", "water", "crosswalk"
]

def main():
    """Run segmentation with default settings for cycling safety project"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run semantic segmentation on cycling safety images')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'Path to directory containing input images (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Path to directory to save segmented images (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--summary_dir', type=str, default=DEFAULT_SUMMARY_DIR,
                        help=f'Path to directory to save summary results (default: {DEFAULT_SUMMARY_DIR})')
    parser.add_argument('--results_file', type=str, default=DEFAULT_RESULTS_FILE,
                        help=f'Path to save combined results CSV file (default: {DEFAULT_RESULTS_FILE})')
    parser.add_argument('--model', type=str, default='custom',
                        choices=['cityscapes', 'pascal_voc', 'ade20k', 'custom'],
                        help='Model to use for segmentation (default: custom)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing images (default: 8)')
    parser.add_argument('--force', action='store_true',
                        help='Force reprocessing even if output files exist')
    
    args = parser.parse_args()
    
    # Create absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    summary_dir = os.path.abspath(args.summary_dir)
    results_file = os.path.abspath(args.results_file)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Check if results already exist
    if not args.force and os.path.exists(results_file):
        logger.warning(f"Results file already exists: {results_file}")
        choice = input("Do you want to reprocess the images? (y/n): ")
        if choice.lower() != 'y':
            logger.info("Exiting without reprocessing")
            sys.exit(0)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Run segmentation
    logger.info(f"Running segmentation on images in {input_dir}")
    logger.info(f"Output images will be saved to {output_dir}")
    logger.info(f"Summary results will be saved to {summary_dir}")
    
    try:
        summary_df = run_segmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            summary_dir=summary_dir,
            model_name=args.model,
            categories=CYCLING_CATEGORIES,
            batch_size=args.batch_size
        )
        
        # Process and combine results
        if summary_df is not None:
            combined_df = process_and_combine_results(
                input_dir=input_dir,
                output_dir=output_dir,
                summary_dir=summary_dir,
                model_name=args.model
            )
            
            # Save combined results
            if combined_df is not None:
                combined_df.to_csv(results_file, index=False)
                logger.info(f"Combined results saved to {results_file}")
                logger.info(f"Found {len(combined_df)} images with segmentation results")
                
                # Show example of the data
                logger.info("Example of the data:")
                print(combined_df.head())
                
                # Count non-zero categories
                nonzero_counts = (combined_df.iloc[:, 1:] > 0).sum()
                top_categories = nonzero_counts.sort_values(ascending=False).head(10)
                logger.info("Top 10 most common categories:")
                for category, count in top_categories.items():
                    logger.info(f"  {category}: {count} images")
    
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        sys.exit(1)
    
    logger.info("Segmentation completed successfully")

if __name__ == "__main__":
    main() 