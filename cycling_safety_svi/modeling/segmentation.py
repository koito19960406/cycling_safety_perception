"""
Semantic segmentation module for cycling safety images.

This module applies semantic segmentation to cycling safety images
using the ZenSVI library's Segmenter class.

Based on the analysis outline in reports/human_made/analysis_outline_20250513.txt
"""

import os
import glob
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from zensvi.cv import Segmenter
except ImportError:
    logger.error("ZenSVI library not found. Please install it with: pip install zensvi")
    logger.info("For more information visit: https://github.com/koito19960406/ZenSVI")
    raise

DEFAULT_INPUT_DIR = "/srv/shared/bicycle_project_roos/images_scaled"
DEFAULT_OUTPUT_DIR = "data/processed/segmented_images"
DEFAULT_SUMMARY_DIR = "data/processed/segmentation_results"

segmenter = Segmenter(
    dataset='mapillary',
    task='semantic',
    device='cuda'
)

segmenter.segment(
    dir_input=DEFAULT_INPUT_DIR,
    dir_summary_output=DEFAULT_SUMMARY_DIR,
    batch_size=64,
    save_format='csv',
    csv_format='wide'
)