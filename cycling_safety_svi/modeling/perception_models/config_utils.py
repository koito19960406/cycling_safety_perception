"""
Configuration utilities for perception models
"""
import os
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def parse_args():
    """Parse command line arguments for the training script"""
    parser = argparse.ArgumentParser(description="Train perception model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        help="Path to perception ratings CSV file (overrides config)"
    )
    parser.add_argument(
        "--img_path", 
        type=str, 
        help="Path to image directory (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save model and results (overrides config)"
    )
    parser.add_argument(
        "--num_categories", 
        type=int, 
        choices=[3, 5],
        help="Number of categories (3 or 5) (overrides config)"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true", 
        help="Enable Weights & Biases logging (overrides config)"
    )
    parser.add_argument(
        "--no_wandb", 
        action="store_true", 
        help="Disable Weights & Biases logging (overrides config)"
    )
    parser.add_argument(
        "--no_optuna", 
        action="store_true", 
        help="Skip Optuna hyperparameter optimization (overrides config)"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def override_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override configuration with command line arguments
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create paths section if it doesn't exist
    if "paths" not in config:
        config["paths"] = {}
    
    # Override paths if provided
    if args.data_file:
        config["paths"]["data_file"] = args.data_file
    
    if args.img_path:
        config["paths"]["img_path"] = args.img_path
        
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    
    # Override num_categories if provided
    if args.num_categories:
        config["dataset"]["num_categories"] = args.num_categories
    
    # Override wandb settings if provided
    if args.use_wandb:
        config["output"]["use_wandb"] = True
    
    if args.no_wandb:
        config["output"]["use_wandb"] = False
    
    # Override optuna settings if provided
    if args.no_optuna:
        config["optuna"]["use_optuna"] = False
    
    return config


def setup_experiment(
    args: Optional[argparse.Namespace] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Set up experiment configuration
    
    Args:
        args: Command line arguments (optional)
        config_path: Path to configuration file (optional)
        
    Returns:
        Configuration dictionary
    """
    # Parse arguments if not provided
    if args is None:
        args = parse_args()
    
    # Use provided config_path if any, else use the one from args
    config_path = config_path or args.config
    
    # Load configuration
    config = load_config(config_path)
    
    # Override with command line arguments
    config = override_config_with_args(config, args)
    
    return config 