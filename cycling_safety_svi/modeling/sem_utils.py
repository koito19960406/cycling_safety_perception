"""
Utility functions for Structural Equation Modeling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from loguru import logger

from cycling_safety_svi.modeling.sem_classes import SEMModel, ModelType, SEMModelRegistry
from cycling_safety_svi.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def prepare_data():
    """
    Prepare the data for the structural equation model by merging the necessary datasets.
    
    Returns:
        pd.DataFrame: Merged dataset with perception ratings, segmentation data, and route utility.
    """
    logger.info("Loading and preparing data...")
    
    # Load datasets
    segmented_images = pd.read_csv(RAW_DATA_DIR / "segmented_images.csv")
    perception_ratings = pd.read_csv(RAW_DATA_DIR / "perceptionratings.csv")
    df_choice_with_Vimg = pd.read_csv(RAW_DATA_DIR / "df_choice_with_Vimg.csv")
    
    # Extract image IDs without the .jpg extension for merging
    segmented_images['image_id'] = segmented_images['image_name'].str.replace('.jpg', '')
    
    # Merge perception ratings with segmented images
    df_merged = pd.merge(perception_ratings, segmented_images, 
                         left_on='imageid', right_on='image_id', how='inner')
    
    # Create a mapping of image_id to V_img from choice data
    vimg_map1 = dict(zip(df_choice_with_Vimg['IMG1'].str.replace('.jpg', ''), df_choice_with_Vimg['V_img1']))
    vimg_map2 = dict(zip(df_choice_with_Vimg['IMG2'].str.replace('.jpg', ''), df_choice_with_Vimg['V_img2']))
    
    # Combine both mappings (might have duplicates, but should have same values)
    vimg_map = {**vimg_map1, **vimg_map2}
    
    # Add V_img to the merged dataframe
    df_merged['V_img'] = df_merged['imageid'].map(vimg_map)
    
    # Drop rows with missing V_img values
    df_merged = df_merged.dropna(subset=['V_img'])
    
    # Rename perception ratings for clarity
    df_merged.rename(columns={
        'traffic_safety': 'traffic_safety_rating',
        'social_safety': 'social_safety_rating',
        'beautiful': 'beauty_rating',
        'traffic sign_pixels': 'traffic_sign_pixels'  # Fix column name with space
    }, inplace=True)
    
    logger.info(f"Prepared dataset with {len(df_merged)} observations")
    
    return df_merged


def compare_models(models: Dict[ModelType, SEMModel], output_dir: Path = None):
    """
    Compare multiple fitted SEM models and generate a comparison table.
    
    Args:
        models: Dictionary of fitted model instances
        output_dir: Directory for saving comparison results
        
    Returns:
        pd.DataFrame: Model comparison table
    """
    logger.info("Comparing SEM models...")
    
    if output_dir is None:
        output_dir = Path(MODELS_DIR)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics to compare
    fit_metrics = ['chi2', 'DoF', 'CFI', 'RMSEA', 'AIC', 'BIC']
    comparison = {
        'Model': [],
        'Description': []
    }
    
    # Add fit metrics to comparison dict
    for metric in fit_metrics:
        comparison[metric] = []
    
    # Add model descriptions
    model_descriptions = {
        ModelType.FULL: "Full model with all paths and cross-paths",
        ModelType.SIMPLE: "Simplified model with fewer cross-paths",
        ModelType.MINIMAL: "Minimal model with essential paths only",
        ModelType.DIRECT_ONLY: "Direct effects only (no mediation)",
        ModelType.MEDIATION_ONLY: "Mediation effects only (no direct effects)",
        ModelType.DIRECT_MEDIATED: "All segmentation variables directly affect perceptions"
    }
    
    # Collect metrics from each model
    for model_type, model in models.items():
        if model.model is not None and model.fit_indices is not None:
            comparison['Model'].append(model_type.value)
            comparison['Description'].append(model_descriptions.get(model_type, ""))
            
            for metric in fit_metrics:
                # Get metric value, with proper handling for string formatting
                metric_value = model.fit_indices.get(metric, None)
                if metric_value is not None:
                    if isinstance(metric_value, str):
                        # Extract numeric value if it's in format like "Value 0.123"
                        import re
                        match = re.search(r'Value\s+([\d\.\-e]+)', metric_value)
                        if match:
                            try:
                                metric_value = float(match.group(1))
                            except ValueError:
                                # If conversion fails, keep the original
                                metric_value = None
                    elif isinstance(metric_value, pd.Series):
                        # If it's a pandas Series, extract the numeric value
                        if len(metric_value) > 0 and 'Value' in metric_value.index:
                            try:
                                metric_value = float(metric_value['Value'])
                            except (ValueError, TypeError):
                                metric_value = None
                        else:
                            metric_value = None
                comparison[metric].append(metric_value)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame(comparison)
    
    # Save comparison to file
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")
    
    # Ensure all numeric columns are properly converted to float for plotting
    for col in fit_metrics:
        # Try to convert to numeric, coercing errors to NaN
        comparison_df[col] = pd.to_numeric(comparison_df[col], errors='coerce')
    
    # Generate comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot AIC and BIC
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='AIC', data=comparison_df)
    plt.title('AIC (lower is better)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='BIC', data=comparison_df)
    plt.title('BIC (lower is better)')
    plt.xticks(rotation=45)
    
    # Plot CFI and RMSEA
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='CFI', data=comparison_df)
    plt.title('CFI (higher is better)')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='RMSEA', data=comparison_df)
    plt.title('RMSEA (lower is better)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = output_dir / "model_comparison.png"
    plt.savefig(plot_path)
    logger.info(f"Model comparison plot saved to {plot_path}")
    
    return comparison_df


def stepwise_model_selection(base_model: SEMModel, data: pd.DataFrame, 
                            variables: List[str], target_mediator: str,
                            criteria: str = 'AIC') -> Tuple[List[str], Dict[str, Any]]:
    """
    Perform stepwise variable selection for a SEM model.
    
    Args:
        base_model: Base SEM model to use as template
        data: Dataset for model fitting
        variables: List of potential variables to select from
        target_mediator: Name of mediator to optimize
        criteria: Selection criteria ('AIC', 'BIC', or 'CFI')
        
    Returns:
        Tuple containing list of selected variables and results dictionary
    """
    logger.info(f"Performing stepwise selection for {target_mediator} using {criteria}")
    
    # Track results
    results = {
        'step': [],
        'variables': [],
        'added_variable': [],
        'removed_variable': [],
        criteria: [],
        'model': []
    }
    
    # Start with no variables
    selected_vars = []
    best_score = None
    best_model = None
    
    # Forward selection phase
    logger.info("Starting forward selection phase...")
    improvement = True
    while improvement and len(selected_vars) < len(variables):
        improvement = False
        best_new_var = None
        
        for var in variables:
            if var in selected_vars:
                continue
                
            # Try adding this variable
            test_vars = selected_vars + [var]
            
            # Create and fit test model
            # This is a placeholder - would need to customize model creation based on variables
            # For this example, we'll assume there's a way to create a model with specific variables
            test_model = base_model.__class__(base_model.model_type, output_dir=base_model.output_dir)
            test_model.fit(data)
            
            # Get score based on criteria
            if criteria == 'CFI':
                score = test_model.fit_indices.get('CFI', 0)
                is_better = (best_score is None or score > best_score)
            else:  # AIC or BIC
                score = test_model.fit_indices.get(criteria, float('inf'))
                is_better = (best_score is None or score < best_score)
            
            if is_better:
                best_score = score
                best_new_var = var
                best_model = test_model
                improvement = True
        
        # Add the best variable if found
        if improvement:
            selected_vars.append(best_new_var)
            results['step'].append(len(results['step']) + 1)
            results['variables'].append(selected_vars.copy())
            results['added_variable'].append(best_new_var)
            results['removed_variable'].append(None)
            results[criteria].append(best_score)
            results['model'].append(best_model)
            logger.info(f"Added variable {best_new_var} with {criteria}={best_score}")
    
    # Backward elimination phase
    logger.info("Starting backward elimination phase...")
    improvement = True
    while improvement and len(selected_vars) > 1:
        improvement = False
        best_remove_var = None
        
        for i, var in enumerate(selected_vars):
            # Try removing this variable
            test_vars = selected_vars.copy()
            test_vars.remove(var)
            
            # Create and fit test model
            test_model = base_model.__class__(base_model.model_type, output_dir=base_model.output_dir)
            test_model.fit(data)
            
            # Get score based on criteria
            if criteria == 'CFI':
                score = test_model.fit_indices.get('CFI', 0)
                is_better = score > best_score
            else:  # AIC or BIC
                score = test_model.fit_indices.get(criteria, float('inf'))
                is_better = score < best_score
            
            if is_better:
                best_score = score
                best_remove_var = var
                best_model = test_model
                improvement = True
        
        # Remove the best variable if found
        if improvement and best_remove_var is not None:
            selected_vars.remove(best_remove_var)
            results['step'].append(len(results['step']) + 1)
            results['variables'].append(selected_vars.copy())
            results['added_variable'].append(None)
            results['removed_variable'].append(best_remove_var)
            results[criteria].append(best_score)
            results['model'].append(best_model)
            logger.info(f"Removed variable {best_remove_var} with {criteria}={best_score}")
    
    logger.info(f"Final selected variables: {selected_vars}")
    return selected_vars, results 