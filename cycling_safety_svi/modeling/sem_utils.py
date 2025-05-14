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
import time
from loguru import logger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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
        ModelType.DIRECT_MEDIATED: "All segmentation variables directly affect perceptions",
        ModelType.BENCHMARK: "Simple benchmark with all variables as direct predictors"
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


def cross_validate_models(
    models_to_test: Dict[ModelType, type], 
    data: pd.DataFrame,
    target_col: str = 'V_img',
    n_splits: int = 5,
    output_dir: Path = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform k-fold cross-validation for multiple SEM models.
    
    Args:
        models_to_test: Dictionary mapping model types to model classes
        data: DataFrame containing all variables needed for SEM models
        target_col: Target column to predict (default: 'V_img')
        n_splits: Number of folds for cross-validation (default: 5)
        output_dir: Directory to save results (default: None)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        DataFrame with cross-validation results
    """
    logger.info(f"Performing {n_splits}-fold cross-validation for {len(models_to_test)} models")
    
    if output_dir is None:
        output_dir = Path(MODELS_DIR) / "cross_validation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataframe to store results
    results = []
    
    # Create cross-validation splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize metrics for each model and fold
    metrics = ['MSE', 'RMSE', 'R2', 'AIC', 'BIC', 'CFI', 'RMSEA']
    
    # For each model type
    for model_type, model_class in models_to_test.items():
        logger.info(f"Cross-validating {model_type.value} model")
        
        model_metrics = {
            'Model': model_type.value,
            'Fold': [],
            'MSE': [],
            'RMSE': [],
            'R2': [],
            'AIC': [],
            'BIC': [],
            'CFI': [],
            'RMSEA': [],
            'Converged': [],
            'Time': []
        }
        
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
            logger.info(f"Fold {fold}/{n_splits}")
            
            # Create train/test sets
            train_data = data.iloc[train_idx].copy()
            test_data = data.iloc[test_idx].copy()
            
            # Record start time
            start_time = time.time()
            
            try:
                # Create and fit model
                model = model_class(model_type, output_dir=output_dir / f"fold_{fold}")
                model.fit(train_data)
                model_converged = True
                
                # Record model fitting time
                fit_time = time.time() - start_time
                
                # Extract fit indices
                if model.fit_indices is not None:
                    fit_indices = model.fit_indices.copy()
                else:
                    fit_indices = {}
                
                # Calculate predictions on test data
                try:
                    # Get predictions using the model
                    test_preds = predict_with_sem_model(model, test_data, target_col)
                    
                    # Store true and predicted values
                    y_true = test_data[target_col].values
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_true, test_preds)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_true, test_preds)
                    
                    # Store metrics
                    model_metrics['Fold'].append(fold)
                    model_metrics['MSE'].append(mse)
                    model_metrics['RMSE'].append(rmse)
                    model_metrics['R2'].append(r2)
                    
                    # Get other metrics from fit indices
                    for metric in ['AIC', 'BIC', 'CFI', 'RMSEA']:
                        if metric in fit_indices:
                            try:
                                # Extract value from semopy format
                                value = fit_indices[metric]
                                if isinstance(value, pd.Series) and 'Value' in value.index:
                                    value = value['Value']
                                model_metrics[metric].append(value)
                            except (ValueError, KeyError):
                                model_metrics[metric].append(np.nan)
                        else:
                            model_metrics[metric].append(np.nan)
                    
                    model_metrics['Converged'].append(True)
                    model_metrics['Time'].append(fit_time)
                    
                except Exception as e:
                    logger.error(f"Error in prediction for fold {fold}: {e}")
                    # Add NaN values for this fold
                    model_metrics['Fold'].append(fold)
                    model_metrics['MSE'].append(np.nan)
                    model_metrics['RMSE'].append(np.nan)
                    model_metrics['R2'].append(np.nan)
                    
                    # Get other metrics from fit indices
                    for metric in ['AIC', 'BIC', 'CFI', 'RMSEA']:
                        if metric in fit_indices:
                            try:
                                # Extract value from semopy format
                                value = fit_indices[metric]
                                if isinstance(value, pd.Series) and 'Value' in value.index:
                                    value = value['Value']
                                model_metrics[metric].append(value)
                            except (ValueError, KeyError):
                                model_metrics[metric].append(np.nan)
                        else:
                            model_metrics[metric].append(np.nan)
                    
                    model_metrics['Converged'].append(False)
                    model_metrics['Time'].append(fit_time)
                
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                # Add NaN values for this fold with explanation
                model_metrics['Fold'].append(fold)
                for metric in metrics:
                    model_metrics[metric].append(np.nan)
                model_metrics['Converged'].append(False)
                model_metrics['Time'].append(time.time() - start_time)
        
        # If we have valid results, calculate overall metrics
        if len(model_metrics['Fold']) > 0 and not all(np.isnan(model_metrics['MSE'])):
            model_metrics['Fold'].append('Overall')
            
            valid_indices = ~np.isnan(model_metrics['MSE'])[:n_splits]
            if np.any(valid_indices):
                # Calculate overall performance metrics as average across folds (excluding NaN values)
                model_metrics['MSE'].append(np.nanmean(model_metrics['MSE'][:n_splits]))
                model_metrics['RMSE'].append(np.nanmean(model_metrics['RMSE'][:n_splits]))
                model_metrics['R2'].append(np.nanmean(model_metrics['R2'][:n_splits]))
                
                # Add overall model fit metrics
                for metric in ['AIC', 'BIC', 'CFI', 'RMSEA']:
                    model_metrics[metric].append(np.nanmean(model_metrics[metric][:n_splits]))
                
                model_metrics['Converged'].append(np.mean(model_metrics['Converged'][:n_splits]))
                model_metrics['Time'].append(np.sum(model_metrics['Time'][:n_splits]))
        
        # Add to results
        for i in range(len(model_metrics['Fold'])):
            row = {'Model': model_type.value, 'Fold': model_metrics['Fold'][i]}
            for metric in metrics + ['Converged', 'Time']:
                if i < len(model_metrics[metric]):
                    row[metric] = model_metrics[metric][i]
                else:
                    row[metric] = np.nan
            results.append(row)
    
    # Convert to dataframe
    results_df = pd.DataFrame(results)
    
    # Create summary with average results per model
    fold_results = results_df[results_df['Fold'] != 'Overall']
    overall_results = results_df[results_df['Fold'] == 'Overall']
    
    # If we have valid results, create visualizations and save
    if len(fold_results) > 0:
        # Save all results
        results_df.to_csv(output_dir / "cv_results.csv", index=False)
        
        # Create summary dataframe if we have overall results
        if len(overall_results) > 0:
            summary_df = pd.DataFrame(index=overall_results['Model'])
            
            for metric in metrics:
                if metric in overall_results.columns:
                    summary_df[f'{metric}_mean'] = overall_results.set_index('Model')[metric]
            
            # Calculate standard deviations for each metric across folds
            for metric in ['MSE', 'RMSE', 'R2']:
                for model in summary_df.index:
                    model_folds = fold_results[(fold_results['Model'] == model)]
                    if len(model_folds) > 0:
                        summary_df.loc[model, f'{metric}_std'] = model_folds[metric].std(skipna=True)
                    else:
                        summary_df.loc[model, f'{metric}_std'] = np.nan
            
            # Save summary
            summary_df.to_csv(output_dir / "cv_summary.csv")
            
            # Create visualizations
            create_cv_visualizations(results_df, summary_df, output_dir)
    
    return results_df


def predict_with_sem_model(model, data, target_col):
    """
    Use a fitted SEM model to predict a target variable.
    
    Args:
        model: Fitted semopy model object or SEMModel instance
        data: DataFrame with predictors
        target_col: Target column to predict
        
    Returns:
        numpy.ndarray: Predicted values
    """
    try:
        # If the model is a SEMModel instance, get the underlying semopy model
        if hasattr(model, 'model'):
            sem_model = model.model
        else:
            sem_model = model
            
        # Get model estimates
        estimates = sem_model.inspect()
        
        # Extract the coefficients for predicting the target column
        target_coeffs = estimates[estimates['lval'] == target_col]
        
        # Create a copy of the data
        X = data.copy()
        
        # Prepare prediction
        y_pred = np.zeros(len(X))
        
        # Apply coefficients to make predictions
        for _, row in target_coeffs.iterrows():
            predictor = row['rval']
            coefficient = row['Estimate']
            
            # Skip if the predictor is not in the data
            if predictor not in X.columns:
                continue
                
            y_pred += X[predictor] * coefficient
            
        return y_pred
        
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise


def create_cv_visualizations(results_df, summary_df, output_dir):
    """
    Create visualizations for cross-validation results.
    
    Args:
        results_df: DataFrame with cross-validation results
        summary_df: DataFrame with summary statistics
        output_dir: Directory to save visualizations
    """
    # Filter out 'Overall' results for fold-specific visualizations
    fold_results = results_df[results_df['Fold'] != 'Overall'].copy()
    
    # Check if we have any valid fold results
    if len(fold_results) == 0:
        logger.warning("No valid fold results for visualization")
        return
    
    # Make sure all fold results have numeric values
    for col in ['MSE', 'RMSE', 'R2']:
        fold_results[col] = pd.to_numeric(fold_results[col], errors='coerce')
    
    # Check if we have sufficient data for boxplots
    valid_models = fold_results.dropna(subset=['MSE', 'RMSE', 'R2'])['Model'].unique()
    if len(valid_models) == 0:
        logger.warning("No valid data for boxplots")
        return

    # Only keep models with valid data
    fold_results = fold_results[fold_results['Model'].isin(valid_models)]
    
    # R² by model and fold
    plt.figure(figsize=(12, 8))
    if len(fold_results) > 0:
        sns.boxplot(x='Model', y='R2', data=fold_results)
        plt.title('R² by Model Type (Higher is Better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "cv_r2_boxplot.png")
        plt.close()
    
    # MSE by model and fold
    plt.figure(figsize=(12, 8))
    if len(fold_results) > 0:
        sns.boxplot(x='Model', y='MSE', data=fold_results)
        plt.title('MSE by Model Type (Lower is Better)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "cv_mse_boxplot.png")
        plt.close()
    
    # Summary plot
    plt.figure(figsize=(15, 10))
    
    # Create a bar plot for mean MSE with standard deviation error bars
    if len(summary_df) > 0:
        plt.subplot(2, 2, 1)
        models = summary_df.index
        x_pos = np.arange(len(models))
        plt.bar(x_pos, summary_df['MSE_mean'], yerr=summary_df['MSE_std'], 
                align='center', alpha=0.7)
        plt.xticks(x_pos, models, rotation=45)
        plt.title('Mean MSE with Standard Deviation')
        plt.ylabel('MSE (Lower is Better)')
        
        # Create a bar plot for mean R² with standard deviation error bars
        plt.subplot(2, 2, 2)
        plt.bar(x_pos, summary_df['R2_mean'], yerr=summary_df['R2_std'], 
                align='center', alpha=0.7)
        plt.xticks(x_pos, models, rotation=45)
        plt.title('Mean R² with Standard Deviation')
        plt.ylabel('R² (Higher is Better)')
        
        plt.tight_layout()
        plt.savefig(output_dir / "cv_summary_plot.png")
        plt.close()

    logger.info(f"Saved cross-validation visualizations to {output_dir}") 