"""
Main script for running Structural Equation Models for cycling safety analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
import typer
from typing import Optional, List
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns

from cycling_safety_svi.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from cycling_safety_svi.modeling.sem_classes import SEMModel, ModelType, SEMModelRegistry
from cycling_safety_svi.modeling.sem_utils import prepare_data, compare_models, stepwise_model_selection, cross_validate_models
# Import model implementations to register them with the registry
from cycling_safety_svi.modeling.sem_models import *

app = typer.Typer()

# Define a simple main function that can be called directly without Typer
def main_function(
    run_all: bool = True,
    specific_model: Optional[str] = None,
    output_dir: Path = Path("reports/models/sem"),
    run_cross_validation: bool = True,
    cv_folds: int = 5
):
    """
    Run SEM analysis without using the CLI interface.
    
    Args:
        run_all: Whether to run all models
        specific_model: Specific model to run (if run_all is False)
        output_dir: Output directory for results
        run_cross_validation: Whether to run cross-validation
        cv_folds: Number of folds for cross-validation
    """
    logger.info("Starting SEM analysis...")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_data()
    logger.info(f"Prepared data with {len(data)} observations")
    
    # Store fitted models
    fitted_models = {}
    
    if run_all:
        # Run all models
        for model_type in ModelType:
            logger.info(f"Running {model_type.value} model...")
            model_dir = output_dir / model_type.value
            os.makedirs(model_dir, exist_ok=True)
            
            # Create and fit model
            model = SEMModelRegistry.get_model(model_type, output_dir=model_dir)
            model.fit(data)
            model.save_results()
            
            fitted_models[model_type] = model
            
        # Compare models
        comparison = compare_models(fitted_models, output_dir=output_dir)
        print("\nModel Comparison:")
        print(comparison)
        
        # Run cross-validation if requested
        if run_cross_validation:
            logger.info("Running cross-validation...")
            cv_output_dir = output_dir / "cross_validation"
            os.makedirs(cv_output_dir, exist_ok=True)
            
            # Create dictionary mapping model types to model classes
            model_classes = {model_type: SEMModelRegistry._models[model_type] 
                            for model_type in ModelType 
                            if model_type in SEMModelRegistry._models}
            
            # Run cross-validation
            cv_results = cross_validate_models(
                models_to_test=model_classes,
                data=data,
                n_splits=cv_folds,
                output_dir=cv_output_dir
            )
            
            # Print a simple summary
            summary = cv_results[cv_results['Fold'] == 'Overall'].set_index('Model')[['R2', 'MSE', 'RMSE']]
            print("\nCross-Validation Summary (Overall Performance):")
            print(summary)
    else:
        # Run specific model
        try:
            model_type = ModelType(specific_model.lower())
            model_dir = output_dir / model_type.value
            os.makedirs(model_dir, exist_ok=True)
            
            model = SEMModelRegistry.get_model(model_type, output_dir=model_dir)
            model.fit(data)
            model.save_results()
            
            logger.info(f"Successfully ran {model_type.value} model")
            
            # Run cross-validation for this model if requested
            if run_cross_validation:
                logger.info(f"Running cross-validation for {model_type.value} model...")
                cv_output_dir = model_dir / "cross_validation"
                os.makedirs(cv_output_dir, exist_ok=True)
                
                # Create dictionary with just this model
                model_classes = {model_type: SEMModelRegistry._models[model_type]}
                
                # Run cross-validation
                cv_results = cross_validate_models(
                    models_to_test=model_classes,
                    data=data,
                    n_splits=cv_folds,
                    output_dir=cv_output_dir
                )
                
                # Print a simple summary
                summary = cv_results[cv_results['Fold'] == 'Overall'].set_index('Model')[['R2', 'MSE', 'RMSE']]
                print("\nCross-Validation Summary:")
                print(summary)
        except Exception as e:
            logger.error(f"Error running model: {e}")
    
    return fitted_models


@app.command()
def run_model(
    model_type: str = typer.Argument(..., help="Type of model to run (full, simple, minimal, direct_only, mediation_only, direct_mediated)"),
    output_dir: Optional[Path] = None,
):
    """
    Run a single SEM model.
    
    Args:
        model_type: Type of model to run
        output_dir: Directory for saving outputs
    """
    # Convert string to ModelType
    try:
        model_type_enum = ModelType(model_type.lower())
    except ValueError:
        logger.error(f"Invalid model type: {model_type}")
        logger.info(f"Available model types: {[m.value for m in ModelType]}")
        return
    
    # Prepare data
    data = prepare_data()
    
    # Create model
    model = SEMModelRegistry.get_model(model_type_enum, output_dir=output_dir)
    
    # Fit model
    model.fit(data)
    
    # Save results
    model.save_results()
    
    # Return model for possible further use
    return model


@app.command()
def run_all_models(
    output_base_dir: Optional[Path] = Path("reports/models/sem"),
    include_models: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
):
    """
    Run all SEM models and compare them.
    
    Args:
        output_base_dir: Base directory for saving model outputs
        include_models: List of model types to include (None for all)
        exclude_models: List of model types to exclude
    """
    # Determine which models to run
    if include_models is None:
        models_to_run = list(ModelType)
    else:
        models_to_run = [ModelType(m.lower()) for m in include_models if m.lower() in [mt.value for mt in ModelType]]
    
    if exclude_models:
        exclude_enums = [ModelType(m.lower()) for m in exclude_models if m.lower() in [mt.value for mt in ModelType]]
        models_to_run = [m for m in models_to_run if m not in exclude_enums]
    
    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_data()
    
    # Track fitted models
    fitted_models = {}
    
    # Fit each model
    for model_type in models_to_run:
        logger.info(f"============= FITTING {model_type.value.upper()} MODEL =============")
        
        # Create output directory for this model
        model_dir = output_base_dir / model_type.value
        os.makedirs(model_dir, exist_ok=True)
        
        # Create and fit model
        model = SEMModelRegistry.get_model(model_type, output_dir=model_dir)
        model.fit(data)
        
        # Save results
        model.save_results()
        
        # Add to dictionary
        fitted_models[model_type] = model
    
    # Compare models
    comparison = compare_models(fitted_models, output_dir=output_base_dir)
    print("\nModel Comparison:")
    print(comparison)
    
    return fitted_models


@app.command()
def stepwise_selection(
    base_model_type: str = typer.Argument("minimal", help="Base model type (full, simple, minimal, direct_only, mediation_only, direct_mediated)"),
    target_mediator: str = "traffic_safety",
    criteria: str = "AIC",
    output_dir: Optional[Path] = Path("reports/models/sem/stepwise"),
):
    """
    Perform stepwise variable selection to find optimal SEM model.
    
    Args:
        base_model_type: Type of model to use as a base
        target_mediator: Name of mediator to optimize ("traffic_safety", "social_safety", or "beauty")
        criteria: Selection criteria ("AIC", "BIC", or "CFI")
        output_dir: Directory for saving results
    """
    # Convert string to ModelType
    try:
        base_model_type_enum = ModelType(base_model_type.lower())
    except ValueError:
        logger.error(f"Invalid model type: {base_model_type}")
        logger.info(f"Available model types: {[m.value for m in ModelType]}")
        return
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_data()
    
    # Create base model
    base_model = SEMModelRegistry.get_model(base_model_type_enum, output_dir=output_dir)
    
    # Define candidate variables for each mediator
    if target_mediator == "traffic_safety":
        variables = ['road', 'car_pixels', 'bicycle_pixels', 'truck_pixels', 'traffic_sign_pixels']
    elif target_mediator == "social_safety":
        variables = ['person_pixels', 'building', 'sidewalk']
    elif target_mediator == "beauty":
        variables = ['trees', 'grass', 'sky', 'water', 'plants']
    else:
        raise ValueError(f"Unknown mediator: {target_mediator}")
    
    # Run stepwise selection
    selected_vars, results = stepwise_model_selection(
        base_model, data, variables, target_mediator, criteria
    )
    
    # Print and save results
    logger.info(f"Selected variables for {target_mediator}: {selected_vars}")
    
    # Convert results to DataFrame for saving
    results_df = pd.DataFrame({
        'Step': results['step'],
        'Variables': [", ".join(vars) for vars in results['variables']],
        'Added': results['added_variable'],
        'Removed': results['removed_variable'],
        criteria: results[criteria]
    })
    
    # Save results to CSV
    results_path = output_dir / f"stepwise_{target_mediator}_{criteria}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Stepwise selection results saved to {results_path}")
    
    # Plot selection criteria over steps
    plt.figure(figsize=(10, 6))
    steps = results['step']
    scores = results[criteria]
    
    plt.plot(steps, scores, marker='o')
    plt.xlabel('Selection Step')
    plt.ylabel(criteria)
    plt.title(f'Stepwise Selection for {target_mediator} using {criteria}')
    
    for i, (step, score, added, removed) in enumerate(zip(
        results['step'], results[criteria], results['added_variable'], results['removed_variable']
    )):
        if added is not None:
            plt.annotate(f"+{added}", (step, score), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        if removed is not None:
            plt.annotate(f"-{removed}", (step, score), textcoords="offset points", 
                        xytext=(0,-15), ha='center')
    
    plt.grid(True)
    plot_path = output_dir / f"stepwise_{target_mediator}_{criteria}.png"
    plt.savefig(plot_path)
    logger.info(f"Stepwise selection plot saved to {plot_path}")
    
    return selected_vars, results


@app.command()
def organize_results(
    source_dir: Path = Path("models"),
    target_dir: Path = Path("reports/models/sem"),
    max_file_size_mb: float = 50.0
):
    """
    Organize SEM model results into a clean structure for reporting.
    
    Args:
        source_dir: Source directory containing model results
        target_dir: Target directory for organized results
        max_file_size_mb: Maximum file size to copy (files larger than this will be skipped)
    """
    logger.info(f"Organizing model results from {source_dir} to {target_dir}")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Define model types for organizing
    model_types = [m.value for m in ModelType]
    
    # Create directories for each model type
    for model_type in model_types:
        os.makedirs(target_dir / model_type, exist_ok=True)
    
    # Track moved files
    moved_files = {model_type: [] for model_type in model_types}
    skipped_files = []
    
    # Process all files in the source directory
    for file_path in source_dir.glob('*'):
        if file_path.is_file():
            # Skip directories and hidden files
            if file_path.name.startswith('.'):
                continue
                
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_file_size_mb:
                logger.warning(f"Skipping large file: {file_path.name} ({file_size_mb:.2f} MB)")
                skipped_files.append((file_path.name, file_size_mb))
                continue
            
            # Determine which model type this file belongs to
            target_model = None
            for model_type in model_types:
                if model_type in file_path.name:
                    target_model = model_type
                    break
            
            if target_model:
                # Copy to appropriate folder
                dest_path = target_dir / target_model / file_path.name
                shutil.copy2(file_path, dest_path)
                moved_files[target_model].append(file_path.name)
                logger.info(f"Copied {file_path.name} to {target_model}/")
            else:
                # Copy to root folder if not associated with a specific model
                dest_path = target_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied {file_path.name} to root output directory")
    
    # Summarize results
    logger.info("Finished organizing model results")
    for model_type, files in moved_files.items():
        if files:
            logger.info(f"Moved {len(files)} files to {model_type}/")
    
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} large files")
        for name, size in skipped_files:
            logger.warning(f"  - {name} ({size:.2f} MB)")
    
    return {
        "moved_files": moved_files,
        "skipped_files": skipped_files
    }


@app.command()
def clean_model_comparison(
    input_file: Path = Path("models/model_comparison.csv"),
    output_file: Path = Path("reports/models/sem/model_comparison.csv")
):
    """
    Clean the model comparison output file for better readability.
    
    Args:
        input_file: Input CSV file with model comparison
        output_file: Output CSV file for cleaned comparison
    """
    logger.info(f"Cleaning model comparison from {input_file}")
    
    # Ensure output directory exists
    os.makedirs(output_file.parent, exist_ok=True)
    
    try:
        # Read the input file
        df = pd.read_csv(input_file)
        
        # Clean up the format
        for col in df.columns:
            if col == 'Model' or col == 'Description':
                continue
                
            # Extract just the numeric value if in format "Value X.XXX"
            if df[col].dtype == 'object':
                # Process each row individually
                cleaned_values = []
                for val in df[col]:
                    if isinstance(val, str):
                        import re
                        match = re.search(r'Value\s+([\d\.\-e]+)', val)
                        if match:
                            try:
                                cleaned_values.append(float(match.group(1)))
                            except ValueError:
                                cleaned_values.append(None)
                        else:
                            cleaned_values.append(None)
                    else:
                        cleaned_values.append(val)
                df[col] = cleaned_values
            
            # Convert to appropriate data type
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add descriptions if not present
        if 'Description' not in df.columns:
            model_descriptions = {
                "full": "Full model with all paths and cross-paths",
                "simple": "Simplified model with fewer cross-paths",
                "minimal": "Minimal model with essential paths only",
                "direct_only": "Direct effects only (no mediation)",
                "mediation_only": "Mediation effects only (no direct effects)",
                "direct_mediated": "All segmentation variables directly affect perceptions"
            }
            
            df['Description'] = df['Model'].map(model_descriptions)
        
        # Reorder columns
        cols = ['Model', 'Description'] + [c for c in df.columns if c not in ['Model', 'Description']]
        df = df[cols]
        
        # Save cleaned file
        df.to_csv(output_file, index=False)
        logger.info(f"Saved cleaned model comparison to {output_file}")
        
        # Check if there's valid data for visualization
        has_valid_data = True
        for col in ['AIC', 'BIC', 'CFI', 'RMSEA']:
            if col not in df.columns or df[col].isna().all():
                logger.warning(f"No valid data for {col}, skipping visualization")
                has_valid_data = False
                break
        
        if has_valid_data:
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Plot AIC and BIC
            plt.subplot(2, 2, 1)
            sns.barplot(x='Model', y='AIC', data=df)
            plt.title('AIC (lower is better)')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            sns.barplot(x='Model', y='BIC', data=df)
            plt.title('BIC (lower is better)')
            plt.xticks(rotation=45)
            
            # Plot CFI and RMSEA
            plt.subplot(2, 2, 3)
            sns.barplot(x='Model', y='CFI', data=df)
            plt.title('CFI (higher is better)')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            sns.barplot(x='Model', y='RMSEA', data=df)
            plt.title('RMSEA (lower is better)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plot_path = output_file.parent / "model_comparison.png"
            plt.savefig(plot_path)
            logger.info(f"Model comparison plot saved to {plot_path}")
        else:
            logger.warning("Skipping visualization due to invalid data")
            
        return df
    
    except Exception as e:
        logger.error(f"Error cleaning model comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


@app.command()
def cross_validate(
    output_dir: Optional[Path] = Path("reports/models/sem/cross_validation"),
    n_splits: int = 5,
    include_models: Optional[List[str]] = None,
    exclude_models: Optional[List[str]] = None,
    random_state: int = 42
):
    """
    Run cross-validation for SEM models to evaluate predictive performance.
    
    Args:
        output_dir: Directory for saving cross-validation results
        n_splits: Number of folds for cross-validation
        include_models: List of model types to include (None for all)
        exclude_models: List of model types to exclude
        random_state: Random seed for reproducibility
    """
    logger.info(f"Starting {n_splits}-fold cross-validation for SEM models")
    
    # Determine which models to run
    if include_models is None:
        models_to_test = list(ModelType)
    else:
        models_to_test = [ModelType(m.lower()) for m in include_models 
                          if m.lower() in [mt.value for mt in ModelType]]
    
    if exclude_models:
        exclude_enums = [ModelType(m.lower()) for m in exclude_models 
                         if m.lower() in [mt.value for mt in ModelType]]
        models_to_test = [m for m in models_to_test if m not in exclude_enums]
    
    # Create dictionary mapping model types to model classes
    model_classes = {model_type: SEMModelRegistry._models[model_type] 
                     for model_type in models_to_test 
                     if model_type in SEMModelRegistry._models}
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    data = prepare_data()
    logger.info(f"Prepared data with {len(data)} observations for cross-validation")
    
    # Run cross-validation
    cv_results = cross_validate_models(
        models_to_test=model_classes,
        data=data,
        n_splits=n_splits,
        output_dir=output_dir,
        random_state=random_state
    )
    
    logger.info("Cross-validation completed")
    
    # Print a simple summary
    summary = cv_results[cv_results['Fold'] == 'Overall'].set_index('Model')[['R2', 'MSE', 'RMSE']]
    print("\nCross-Validation Summary (Overall Performance):")
    print(summary)
    
    return cv_results


if __name__ == "__main__":
    # Use the direct function instead of Typer app
    main_function(
        run_all=True,
        run_cross_validation=True,
        cv_folds=5,
        output_dir=Path("reports/models/sem")
    ) 