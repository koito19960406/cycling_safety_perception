import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import semopy
from loguru import logger
import typer
from typing import Optional, List, Dict
import os
from sklearn.preprocessing import StandardScaler
import json

from cycling_safety_svi.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, FIGURES_DIR

app = typer.Typer()

def prepare_data():
    """
    Prepare the data for the structural equation model by merging the necessary datasets.
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
        'beautiful': 'beauty_rating'
    }, inplace=True)
    
    logger.info(f"Prepared dataset with {len(df_merged)} observations")
    
    return df_merged

def explore_data(df, output_dir=None):
    """
    Perform exploratory data analysis on the dataset.
    
    Args:
        df: DataFrame to explore
        output_dir: Directory to save plots (if None, plots will only be displayed)
    """
    logger.info("Performing exploratory data analysis...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Summary statistics
    summary = df.describe().T
    logger.info("Data summary statistics:")
    print(summary)
    
    # Correlation matrix for key variables
    pixel_cols = [col for col in df.columns if '_pixels' in col]
    rating_cols = ['traffic_safety_rating', 'social_safety_rating', 'beauty_rating']
    segment_cols = ['building', 'road', 'sky', 'trees', 'sidewalk', 'grass', 'water', 'plants']
    key_cols = pixel_cols + rating_cols + segment_cols + ['V_img']
    
    # Filter to only include columns that exist in the dataframe
    key_cols = [col for col in key_cols if col in df.columns]
    
    corr = df[key_cols].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                center=0, square=True, linewidths=.5)
    plt.title('Correlation Matrix of Key Variables')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    
    # Distribution of perception ratings
    if all(col in df.columns for col in rating_cols):
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(rating_cols):
            plt.subplot(1, 3, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'rating_distributions.png'), dpi=300)
        plt.close()
    
    # Relationship between perception ratings and V_img
    if all(col in df.columns for col in rating_cols):
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(rating_cols):
            plt.subplot(1, 3, i+1)
            sns.scatterplot(x=col, y='V_img', data=df)
            plt.title(f'{col.replace("_", " ").title()} vs. V_img')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'ratings_vs_vimg.png'), dpi=300)
        plt.close()
    
    # Analyze relationships between segmentation variables and ratings
    if all(col in df.columns for col in rating_cols):
        for rating in rating_cols:
            plt.figure(figsize=(15, 10))
            for i, segment in enumerate(segment_cols[:min(9, len(segment_cols))]):
                if segment in df.columns:
                    plt.subplot(3, 3, i+1)
                    sns.scatterplot(x=segment, y=rating, data=df)
                    plt.title(f'{segment} vs. {rating}')
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'segments_vs_{rating}.png'), dpi=300)
            plt.close()
    
    # Create pairplots for important relationships
    if len(segment_cols) > 3:
        key_segment_cols = ['building', 'road', 'trees', 'sidewalk']
        key_segment_cols = [col for col in key_segment_cols if col in df.columns]
        if len(key_segment_cols) > 0:
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[key_segment_cols + ['V_img']])
            plt.suptitle('Relationships between Key Segmentation Features and V_img', y=1.02)
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'segmentation_pairplot.png'), dpi=300)
            plt.close()
    
    logger.info("Exploratory data analysis completed")

def define_sem_models():
    """
    Define multiple structural equation model specifications to test different mediation hypotheses.
    """
    models = {}
    
    # Model 1: Full mediation model with latent constructs as mediators
    models['full_mediation'] = """
    # Defining key exogenous segmentation variables
    # Traffic safety related segmentation variables
    traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic sign_pixels
    
    # Social safety related segmentation variables
    social_seg =~ person_pixels + building + sidewalk
    
    # Beauty related segmentation variables
    beauty_seg =~ trees + grass + sky + water + plants
    
    # Measurement model for latent mediators
    traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
    social_safety =~ 1*social_safety_rating    # Anchor to observed rating
    beauty =~ 1*beauty_rating                  # Anchor to observed rating
    
    # Mediation paths: segmentation affects latent perceptions
    traffic_safety ~ traffic_seg
    social_safety ~ social_seg
    beauty ~ beauty_seg
    
    # Cross-paths: segmentation variables can affect multiple perceptions
    traffic_safety ~ social_seg + beauty_seg
    social_safety ~ traffic_seg + beauty_seg
    beauty ~ traffic_seg + social_seg
    
    # Outcome model: perceptions affect V_img (full mediation)
    V_img ~ traffic_safety + social_safety + beauty
    
    # Allow correlation between exogenous segmentation variables
    traffic_seg ~~ social_seg
    traffic_seg ~~ beauty_seg
    social_seg ~~ beauty_seg
    """
    
    # Model 2: Partial mediation model with both direct and indirect paths
    models['partial_mediation'] = """
    # Defining key exogenous segmentation variables
    # Traffic safety related segmentation variables
    traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic sign_pixels
    
    # Social safety related segmentation variables
    social_seg =~ person_pixels + building + sidewalk
    
    # Beauty related segmentation variables
    beauty_seg =~ trees + grass + sky + water + plants
    
    # Measurement model for latent mediators
    traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
    social_safety =~ 1*social_safety_rating    # Anchor to observed rating
    beauty =~ 1*beauty_rating                  # Anchor to observed rating
    
    # Mediation paths: segmentation affects latent perceptions
    traffic_safety ~ traffic_seg
    social_safety ~ social_seg
    beauty ~ beauty_seg
    
    # Cross-paths: segmentation variables can affect multiple perceptions
    traffic_safety ~ social_seg + beauty_seg
    social_safety ~ traffic_seg + beauty_seg
    beauty ~ traffic_seg + social_seg
    
    # Outcome model: perceptions affect V_img
    V_img ~ traffic_safety + social_safety + beauty
    
    # Direct effects: test if segmentation has direct effects beyond perceptions
    V_img ~ traffic_seg + social_seg + beauty_seg
    
    # Allow correlation between exogenous segmentation variables
    traffic_seg ~~ social_seg
    traffic_seg ~~ beauty_seg
    social_seg ~~ beauty_seg
    """
    
    # Model 3: Simplified mediation model with fewer cross-paths
    models['simplified_mediation'] = """
    # Defining key exogenous segmentation variables with simpler structure
    # Traffic safety related segmentation variables
    traffic_seg =~ road + car_pixels + truck_pixels
    
    # Social safety related segmentation variables
    social_seg =~ person_pixels + sidewalk
    
    # Beauty related segmentation variables
    beauty_seg =~ trees + grass + sky
    
    # Measurement model for latent mediators
    traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
    social_safety =~ 1*social_safety_rating    # Anchor to observed rating
    beauty =~ 1*beauty_rating                  # Anchor to observed rating
    
    # Mediation paths: segmentation affects corresponding latent perceptions
    traffic_safety ~ traffic_seg
    social_safety ~ social_seg
    beauty ~ beauty_seg
    
    # Outcome model: perceptions affect V_img
    V_img ~ traffic_safety + social_safety + beauty
    
    # Direct effects: test if segmentation has direct effects beyond perceptions
    V_img ~ traffic_seg + social_seg + beauty_seg
    
    # Allow correlation between exogenous segmentation variables
    traffic_seg ~~ social_seg
    traffic_seg ~~ beauty_seg
    social_seg ~~ beauty_seg
    """
    
    # Model 4: Direct effects only (no mediation)
    models['direct_only'] = """
    # Defining key exogenous segmentation variables
    # Traffic safety related segmentation variables
    traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic sign_pixels
    
    # Social safety related segmentation variables
    social_seg =~ person_pixels + building + sidewalk
    
    # Beauty related segmentation variables
    beauty_seg =~ trees + grass + sky + water + plants
    
    # Direct effects only
    V_img ~ traffic_seg + social_seg + beauty_seg
    
    # Allow correlation between exogenous segmentation variables
    traffic_seg ~~ social_seg
    traffic_seg ~~ beauty_seg
    social_seg ~~ beauty_seg
    """
    
    return models

def fit_and_evaluate_model(df, model_spec, model_name, output_dir=None):
    """
    Fit a SEM model and evaluate its performance.
    
    Args:
        df: DataFrame containing the data
        model_spec: Model specification in semopy syntax
        model_name: Name of the model for saving results
        output_dir: Directory to save results and plots
    
    Returns:
        Tuple containing (model, standardized estimates, fit indices)
    """
    logger.info(f"Fitting model: {model_name}")
    
    # Create output directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and fit the model
    model = semopy.Model(model_spec)
    
    try:
        result = model.fit(df)
        
        # Extract and display results
        logger.info(f"Model {model_name} estimation complete.")
        summary = model.inspect()
        
        # Extract standardized estimates
        std_estimates = model.standardized()
        
        # Calculate model fit indices
        fit_indices = semopy.calc_stats(model)
        logger.info(f"Model {model_name} fit indices:")
        for key, value in fit_indices.items():
            logger.info(f"{key}: {value}")
        
        # Save results if output directory is provided
        if output_dir:
            # Save standardized estimates
            std_estimates.to_csv(os.path.join(output_dir, f"{model_name}_std_estimates.csv"), index=True)
            
            # Save model summary
            with open(os.path.join(output_dir, f"{model_name}_summary.txt"), 'w') as f:
                f.write(str(summary))
            
            # Save fit indices
            with open(os.path.join(output_dir, f"{model_name}_fit_indices.json"), 'w') as f:
                json.dump(fit_indices, f, indent=4)
            
            # Calculate and save mediation effects if applicable
            if 'mediation' in model_name:
                effects_df = calculate_mediation_effects(model, std_estimates)
                if effects_df is not None:
                    effects_df.to_csv(os.path.join(output_dir, f"{model_name}_mediation_effects.csv"), index=False)
            
            # Generate and save path diagram
            try:
                fig = plt.figure(figsize=(14, 12))
                semopy.semplot(model, "std", fig=fig)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_path_diagram.png"), dpi=300)
            except Exception as e:
                logger.error(f"Error generating path diagram for model {model_name}: {e}")
        
        return model, std_estimates, fit_indices
    
    except Exception as e:
        logger.error(f"Error fitting model {model_name}: {e}")
        return None, None, None

def calculate_mediation_effects(model, std_estimates):
    """
    Calculate direct, indirect, and total effects for mediation models.
    
    Args:
        model: Fitted SEM model
        std_estimates: Standardized estimates from the model
    
    Returns:
        DataFrame containing mediation effects or None if not applicable
    """
    try:
        # Extract paths from standardized estimates
        paths = std_estimates.copy()
        
        # Check if this is a mediation model by looking for necessary paths
        mediators = ['traffic_safety', 'social_safety', 'beauty']
        predictors = ['traffic_seg', 'social_seg', 'beauty_seg']
        
        has_predictor_paths = any((paths['lval'] == pred).any() for pred in predictors)
        has_mediator_paths = any((paths['lval'] == med).any() for med in mediators)
        
        if not (has_predictor_paths and has_mediator_paths):
            logger.info("Not a mediation model or missing required paths")
            return None
        
        # Initialize effects dictionary
        effects = {
            'Path': [],
            'Direct_Effect': [],
            'Indirect_Effect': [],
            'Total_Effect': [],
            'Proportion_Mediated': []
        }
        
        # Calculate mediation effects for each segmentation variable group
        for pred in predictors:
            # Skip if predictor is not in the model
            if not (paths['lval'] == pred).any():
                continue
                
            # Direct effect on V_img (if exists)
            direct_effect = 0
            if (paths['lval'] == pred) & (paths['rval'] == 'V_img').any():
                direct_effect = paths.loc[(paths['lval'] == pred) & (paths['rval'] == 'V_img'), 'est'].values[0]
            
            # Calculate indirect effects through each mediator
            indirect_effects = []
            for med in mediators:
                # Skip if mediator is not in the model
                if not ((paths['lval'] == pred) & (paths['rval'] == med)).any() or not ((paths['lval'] == med) & (paths['rval'] == 'V_img')).any():
                    continue
                    
                # Effect of predictor on mediator
                a_path = paths.loc[(paths['lval'] == pred) & (paths['rval'] == med), 'est'].values[0]
                
                # Effect of mediator on V_img
                b_path = paths.loc[(paths['lval'] == med) & (paths['rval'] == 'V_img'), 'est'].values[0]
                
                # Calculate indirect effect
                indirect_effect = a_path * b_path
                indirect_effects.append(indirect_effect)
                
                # Add specific indirect path
                effects['Path'].append(f"{pred} → {med} → V_img")
                effects['Direct_Effect'].append(0)  # No direct effect for specific paths
                effects['Indirect_Effect'].append(indirect_effect)
                effects['Total_Effect'].append(indirect_effect)
                effects['Proportion_Mediated'].append(1.0)
            
            # Total indirect effect
            total_indirect = sum(indirect_effects) if indirect_effects else 0
            
            # Total effect
            total_effect = direct_effect + total_indirect
            
            # Proportion mediated
            prop_mediated = total_indirect / total_effect if total_effect != 0 else 0
            
            # Add overall effect
            effects['Path'].append(f"{pred} → V_img (Total)")
            effects['Direct_Effect'].append(direct_effect)
            effects['Indirect_Effect'].append(total_indirect)
            effects['Total_Effect'].append(total_effect)
            effects['Proportion_Mediated'].append(prop_mediated)
        
        # Convert to DataFrame
        effects_df = pd.DataFrame(effects)
        
        return effects_df
    
    except Exception as e:
        logger.error(f"Error calculating mediation effects: {e}")
        return None

def compare_models(model_results, output_path=None):
    """
    Compare multiple SEM models based on their fit indices.
    
    Args:
        model_results: Dictionary mapping model names to (model, std_estimates, fit_indices)
        output_path: Path to save comparison results
    """
    logger.info("Comparing models...")
    
    # Extract fit indices for comparison
    comparison = {}
    for model_name, (_, _, fit_indices) in model_results.items():
        if fit_indices is not None:
            # Select key fit indices for comparison
            key_indices = {
                'CFI': fit_indices.get('CFI', None),
                'TLI': fit_indices.get('TLI', None),
                'RMSEA': fit_indices.get('RMSEA', None),
                'chi2': fit_indices.get('chi2', None),
                'AIC': fit_indices.get('AIC', None),
                'BIC': fit_indices.get('BIC', None)
            }
            comparison[model_name] = key_indices
    
    # Convert to DataFrame for easier comparison
    comparison_df = pd.DataFrame(comparison).T if comparison else None
    
    if comparison_df is not None:
        logger.info("Model comparison results:")
        print(comparison_df)
        
        # Save comparison if output path is provided
        if output_path:
            comparison_df.to_csv(output_path)
            logger.info(f"Saved model comparison to {output_path}")
        
        # Identify best model based on fit indices
        best_model = None
        best_score = float('-inf')
        
        for model_name, indices in comparison.items():
            # Simple scoring: higher CFI and TLI are better, lower RMSEA, AIC, and BIC are better
            score = 0
            if indices['CFI'] is not None:
                score += indices['CFI']
            if indices['TLI'] is not None:
                score += indices['TLI']
            if indices['RMSEA'] is not None:
                score -= indices['RMSEA']
            if indices['AIC'] is not None:
                score -= indices['AIC'] / 1000  # Scaling to prevent domination
            if indices['BIC'] is not None:
                score -= indices['BIC'] / 1000  # Scaling to prevent domination
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model:
            logger.info(f"Best model based on fit indices: {best_model}")
        
        return comparison_df, best_model
    else:
        logger.error("No valid models to compare")
        return None, None

def compare_mediation_effects(model_results, output_dir=None):
    """
    Compare mediation effects across different models.
    
    Args:
        model_results: Dictionary mapping model names to (model, std_estimates, fit_indices)
        output_dir: Directory to save comparison visualizations
    """
    logger.info("Comparing mediation effects across models...")
    
    # Extract models with mediation effects
    mediation_models = {}
    
    for model_name, (model, std_estimates, _) in model_results.items():
        if model is not None and 'mediation' in model_name:
            effects_df = calculate_mediation_effects(model, std_estimates)
            if effects_df is not None:
                mediation_models[model_name] = effects_df
    
    if not mediation_models:
        logger.info("No mediation models available for comparison")
        return
    
    # Compare total effects across models
    total_effects = {}
    
    for model_name, effects_df in mediation_models.items():
        # Extract total effects for paths ending with "(Total)"
        total_paths = effects_df[effects_df['Path'].str.contains('(Total)')]
        
        for _, row in total_paths.iterrows():
            path = row['Path'].replace(' (Total)', '')
            if path not in total_effects:
                total_effects[path] = {}
            
            total_effects[path][model_name] = {
                'Direct': row['Direct_Effect'],
                'Indirect': row['Indirect_Effect'],
                'Total': row['Total_Effect'],
                'Proportion': row['Proportion_Mediated']
            }
    
    # Create visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize proportion mediated across models
        plt.figure(figsize=(12, 8))
        data = []
        labels = []
        models = []
        
        for path, model_effects in total_effects.items():
            for model_name, effects in model_effects.items():
                data.append(effects['Proportion'])
                labels.append(path)
                models.append(model_name)
        
        df_plot = pd.DataFrame({
            'Path': labels,
            'Model': models,
            'Proportion Mediated': data
        })
        
        sns.barplot(x='Path', y='Proportion Mediated', hue='Model', data=df_plot)
        plt.title('Proportion of Effect Mediated by Path and Model')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'proportion_mediated_comparison.png'), dpi=300)
        plt.close()
        
        # Visualize direct vs indirect effects
        for path in total_effects:
            plt.figure(figsize=(10, 6))
            models = []
            direct = []
            indirect = []
            
            for model_name, effects in total_effects[path].items():
                models.append(model_name)
                direct.append(effects['Direct'])
                indirect.append(effects['Indirect'])
            
            x = np.arange(len(models))
            width = 0.35
            
            plt.bar(x - width/2, direct, width, label='Direct Effect')
            plt.bar(x + width/2, indirect, width, label='Indirect Effect')
            
            plt.xlabel('Model')
            plt.ylabel('Effect Size')
            plt.title(f'Direct vs. Indirect Effects: {path}')
            plt.xticks(x, models, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'effects_comparison_{path.replace(" → ", "_to_")}.png'), dpi=300)
            plt.close()
    
    logger.info("Mediation comparison completed")

@app.command()
def main(
    explore: bool = True,
    output_dir: Optional[Path] = MODELS_DIR / "sem_models",
    figures_dir: Optional[Path] = FIGURES_DIR / "sem_analysis",
):
    """
    Build and evaluate multiple structural equation models that use latent constructs
    as mediators between segmentation results and the utility of the image (V_img).
    
    Args:
        explore: Whether to perform exploratory data analysis
        output_dir: Directory to save model results
        figures_dir: Directory to save figures from EDA
    """
    # Prepare data
    df = prepare_data()
    
    # Standardize numerical variables for better SEM performance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Perform exploratory data analysis if requested
    if explore:
        explore_data(df, figures_dir)
    
    # Define SEM models
    model_specs = define_sem_models()
    
    # Fit and evaluate each model
    model_results = {}
    for model_name, model_spec in model_specs.items():
        model_output_dir = output_dir / model_name if output_dir else None
        model_result = fit_and_evaluate_model(df, model_spec, model_name, model_output_dir)
        model_results[model_name] = model_result
    
    # Compare models
    comparison_df, best_model = compare_models(
        model_results, 
        output_path=output_dir / "model_comparison.csv" if output_dir else None
    )
    
    # Compare mediation effects
    compare_mediation_effects(model_results, output_dir=output_dir / "mediation_comparison" if output_dir else None)
    
    # Return the best model
    if best_model:
        best_model_result = model_results[best_model]
        return best_model_result[0], best_model_result[1]
    
    # If no best model, return the first successful model
    for model_name, (model, std_estimates, _) in model_results.items():
        if model is not None:
            return model, std_estimates
    
    logger.error("No models were successfully fit.")
    return None, None

if __name__ == "__main__":
    app() 