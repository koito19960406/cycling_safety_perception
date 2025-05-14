import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import semopy
from loguru import logger
import typer
from typing import Optional, List
from sklearn.preprocessing import StandardScaler
from enum import Enum

from cycling_safety_svi.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

class ModelType(str, Enum):
    """Model types for Structural Equation Modeling."""
    FULL = "full"                    # Full model with all paths and cross-paths
    SIMPLE = "simple"                # Simplified model with fewer cross-paths 
    MINIMAL = "minimal"              # Minimal model with essential paths only
    DIRECT_ONLY = "direct_only"      # Direct effects only (no mediation)
    MEDIATION_ONLY = "mediation_only"  # Mediation effects only (no direct effects)
    DIRECT_MEDIATED = "direct_mediated"  # All segmentation variables directly affect perceptions
    BENCHMARK = "benchmark"          # Simple benchmark model with direct effects only (no SEM structure)

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

def get_model_spec(model_type: ModelType = ModelType.FULL):
    """
    Generate a structural equation model specification with the specified complexity level.
    
    Args:
        model_type: Type of model to generate (full, simple, minimal, direct_only, mediation_only, or direct_mediated)
    
    Returns:
        str: Model specification in semopy syntax
    """
    if model_type == ModelType.FULL:
        # Full model with all paths and cross-paths
        return """
        # Defining key exogenous segmentation variables
        # Traffic safety related segmentation variables
        traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels
        
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
    
    elif model_type == ModelType.SIMPLE:
        # Simplified model with fewer cross-paths
        return """
        # Defining key exogenous segmentation variables
        # Traffic safety related segmentation variables
        traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels
        
        # Social safety related segmentation variables
        social_seg =~ person_pixels + building + sidewalk
        
        # Beauty related segmentation variables
        beauty_seg =~ trees + grass + sky + water + plants
        
        # Measurement model for latent mediators
        traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
        social_safety =~ 1*social_safety_rating    # Anchor to observed rating
        beauty =~ 1*beauty_rating                  # Anchor to observed rating
        
        # Mediation paths: segmentation affects only its corresponding latent perception
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
    
    elif model_type == ModelType.MINIMAL:
        # Minimal model with essential paths only
        return """
        # Defining key exogenous segmentation variables with fewer indicators
        # Traffic safety related segmentation variables
        traffic_seg =~ road + car_pixels  # Only essential indicators
        
        # Social safety related segmentation variables
        social_seg =~ person_pixels + building  # Only essential indicators
        
        # Beauty related segmentation variables
        beauty_seg =~ trees + grass  # Only essential indicators
        
        # Measurement model for latent mediators
        traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
        social_safety =~ 1*social_safety_rating    # Anchor to observed rating
        beauty =~ 1*beauty_rating                  # Anchor to observed rating
        
        # Mediation paths: segmentation affects only its corresponding latent perception
        traffic_safety ~ traffic_seg
        social_safety ~ social_seg
        beauty ~ beauty_seg
        
        # Outcome model: perceptions affect V_img
        V_img ~ traffic_safety + social_safety + beauty
        
        # Allow correlation between exogenous segmentation variables
        traffic_seg ~~ social_seg
        traffic_seg ~~ beauty_seg
        social_seg ~~ beauty_seg
        """
    
    elif model_type == ModelType.DIRECT_ONLY:
        # Direct effects only (no mediation)
        return """
        # Defining key exogenous segmentation variables
        # Traffic safety related segmentation variables
        traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels
        
        # Social safety related segmentation variables
        social_seg =~ person_pixels + building + sidewalk
        
        # Beauty related segmentation variables
        beauty_seg =~ trees + grass + sky + water + plants
        
        # Direct effects only: segmentation directly affects V_img
        V_img ~ traffic_seg + social_seg + beauty_seg
        
        # Allow correlation between exogenous segmentation variables
        traffic_seg ~~ social_seg
        traffic_seg ~~ beauty_seg
        social_seg ~~ beauty_seg
        """
    
    elif model_type == ModelType.MEDIATION_ONLY:
        # Mediation effects only (no direct effects)
        return """
        # Defining key exogenous segmentation variables
        # Traffic safety related segmentation variables
        traffic_seg =~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels
        
        # Social safety related segmentation variables
        social_seg =~ person_pixels + building + sidewalk
        
        # Beauty related segmentation variables
        beauty_seg =~ trees + grass + sky + water + plants
        
        # Measurement model for latent mediators
        traffic_safety =~ 1*traffic_safety_rating  # Anchor to observed rating
        social_safety =~ 1*social_safety_rating    # Anchor to observed rating
        beauty =~ 1*beauty_rating                  # Anchor to observed rating
        
        # Mediation paths: segmentation affects only its corresponding latent perception
        traffic_safety ~ traffic_seg
        social_safety ~ social_seg
        beauty ~ beauty_seg
        
        # Outcome model: perceptions affect V_img (no direct effects)
        V_img ~ traffic_safety + social_safety + beauty
        
        # Allow correlation between exogenous segmentation variables
        traffic_seg ~~ social_seg
        traffic_seg ~~ beauty_seg
        social_seg ~~ beauty_seg
        """
    
    elif model_type == ModelType.DIRECT_MEDIATED:
        # Direct mediated model - all segmentation variables directly affect perceptions,
        # which then mediate the effect on V_img
        # Fixed syntax: removed trailing '+' characters causing parsing errors
        return """
        # Direct effects of segmentation variables on traffic safety perception
        traffic_safety_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Direct effects of segmentation variables on social safety perception
        social_safety_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Direct effects of segmentation variables on beauty perception
        beauty_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Mediation: perception ratings affect V_img
        V_img ~ traffic_safety_rating + social_safety_rating + beauty_rating
        
        # Allow for direct effects of segmentation variables on V_img to test for partial mediation
        V_img ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Allow correlations between perception ratings
        traffic_safety_rating ~~ social_safety_rating
        traffic_safety_rating ~~ beauty_rating
        social_safety_rating ~~ beauty_rating
        """
    
    elif model_type == ModelType.BENCHMARK:
        # Benchmark model - simple direct effects model without SEM structure
        # All segmentation variables and perception ratings directly predict utility
        return """
        # Direct effects of all variables on V_img 
        # Dropping 'water' to avoid perfect multicollinearity
        V_img ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + 
                person_pixels + building + sidewalk + trees + grass + sky + plants + 
                traffic_safety_rating + social_safety_rating + beauty_rating
        
        # Allow correlations between perception ratings
        traffic_safety_rating ~~ social_safety_rating
        traffic_safety_rating ~~ beauty_rating
        social_safety_rating ~~ beauty_rating
        """
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def calculate_indirect_effects(model, std_estimates, model_type: ModelType = ModelType.FULL):
    """
    Calculate indirect effects for the mediation model.
    
    Args:
        model: Fitted SEM model
        std_estimates: Standardized estimates from the model
        model_type: Type of model to calculate effects for
    
    Returns:
        pd.DataFrame: DataFrame containing direct, indirect, and total effects
    """
    logger.info("Calculating mediation effects...")
    
    # Extract paths from standardized estimates
    paths = std_estimates.copy()
    
    # Initialize effects dictionary
    effects = {
        'Path': [],
        'Direct_Effect': [],
        'Indirect_Effect': [],
        'Total_Effect': [],
        'Proportion_Mediated': []
    }
    
    # Calculate effects based on model type
    if model_type == ModelType.DIRECT_MEDIATED:
        # For direct_mediated model, we analyze each segmentation variable separately
        seg_vars = ['road', 'car_pixels', 'bicycle_pixels', 'truck_pixels', 'traffic_sign_pixels',
                   'person_pixels', 'building', 'sidewalk', 'trees', 'grass', 'sky', 'water']
        
        mediators = ['traffic_safety_rating', 'social_safety_rating', 'beauty_rating']
        
        for seg_var in seg_vars:
            # Get direct effect on V_img (if present)
            direct_effect_rows = paths[(paths['lval'] == 'V_img') & (paths['rval'] == seg_var)]
            direct_effect = direct_effect_rows['Estimate'].values[0] if not direct_effect_rows.empty else 0
            
            # Calculate indirect effects through each mediator
            indirect_effects = []
            for mediator in mediators:
                # Effect of segmentation variable on mediator
                a_path_rows = paths[(paths['lval'] == mediator) & (paths['rval'] == seg_var)]
                a_path = a_path_rows['Estimate'].values[0] if not a_path_rows.empty else 0
                
                # Effect of mediator on V_img
                b_path_rows = paths[(paths['lval'] == 'V_img') & (paths['rval'] == mediator)]
                b_path = b_path_rows['Estimate'].values[0] if not b_path_rows.empty else 0
                
                # Calculate indirect effect through this mediator
                indirect_effect = a_path * b_path
                indirect_effects.append(indirect_effect)
                
                # Add specific indirect path
                effects['Path'].append(f"{seg_var} → {mediator} → V_img")
                effects['Direct_Effect'].append(0)
                effects['Indirect_Effect'].append(indirect_effect)
                effects['Total_Effect'].append(indirect_effect)
                effects['Proportion_Mediated'].append(1.0)
            
            # Total indirect effect
            total_indirect = sum(indirect_effects)
            
            # Total effect
            total_effect = direct_effect + total_indirect
            
            # Proportion mediated
            prop_mediated = total_indirect / total_effect if total_effect != 0 else 0
            
            # Add overall effect for this segmentation variable
            effects['Path'].append(f"{seg_var} → V_img (Total)")
            effects['Direct_Effect'].append(direct_effect)
            effects['Indirect_Effect'].append(total_indirect)
            effects['Total_Effect'].append(total_effect)
            effects['Proportion_Mediated'].append(prop_mediated)
    elif model_type == ModelType.BENCHMARK:
        # For benchmark model, there are only direct effects
        seg_vars = ['road', 'car_pixels', 'bicycle_pixels', 'truck_pixels', 'traffic_sign_pixels',
                   'person_pixels', 'building', 'sidewalk', 'trees', 'grass', 'sky', 'plants',
                   'traffic_safety_rating', 'social_safety_rating', 'beauty_rating']
        
        for var in seg_vars:
            # Get direct effect on V_img
            direct_effect_rows = paths[(paths['lval'] == 'V_img') & (paths['rval'] == var)]
            direct_effect = direct_effect_rows['Estimate'].values[0] if not direct_effect_rows.empty else 0
            
            # Add direct effect for this variable
            effects['Path'].append(f"{var} → V_img")
            effects['Direct_Effect'].append(direct_effect)
            effects['Indirect_Effect'].append(0)  # No indirect effects
            effects['Total_Effect'].append(direct_effect)
            effects['Proportion_Mediated'].append(0.0)  # 0% mediated
    else:
        # For standard latent variable models
        for seg in ['traffic_seg', 'social_seg', 'beauty_seg']:
            # Direct effect on V_img
            direct_effect_rows = paths[(paths['lval'] == 'V_img') & (paths['rval'] == seg)]
            direct_effect = direct_effect_rows['Estimate'].values[0] if not direct_effect_rows.empty else 0
            
            # Calculate indirect effects through each mediator
            indirect_effects = []
            for mediator in ['traffic_safety', 'social_safety', 'beauty']:
                # Effect of segmentation on mediator
                a_path_rows = paths[(paths['lval'] == mediator) & (paths['rval'] == seg)]
                a_path = a_path_rows['Estimate'].values[0] if not a_path_rows.empty else 0
                
                # Effect of mediator on V_img
                b_path_rows = paths[(paths['lval'] == 'V_img') & (paths['rval'] == mediator)]
                b_path = b_path_rows['Estimate'].values[0] if not b_path_rows.empty else 0
                
                # Calculate indirect effect through this mediator
                indirect_effect = a_path * b_path
                indirect_effects.append(indirect_effect)
                
                # Add specific indirect path
                effects['Path'].append(f"{seg} → {mediator} → V_img")
                effects['Direct_Effect'].append(0)  # No direct effect for specific indirect paths
                effects['Indirect_Effect'].append(indirect_effect)
                effects['Total_Effect'].append(indirect_effect)
                effects['Proportion_Mediated'].append(1.0)  # 100% mediated
            
            # Total indirect effect
            total_indirect = sum(indirect_effects)
            
            # Total effect
            total_effect = direct_effect + total_indirect
            
            # Proportion mediated
            prop_mediated = total_indirect / total_effect if total_effect != 0 else 0
            
            # Add overall effect for this segmentation variable
            effects['Path'].append(f"{seg} → V_img (Total)")
            effects['Direct_Effect'].append(direct_effect)
            effects['Indirect_Effect'].append(total_indirect)
            effects['Total_Effect'].append(total_effect)
            effects['Proportion_Mediated'].append(prop_mediated)
    
    # Convert to DataFrame
    effects_df = pd.DataFrame(effects)
    
    return effects_df

def fit_model(df, model_type: ModelType, output_path: Path, plot_path: Path, effects_path: Path):
    """
    Fit a structural equation model with the specified complexity level.
    
    Args:
        df: Prepared data frame
        model_type: Type of model to fit
        output_path: Path to save model results
        plot_path: Path to save plot
        effects_path: Path to save mediation effects
        
    Returns:
        tuple: (model, std_estimates, effects_df) or (None, None, None) if error
    """
    # Define SEM model based on specified type
    model_spec = get_model_spec(model_type)
    
    # Create and fit the model
    logger.info(f"Fitting {model_type.value} SEM model...")
    try:
        model = semopy.Model(model_spec)
        result = model.fit(df)
        
        # Extract and display results
        logger.info("Model estimation complete. Summary of results:")
        summary = model.inspect()
        print(summary)
        
        # Get standardized estimates
        std_estimates = model.inspect(std_est=True)
        
        # Create model-specific file paths
        model_output_path = output_path.parent / f"{model_type.value}_{output_path.name}"
        model_plot_path = plot_path.parent / f"{model_type.value}_{plot_path.name}"
        model_effects_path = effects_path.parent / f"{model_type.value}_{effects_path.name}"
        
        # Save results to file
        std_estimates.to_csv(model_output_path, index=True)
        logger.info(f"Saved standardized estimates to {model_output_path}")
        
        # Calculate and save mediation effects (if model has mediation)
        if model_type not in [ModelType.DIRECT_ONLY, ModelType.BENCHMARK]:
            effects_df = calculate_indirect_effects(model, std_estimates, model_type)
            effects_df.to_csv(model_effects_path, index=False)
            logger.info(f"Saved mediation effects to {model_effects_path}")
        elif model_type == ModelType.BENCHMARK:
            # For benchmark model, calculate direct effects only
            effects_df = calculate_indirect_effects(model, std_estimates, model_type)
            effects_df.to_csv(model_effects_path, index=False)
            logger.info(f"Saved direct effects to {model_effects_path}")
        else:
            effects_df = None
            logger.info("Skipping mediation effects calculation for direct-only model")
        
        # Create and save path diagram
        logger.info("Generating path diagram...")
        try:
            model_plot_path = model_plot_path.with_suffix('.png')  # Ensure proper file extension
            semopy.semplot(model, str(model_plot_path), plot_covs=True, std_ests=True)
            logger.info(f"Saved path diagram to {model_plot_path}")
        except Exception as e:
            logger.error(f"Error generating path diagram: {e}")
        
        # Calculate model fit indices
        fit_indices = semopy.calc_stats(model)
        logger.info(f"Model fit indices for {model_type.value} model:")
        for key, value in fit_indices.items():
            logger.info(f"{key}: {value}")
            
        return model, std_estimates, effects_df
        
    except Exception as e:
        logger.error(f"Error fitting {model_type.value} model: {e}")
        return None, None, None

@app.command()
def main(
    model_types: Optional[List[ModelType]] = None,
    output_path: Optional[Path] = MODELS_DIR / "sem_results.csv",
    plot_path: Optional[Path] = MODELS_DIR / "sem_plot.png",
    effects_path: Optional[Path] = MODELS_DIR / "mediation_effects.csv"
):
    """
    Build and estimate structural equation models that use latent constructs
    as mediators between segmentation results and the utility of the image (V_img).
    
    Args:
        model_types: List of model types to fit. If None, fit all models.
        output_path: Base path for saving model results
        plot_path: Base path for saving plots
        effects_path: Base path for saving mediation effects
    """
    # Prepare data
    df = prepare_data()
    
    # Standardize numerical variables for better SEM performance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Default to all model types if none specified
    if model_types is None:
        model_types = list(ModelType)
    elif isinstance(model_types, str):
        model_types = [ModelType(model_types)]
    
    results = {}
    
    # Fit each specified model type
    for model_type in model_types:
        logger.info(f"============= FITTING {model_type.value.upper()} MODEL =============")
        model_result = fit_model(df, model_type, output_path, plot_path, effects_path)
        results[model_type] = model_result
    
    # Compare model fit if multiple models were fit
    if len(model_types) > 1:
        logger.info("============= MODEL COMPARISON =============")
        fit_metrics = ['chi2', 'DoF', 'CFI', 'RMSEA', 'AIC', 'BIC']
        comparison = {metric: [] for metric in fit_metrics}
        comparison['Model'] = []
        
        for model_type in model_types:
            model, _, _ = results[model_type]
            if model is not None:
                comparison['Model'].append(model_type.value)
                fit_indices = semopy.calc_stats(model)
                for metric in fit_metrics:
                    if metric in fit_indices:
                        comparison[metric].append(fit_indices[metric])
                    else:
                        comparison[metric].append(None)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison)
        comparison_path = MODELS_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Model comparison saved to {comparison_path}")
        print("\nModel Comparison:")
        print(comparison_df)
    
    return results

if __name__ == "__main__":
    app() 