import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import semopy
from loguru import logger
import typer
from typing import Optional
from sklearn.preprocessing import StandardScaler

from cycling_safety_svi.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

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

def define_sem_model():
    """
    Define the structural equation model specification using a mediation framework.
    """
    # Model specification in semopy syntax with mediation structure
    model_spec = """
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
    
    return model_spec

def calculate_indirect_effects(model, std_estimates):
    """
    Calculate indirect effects for the mediation model.
    
    Args:
        model: Fitted SEM model
        std_estimates: Standardized estimates from the model
    
    Returns:
        DataFrame containing direct, indirect, and total effects
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
    
    # Calculate mediation effects for each segmentation variable group
    for seg in ['traffic_seg', 'social_seg', 'beauty_seg']:
        # Direct effect on V_img
        direct_effect = paths.loc[(paths['lval'] == seg) & (paths['rval'] == 'V_img'), 'est'].values[0] \
            if (paths['lval'] == seg) & (paths['rval'] == 'V_img').any() else 0
        
        # Calculate indirect effects through each mediator
        indirect_effects = []
        for mediator in ['traffic_safety', 'social_safety', 'beauty']:
            # Effect of segmentation on mediator
            a_path = paths.loc[(paths['lval'] == seg) & (paths['rval'] == mediator), 'est'].values[0] \
                if (paths['lval'] == seg) & (paths['rval'] == mediator).any() else 0
            
            # Effect of mediator on V_img
            b_path = paths.loc[(paths['lval'] == mediator) & (paths['rval'] == 'V_img'), 'est'].values[0] \
                if (paths['lval'] == mediator) & (paths['rval'] == 'V_img').any() else 0
            
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

@app.command()
def main(
    output_path: Optional[Path] = MODELS_DIR / "sem_results.csv",
    plot_path: Optional[Path] = MODELS_DIR / "sem_plot.png",
    effects_path: Optional[Path] = MODELS_DIR / "mediation_effects.csv"
):
    """
    Build and estimate a structural equation model (SEM) that uses latent constructs
    as mediators between segmentation results and the utility of the image (V_img).
    """
    # Prepare data
    df = prepare_data()
    
    # Standardize numerical variables for better SEM performance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Define SEM model
    model_spec = define_sem_model()
    
    # Create and fit the model
    logger.info("Fitting SEM mediation model...")
    model = semopy.Model(model_spec)
    try:
        result = model.fit(df)
        
        # Extract and display results
        logger.info("Model estimation complete. Summary of results:")
        summary = model.inspect()
        print(summary)
        
        # Extract standardized estimates
        std_estimates = model.standardized()
        
        # Save results to file
        std_estimates.to_csv(output_path, index=True)
        logger.info(f"Saved standardized estimates to {output_path}")
        
        # Calculate and save mediation effects
        effects_df = calculate_indirect_effects(model, std_estimates)
        effects_df.to_csv(effects_path, index=False)
        logger.info(f"Saved mediation effects to {effects_path}")
        
        # Create and save path diagram
        logger.info("Generating path diagram...")
        try:
            fig = plt.figure(figsize=(12, 10))
            semopy.semplot(model, "std", fig=fig)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            logger.info(f"Saved path diagram to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating path diagram: {e}")
        
        # Calculate model fit indices
        fit_indices = semopy.calc_stats(model)
        logger.info("Model fit indices:")
        for key, value in fit_indices.items():
            logger.info(f"{key}: {value}")
        
        return model, std_estimates, effects_df
    
    except Exception as e:
        logger.error(f"Error fitting model: {e}")
        return None, None, None

if __name__ == "__main__":
    app() 