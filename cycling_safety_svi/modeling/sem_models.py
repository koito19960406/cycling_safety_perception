"""
Standard Structural Equation Models implementation for cycling safety analysis.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Optional

from cycling_safety_svi.modeling.sem_classes import SEMModel, ModelType, SEMModelRegistry


@SEMModelRegistry.register(ModelType.FULL)
class FullSEMModel(SEMModel):
    """Full SEM model with all paths and cross-paths."""
    
    def get_model_spec(self) -> str:
        """Get the full model specification."""
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
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate mediation effects for the full model."""
        logger.info("Calculating mediation effects...")
        
        # Extract paths from standardized estimates
        paths = self.std_estimates.copy()
        
        # Initialize effects dictionary
        effects = {
            'Path': [],
            'Direct_Effect': [],
            'Indirect_Effect': [],
            'Total_Effect': [],
            'Proportion_Mediated': []
        }
        
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


@SEMModelRegistry.register(ModelType.SIMPLE)
class SimpleSEMModel(SEMModel):
    """Simple SEM model with fewer cross-paths."""
    
    def get_model_spec(self) -> str:
        """Get the simple model specification."""
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
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate mediation effects for the simple model."""
        # Uses the same calculation method as the full model
        return FullSEMModel.calculate_indirect_effects(self)


@SEMModelRegistry.register(ModelType.MINIMAL)
class MinimalSEMModel(SEMModel):
    """Minimal SEM model with essential paths only."""
    
    def get_model_spec(self) -> str:
        """Get the minimal model specification."""
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
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate mediation effects for the minimal model."""
        # Uses the same calculation method as the full model
        return FullSEMModel.calculate_indirect_effects(self)


@SEMModelRegistry.register(ModelType.DIRECT_ONLY)
class DirectOnlySEMModel(SEMModel):
    """Direct-only SEM model with no mediation."""
    
    def get_model_spec(self) -> str:
        """Get the direct-only model specification."""
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


@SEMModelRegistry.register(ModelType.MEDIATION_ONLY)
class MediationOnlySEMModel(SEMModel):
    """Mediation-only SEM model with no direct effects."""
    
    def get_model_spec(self) -> str:
        """Get the mediation-only model specification."""
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
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate mediation effects for the mediation-only model."""
        # Uses the same calculation method as the full model
        return FullSEMModel.calculate_indirect_effects(self)


@SEMModelRegistry.register(ModelType.DIRECT_MEDIATED)
class DirectMediatedSEMModel(SEMModel):
    """Direct-mediated SEM model with direct effects of segmentation on perceptions."""
    
    def get_model_spec(self) -> str:
        """Get the direct-mediated model specification."""
        return """
        # Direct effects of segmentation variables on traffic safety perception
        traffic_safety_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Direct effects of segmentation variables on social safety perception
        social_safety_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Direct effects of segmentation variables on beauty perception
        beauty_rating ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + water + plants
        
        # Mediation: perception ratings affect V_img
        V_img ~ traffic_safety_rating + social_safety_rating + beauty_rating
        
        # Allow correlations between perception ratings
        traffic_safety_rating ~~ social_safety_rating
        traffic_safety_rating ~~ beauty_rating
        social_safety_rating ~~ beauty_rating
        """
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate mediation effects for the direct-mediated model."""
        logger.info("Calculating mediation effects...")
        
        # Extract paths from standardized estimates
        paths = self.std_estimates.copy()
        
        # Initialize effects dictionary
        effects = {
            'Path': [],
            'Direct_Effect': [],
            'Indirect_Effect': [],
            'Total_Effect': [],
            'Proportion_Mediated': []
        }
        
        # For direct_mediated model, we analyze each segmentation variable separately
        seg_vars = ['road', 'car_pixels', 'bicycle_pixels', 'truck_pixels', 'traffic_sign_pixels',
                   'person_pixels', 'building', 'sidewalk', 'trees', 'grass', 'sky', 'water', 'plants']
        
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
        
        # Convert to DataFrame
        effects_df = pd.DataFrame(effects)
        
        return effects_df 


@SEMModelRegistry.register(ModelType.BENCHMARK)
class BenchmarkSEMModel(SEMModel):
    """Benchmark model with direct effects only (no SEM structure).
    
    This model treats all segmentation variables and perception variables as 
    independent predictors of utility (V_img), without any latent variable structure.
    It serves as a useful baseline to compare against the more complex SEM models.
    """
    
    def get_model_spec(self) -> str:
        """Get the benchmark model specification."""
        return """
        # Direct effects of all variables on V_img 
        # Dropping 'water' to avoid perfect multicollinearity
        V_img ~ road + car_pixels + bicycle_pixels + truck_pixels + traffic_sign_pixels + person_pixels + building + sidewalk + trees + grass + sky + plants + traffic_safety_rating + social_safety_rating + beauty_rating
        
        # Allow correlations between perception ratings
        traffic_safety_rating ~~ social_safety_rating
        traffic_safety_rating ~~ beauty_rating
        social_safety_rating ~~ beauty_rating
        """
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate direct effects for the benchmark model."""
        logger.info("Calculating direct effects for benchmark model...")
        
        # Extract paths from standardized estimates
        paths = self.std_estimates.copy()
        
        # Initialize effects dictionary
        effects = {
            'Path': [],
            'Direct_Effect': [],
            'Indirect_Effect': [],
            'Total_Effect': [],
            'Proportion_Mediated': []
        }
        
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
        
        # Convert to DataFrame
        effects_df = pd.DataFrame(effects)
        
        return effects_df 