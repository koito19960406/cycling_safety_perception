"""
Object-oriented implementation of Structural Equation Models for cycling safety analysis.
This module provides classes for different types of SEM models with consistent interfaces.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import semopy
from loguru import logger
from sklearn.preprocessing import StandardScaler
from enum import Enum
from typing import Optional, Dict, List, Tuple, Union, Any

from cycling_safety_svi.config import MODELS_DIR


class ModelType(str, Enum):
    """Model types for Structural Equation Modeling."""
    FULL = "full"                    # Full model with all paths and cross-paths
    SIMPLE = "simple"                # Simplified model with fewer cross-paths 
    MINIMAL = "minimal"              # Minimal model with essential paths only
    DIRECT_ONLY = "direct_only"      # Direct effects only (no mediation)
    MEDIATION_ONLY = "mediation_only"  # Mediation effects only (no direct effects)
    DIRECT_MEDIATED = "direct_mediated"  # All segmentation variables directly affect perceptions
    BENCHMARK = "benchmark"          # Simple benchmark model with direct effects only (no SEM structure)


class SEMModel:
    """Base class for Structural Equation Models.
    
    This class provides common functionality for all SEM models.
    """
    
    def __init__(
        self, 
        model_type: ModelType,
        output_dir: Path = None,
        standardize_data: bool = True
    ):
        """Initialize the SEM model.
        
        Args:
            model_type: Type of model to fit
            output_dir: Directory for saving model outputs
            standardize_data: Whether to standardize data before fitting
        """
        self.model_type = model_type
        self.name = model_type.value
        self.output_dir = output_dir or Path(MODELS_DIR) / self.name
        self.standardize_data = standardize_data
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Will be initialized during fitting
        self.model = None
        self.model_spec = None
        self.raw_data = None
        self.data = None
        self.std_estimates = None
        self.effects_df = None
        self.fit_indices = None
        self.scaler = StandardScaler() if standardize_data else None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for modeling, including standardization if specified.
        
        Args:
            df: Raw dataframe with model variables
            
        Returns:
            Processed dataframe ready for modeling
        """
        self.raw_data = df.copy()
        data = df.copy()
        
        # Standardize numerical variables if specified
        if self.standardize_data:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data[numeric_cols] = self.scaler.fit_transform(data[numeric_cols])
        
        self.data = data
        return data
    
    def get_model_spec(self) -> str:
        """Get the model specification string.
        
        This method should be implemented by subclasses.
        
        Returns:
            Model specification in semopy syntax
        """
        raise NotImplementedError("Subclasses must implement get_model_spec()")
    
    def fit(self, df: pd.DataFrame) -> semopy.Model:
        """Fit the model to data.
        
        Args:
            df: Dataframe with model variables
            
        Returns:
            Fitted semopy.Model instance
        """
        # Prepare data
        data = self.prepare_data(df)
        
        # Get model specification
        self.model_spec = self.get_model_spec()
        
        # Create and fit the model
        logger.info(f"Fitting {self.name} SEM model...")
        try:
            self.model = semopy.Model(self.model_spec)
            result = self.model.fit(data)
            
            # Extract results
            logger.info("Model estimation complete. Summary of results:")
            summary = self.model.inspect()
            print(summary)
            
            # Get standardized estimates
            self.std_estimates = self.model.inspect(std_est=True)
            
            # Calculate model fit indices
            self.fit_indices = semopy.calc_stats(self.model)
            logger.info(f"Model fit indices for {self.name} model:")
            for key, value in self.fit_indices.items():
                logger.info(f"{key}: {value}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error fitting {self.name} model: {e}")
            return None
    
    def save_results(self) -> Dict[str, Path]:
        """Save model results to files.
        
        Returns:
            Dictionary of saved file paths
        """
        if self.model is None:
            logger.error("Model has not been fit yet")
            return {}
        
        saved_files = {}
        
        # Save standardized estimates
        estimates_path = self.output_dir / f"{self.name}_estimates.csv"
        self.std_estimates.to_csv(estimates_path, index=True)
        saved_files['estimates'] = estimates_path
        logger.info(f"Saved standardized estimates to {estimates_path}")
        
        # Save path diagram
        try:
            plot_path = self.output_dir / f"{self.name}_plot.png"
            semopy.semplot(self.model, str(plot_path), plot_covs=True, std_ests=True)
            saved_files['plot'] = plot_path
            logger.info(f"Saved path diagram to {plot_path}")
        except Exception as e:
            logger.error(f"Error generating path diagram: {e}")
        
        # Save fit indices
        fit_path = self.output_dir / f"{self.name}_fit_indices.csv"
        pd.DataFrame({k: [v] for k, v in self.fit_indices.items()}).to_csv(fit_path, index=False)
        saved_files['fit_indices'] = fit_path
        logger.info(f"Saved fit indices to {fit_path}")
        
        # Calculate and save mediation effects if model has mediation
        if hasattr(self, 'calculate_indirect_effects') and self.model_type != ModelType.DIRECT_ONLY:
            try:
                self.effects_df = self.calculate_indirect_effects()
                effects_path = self.output_dir / f"{self.name}_mediation_effects.csv"
                self.effects_df.to_csv(effects_path, index=False)
                saved_files['mediation_effects'] = effects_path
                logger.info(f"Saved mediation effects to {effects_path}")
            except Exception as e:
                logger.error(f"Error calculating mediation effects: {e}")
        
        return saved_files
    
    def calculate_indirect_effects(self) -> pd.DataFrame:
        """Calculate indirect effects for the mediation model.
        
        This method should be implemented by subclasses that support mediation.
        
        Returns:
            DataFrame containing direct, indirect, and total effects
        """
        raise NotImplementedError("Subclasses should implement calculate_indirect_effects()")


class SEMModelRegistry:
    """Registry for all available SEM model classes."""
    
    _models = {}
    
    @classmethod
    def register(cls, model_type: ModelType):
        """Decorator to register a model class for a specific model type."""
        def decorator(model_class):
            cls._models[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, model_type: ModelType, **kwargs) -> SEMModel:
        """Get model instance for specified type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type](model_type, **kwargs)
    
    @classmethod
    def available_models(cls) -> List[ModelType]:
        """List all available model types."""
        return list(cls._models.keys())


# Import model implementations that will register themselves
# This will be implemented in separate files to keep the code organized 