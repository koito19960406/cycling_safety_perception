"""
GOLEM-DC: Joint Causal Discovery and Discrete Choice Modeling

This package implements GOLEM-DC, which combines causal structure learning
with discrete choice modeling using joint optimization.
"""

from .golem_dc_model import GOLEMDCModel
from .golem_dc_data import GOLEMDCDataLoader, ChoiceDataset
from .golem_dc_trainer import GOLEMDCTrainer

__version__ = "1.0.0"
__author__ = "GOLEM-DC Development Team"

__all__ = [
    "GOLEMDCModel",
    "GOLEMDCDataLoader", 
    "ChoiceDataset",
    "GOLEMDCTrainer"
] 