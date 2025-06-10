"""
Test script for GOLEM-DC implementation

This script tests the basic functionality of the GOLEM-DC model
with a small sample of data to ensure everything works correctly.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from golem_dc_model import GOLEMDCModel
from golem_dc_data import GOLEMDCDataLoader, ChoiceDataset
from golem_dc_trainer import GOLEMDCTrainer


def test_model_initialization():
    """Test model initialization"""
    print("Testing model initialization...")
    
    n_features = 5
    model = GOLEMDCModel(n_features=n_features, hidden_dim=32)
    
    # Check parameters
    assert hasattr(model, 'adjacency'), "Model missing adjacency matrix"
    assert hasattr(model, 'utility_net'), "Model missing utility network"
    assert hasattr(model, 'log_gumbel_scale'), "Model missing Gumbel scale parameter"
    
    # Check dimensions
    assert model.adjacency.shape == (n_features, n_features), "Incorrect adjacency matrix shape"
    
    print("✓ Model initialization successful")
    return model


def test_forward_pass():
    """Test forward pass"""
    print("\nTesting forward pass...")
    
    # Create dummy data
    batch_size = 10
    n_alternatives = 2
    n_features = 5
    
    model = GOLEMDCModel(n_features=n_features)
    X = torch.randn(batch_size, n_alternatives, n_features)
    choice_sets = torch.ones(batch_size, n_alternatives)
    
    # Forward pass
    probs, utilities = model(X, choice_sets)
    
    # Check outputs
    assert probs.shape == (batch_size, n_alternatives), "Incorrect probability shape"
    assert utilities.shape == (batch_size, n_alternatives), "Incorrect utilities shape"
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6), "Probabilities don't sum to 1"
    
    print("✓ Forward pass successful")


def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")
    
    # Create dummy data
    batch_size = 10
    n_alternatives = 2
    n_features = 5
    
    model = GOLEMDCModel(n_features=n_features)
    X = torch.randn(batch_size, n_alternatives, n_features)
    choices = torch.randint(0, n_alternatives, (batch_size,))
    choice_sets = torch.ones(batch_size, n_alternatives)
    
    # Compute loss
    loss_dict = model.compute_loss(X, choices, choice_sets)
    
    # Check outputs
    required_keys = ['total_loss', 'choice_loss', 'dag_penalty', 'l1_penalty']
    for key in required_keys:
        assert key in loss_dict, f"Missing {key} in loss dictionary"
        assert not torch.isnan(loss_dict[key]), f"{key} is NaN"
    
    print("✓ Loss computation successful")


def test_data_loading():
    """Test data loading with small sample"""
    print("\nTesting data loading...")
    
    # Check if data files exist
    choice_data_path = 'data/raw/cv_dcm.csv'
    safety_scores_path = 'data/processed/predicted_danish/cycling_safety_scores.csv'
    
    if not os.path.exists(choice_data_path) or not os.path.exists(safety_scores_path):
        print("⚠ Skipping data loading test - data files not found")
        return None
    
    # Load small sample
    data_loader = GOLEMDCDataLoader(
        choice_data_path=choice_data_path,
        safety_scores_path=safety_scores_path,
        segmentation_path=None  # Skip segmentation for speed
    )
    
    # Test feature preparation
    features_alt1, features_alt2, feature_names, scaler = data_loader.prepare_features()
    
    print(f"  - Loaded {len(data_loader.choice_data)} choice observations")
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Feature names: {', '.join(feature_names)}")
    
    print("✓ Data loading successful")
    return data_loader


def test_training_step():
    """Test single training step"""
    print("\nTesting training step...")
    
    # Create small dataset
    n_samples = 50
    n_features = 3
    
    # Create synthetic data
    features_alt1 = np.random.randn(n_samples, n_features)
    features_alt2 = np.random.randn(n_samples, n_features)
    
    # Create choices based on simple utility
    utilities_1 = features_alt1.sum(axis=1) + np.random.normal(0, 0.1, n_samples)
    utilities_2 = features_alt2.sum(axis=1) + np.random.normal(0, 0.1, n_samples)
    choices = (utilities_2 > utilities_1).astype(int) + 1
    
    # Create dataset
    dataset = ChoiceDataset(
        pd.DataFrame({'dummy': range(n_samples)}),
        features_alt1,
        features_alt2,
        choices
    )
    
    # Create data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize model and trainer
    model = GOLEMDCModel(n_features=n_features, hidden_dim=16)
    trainer = GOLEMDCTrainer(model, device='cpu')
    
    # Run one training epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    metrics = trainer._train_epoch(loader, optimizer)
    
    # Check metrics
    assert 'loss' in metrics, "Missing loss in metrics"
    assert 'accuracy' in metrics, "Missing accuracy in metrics"
    assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1, "Invalid accuracy"
    
    print(f"  - Training loss: {metrics['loss']:.4f}")
    print(f"  - Training accuracy: {metrics['accuracy']:.4f}")
    print("✓ Training step successful")


def test_causal_matrix():
    """Test causal matrix extraction"""
    print("\nTesting causal matrix extraction...")
    
    n_features = 4
    model = GOLEMDCModel(n_features=n_features)
    
    # Get causal matrix
    causal_matrix = model.get_causal_matrix()
    
    # Check properties
    assert causal_matrix.shape == (n_features, n_features), "Incorrect causal matrix shape"
    assert np.allclose(np.diag(causal_matrix), 0), "Diagonal should be zero"
    
    print("✓ Causal matrix extraction successful")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running GOLEM-DC Tests")
    print("=" * 60)
    
    try:
        # Test 1: Model initialization
        model = test_model_initialization()
        
        # Test 2: Forward pass
        test_forward_pass()
        
        # Test 3: Loss computation
        test_loss_computation()
        
        # Test 4: Data loading
        data_loader = test_data_loading()
        
        # Test 5: Training step
        test_training_step()
        
        # Test 6: Causal matrix
        test_causal_matrix()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error: {str(e)}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 