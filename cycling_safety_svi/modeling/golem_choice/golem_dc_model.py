"""
GOLEM-DC (GOLEM for Discrete Choice) Model Implementation

This model combines causal structure learning with discrete choice modeling
using a joint optimization objective:
L(A, θ) = -Choice_Likelihood(A, θ) + λ₁||A||₁ + λ₂h(A) + λ₃||θ||₁

Key innovations:
- Joint optimization of causal structure and choice parameters
- Causal transformation of features before choice modeling
- Gumbel noise for random utility model (not Gaussian)
- Z-score standardization for features
- L1 penalty on utility network weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GOLEMDCModel(nn.Module):
    """
    GOLEM for Discrete Choice Model
    
    Jointly learns:
    1. Causal structure between features (adjacency matrix A)
    2. Choice model parameters (utility function θ)
    
    Features include:
    - Safety scores (predicted cycling safety)
    - Pixel ratios from segmentation
    - Traditional attributes (travel time, traffic lights)
    """
    
    def __init__(self, n_features, hidden_dim=64, lambda_1=0.01, lambda_2=1.0, lambda_3=0.0, 
                 standardize=True, segmentation_start_idx=None):
        """
        Initialize GOLEM-DC model
        
        Args:
            n_features: Number of features per alternative
            hidden_dim: Hidden dimension for utility network
            lambda_1: L1 penalty weight for sparsity on adjacency matrix
            lambda_2: DAG constraint penalty weight
            lambda_3: L1 penalty weight for utility network weights
            standardize: Whether to standardize features
            segmentation_start_idx: Starting index of segmentation features for selective standardization
        """
        super(GOLEMDCModel, self).__init__()
        
        self.n_features = n_features
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.standardize = standardize
        self.segmentation_start_idx = segmentation_start_idx
        
        # Learnable causal adjacency matrix
        self.adjacency = nn.Parameter(torch.randn(n_features, n_features) * 0.01)
        
        # Utility function neural network with separate layers for L1 penalty
        self.utility_layer1 = nn.Linear(n_features, hidden_dim)
        self.utility_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.utility_layer3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Learnable Gumbel scale parameter
        self.log_gumbel_scale = nn.Parameter(torch.zeros(1))
        
        # Standardization parameters (will be set during training)
        self.register_buffer('feature_mean', torch.zeros(n_features))
        self.register_buffer('feature_std', torch.ones(n_features))
        self.standardization_fitted = False
        
    def fit_standardization(self, X):
        """
        Fit standardization parameters from training data
        
        Args:
            X: Features tensor (batch_size, n_alternatives, n_features)
        """
        if not self.standardize:
            return
            
        with torch.no_grad():
            # Reshape to (batch_size * n_alternatives, n_features)
            X_flat = X.view(-1, self.n_features)
            
            # Compute mean and std
            self.feature_mean = X_flat.mean(dim=0)
            self.feature_std = X_flat.std(dim=0)
            
            # Avoid division by zero
            self.feature_std[self.feature_std < 1e-6] = 1.0
            
            self.standardization_fitted = True
    
    def standardize_features(self, X):
        """
        Apply z-score standardization to features
        
        Args:
            X: Features tensor (batch_size, n_alternatives, n_features)
            
        Returns:
            X_standardized: Standardized features
        """
        if not self.standardize or not self.standardization_fitted:
            return X
            
        # Standardize all features or only segmentation features
        if self.segmentation_start_idx is not None:
            X_standardized = X.clone()
            # Only standardize segmentation features
            X_standardized[..., self.segmentation_start_idx:] = (
                X[..., self.segmentation_start_idx:] - self.feature_mean[self.segmentation_start_idx:]
            ) / self.feature_std[self.segmentation_start_idx:]
        else:
            # Standardize all features
            X_standardized = (X - self.feature_mean) / self.feature_std
            
        return X_standardized
        
    def apply_causal_transformation(self, X):
        """
        Apply causal transformation: X_causal = X(I - A^T)^(-1)
        
        Args:
            X: Features tensor (batch_size, n_alternatives, n_features)
            
        Returns:
            X_causal: Causally transformed features
        """
        batch_size, n_alt, n_feat = X.shape
        
        # Mask diagonal to prevent self-loops
        A_masked = self.adjacency * (1 - torch.eye(self.n_features, device=X.device))
        
        # Identity matrix
        I = torch.eye(self.n_features, device=X.device)
        
        try:
            # Compute (I - A^T)^(-1)
            inv_matrix = torch.linalg.inv(I - A_masked.T)
            
            # Apply transformation
            X_causal = torch.matmul(X, inv_matrix)
            
        except RuntimeError:
            # Fallback for singular matrix: first-order approximation
            # X_causal ≈ X + X * A^T
            X_causal = X + torch.matmul(X, A_masked.T)
            
        return X_causal
    
    def compute_utilities(self, X_causal):
        """
        Compute utilities from causally transformed features
        
        Args:
            X_causal: Causally transformed features (batch_size, n_alternatives, n_features)
            
        Returns:
            utilities: Utility values (batch_size, n_alternatives)
        """
        batch_size, n_alt, n_feat = X_causal.shape
        
        # Reshape for batch processing
        X_flat = X_causal.view(-1, n_feat)
        
        # Forward through utility network
        h1 = F.relu(self.utility_layer1(X_flat))
        h1 = self.dropout(h1)
        h2 = F.relu(self.utility_layer2(h1))
        h2 = self.dropout(h2)
        utilities_flat = self.utility_layer3(h2).squeeze(-1)
        
        utilities = utilities_flat.view(batch_size, n_alt)
        
        return utilities
    
    def forward(self, X, choice_sets=None):
        """
        Forward pass through the model
        
        Args:
            X: Features (batch_size, n_alternatives, n_features)
            choice_sets: Availability mask (batch_size, n_alternatives)
            
        Returns:
            choice_probs: Choice probabilities (batch_size, n_alternatives)
            utilities: Utility values (batch_size, n_alternatives)
        """
        # Standardize features
        X_standardized = self.standardize_features(X)
        
        # Apply causal transformation
        X_causal = self.apply_causal_transformation(X_standardized)
        
        # Compute utilities
        utilities = self.compute_utilities(X_causal)
        
        # Add Gumbel noise during training
        if self.training:
            gumbel_scale = torch.exp(self.log_gumbel_scale)
            # Sample Gumbel(0, scale) noise
            uniform = torch.rand_like(utilities)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-20) + 1e-20) * gumbel_scale
            utilities = utilities + gumbel_noise
        
        # Apply choice set constraints if provided
        if choice_sets is not None:
            # Mask unavailable alternatives with large negative utility
            utilities_masked = utilities * choice_sets + (-1e10) * (1 - choice_sets)
        else:
            utilities_masked = utilities
            
        # Compute choice probabilities (softmax)
        choice_probs = F.softmax(utilities_masked, dim=1)
        
        return choice_probs, utilities
    
    def dag_constraint_penalty(self):
        """
        Compute DAG constraint penalty: h(A) = tr(exp(A ⊙ A)) - n
        """
        # Mask diagonal
        A_masked = self.adjacency * (1 - torch.eye(self.n_features, device=self.adjacency.device))
        
        # Element-wise square
        A_squared = A_masked * A_masked
        
        # Matrix exponential trace
        try:
            exp_A = torch.matrix_exp(A_squared)
            h = torch.trace(exp_A) - self.n_features
        except RuntimeError:
            # Approximation using power series
            I = torch.eye(self.n_features, device=self.adjacency.device)
            A2 = torch.matmul(A_squared, A_squared)
            exp_approx = I + A_squared + A2 / 2.0
            h = torch.trace(exp_approx) - self.n_features
            
        return h
    
    def l1_penalty(self):
        """
        Compute L1 penalty for sparsity: ||A||₁
        """
        # Mask diagonal to be consistent with DAG penalty and original GOLEM
        A_masked = self.adjacency * (1 - torch.eye(self.n_features, device=self.adjacency.device))
        return torch.sum(torch.abs(A_masked))
    
    def utility_l1_penalty(self):
        """
        Compute L1 penalty for utility network weights: ||θ||₁
        """
        l1_penalty = 0.0
        
        # Add L1 penalty for each layer's weights
        l1_penalty += torch.sum(torch.abs(self.utility_layer1.weight))
        l1_penalty += torch.sum(torch.abs(self.utility_layer2.weight))
        l1_penalty += torch.sum(torch.abs(self.utility_layer3.weight))
        
        return l1_penalty
    
    def compute_loss(self, X, choices, choice_sets=None):
        """
        Compute joint GOLEM-DC loss
        
        Args:
            X: Features (batch_size, n_alternatives, n_features)
            choices: Chosen alternatives (batch_size,)
            choice_sets: Availability mask (batch_size, n_alternatives)
            
        Returns:
            Dictionary with loss components
        """
        # Get choice probabilities and utilities
        choice_probs, utilities = self.forward(X, choice_sets)
        
        # Compute negative log-likelihood (cross-entropy)
        if choice_sets is not None:
            utilities_masked = utilities * choice_sets + (-1e10) * (1 - choice_sets)
        else:
            utilities_masked = utilities
            
        choice_loss = F.cross_entropy(utilities_masked, choices)
        
        # Compute regularization penalties
        dag_penalty = self.dag_constraint_penalty()
        l1_penalty = self.l1_penalty()
        utility_l1_penalty = self.utility_l1_penalty()
        
        # Total loss with all penalties
        total_loss = (choice_loss + 
                     self.lambda_1 * l1_penalty + 
                     self.lambda_2 * dag_penalty +
                     self.lambda_3 * utility_l1_penalty)
        
        return {
            'total_loss': total_loss,
            'choice_loss': choice_loss,
            'dag_penalty': dag_penalty,
            'l1_penalty': l1_penalty,
            'utility_l1_penalty': utility_l1_penalty,
            'choice_probs': choice_probs,
            'utilities': utilities
        }
    
    def get_causal_matrix(self):
        """
        Get the learned causal adjacency matrix
        
        Returns:
            Numpy array of causal relationships
        """
        with torch.no_grad():
            A = self.adjacency * (1 - torch.eye(self.n_features, device=self.adjacency.device))
            return A.cpu().numpy()
    
    def get_utility_weights(self):
        """
        Get the weights of the utility network
        
        Returns:
            Dictionary with weight matrices for each layer
        """
        with torch.no_grad():
            return {
                'layer1_weight': self.utility_layer1.weight.cpu().numpy(),
                'layer1_bias': self.utility_layer1.bias.cpu().numpy(),
                'layer2_weight': self.utility_layer2.weight.cpu().numpy(),
                'layer2_bias': self.utility_layer2.bias.cpu().numpy(),
                'layer3_weight': self.utility_layer3.weight.cpu().numpy(),
                'layer3_bias': self.utility_layer3.bias.cpu().numpy()
            }
    
    def predict(self, X, choice_sets=None):
        """
        Make predictions on new data
        
        Args:
            X: Features (batch_size, n_alternatives, n_features)
            choice_sets: Availability mask (batch_size, n_alternatives)
            
        Returns:
            predicted_choices: Predicted alternative indices (batch_size,)
            choice_probs: Choice probabilities (batch_size, n_alternatives)
        """
        self.eval()
        with torch.no_grad():
            choice_probs, _ = self.forward(X, choice_sets)
            predicted_choices = torch.argmax(choice_probs, dim=1)
            
        return predicted_choices, choice_probs 