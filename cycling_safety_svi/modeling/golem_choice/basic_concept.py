"""
Joint GOLEM Choice Optimization - Correct Implementation
This shows the proper way to integrate GOLEM with choice modeling
using a single joint objective function.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class JointGOLEMChoiceModel(nn.Module):
    """
    Joint GOLEM Choice Model with single objective function
    
    This model simultaneously learns:
    1. Causal structure (adjacency matrix A)
    2. Utility function parameters (θ)
    
    Using the joint objective:
    L(A, θ) = -Choice_Likelihood(A, θ) + λ₁||A||₁ + λ₂h(A)
    """
    
    def __init__(self, n_features, hidden_dim=64, lambda_1=0.02, lambda_2=5.0):
        super(JointGOLEMChoiceModel, self).__init__()
        
        self.n_features = n_features
        self.lambda_1 = lambda_1  # L1 penalty
        self.lambda_2 = lambda_2  # DAG penalty
        
        # Learnable causal adjacency matrix
        self.adjacency = nn.Parameter(torch.randn(n_features, n_features) * 0.1)
        
        # Utility function neural network
        self.utility_net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable Gumbel noise scale for random utility model
        # Note: In choice modeling, we assume Gumbel noise on utilities, not Gaussian
        self.log_sigma = nn.Parameter(torch.zeros(1))
        
    def apply_causal_transformation(self, X):
        """
        Apply learned causal structure to transform features
        X_causal = X * (I - A^T)^(-1)
        """
        batch_size, n_alt, n_feat = X.shape
        
        # Mask diagonal to prevent self-loops
        A_masked = self.adjacency * (1 - torch.eye(self.n_features))
        
        # Compute causal transformation matrix
        I = torch.eye(self.n_features)
        
        try:
            # Try exact inverse
            inv_matrix = torch.linalg.inv(I - A_masked.T)
            
            # Apply transformation to each observation
            X_causal = torch.zeros_like(X)
            for i in range(batch_size):
                X_causal[i] = torch.matmul(X[i], inv_matrix)
                
        except RuntimeError:
            # Fallback: Use approximate transformation if matrix is singular
            # X_causal ≈ X + X * A^T (first-order approximation)
            X_causal = torch.zeros_like(X)
            for i in range(batch_size):
                X_causal[i] = X[i] + torch.matmul(X[i], A_masked.T)
        
        return X_causal
    
    def compute_utilities(self, X_causal):
        """Compute utilities using neural network"""
        batch_size, n_alt, n_feat = X_causal.shape
        
        # Flatten for batch processing
        X_flat = X_causal.view(-1, n_feat)
        utilities_flat = self.utility_net(X_flat).squeeze(-1)
        utilities = utilities_flat.view(batch_size, n_alt)
        
        return utilities
    
    def forward(self, X, choice_sets):
        """
        Forward pass: X -> X_causal -> utilities -> choice probabilities
        
        Parameters:
        -----------
        X : tensor, shape (batch_size, n_alternatives, n_features)
        choice_sets : tensor, shape (batch_size, n_alternatives)
        
        Returns:
        --------
        choice_probs : tensor, shape (batch_size, n_alternatives)
        utilities : tensor, shape (batch_size, n_alternatives)
        """
        # Apply causal transformation
        X_causal = self.apply_causal_transformation(X)
        
        # Compute utilities
        utilities = self.compute_utilities(X_causal)
        
        # Add Gumbel noise to utilities (only during training)
        # This is crucial: choice models assume Gumbel noise, NOT Gaussian!
        if self.training:
            sigma = torch.exp(self.log_sigma)
            # Generate Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
            uniform = torch.rand_like(utilities)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-8) + 1e-8) * sigma
            utilities = utilities + gumbel_noise
        
        # Apply choice set constraints
        utilities_masked = utilities * choice_sets + (-1e8) * (1 - choice_sets)
        
        # Compute choice probabilities (softmax)
        choice_probs = torch.softmax(utilities_masked, dim=1)
        
        return choice_probs, utilities
    
    def dag_constraint_penalty(self):
        """
        DAG constraint using matrix exponential trace
        h(A) = tr(exp(A ⊙ A)) - p
        """
        A_masked = self.adjacency * (1 - torch.eye(self.n_features))
        A_squared = A_masked * A_masked  # Element-wise square
        
        # Matrix exponential trace (using eigendecomposition for stability)
        try:
            exp_A = torch.matrix_exp(A_squared)
            trace_exp = torch.trace(exp_A)
        except RuntimeError:
            # Fallback: Use series expansion approximation
            # exp(A) ≈ I + A + A²/2! + A³/3! + ...
            I = torch.eye(self.n_features)
            exp_approx = I + A_squared + torch.matmul(A_squared, A_squared) / 2
            trace_exp = torch.trace(exp_approx)
        
        return trace_exp - self.n_features
    
    def l1_penalty(self):
        """L1 penalty for sparsity: ||A||₁"""
        return torch.sum(torch.abs(self.adjacency))
    
    def compute_joint_loss(self, X, choices, choice_sets):
        """
        Compute the joint GOLEM-Choice objective:
        L(A, θ) = -log p(y | X_causal(A), θ) + λ₁||A||₁ + λ₂h(A)
        
        Where -log p(y | X_causal, θ) is the cross-entropy loss, which is 
        mathematically equivalent to the negative multinomial likelihood.
        """
        # Forward pass to get choice probabilities
        choice_probs, utilities = self.forward(X, choice_sets)
        
        # Apply choice set mask to utilities for proper cross-entropy computation
        utilities_masked = utilities * choice_sets + (-1e8) * (1 - choice_sets)
        
        # Cross-entropy loss = -multinomial log-likelihood
        # This directly corresponds to: -Σᵢ log P(yᵢ | X_causal,ᵢ, θ)
        choice_loss = nn.CrossEntropyLoss()(utilities_masked, choices)
        
        # Alternative: Manual multinomial likelihood computation (equivalent)
        # log_probs = torch.log_softmax(utilities_masked, dim=1)
        # multinomial_likelihood = torch.gather(log_probs, 1, choices.unsqueeze(1)).mean()
        # choice_loss = -multinomial_likelihood  # This equals the cross-entropy above
        
        # GOLEM regularization terms
        dag_penalty = self.dag_constraint_penalty()  # h(A) = tr(exp(A⊙A)) - p
        l1_penalty = self.l1_penalty()              # ||A||₁
        
        # Joint objective: minimize negative log-likelihood + regularization
        total_loss = choice_loss + self.lambda_1 * l1_penalty + self.lambda_2 * dag_penalty
        
        return {
            'total_loss': total_loss,
            'choice_loss': choice_loss,           # Cross-entropy = -log p(y|X_causal,θ)
            'dag_penalty': dag_penalty,           # DAG constraint h(A)
            'l1_penalty': l1_penalty,            # Sparsity penalty ||A||₁
            'neg_log_likelihood': choice_loss     # Same as choice_loss, for clarity
        }


def train_joint_golem_choice(model, X_train, y_train, choice_sets_train,
                           X_val=None, y_val=None, choice_sets_val=None,
                           n_epochs=2000, lr=0.001, patience=50):
    """
    Train the joint GOLEM choice model with early stopping
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2)
    
    # Training history
    history = {
        'train_loss': [], 'train_choice_loss': [], 'train_dag_penalty': [], 
        'train_l1_penalty': [], 'train_accuracy': []
    }
    
    if X_val is not None:
        history.update({
            'val_loss': [], 'val_choice_loss': [], 'val_accuracy': []
        })
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        loss_dict = model.compute_joint_loss(X_train, y_train, choice_sets_train)
        loss_dict['total_loss'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Record training metrics
        history['train_loss'].append(loss_dict['total_loss'].item())
        history['train_choice_loss'].append(loss_dict['choice_loss'].item())
        history['train_dag_penalty'].append(loss_dict['dag_penalty'].item())
        history['train_l1_penalty'].append(loss_dict['l1_penalty'].item())
        
        # Training accuracy
        with torch.no_grad():
            probs, _ = model(X_train, choice_sets_train)
            pred_choices = torch.argmax(probs, dim=1)
            train_acc = (pred_choices == y_train).float().mean().item()
            history['train_accuracy'].append(train_acc)
        
        # Validation phase
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss_dict = model.compute_joint_loss(X_val, y_val, choice_sets_val)
                val_probs, _ = model(X_val, choice_sets_val)
                val_pred = torch.argmax(val_probs, dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
                
                history['val_loss'].append(val_loss_dict['total_loss'].item())
                history['val_choice_loss'].append(val_loss_dict['choice_loss'].item())
                history['val_accuracy'].append(val_acc)
                
                # Early stopping
                current_val_loss = val_loss_dict['total_loss'].item()
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(current_val_loss)
        
        # Logging
        if epoch % 200 == 0:
            log_str = f'Epoch {epoch}: Total Loss={loss_dict["total_loss"].item():.4f}, '
            log_str += f'Choice Loss={loss_dict["choice_loss"].item():.4f}, '
            log_str += f'DAG Penalty={loss_dict["dag_penalty"].item():.4f}, '
            log_str += f'Train Acc={train_acc:.4f}'
            
            if X_val is not None:
                log_str += f', Val Acc={val_acc:.4f}'
            
            print(log_str)
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    return history


def plot_joint_results(model, history, true_causal_matrix):
    """Plot comprehensive results for joint optimization"""
    
    # Extract learned causal matrix
    learned_matrix = model.adjacency.detach().numpy()
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Training loss components
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].plot(history['train_choice_loss'], label='Choice Loss')
    axes[0, 0].set_title('Training Loss Components')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # GOLEM penalties
    axes[0, 1].plot(history['train_dag_penalty'], label='DAG Penalty', color='red')
    axes[0, 1].plot(history['train_l1_penalty'], label='L1 Penalty', color='orange')
    axes[0, 1].set_title('GOLEM Penalties')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Penalty Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy
    axes[0, 2].plot(history['train_accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history:
        axes[0, 2].plot(history['val_accuracy'], label='Val Accuracy')
    axes[0, 2].set_title('Model Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # True causal structure
    sns.heatmap(true_causal_matrix, annot=True, cmap='RdBu_r', center=0,
                ax=axes[1, 0], cbar_kws={'label': 'Causal Strength'})
    axes[1, 0].set_title('True Causal Structure')
    axes[1, 0].set_xlabel('To Feature')
    axes[1, 0].set_ylabel('From Feature')
    
    # Learned causal structure  
    sns.heatmap(learned_matrix, annot=True, cmap='RdBu_r', center=0,
                ax=axes[1, 1], cbar_kws={'label': 'Learned Strength'})
    axes[1, 1].set_title('Learned Causal Structure')
    axes[1, 1].set_xlabel('To Feature')
    axes[1, 1].set_ylabel('From Feature')
    
    # Absolute difference
    diff_matrix = np.abs(learned_matrix - true_causal_matrix)
    sns.heatmap(diff_matrix, annot=True, cmap='Reds',
                ax=axes[1, 2], cbar_kws={'label': 'Absolute Difference'})
    axes[1, 2].set_title('|Learned - True|')
    axes[1, 2].set_xlabel('To Feature')
    axes[1, 2].set_ylabel('From Feature')
    
    # Edge strength distribution
    true_edges = true_causal_matrix[true_causal_matrix != 0]
    learned_edges = learned_matrix[true_causal_matrix != 0]  # Same positions
    
    axes[2, 0].scatter(true_edges, learned_edges, alpha=0.7)
    axes[2, 0].plot([true_edges.min(), true_edges.max()], 
                    [true_edges.min(), true_edges.max()], 'r--', alpha=0.5)
    axes[2, 0].set_xlabel('True Edge Strength')
    axes[2, 0].set_ylabel('Learned Edge Strength')
    axes[2, 0].set_title('Edge Strength Correlation')
    axes[2, 0].grid(True)
    
    # Sparsity comparison
    true_sparsity = np.mean(np.abs(true_causal_matrix) < 0.01)
    learned_sparsity = np.mean(np.abs(learned_matrix) < 0.1)
    
    axes[2, 1].bar(['True Graph', 'Learned Graph'], [true_sparsity, learned_sparsity],
                   color=['blue', 'orange'])
    axes[2, 1].set_title('Graph Sparsity')
    axes[2, 1].set_ylabel('Fraction of Zero Edges')
    axes[2, 1].set_ylim(0, 1)
    
    # Performance summary
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
    
    metrics_text = f'Final Train Accuracy: {final_train_acc:.3f}\n'
    if final_val_acc > 0:
        metrics_text += f'Final Val Accuracy: {final_val_acc:.3f}\n'
    metrics_text += f'True Sparsity: {true_sparsity:.3f}\n'
    metrics_text += f'Learned Sparsity: {learned_sparsity:.3f}\n'
    metrics_text += f'Mean Absolute Error: {np.mean(diff_matrix):.3f}'
    
    axes[2, 2].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', transform=axes[2, 2].transAxes)
    axes[2, 2].set_title('Performance Summary')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()