"""
Training module for GOLEM-DC model

Handles:
1. Training loop with joint optimization
2. Model evaluation and metrics
3. Visualization of causal structure and results
4. Comparison with baseline models
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import os
import json


class GOLEMDCTrainer:
    """
    Trainer for GOLEM-DC model
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: GOLEMDCModel instance
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_choice_loss': [], 'train_dag_penalty': [],
            'train_l1_penalty': [], 'train_accuracy': [],
            'val_loss': [], 'val_choice_loss': [], 'val_accuracy': []
        }
        
    def train(self, train_loader, val_loader=None, n_epochs=100, lr=0.001, 
              patience=20, verbose=True):
        """
        Train the model
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            n_epochs: Number of epochs
            lr: Learning rate
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            history: Training history dictionary
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience//2, factor=0.5
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_metrics = self._train_epoch(train_loader, optimizer)
            
            # Record training metrics
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            
            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_metrics = self._validate_epoch(val_loader)
                
                # Record validation metrics
                for key, value in val_metrics.items():
                    self.history[f'val_{key}'].append(value)
                
                # Learning rate scheduling
                scheduler.step(val_metrics['loss'])
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if verbose and epoch % 10 == 0:
                log_str = f"Epoch {epoch}/{n_epochs}"
                log_str += f" - Train Loss: {train_metrics['loss']:.4f}"
                log_str += f" - Train Acc: {train_metrics['accuracy']:.4f}"
                if val_loader is not None:
                    log_str += f" - Val Loss: {val_metrics['loss']:.4f}"
                    log_str += f" - Val Acc: {val_metrics['accuracy']:.4f}"
                print(log_str)
                
        # Load best model if available
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        total_loss = 0
        total_choice_loss = 0
        total_dag_penalty = 0
        total_l1_penalty = 0
        all_predictions = []
        all_targets = []
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            choices = batch['choice'].to(self.device)
            choice_sets = batch['choice_set'].to(self.device)
            
            optimizer.zero_grad()
            
            # Compute loss
            loss_dict = self.model.compute_loss(features, choices, choice_sets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_choice_loss += loss_dict['choice_loss'].item()
            total_dag_penalty += loss_dict['dag_penalty'].item()
            total_l1_penalty += loss_dict['l1_penalty'].item()
            
            # Predictions for accuracy
            with torch.no_grad():
                probs = loss_dict['choice_probs']
                predictions = torch.argmax(probs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(choices.cpu().numpy())
        
        # Compute metrics
        n_batches = len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return {
            'loss': total_loss / n_batches,
            'choice_loss': total_choice_loss / n_batches,
            'dag_penalty': total_dag_penalty / n_batches,
            'l1_penalty': total_l1_penalty / n_batches,
            'accuracy': accuracy
        }
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        total_loss = 0
        total_choice_loss = 0
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                choices = batch['choice'].to(self.device)
                choice_sets = batch['choice_set'].to(self.device)
                
                # Compute loss
                loss_dict = self.model.compute_loss(features, choices, choice_sets)
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item()
                total_choice_loss += loss_dict['choice_loss'].item()
                
                # Predictions
                probs = loss_dict['choice_probs']
                predictions = torch.argmax(probs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(choices.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        n_batches = len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return {
            'loss': total_loss / n_batches,
            'choice_loss': total_choice_loss / n_batches,
            'accuracy': accuracy
        }
    
    def evaluate(self, test_loader):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test DataLoader
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probs = []
        total_log_likelihood = 0
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                choices = batch['choice'].to(self.device)
                choice_sets = batch['choice_set'].to(self.device)
                
                # Get predictions
                predictions, probs = self.model.predict(features, choice_sets)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(choices.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Log-likelihood
                batch_size = features.shape[0]
                for i in range(batch_size):
                    chosen_prob = probs[i, choices[i]].item()
                    total_log_likelihood += np.log(chosen_prob + 1e-10)
        
        # Compute metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Create one-hot encoded targets for log loss
        n_classes = all_probs.shape[1]
        targets_onehot = np.zeros((len(all_targets), n_classes))
        targets_onehot[np.arange(len(all_targets)), all_targets] = 1
        
        # Calculate metrics
        avg_log_likelihood = total_log_likelihood / len(all_targets)
        
        # Calculate AIC and BIC
        n_params = sum(p.numel() for p in self.model.parameters())
        n_samples = len(all_targets)
        aic = 2 * n_params - 2 * total_log_likelihood
        bic = n_params * np.log(n_samples) - 2 * total_log_likelihood
        
        # McFadden's Pseudo R²
        # Null model: equal probability for each alternative
        null_log_likelihood = n_samples * np.log(1.0 / n_classes)
        pseudo_r2 = 1 - (total_log_likelihood / null_log_likelihood)
        
        metrics = {
            'accuracy': accuracy,
            'log_likelihood': total_log_likelihood,
            'avg_log_likelihood': avg_log_likelihood,
            'aic': aic,
            'bic': bic,
            'pseudo_r2': pseudo_r2,
            'n_parameters': n_params
        }
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save figure (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss components
        axes[0, 0].plot(self.history['train_loss'], label='Total Loss')
        axes[0, 0].plot(self.history['train_choice_loss'], label='Choice Loss')
        if 'val_loss' in self.history:
            axes[0, 0].plot(self.history['val_loss'], label='Val Total Loss', linestyle='--')
            axes[0, 0].plot(self.history['val_choice_loss'], label='Val Choice Loss', linestyle='--')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # GOLEM penalties
        axes[0, 1].plot(self.history['train_dag_penalty'], label='DAG Penalty')
        axes[0, 1].plot(self.history['train_l1_penalty'], label='L1 Penalty')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Penalty')
        axes[0, 1].set_title('GOLEM Penalties')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(self.history['train_accuracy'], label='Train')
        if 'val_accuracy' in self.history:
            axes[1, 0].plot(self.history['val_accuracy'], label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Model Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning curves
        if 'val_loss' in self.history:
            train_loss = self.history['train_choice_loss']
            val_loss = self.history['val_choice_loss']
            
            axes[1, 1].plot(train_loss, label='Train')
            axes[1, 1].plot(val_loss, label='Validation')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Choice Loss')
            axes[1, 1].set_title('Learning Curves')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_causal_structure(self, feature_names, threshold=0.1, save_path=None):
        """
        Visualize learned causal structure
        
        Args:
            feature_names: List of feature names
            threshold: Threshold for displaying edges
            save_path: Path to save figure (optional)
        """
        # Get causal matrix
        causal_matrix = self.model.get_causal_matrix()
        
        # Apply threshold
        causal_matrix_display = causal_matrix.copy()
        causal_matrix_display[np.abs(causal_matrix_display) < threshold] = 0
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(causal_matrix_display, 
                    xticklabels=feature_names,
                    yticklabels=feature_names,
                    cmap='RdBu_r',
                    center=0,
                    annot=True,
                    fmt='.3f',
                    square=True,
                    cbar_kws={'label': 'Causal Strength'})
        
        plt.title('Learned Causal Structure')
        plt.xlabel('To Feature')
        plt.ylabel('From Feature')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        n_edges = np.sum(np.abs(causal_matrix_display) > 0)
        sparsity = 1 - (n_edges / (len(feature_names) ** 2))
        
        print(f"\nCausal Structure Summary:")
        print(f"Number of features: {len(feature_names)}")
        print(f"Number of causal edges (threshold={threshold}): {n_edges}")
        print(f"Sparsity: {sparsity:.3f}")
        
        # Find strongest causal relationships
        print("\nStrongest causal relationships:")
        edges = []
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if i != j and abs(causal_matrix[i, j]) > threshold:
                    edges.append((feature_names[i], feature_names[j], causal_matrix[i, j]))
        
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        for from_feat, to_feat, strength in edges[:10]:
            print(f"  {from_feat} → {to_feat}: {strength:.3f}")
    
    def save_results(self, output_dir, feature_names, test_metrics=None):
        """
        Save model and results
        
        Args:
            output_dir: Directory to save results
            feature_names: List of feature names
            test_metrics: Test evaluation metrics (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), 
                   os.path.join(output_dir, 'golem_dc_model.pt'))
        
        # Save training history
        # Handle different lengths in history dictionary (when validation not used)
        history_data = {}
        max_length = 0
        
        # Find maximum length
        for key, values in self.history.items():
            if len(values) > 0:
                max_length = max(max_length, len(values))
                history_data[key] = values
        
        # Pad shorter lists with NaN
        for key, values in history_data.items():
            if len(values) < max_length:
                history_data[key] = values + [np.nan] * (max_length - len(values))
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            history_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
        
        # Save causal matrix
        causal_matrix = self.model.get_causal_matrix()
        causal_df = pd.DataFrame(causal_matrix, 
                                 index=feature_names,
                                 columns=feature_names)
        causal_df.to_csv(os.path.join(output_dir, 'causal_matrix.csv'))
        
        # Save test metrics if provided
        if test_metrics:
            with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
                json.dump(test_metrics, f, indent=4)
                
        print(f"Results saved to {output_dir}") 