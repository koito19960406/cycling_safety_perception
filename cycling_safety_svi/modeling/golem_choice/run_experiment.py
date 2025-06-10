"""
Hyperparameter Tuning for GOLEM-DC Model using Optuna

This script performs hyperparameter optimization for:
- lambda_1: L1 penalty for adjacency matrix sparsity
- lambda_2: DAG constraint penalty
- lambda_3: L1 penalty for utility network weights

The optimization maximizes validation set log-likelihood while maintaining
model interpretability through appropriate regularization.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from golem_dc_model import GOLEMDCModel


def load_and_prepare_data():
    """
    Load and prepare the choice modeling data
    
    Returns:
        Dictionary with train/val/test splits and feature information
    """
    # Load main choice data
    choice_data = pd.read_csv('data/raw/cv_dcm.csv')
    
    # Load safety scores
    safety_scores = pd.read_csv('data/processed/predicted_danish/cycling_safety_scores.csv')
    safety_dict = dict(zip(safety_scores['image_name'], safety_scores['safety_score']))
    
    # Load segmentation pixel ratios
    pixel_ratios = pd.read_csv('data/processed/segmentation_results/pixel_ratios.csv')
    
    # Get segmentation feature names (excluding filename_key)
    segmentation_features = [col for col in pixel_ratios.columns if col != 'filename_key']
    pixel_dict = pixel_ratios.set_index('filename_key')[segmentation_features].to_dict('index')
    
    # Prepare features for each choice set
    features_list = []
    choices_list = []
    
    for _, row in choice_data.iterrows():
        # Features for alternative 1
        img1 = row['IMG1']
        features1 = [
            row['TT1'],  # Travel time
            row['TL1'],  # Traffic lights
            safety_dict.get(img1, 0.0)  # Safety score
        ]
        # Add segmentation features
        if img1 in pixel_dict:
            features1.extend([pixel_dict[img1].get(feat, 0.0) for feat in segmentation_features])
        else:
            features1.extend([0.0] * len(segmentation_features))
        
        # Features for alternative 2
        img2 = row['IMG2']
        features2 = [
            row['TT2'],  # Travel time
            row['TL2'],  # Traffic lights
            safety_dict.get(img2, 0.0)  # Safety score
        ]
        # Add segmentation features
        if img2 in pixel_dict:
            features2.extend([pixel_dict[img2].get(feat, 0.0) for feat in segmentation_features])
        else:
            features2.extend([0.0] * len(segmentation_features))
        
        # Stack alternatives
        features_list.append([features1, features2])
        choices_list.append(row['CHOICE'] - 1)  # Convert to 0-indexed
    
    # Convert to numpy arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(choices_list, dtype=np.int64)
    
    # Split by train/test flag in original data
    train_mask = choice_data['train'] == 1
    X_train_val = X[train_mask]
    y_train_val = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    # Create torch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'n_features': X.shape[2],
        'segmentation_start_idx': 3,  # Segmentation features start after TT, TL, safety
        'feature_names': ['travel_time', 'traffic_lights', 'safety_score'] + segmentation_features
    }


def train_model(model, train_loader, val_loader, n_epochs=100, patience=10, device='cuda'):
    """
    Train GOLEM-DC model
    
    Returns:
        Best validation loss
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Fit standardization on training data
    if model.standardize:
        all_X = []
        for batch_X, _ in train_loader:
            all_X.append(batch_X)
        all_X = torch.cat(all_X, dim=0).to(device)
        model.fit_standardization(all_X)
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            loss_dict = model.compute_loss(batch_X, batch_y)
            loss = loss_dict['total_loss']
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_choice_losses = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                loss_dict = model.compute_loss(batch_X, batch_y)
                val_losses.append(loss_dict['total_loss'].item())
                val_choice_losses.append(loss_dict['choice_loss'].item())
        
        avg_val_loss = np.mean(val_losses)
        avg_val_choice_loss = np.mean(val_choice_losses)
        
        # Early stopping
        if avg_val_choice_loss < best_val_loss:
            best_val_loss = avg_val_choice_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_val_loss


def objective(trial):
    """
    Optuna objective function for hyperparameter optimization
    """
    # Suggest hyperparameters
    lambda_1 = trial.suggest_float('lambda_1', 1e-4, 1.0, log=True)
    lambda_2 = trial.suggest_float('lambda_2', 0.1, 10.0, log=True)
    lambda_3 = trial.suggest_float('lambda_3', 1e-5, 0.1, log=True)
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    
    # Load data
    data = load_and_prepare_data()
    
    # Create data loaders
    train_loader = DataLoader(data['train_dataset'], batch_size=32, shuffle=True)
    val_loader = DataLoader(data['val_dataset'], batch_size=32, shuffle=False)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GOLEMDCModel(
        n_features=data['n_features'],
        hidden_dim=hidden_dim,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        lambda_3=lambda_3,
        standardize=True,
        segmentation_start_idx=data['segmentation_start_idx']
    ).to(device)
    
    # Train model
    val_loss = train_model(model, train_loader, val_loader, n_epochs=50, device=device)
    
    # Report intermediate value for pruning
    trial.report(val_loss, 0)
    
    # Handle pruning
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return val_loss


def run_hyperparameter_search(n_trials=100):
    """
    Run Optuna hyperparameter optimization
    """
    # Create study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'reports/models/golem_dc_optuna_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save best parameters
    best_params = study.best_params
    with open(f'{results_dir}/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Save study results
    study_df = study.trials_dataframe()
    study_df.to_csv(f'{results_dir}/study_results.csv', index=False)
    
    # Plot optimization history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f'{results_dir}/optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parameter importances
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f'{results_dir}/param_importances.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot parallel coordinate
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig(f'{results_dir}/parallel_coordinate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nBest parameters found:")
    print(json.dumps(best_params, indent=2))
    print(f"\nBest validation loss: {study.best_value:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    return study, best_params


if __name__ == "__main__":
    # Run hyperparameter search
    study, best_params = run_hyperparameter_search(n_trials=100)
    
    # Train final model with best parameters on full training data
    print("\nTraining final model with best parameters...")
    
    data = load_and_prepare_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Combine train and val for final training
    full_train_X = torch.cat([data['train_dataset'].tensors[0], 
                              data['val_dataset'].tensors[0]], dim=0)
    full_train_y = torch.cat([data['train_dataset'].tensors[1], 
                              data['val_dataset'].tensors[1]], dim=0)
    full_train_dataset = TensorDataset(full_train_X, full_train_y)
    full_train_loader = DataLoader(full_train_dataset, batch_size=32, shuffle=True)
    
    # Create final model
    final_model = GOLEMDCModel(
        n_features=data['n_features'],
        hidden_dim=best_params['hidden_dim'],
        lambda_1=best_params['lambda_1'],
        lambda_2=best_params['lambda_2'],
        lambda_3=best_params['lambda_3'],
        standardize=True,
        segmentation_start_idx=data['segmentation_start_idx']
    ).to(device)
    
    # Train final model
    test_loader = DataLoader(data['test_dataset'], batch_size=32, shuffle=False)
    train_model(final_model, full_train_loader, test_loader, n_epochs=100, device=device)
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'reports/models/golem_dc_final_{timestamp}.pt'
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'hyperparameters': best_params,
        'feature_names': data['feature_names'],
        'n_features': data['n_features'],
        'segmentation_start_idx': data['segmentation_start_idx']
    }, model_path)
    
    print(f"Final model saved to: {model_path}") 