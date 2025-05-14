import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import platform
import sys
from pathlib import Path
from datetime import datetime
import optuna
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from perception_model import SinglePerceptionModel
from perception_dataset import PerceptionDataset
from train_single import train_single, evaluate_model_single


def run_training_separate(data_file, img_path, output_dir, batch_size=16, learning_rate=1e-4,
               weight_decay=0.01, num_epochs=50, patience=5, workers=2, num_categories=5,
               use_optuna=False, n_trials=10, device=None, warmup_epochs=2, trial_epochs=10,
               final_epochs=80, lr_decay_factor=0.1, scheduler_factor=0.5, scheduler_patience=3,
               hidden_layers=2, hidden_dims=None, dropout_rates=None, freeze_backbone=False):
    """
    Train separate perception prediction models for each perception variable
    
    Args:
        data_file: Path to perception ratings data
        img_path: Path to image directory
        output_dir: Directory to save model and results
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: Weight decay parameter
        num_epochs: Maximum number of epochs (if not using Optuna)
        patience: Early stopping patience
        workers: Number of workers for data loading
        num_categories: Number of ordinal categories (3 or 5)
        use_optuna: Whether to use Optuna for hyperparameter optimization
        n_trials: Number of Optuna trials
        device: Device to use for training
        warmup_epochs: Number of warmup epochs for Optuna trials
        trial_epochs: Number of epochs for each Optuna trial after warmup
        final_epochs: Number of epochs for final training after optimization
        lr_decay_factor: Factor for layer-wise learning rate decay
        scheduler_factor: Factor for learning rate scheduler
        scheduler_patience: Patience for learning rate scheduler
        hidden_layers: Number of hidden layers
        hidden_dims: List of hidden dimensions (default: [512, 256])
        dropout_rates: List of dropout rates (default: [0.3, 0.2])
        freeze_backbone: Whether to freeze the vision model backbone
    """
    # Set default values for hidden_dims and dropout_rates
    if hidden_dims is None:
        hidden_dims = [512, 256]
    if dropout_rates is None:
        dropout_rates = [0.3, 0.2]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%H%M_%d_%m_%Y")

    # Keep diary
    output_filen = f'{dateStr}_perception_output.txt'
    
    def printLog(*args, **kwargs):
        print(*args, **kwargs)
        with open(os.path.join(output_dir, output_filen), 'a') as file:
            print(*args, **kwargs, file=file)
    
    # Display code started
    printLog("PERCEPTION MODEL TRAINING STARTED (SEPARATE MODELS) at:", dateStr)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    printLog(f'Using device: {device}')

    # Load data
    printLog(f'Loading data from {data_file}')
    dataset_train = PerceptionDataset(data_file=data_file, img_path=img_path, set_type='train', 
                                    transform=True, num_categories=num_categories)
    dataset_test = PerceptionDataset(data_file=data_file, img_path=img_path, set_type='test', 
                                   transform=True, num_categories=num_categories)
    
    printLog(f'Dataset loaded: {len(dataset_train)} training samples, {len(dataset_test)} test samples')

    # Create dataloaders
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, 
                             num_workers=workers, pin_memory=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False,
                           num_workers=workers, pin_memory=True)

    # List of perception types to train
    perception_types = ['traffic_safety', 'social_safety', 'beautiful']
    
    # Dictionary to store results for each perception model
    best_models = {}
    best_params = {}
    all_metrics = {}

    # Train a separate model for each perception variable
    for perception_type in perception_types:
        printLog(f"\n{'=' * 50}")
        printLog(f"TRAINING MODEL FOR {perception_type.upper()}")
        printLog(f"{'=' * 50}\n")
        
        model_params = {
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'lr_decay_factor': lr_decay_factor,
            'scheduler_factor': scheduler_factor,
            'scheduler_patience': scheduler_patience,
            'hidden_layers': hidden_layers,
            'hidden_dims': hidden_dims,
            'dropout_rates': dropout_rates,
            'freeze_backbone': freeze_backbone
        }
        
        if use_optuna:
            # Define the objective function for Optuna
            def objective(trial):
                # Suggest hyperparameters
                lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
                wd = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
                lr_decay = trial.suggest_float('lr_decay_factor', 0.05, 0.5, log=True)
                sched_factor = trial.suggest_float('scheduler_factor', 0.1, 0.8)
                sched_patience = trial.suggest_int('scheduler_patience', 2, 5)
                
                # Network architecture hyperparameters
                num_hidden = trial.suggest_int('hidden_layers', 1, 3)
                
                # Suggest dimensions based on number of layers
                hidden_dims_trial = []
                dropout_rates_trial = []
                
                for i in range(num_hidden):
                    # Decreasing dimensions for deeper layers
                    dim_name = f'hidden_dim_{i}'
                    dropout_name = f'dropout_rate_{i}'
                    
                    if i == 0:
                        # First hidden layer
                        dim = trial.suggest_int(dim_name, 128, 1024, step=128)
                    else:
                        # Subsequent layers (smaller than previous)
                        dim = trial.suggest_int(dim_name, 64, hidden_dims_trial[i-1], step=64)
                    
                    dropout = trial.suggest_float(dropout_name, 0.1, 0.5)
                    
                    hidden_dims_trial.append(dim)
                    dropout_rates_trial.append(dropout)
                
                # Backbone freezing
                freeze_backbone_trial = trial.suggest_categorical('freeze_backbone', [True, False])
                
                printLog(f"Trial {trial.number} for {perception_type}: Testing hyperparameters:")
                printLog(f"  Learning rate: {lr}")
                printLog(f"  Weight decay: {wd}")
                printLog(f"  LR decay factor: {lr_decay}")
                printLog(f"  Scheduler factor: {sched_factor}")
                printLog(f"  Scheduler patience: {sched_patience}")
                printLog(f"  Hidden layers: {num_hidden}")
                printLog(f"  Hidden dimensions: {hidden_dims_trial}")
                printLog(f"  Dropout rates: {dropout_rates_trial}")
                printLog(f"  Freeze backbone: {freeze_backbone_trial}")
                
                # Create model with these hyperparameters
                model = SinglePerceptionModel(
                    perception_type=perception_type,
                    ordinal_levels=num_categories,
                    hidden_layers=num_hidden,
                    hidden_dims=hidden_dims_trial,
                    dropout_rates=dropout_rates_trial,
                    freeze_backbone=freeze_backbone_trial
                ).to(device)
                
                # Define loss function
                criterion = nn.CrossEntropyLoss()
                
                # First do warmup training (2 epochs as specified)
                printLog(f"Warmup training for {warmup_epochs} epochs")
                train_single(model, train_loader, test_loader, criterion, 
                      wd, lr, patience=100,  # High patience to prevent early stopping
                      n_epochs=warmup_epochs, device=device, 
                      printLog=printLog, perception_name=perception_type,
                      optuna_trial=None,
                      lr_decay_factor=lr_decay,
                      scheduler_factor=sched_factor,
                      scheduler_patience=100)  # Prevent scheduler from adjusting during warmup
                
                # Then do the actual trial training (10 epochs as specified)
                printLog(f"Trial training for {trial_epochs} epochs")
                best_model, train_loss, test_loss = train_single(model, train_loader, test_loader, criterion, 
                                                       wd, lr, patience=trial_epochs,  # Set patience to max epochs
                                                       n_epochs=trial_epochs, device=device, 
                                                       printLog=printLog, perception_name=perception_type,
                                                       optuna_trial=trial,
                                                       lr_decay_factor=lr_decay,
                                                       scheduler_factor=sched_factor,
                                                       scheduler_patience=sched_patience)
                
                # Return minimum test loss
                return min(test_loss)
            
            # Create a study object and optimize
            printLog(f"Starting Optuna hyperparameter optimization for {perception_type} model")
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # Get best hyperparameters
            printLog(f"Best hyperparameters for {perception_type} model:")
            for key, value in study.best_params.items():
                printLog(f"{key}: {value}")
                model_params[key] = value
            
            # Update parameters with best values for final training
            learning_rate = model_params['learning_rate']
            weight_decay = model_params['weight_decay']
            lr_decay_factor = model_params['lr_decay_factor']
            scheduler_factor = model_params['scheduler_factor']
            scheduler_patience = model_params['scheduler_patience']
            hidden_layers = model_params['hidden_layers']
            freeze_backbone = model_params['freeze_backbone']
            
            # Reconstruct hidden_dims and dropout_rates from best trial
            hidden_dims = [model_params[f'hidden_dim_{i}'] for i in range(hidden_layers)]
            dropout_rates = [model_params[f'dropout_rate_{i}'] for i in range(hidden_layers)]
            model_params['hidden_dims'] = hidden_dims
            model_params['dropout_rates'] = dropout_rates
        
        # Log hyperparameters for final training
        printLog(f"Final training for {perception_type} model with hyperparameters:")
        for key, value in model_params.items():
            printLog(f"  {key}: {value}")
        
        # Create model with best hyperparameters
        model = SinglePerceptionModel(
            perception_type=perception_type,
            ordinal_levels=num_categories,
            hidden_layers=hidden_layers,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
            freeze_backbone=freeze_backbone
        ).to(device)
        
        # Loss function (CrossEntropyLoss for classification)
        criterion = nn.CrossEntropyLoss()
        
        # Train model with best hyperparameters for more epochs
        printLog(f"Final training for {perception_type} model for up to {final_epochs} epochs with early stopping (patience={patience})")
        best_model, train_loss, test_loss = train_single(model, train_loader, test_loader, criterion, 
                                               weight_decay, learning_rate, patience, final_epochs, 
                                               device, printLog, perception_name=perception_type,
                                               lr_decay_factor=lr_decay_factor,
                                               scheduler_factor=scheduler_factor,
                                               scheduler_patience=scheduler_patience)
        
        # Save loss history to CSV for later plotting
        loss_df = pd.DataFrame({
            'epoch': range(1, len(train_loss) + 1),
            'train_loss': train_loss,
            'test_loss': test_loss
        })
        loss_history_file = os.path.join(output_dir, f'{dateStr}_{perception_type}_loss_history.csv')
        loss_df.to_csv(loss_history_file, index=False)
        printLog(f"Loss history saved to {loss_history_file}")
        
        # Evaluate model on test set
        metrics = evaluate_model_single(best_model, test_loader, device, perception_type)
        
        # Print metrics
        printLog(f"\nEvaluation metrics for {perception_type} model:")
        for metric_name, value in metrics.items():
            printLog(f"{metric_name}: {value:.4f}")
        
        # Save metrics to JSON
        metrics_file = os.path.join(output_dir, f'{dateStr}_{perception_type}_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save best parameters to JSON
        params_file = os.path.join(output_dir, f'{dateStr}_{perception_type}_params.json')
        
        # Convert any numpy types to native Python types for JSON serialization
        best_params_json = {}
        for k, v in model_params.items():
            if isinstance(v, np.integer):
                best_params_json[k] = int(v)
            elif isinstance(v, np.floating):
                best_params_json[k] = float(v)
            elif isinstance(v, np.ndarray):
                best_params_json[k] = v.tolist()
            else:
                best_params_json[k] = v
        
        with open(params_file, 'w') as f:
            json.dump(best_params_json, f, indent=4)
        
        # Save model
        model_name = f'{dateStr}_{perception_type}_model_acc{metrics[f"{perception_type}_acc"]:.4f}.pt'
        model_path = os.path.join(output_dir, model_name)
        torch.save(best_model.state_dict(), model_path)
        printLog(f'Saved {perception_type} model to {model_path}')
        
        # Store results for this perception type
        best_models[perception_type] = best_model
        best_params[perception_type] = model_params
        all_metrics[perception_type] = metrics
    
    # Overall summary
    printLog("\n" + "=" * 50)
    printLog("TRAINING COMPLETE - SUMMARY OF RESULTS")
    printLog("=" * 50)
    
    # Print summary of all models
    for perception_type in perception_types:
        printLog(f"{perception_type} model - Accuracy: {all_metrics[perception_type][f'{perception_type}_acc']:.4f}")
    
    return best_models, all_metrics, best_params


if __name__ == "__main__":
    # Initialize paths
    working_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    output_dir = working_folder / 'output'
    
    # Determine paths based on OS
    sys_os = platform.system()
    if sys_os == 'Darwin':  # macOS
        img_path = Path('/Users/sandervancranenburgh/Documents/Repos_and_data/Data/bicycle_project_roos/images')
        data_file = Path(os.getcwd()) / 'data' / 'raw' / 'perceptionratings.csv'
    elif sys_os == 'Linux':
        img_path = Path('/srv/shared/bicycle_project_roos/images_scaled')
        data_file = Path(os.getcwd()) / 'data' / 'raw' / 'perceptionratings.csv'
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    weight_decay = 0.01
    patience = 10
    workers = 2
    num_categories = 5  # Use 3 or 5 categories
    use_optuna = True
    n_trials = 5  # Reduced trials per model since we're training three separate models
    
    # Specific settings from analysis outline
    warmup_epochs = 2     # 2 warmup epochs as specified
    trial_epochs = 8      # 8 epochs for each trial (slightly reduced)
    final_epochs = 50     # More epochs for final training after optimization
    
    # Architecture hyperparameters
    lr_decay_factor = 0.1
    scheduler_factor = 0.5
    scheduler_patience = 3
    hidden_layers = 2
    hidden_dims = [512, 256]
    dropout_rates = [0.3, 0.2]
    
    # Backbone hyperparameters
    freeze_backbone = False
    
    # Run training for separate models
    best_models, metrics, best_params = run_training_separate(
        data_file=data_file,
        img_path=img_path,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        workers=workers,
        num_categories=num_categories,
        use_optuna=use_optuna,
        n_trials=n_trials,
        warmup_epochs=warmup_epochs,
        trial_epochs=trial_epochs,
        final_epochs=final_epochs,
        lr_decay_factor=lr_decay_factor,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        hidden_layers=hidden_layers,
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates,
        freeze_backbone=freeze_backbone
    ) 