import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import platform
import sys
import json
from pathlib import Path
from datetime import datetime
import optuna
import matplotlib.pyplot as plt
import logging
from copy import deepcopy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from perception_model import PerceptionModel
from perception_dataset import PerceptionDataset
from train import train, evaluate_model
from config_utils import setup_experiment
from wandb_utils import init_wandb, log_evaluation_metrics, log_loss_plot, log_hyperparameters, finish_wandb


def calculate_class_weights(dataset, num_categories):
    """
    Calculate class weights to deal with class imbalance
    
    Args:
        dataset: PerceptionDataset instance
        num_categories: Number of ordinal categories
        
    Returns:
        Dictionary of class weights tensors for each perception variable
    """
    perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
    weights = {}
    
    # Count class occurrences for each perception variable
    for var in perception_vars:
        # Initialize count array
        counts = np.zeros(num_categories)
        
        # Count occurrences
        for i in range(len(dataset)):
            label = dataset.annotations.iloc[i][f'{var}_cat']
            counts[int(label)] += 1
        
        # Calculate weights (inversely proportional to class frequency)
        class_weights = 1.0 / counts
        # Normalize weights
        class_weights = class_weights / np.sum(class_weights) * num_categories
        
        # Convert to tensor
        weights[var] = torch.FloatTensor(class_weights)
    
    return weights


def print_formatted_metrics(metrics, printLog):
    """
    Print evaluation metrics in a formatted way
    
    Args:
        metrics: Dictionary of evaluation metrics
        printLog: Function for logging
    """
    printLog("\nEvaluation metrics:")
    
    # Print accuracy metrics
    for metric_name in ['traffic_safety_acc', 'social_safety_acc', 'beautiful_acc', 'overall_acc']:
        printLog(f"{metric_name}: {metrics[metric_name]:.4f}")
    
    # Print per-class metrics for each perception variable
    perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
    
    for i, var in enumerate(perception_vars):
        printLog(f"\nPer-class metrics for {var}:")
        
        # Determine number of classes and labels
        num_classes = len(metrics['precision'][i])
        
        # Generate appropriate labels based on number of classes
        if num_classes == 3:
            labels = ["low", "medium", "high"]
        else:
            labels = [str(j+1) for j in range(num_classes)]
        
        # Print header
        printLog(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
        
        # Print metrics for each class
        for j in range(num_classes):
            printLog(
                f"{labels[j]:<10} "
                f"{metrics['precision'][i][j]:.4f}     "
                f"{metrics['recall'][i][j]:.4f}     "
                f"{metrics['f1_score'][i][j]:.4f}"
            )


def save_confusion_matrices(metrics, output_dir, timestamp):
    """
    Save confusion matrices as images
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save images
        timestamp: Timestamp string for file naming
    """
    perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
    
    for i, var in enumerate(perception_vars):
        # Get confusion matrix
        cm = metrics['confusion_matrices'][i]
        
        # Determine number of classes and labels
        num_classes = cm.shape[0]
        
        # Generate appropriate labels based on number of classes
        if num_classes == 3:
            labels = ["low", "medium", "high"]
        else:
            labels = [str(j+1) for j in range(num_classes)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=labels,
            yticklabels=labels,
            xlabel='Predicted label',
            ylabel='True label',
            title=f'Confusion Matrix - {var}'
        )
        
        # Rotate x tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for j in range(num_classes):
            for k in range(num_classes):
                ax.text(
                    k, j, f"{cm[j, k]}", 
                    ha="center", va="center",
                    color="white" if cm[j, k] > thresh else "black"
                )
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f"{timestamp}_confusion_matrix_{var}.png"))
        plt.close(fig)


def plot_loss_history(train_loss, val_loss, output_dir, timestamp):
    """
    Plot training and validation loss history
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        output_dir: Directory to save the plot
        timestamp: Timestamp for the filename
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{timestamp}_loss_history.png"))
    plt.close()


def run_training(config):
    """
    Train a perception prediction model
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing:
        - best_model: Trained model
        - metrics: Evaluation metrics
        - best_params: Best hyperparameters
        - loss_history: Training and validation loss history
    """
    # Extract paths from config
    data_file = config["paths"]["data_file"]
    img_path = config["paths"]["img_path"]
    output_dir = config["paths"]["output_dir"]
    
    # Log the paths for debugging
    print(f"Using data file: {data_file}")
    print(f"Using image path: {img_path}")
    print(f"Using output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract dataset parameters
    num_categories = config["dataset"]["num_categories"]
    
    # Extract training parameters
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]
    workers = config["training"]["workers"]
    num_epochs = config["training"]["num_epochs"]
    lr_decay_factor = config["training"]["lr_decay_factor"]
    scheduler_factor = config["training"]["scheduler_factor"]
    scheduler_patience = config["training"]["scheduler_patience"]
    grad_clip_value = config["training"]["grad_clip_value"]
    warmup_epochs = config["training"]["warmup_epochs"]
    
    # Extract warmup-specific parameters with defaults if they don't exist yet
    warmup_type = config["training"].get("warmup_type", "linear")
    warmup_initial_factor = config["training"].get("warmup_initial_factor", 0.1)
    
    # Extract model parameters
    hidden_layers = config["model"]["hidden_layers"]
    hidden_dims = config["model"]["hidden_dims"]
    dropout_rates = config["model"]["dropout_rates"]
    freeze_backbone = config["model"]["freeze_backbone"]
    
    # Extract additional model parameters with defaults
    backbone_dropout = config["model"].get("backbone_dropout", 0.0)
    stochastic_depth_rate = config["model"].get("stochastic_depth_rate", 0.0)
    
    # Extract dataset augmentation parameters
    mixup_alpha = config["dataset"].get("mixup_alpha", 0.0)
    cutmix_alpha = config["dataset"].get("cutmix_alpha", 0.0)
    
    # Extract training advanced parameters
    mixed_precision = config["training"].get("mixed_precision", False)
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    
    # Extract optuna parameters
    use_optuna = config["optuna"]["use_optuna"]
    n_trials = config["optuna"]["n_trials"]
    trial_epochs = config["optuna"]["trial_epochs"]
    final_epochs = config["optuna"]["final_epochs"]
    
    # Get timestamp
    timestamp = datetime.now().strftime("%H%M_%d_%m_%Y")

    # Initialize log file
    output_file = f'{timestamp}_perception_output.txt'
    log_file_path = os.path.join(output_dir, output_file)
    
    # Define logging function
    def printLog(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_file_path, 'a') as file:
            print(*args, **kwargs, file=file)
    
    # Display code started
    printLog("PERCEPTION MODEL TRAINING STARTED at:", timestamp)
    printLog(f"Number of categories: {num_categories} (using {'3 (low, medium, high)' if num_categories == 3 else '5 (1-5)'} classes)")

    # Initialize WandB if enabled
    wandb_run = init_wandb(config, name=f"perception_{timestamp}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    printLog(f'Using device: {device}')

    # Load data
    printLog(f'Loading data from {data_file}')
    
    dataset_train = PerceptionDataset(
        data_file=data_file, 
        img_path=img_path, 
        set_type='train', 
        transform=True, 
        num_categories=num_categories
    )
    
    dataset_val = PerceptionDataset(
        data_file=data_file, 
        img_path=img_path, 
        set_type='val', 
        transform=False, 
        num_categories=num_categories
    )
    
    dataset_test = PerceptionDataset(
        data_file=data_file, 
        img_path=img_path, 
        set_type='test', 
        transform=False, 
        num_categories=num_categories
    )
    
    printLog(f'Dataset loaded: {len(dataset_train)} training samples, '
             f'{len(dataset_val)} validation samples, '
             f'{len(dataset_test)} test samples')

    # Calculate class weights to handle imbalance
    class_weights = calculate_class_weights(dataset_train, num_categories)
    printLog("Class weights calculated for balancing:")
    for var, weights in class_weights.items():
        printLog(f"  {var}: {weights.numpy().tolist()}")

    # Create dataloaders
    train_loader = DataLoader(
        dataset=dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=dataset_val, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True
    )

    # Initialize best parameters dictionary
    best_params = {
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'lr_decay_factor': lr_decay_factor,
        'scheduler_factor': scheduler_factor,
        'scheduler_patience': scheduler_patience,
        'hidden_layers': hidden_layers,
        'hidden_dims': hidden_dims,
        'dropout_rates': dropout_rates,
        'freeze_backbone': freeze_backbone,
        'grad_clip_value': grad_clip_value
    }

    # Optuna hyperparameter optimization
    if use_optuna:
        printLog("Starting Optuna hyperparameter optimization")
        
        # Define the objective function for Optuna
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float('learning_rate', 5e-6, 5e-4, log=True)
            wd = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)
            lr_decay = trial.suggest_float('lr_decay_factor', 0.05, 0.5, log=True)
            sched_factor = trial.suggest_float('scheduler_factor', 0.1, 0.8)
            sched_patience = trial.suggest_int('scheduler_patience', 2, 5)
            grad_clip = trial.suggest_float('grad_clip_value', 0.5, 3.0)
            
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
                    # First hidden layer - reduced max size
                    dim = trial.suggest_int(dim_name, 128, 512, step=64)
                else:
                    # Subsequent layers (smaller than previous)
                    dim = trial.suggest_int(dim_name, 64, hidden_dims_trial[i-1], step=64)
                
                # Increased minimum dropout rate for regularization
                dropout = trial.suggest_float(dropout_name, 0.3, 0.6)
                
                hidden_dims_trial.append(dim)
                dropout_rates_trial.append(dropout)
            
            # Backbone freezing
            freeze_backbone_trial = trial.suggest_categorical('freeze_backbone', [True, False])
            
            printLog(f"Trial {trial.number}: Testing hyperparameters:")
            printLog(f"  Learning rate: {lr}")
            printLog(f"  Weight decay: {wd}")
            printLog(f"  LR decay factor: {lr_decay}")
            printLog(f"  Scheduler factor: {sched_factor}")
            printLog(f"  Scheduler patience: {sched_patience}")
            printLog(f"  Gradient clip value: {grad_clip}")
            printLog(f"  Hidden layers: {num_hidden}")
            printLog(f"  Hidden dimensions: {hidden_dims_trial}")
            printLog(f"  Dropout rates: {dropout_rates_trial}")
            printLog(f"  Freeze backbone: {freeze_backbone_trial}")
            
            # Create model with these hyperparameters
            model = PerceptionModel(
                num_classes=3,
                ordinal_levels=num_categories,
                hidden_layers=num_hidden,
                hidden_dims=hidden_dims_trial,
                dropout_rates=dropout_rates_trial,
                freeze_backbone=freeze_backbone_trial
            ).to(device)
            
            # Define loss function with class weights
            criterion = nn.CrossEntropyLoss(weight=class_weights['traffic_safety'].to(device))
            
            # First do warmup training
            printLog(f"Warmup training for {warmup_epochs} epochs")
            train(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                wd, 
                lr, 
                patience=100,  # High patience to prevent early stopping
                n_epochs=warmup_epochs, 
                device=device, 
                printLog=printLog, 
                optuna_trial=None,
                lr_decay_factor=lr_decay,
                scheduler_factor=sched_factor,
                scheduler_patience=100,  # Prevent scheduler from adjusting during warmup
                grad_clip_value=grad_clip,
                wandb_run=None  # No WandB logging during trials
            )
            
            # Then do the actual trial training
            printLog(f"Trial training for {trial_epochs} epochs")
            best_model, train_loss, val_loss = train(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                wd, 
                lr, 
                patience=trial_epochs,  # Set patience to max epochs
                n_epochs=trial_epochs, 
                device=device, 
                printLog=printLog, 
                optuna_trial=trial,
                lr_decay_factor=lr_decay,
                scheduler_factor=sched_factor,
                scheduler_patience=sched_patience,
                grad_clip_value=grad_clip,
                wandb_run=None  # No WandB logging during trials
            )
            
            # Check for overfitting
            final_train_loss = train_loss[-1]
            final_val_loss = val_loss[-1]
            
            # Calculate the overfitting penalty
            # If the gap between train and val loss is too large, we penalize the trial
            overfitting_penalty = 0
            if final_train_loss < final_val_loss * 0.7:
                overfitting_penalty = (final_val_loss - final_train_loss) / final_val_loss
                printLog(f"Applied overfitting penalty: {overfitting_penalty:.4f}")
            
            # Return minimum validation loss plus the overfitting penalty
            best_val_loss = min(val_loss)
            return best_val_loss + overfitting_penalty
        
        # Create a study object and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best hyperparameters
        printLog("Best hyperparameters:")
        for key, value in study.best_params.items():
            printLog(f"{key}: {value}")
            best_params[key] = value
        
        # Update parameters with best values for final training
        learning_rate = best_params['learning_rate']
        weight_decay = best_params['weight_decay']
        lr_decay_factor = best_params['lr_decay_factor']
        scheduler_factor = best_params['scheduler_factor']
        scheduler_patience = best_params['scheduler_patience']
        hidden_layers = best_params['hidden_layers']
        freeze_backbone = best_params['freeze_backbone']
        grad_clip_value = best_params['grad_clip_value']
        
        # Reconstruct hidden_dims and dropout_rates from best trial
        hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(hidden_layers)]
        dropout_rates = [best_params[f'dropout_rate_{i}'] for i in range(hidden_layers)]
        best_params['hidden_dims'] = hidden_dims
        best_params['dropout_rates'] = dropout_rates
    
    # Log hyperparameters for final training
    printLog("Final training with hyperparameters:")
    for key, value in best_params.items():
        printLog(f"  {key}: {value}")
        
    # Log to WandB if enabled
    if wandb_run is not None:
        log_hyperparameters(wandb_run, best_params)
    
    # Create model with best hyperparameters
    model = PerceptionModel(
        num_classes=3,
        ordinal_levels=num_categories,
        hidden_layers=hidden_layers,
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates,
        freeze_backbone=freeze_backbone,
        backbone_dropout=backbone_dropout,
        stochastic_depth_rate=stochastic_depth_rate
    ).to(device)
    
    # Loss function (CrossEntropyLoss with class weights and label smoothing if specified)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights['traffic_safety'].to(device),
        label_smoothing=label_smoothing
    )
    
    # First do warmup training for the final model
    if warmup_epochs > 0:
        printLog(f"Performing warmup training for {warmup_epochs} epochs using {warmup_type} scheduler")
        
        # Setup optimizer with initial smaller learning rate
        initial_lr = learning_rate * warmup_initial_factor
        warmup_optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay/2)  # Lower weight decay during warmup
        
        # Track learning rate to verify it's increasing
        lr_history = []
        
        # Train for warmup epochs
        model.train()
        for epoch in range(1, warmup_epochs + 1):
            # Calculate warmup factor (0 to 1) based on specified warmup type
            progress = epoch / warmup_epochs
            
            if warmup_type == "linear":
                # Linear warmup: factor increases linearly from 0 to 1
                warmup_factor = progress
            elif warmup_type == "exponential":
                # Exponential warmup: faster increase towards the end
                warmup_factor = progress ** 2
            elif warmup_type == "cosine":
                # Cosine warmup: smooth increase based on cosine curve
                import math
                warmup_factor = 0.5 * (1 - math.cos(math.pi * progress))
            else:
                # Default to linear if unknown type
                printLog(f"Warning: Unknown warmup type '{warmup_type}', using linear instead")
                warmup_factor = progress
            
            # Update learning rate for all parameter groups
            current_lr = initial_lr + (learning_rate - initial_lr) * warmup_factor
            for param_group in warmup_optimizer.param_groups:
                param_group['lr'] = current_lr
            
            lr_history.append(current_lr)
            printLog(f"Warmup Epoch {epoch}/{warmup_epochs}, LR: {current_lr:.6f}, Factor: {warmup_factor:.4f}")
            
            # Training for one epoch
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Get inputs and labels from batch dictionary
                images = batch['image'].to(device)
                
                # Create labels dictionary
                labels = {
                    'traffic_safety_cat': batch['traffic_safety'].to(device),
                    'social_safety_cat': batch['social_safety'].to(device),
                    'beautiful_cat': batch['beautiful'].to(device)
                }
                
                # Zero the parameter gradients
                warmup_optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                batch_loss = 0
                for i, var in enumerate(['traffic_safety', 'social_safety', 'beautiful']):
                    batch_loss += criterion(outputs[i], labels[f'{var}_cat'])
                
                # Backward pass and optimize
                batch_loss.backward()
                
                # Gradient clipping (if specified)
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                warmup_optimizer.step()
                
                # Update statistics
                running_loss += batch_loss.item()
            
            epoch_loss = running_loss / len(train_loader)
            printLog(f"Warmup Epoch {epoch}/{warmup_epochs}, Loss: {epoch_loss:.4f}")
            
            # Log to WandB if enabled
            if wandb_run:
                wandb_run.log({
                    "warmup_epoch": epoch,
                    "warmup_loss": epoch_loss,
                    "warmup_lr": current_lr,
                    "warmup_factor": warmup_factor
                })
        
        printLog(f"Warmup completed with {warmup_type} schedule")
        printLog(f"Learning rate progression: {lr_history[0]:.6f} → {lr_history[-1]:.6f}")
        printLog("Starting main training")
        
    # Train model with best hyperparameters for more epochs
    printLog(f"Final training for up to {final_epochs} epochs with early stopping (patience={patience})")
    best_model, train_loss, val_loss = train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        weight_decay, 
        learning_rate, 
        patience, 
        final_epochs, 
        device, 
        printLog,
        lr_decay_factor=lr_decay_factor,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        grad_clip_value=grad_clip_value,
        wandb_run=wandb_run,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mixed_precision=mixed_precision
    )
    
    # Save loss history to CSV for later plotting
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_loss) + 1),
        'train_loss': train_loss,
        'val_loss': val_loss
    })
    loss_history_file = os.path.join(output_dir, f'{timestamp}_loss_history.csv')
    loss_df.to_csv(loss_history_file, index=False)
    printLog(f"Loss history saved to {loss_history_file}")
    
    # Plot loss history
    plot_loss_history(train_loss, val_loss, output_dir, timestamp)
    
    # Log loss plot to WandB if enabled
    if wandb_run is not None:
        log_loss_plot(wandb_run, train_loss, val_loss)
    
    # Evaluate model on test set
    metrics = evaluate_model(best_model, test_loader, device)
    
    # Print metrics
    print_formatted_metrics(metrics, printLog)
    
    # Save confusion matrices
    save_confusion_matrices(metrics, output_dir, timestamp)
    
    # Log evaluation metrics to WandB if enabled
    if wandb_run is not None:
        log_evaluation_metrics(wandb_run, metrics)
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, f'{timestamp}_metrics.json')
    
    # Remove numpy arrays and matrices from metrics for JSON serialization
    metrics_json = {k: v for k, v in metrics.items() 
                   if k not in ['confusion_matrices', 'precision', 'recall', 'f1_score']}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save best parameters to JSON
    params_file = os.path.join(output_dir, f'{timestamp}_best_params.json')
    
    # Convert any numpy types to native Python types for JSON serialization
    best_params_json = {}
    for k, v in best_params.items():
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
    model_name = f'{timestamp}_PerceptionModel_acc{metrics["overall_acc"]:.4f}.pt'
    model_path = os.path.join(output_dir, model_name)
    torch.save(best_model.state_dict(), model_path)
    printLog(f'Saved model to {model_path}')
    
    # Finish WandB run if enabled
    if wandb_run is not None:
        finish_wandb(wandb_run)
    
    return best_model, metrics, best_params, (train_loss, val_loss)


if __name__ == "__main__":
    # Get configuration
    config = setup_experiment()
    
    # Initialize paths if not set in config
    if "paths" not in config:
        config["paths"] = {}
        
    # Set default paths based on OS
    # Get base paths
    working_folder = Path(os.path.dirname(os.path.realpath(__file__)))
    
    # Set default output_dir if not specified
    if "output_dir" not in config["paths"] or config["paths"]["output_dir"] is None:
        config["paths"]["output_dir"] = str(working_folder / 'output')
        
    # Set default data paths based on OS
    sys_os = platform.system()
    if sys_os == 'Darwin':  # macOS
        if "img_path" not in config["paths"] or config["paths"]["img_path"] is None:
            config["paths"]["img_path"] = str(Path('/Users/sandervancranenburgh/Documents/Repos_and_data/Data/bicycle_project_roos/images'))
        if "data_file" not in config["paths"] or config["paths"]["data_file"] is None:
            config["paths"]["data_file"] = str(Path(os.getcwd()).parent.parent.parent / 'data' / 'raw' / 'perceptionratings.csv')
    elif sys_os == 'Linux':
        if "img_path" not in config["paths"] or config["paths"]["img_path"] is None:
            config["paths"]["img_path"] = str(Path('/srv/shared/bicycle_project_roos/images_scaled'))
        if "data_file" not in config["paths"] or config["paths"]["data_file"] is None:
            config["paths"]["data_file"] = str(Path(os.getcwd()).parent.parent.parent / 'data' / 'raw' / 'perceptionratings.csv')
    
    # Validate that all required paths exist
    if not config["paths"].get("data_file"):
        raise ValueError("data_file path is not specified in config")
    
    if not config["paths"].get("img_path"):
        raise ValueError("img_path is not specified in config")
    
    if not config["paths"].get("output_dir"):
        raise ValueError("output_dir is not specified in config")
    
    # Run training
    best_model, metrics, best_params, loss_history = run_training(config) 