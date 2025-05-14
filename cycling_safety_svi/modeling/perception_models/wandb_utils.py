"""
Weights & Biases integration utilities for perception models
"""
import os
import json
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


def init_wandb(
    config: Dict[str, Any], 
    project: str = "cycling-perception", 
    entity: Optional[str] = None,
    name: Optional[str] = None,
) -> Optional[wandb.run]:
    """Initialize Weights & Biases
    
    Args:
        config: Configuration dictionary
        project: WandB project name
        entity: WandB entity (username/team name)
        name: Run name (optional)
        
    Returns:
        WandB run object if initialized, None otherwise
    """
    # Check if WandB is enabled
    use_wandb = config["output"].get("use_wandb", False)
    if not use_wandb:
        return None
    
    # Get project and entity from config if specified
    project = config["output"].get("wandb_project", project)
    entity = config["output"].get("wandb_entity", entity)
    
    # Generate a run name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%H%M_%d_%m_%Y")
        name = f"perception_{timestamp}"
    
    # Initialize WandB
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
    )
    
    return run


def log_epoch_metrics(
    run: wandb.run,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    learning_rate: float,
) -> None:
    """Log epoch metrics to WandB
    
    Args:
        run: WandB run object
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        best_val_loss: Best validation loss so far
        learning_rate: Current learning rate
    """
    if run is None:
        return
    
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "learning_rate": learning_rate,
    }
    
    run.log(metrics)


def log_evaluation_metrics(
    run: wandb.run,
    metrics: Dict[str, float],
) -> None:
    """Log evaluation metrics to WandB
    
    Args:
        run: WandB run object
        metrics: Dictionary of evaluation metrics
    """
    if run is None:
        return
    
    run.log(metrics)


def log_loss_plot(
    run: wandb.run,
    train_loss: List[float],
    val_loss: List[float],
) -> None:
    """Create and log a loss plot to WandB
    
    Args:
        run: WandB run object
        train_loss: List of training losses
        val_loss: List of validation losses
    """
    if run is None:
        return
    
    # Create figure
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot training and validation loss
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, 'b-', label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    # Add labels and legend
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Log to WandB
    run.log({"loss_plot": wandb.Image(fig)})
    
    # Close the figure to free memory
    plt.close(fig)


def log_hyperparameters(
    run: wandb.run,
    best_params: Dict[str, Any],
) -> None:
    """Log best hyperparameters to WandB
    
    Args:
        run: WandB run object
        best_params: Dictionary of best hyperparameters
    """
    if run is None:
        return
    
    # Convert any numpy types to native Python types
    best_params_dict = {}
    for k, v in best_params.items():
        if isinstance(v, np.integer):
            best_params_dict[k] = int(v)
        elif isinstance(v, np.floating):
            best_params_dict[k] = float(v)
        elif isinstance(v, np.ndarray):
            best_params_dict[k] = v.tolist()
        else:
            best_params_dict[k] = v
    
    # Log to WandB
    run.summary.update(best_params_dict)


def finish_wandb(run: Optional[wandb.run]) -> None:
    """Finish WandB run
    
    Args:
        run: WandB run object
    """
    if run is not None:
        run.finish() 