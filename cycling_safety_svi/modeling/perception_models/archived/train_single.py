import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from datetime import datetime
from copy import deepcopy
import optuna
from perception_dataset import data_to_device


def train_single(model, train_loader, test_loader, criterion, wd, learning_rate, patience, n_epochs, 
          device, printLog, perception_name, optuna_trial=None, lr_decay_factor=0.1, scheduler_factor=0.5, 
          scheduler_patience=3):
    """
    Train a single perception prediction model
    
    Args:
        model: SinglePerceptionModel instance
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function (CrossEntropyLoss)
        wd: Weight decay parameter
        learning_rate: Base learning rate
        patience: Early stopping patience
        n_epochs: Maximum number of epochs
        device: Computation device
        printLog: Function for logging
        perception_name: Name of the perception variable to train for
        optuna_trial: Optional Optuna trial object for hyperparameter tuning
        lr_decay_factor: Factor for layer-wise learning rate decay (lower means more decay)
        scheduler_factor: Factor for reducing learning rate on plateau
        scheduler_patience: Number of epochs with no improvement before reducing LR
        
    Returns:
        best_model: The model with best validation performance
        train_loss: List of training losses
        val_loss: List of validation losses
    """
    # Report progress
    printLog(f"Starting {perception_name} perception model training")

    # Initialize optimizer with layer-wise learning rate decay
    param_groups = []
    
    # Vision model gets lower learning rate based on lr_decay_factor
    param_groups.append({'params': model.vision_model.parameters(), 'lr': learning_rate * lr_decay_factor, 'weight_decay': wd})
    
    # Perception head gets the base learning rate
    param_groups.append({'params': model.perception_head.parameters(), 'lr': learning_rate, 'weight_decay': wd})
    
    optimizer = optim.AdamW(param_groups, lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, 
        patience=scheduler_patience
    )

    # Initialize lists and variables for tracking performance
    train_loss_all, test_loss_all = [], []
    best_test_loss = np.inf
    counter = 0
    best_model = None

    # Train model
    for epoch in range(1, n_epochs + 1):
        
        # Train for one epoch
        train_loss = train_epoch_single(model, train_loader, optimizer, criterion, device, epoch, perception_name)
        train_loss_all.append(train_loss)

        # Evaluate on test set
        test_loss = eval_epoch_single(model, test_loader, criterion, device, epoch, perception_name)
        test_loss_all.append(test_loss)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Model selection - save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = deepcopy(model)
            counter = 0
        else:
            counter += 1
        
        # Print progress
        if epoch % 1 == 0:
            # Clear line
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.flush()
            
            # Get timestamp
            dateTimeObj = datetime.now()
            dateStr = dateTimeObj.strftime("%H%M_%d_%m_%Y")

            # Print progress
            printLog(f"{dateStr}\t{perception_name} - Epoch {epoch:03d}  Train Loss | Test Loss | Best Test Loss\t{train_loss:0.3f} | {test_loss:0.3f} | {min(test_loss_all):0.3f} {'+' * counter}")
            
        # Early stopping
        if counter >= patience:
            printLog(f"Early stopping at epoch {epoch}")
            break
            
        # Report to Optuna for hyperparameter optimization if provided
        if optuna_trial is not None and epoch >= 2:  # Skip first epoch warmup
            optuna_trial.report(test_loss, epoch)
            
            # Handle pruning based on intermediate results
            if optuna_trial.should_prune():
                printLog("Pruning trial...")
                raise optuna.exceptions.TrialPruned()
    
    if best_model is None:
        best_model = model
        
    # Return best model and loss history
    return best_model, train_loss_all, test_loss_all


def train_epoch_single(model, data_loader, optimizer, criterion, device, epoch, perception_name):
    """
    Train the single perception model for one epoch
    
    Args:
        model: Model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        perception_name: Name of the perception variable ('traffic_safety', 'social_safety', or 'beautiful')
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    # Set model to training mode
    model.train()
    
    # Initialize variables
    total_loss = 0
    num_instances = 0

    # Iterate over batches
    for j, batch in enumerate(data_loader):
        # Show progress every few batches
        if j % (len(data_loader) // np.min((20, len(data_loader)))) == 0:
            sys.stdout.write(f"\r{perception_name} - Epoch {epoch:03d} -- Training | {(j / len(data_loader)) * 100:3.0f}% of batches completed")
            sys.stdout.flush() 

        # Move data to device
        batch = data_to_device(batch, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch['image'])
        
        # Calculate loss
        loss = criterion(output, batch[perception_name])
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update total loss and instance count
        total_loss += loss.item()
        num_instances += len(batch['image'])
    
    # Calculate average loss
    avg_loss = total_loss / num_instances
    
    # Report
    sys.stdout.write(f"\r{perception_name} - Train loss epoch {avg_loss:0.3f} {' ' * 60} ")
    sys.stdout.flush()
    
    return avg_loss


def eval_epoch_single(model, data_loader, criterion, device, epoch, perception_name):
    """
    Evaluate the single perception model on the validation set
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        perception_name: Name of the perception variable ('traffic_safety', 'social_safety', or 'beautiful')
        
    Returns:
        avg_loss: Average loss for the epoch
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    total_loss = 0
    num_instances = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        # Iterate over batches
        for j, batch in enumerate(data_loader):
            # Show progress every few batches
            if j % (len(data_loader) // np.min((20, len(data_loader)))) == 0:
                sys.stdout.write(f"\r{perception_name} - Epoch {epoch:03d} -- Evaluation | {(j / len(data_loader)) * 100:3.0f}% of batches completed")
                sys.stdout.flush() 
            
            # Move data to device
            batch = data_to_device(batch, device)
            
            # Forward pass
            output = model(batch['image'])
            
            # Calculate loss
            loss = criterion(output, batch[perception_name])
            
            # Update total loss and instance count
            total_loss += loss.item()
            num_instances += len(batch['image'])
    
    # Calculate average loss
    avg_loss = total_loss / num_instances
    
    return avg_loss


def evaluate_model_single(model, data_loader, device, perception_name):
    """
    Evaluate single perception model performance with metrics
    
    Args:
        model: SinglePerceptionModel to evaluate
        data_loader: DataLoader for evaluation data
        device: Computation device
        perception_name: Name of the perception variable ('traffic_safety', 'social_safety', or 'beautiful')
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = data_to_device(batch, device)
            
            # Get predictions
            output = model(batch['image'])
            pred = torch.argmax(output, dim=1)
            correct += (pred == batch[perception_name]).sum().item()
            total += len(batch['image'])
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Return results
    return {
        f'{perception_name}_acc': accuracy,
    } 