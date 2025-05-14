import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from datetime import datetime
from copy import deepcopy
import optuna
from perception_dataset import data_to_device
from typing import Dict, Any, Optional, List, Tuple, Callable
import random


def mixup_data(x, y_dict, alpha=1.0):
    """
    Applies mixup augmentation to the batch.
    
    Args:
        x: Input tensors [batch_size, ...]
        y_dict: Dictionary of label tensors
        alpha: Mixup interpolation coefficient
        
    Returns:
        Mixed inputs, dictionary of mixed targets, and lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    mixed_y = {}
    for key, y in y_dict.items():
        mixed_y[key] = y.clone()  # Create a copy to avoid modifying original labels
        
    return mixed_x, mixed_y, lam, index


def cutmix_data(x, y_dict, alpha=1.0):
    """
    Applies CutMix augmentation to the batch.
    
    Args:
        x: Input tensors [batch_size, ...]
        y_dict: Dictionary of label tensors
        alpha: CutMix interpolation coefficient
        
    Returns:
        Mixed inputs, dictionary of mixed targets, and lambda value
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Get dimensions
    _, c, h, w = x.shape
    
    # Calculate cut size and center position
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Ensure bounding box stays within image
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Create mixed image
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = mixed_x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to account for actual area of cut
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
    
    mixed_y = {}
    for key, y in y_dict.items():
        mixed_y[key] = y.clone()  # Create a copy to avoid modifying original labels
        
    return mixed_x, mixed_y, lam, index


def train(model, train_loader, val_loader, criterion, weight_decay, learning_rate, 
         patience, n_epochs, device, printLog, optuna_trial=None, 
         lr_decay_factor=0.1, scheduler_factor=0.5, scheduler_patience=5, 
         grad_clip_value=1.0, wandb_run=None, mixup_alpha=0.0, cutmix_alpha=0.0, 
         mixed_precision=False):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        weight_decay: Weight decay for optimizer
        learning_rate: Learning rate
        patience: Patience for early stopping
        n_epochs: Maximum number of epochs
        device: Device to use for training
        printLog: Logging function
        optuna_trial: Optuna trial object (for hyperparameter optimization)
        lr_decay_factor: Factor to decay learning rate on restart
        scheduler_factor: Factor to reduce learning rate when loss plateaus
        scheduler_patience: Patience for scheduler
        grad_clip_value: Gradient clipping value
        wandb_run: Weights & Biases run object
        mixup_alpha: Mixup alpha parameter (0 = disabled)
        cutmix_alpha: CutMix alpha parameter (0 = disabled)
        mixed_precision: Whether to use mixed precision training
        
    Returns:
        Tuple containing:
        - best_model: Model with best validation loss
        - train_loss: List of training losses
        - val_loss: List of validation losses
    """
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=scheduler_factor, 
        patience=scheduler_patience, 
        verbose=True
    )
    
    # Initialize lists to store loss
    train_loss = []
    val_loss = []
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    best_model = None
    early_stop_counter = 0
    
    # Training loop
    printLog(f"Starting perception model training")
    
    # Record start time
    timestamp = datetime.now().strftime("%H%M_%d_%m_%Y")
    
    for epoch in range(1, n_epochs + 1):
        # Train one epoch
        epoch_train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, 
            grad_clip_value=grad_clip_value,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixed_precision=mixed_precision
        )
        train_loss.append(epoch_train_loss)

        # Evaluate model
        epoch_val_loss = eval_epoch(model, val_loader, criterion, device, epoch)
        
        # Store losses
        val_loss.append(epoch_val_loss)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to WandB if enabled
        if wandb_run is not None:
            from wandb_utils import log_epoch_metrics
            log_epoch_metrics(
                wandb_run,
                epoch=epoch,
                train_loss=epoch_train_loss,
                val_loss=epoch_val_loss,
                best_val_loss=min(val_loss),
                learning_rate=current_lr,
            )
        
        # Model selection - save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model = deepcopy(model)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Print progress
        if epoch % 1 == 0:
            # Clear line
            sys.stdout.write('\r' + ' ' * 100 + '\r')
            sys.stdout.flush()
            
            # Get timestamp
            dateTimeObj = datetime.now()
            dateStr = dateTimeObj.strftime("%H%M_%d_%m_%Y")

            # Print progress
            printLog(f"{dateStr}\tEpoch {epoch:03d}  Train Loss | Val Loss | Best Val Loss\t{epoch_train_loss:0.3f} | {epoch_val_loss:0.3f} | {min(val_loss):0.3f} {'+' * early_stop_counter}")
            
            # Check if gap between train and validation loss is too large (sign of overfitting)
            if epoch_train_loss < epoch_val_loss * 0.7 and epoch > 5:  # If train loss is much lower than validation loss
                printLog(f"Detected potential overfitting: train_loss={epoch_train_loss:.3f}, val_loss={epoch_val_loss:.3f}")
                if early_stop_counter >= patience // 2:  # If we're already halfway to early stopping
                    printLog(f"Early stopping at epoch {epoch} due to overfitting")
                    break
            
        # Early stopping
        if early_stop_counter >= patience:
            printLog(f"Early stopping at epoch {epoch}")
            break
            
        # Report to Optuna for hyperparameter optimization if provided
        if optuna_trial is not None and epoch >= 2:  # Skip first epoch warmup
            optuna_trial.report(epoch_val_loss, epoch)
            
            # Handle pruning based on intermediate results
            if optuna_trial.should_prune():
                printLog("Pruning trial...")
                raise optuna.exceptions.TrialPruned()
    
    if best_model is None:
        best_model = model
        
    # Return best model and loss history
    return best_model, train_loss, val_loss


def train_epoch(model, data_loader, optimizer, criterion, device, epoch, grad_clip_value=1.0, 
               mixup_alpha=0.0, cutmix_alpha=0.0, mixed_precision=False):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        data_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        grad_clip_value: Value for gradient clipping
        mixup_alpha: Mixup alpha parameter (0 = disabled)
        cutmix_alpha: CutMix alpha parameter (0 = disabled)
        mixed_precision: Whether to use mixed precision training
        
    Returns:
        Average loss for this epoch
    """
    # Set model to training mode
    model.train()
    
    total_loss = 0.0
    batch_count = len(data_loader)
    
    # Set up mixed precision scaler if enabled
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    
    for batch_idx, batch in enumerate(data_loader):
        # Move batch to device
        images = batch['image'].to(device)
        
        # Create labels dictionary
        labels = {
            'traffic_safety_cat': batch['traffic_safety'].to(device),
            'social_safety_cat': batch['social_safety'].to(device),
            'beautiful_cat': batch['beautiful_cat'].to(device)
        }
        
        # Apply data augmentation randomly (with 50% probability for each if enabled)
        use_mixup = mixup_alpha > 0 and random.random() < 0.5
        use_cutmix = cutmix_alpha > 0 and random.random() < 0.5 and not use_mixup  # Don't use both at once
        
        # Save original labels for loss calculation
        original_labels = {k: v.clone() for k, v in labels.items()}
        
        # Apply mixup augmentation
        if use_mixup:
            images, _, lam, index = mixup_data(images, labels, mixup_alpha)
        
        # Apply cutmix augmentation
        if use_cutmix:
            images, _, lam, index = cutmix_data(images, labels, cutmix_alpha)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                
                # Calculate loss
                loss = 0
                for i, var in enumerate(['traffic_safety', 'social_safety', 'beautiful']):
                    # If using mixup or cutmix, interpolate between original and shuffled targets
                    if use_mixup or use_cutmix:
                        target_a = original_labels[f'{var}_cat']
                        target_b = original_labels[f'{var}_cat'][index]
                        curr_loss = lam * criterion(outputs[i], target_a) + (1 - lam) * criterion(outputs[i], target_b)
                    else:
                        curr_loss = criterion(outputs[i], original_labels[f'{var}_cat'])
                    
                    loss += curr_loss
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Clip gradients
            if grad_clip_value > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            outputs = model(images)
            
            # Calculate loss
            loss = 0
            for i, var in enumerate(['traffic_safety', 'social_safety', 'beautiful']):
                # If using mixup or cutmix, interpolate between original and shuffled targets
                if use_mixup or use_cutmix:
                    target_a = original_labels[f'{var}_cat']
                    target_b = original_labels[f'{var}_cat'][index]
                    curr_loss = lam * criterion(outputs[i], target_a) + (1 - lam) * criterion(outputs[i], target_b)
                else:
                    curr_loss = criterion(outputs[i], original_labels[f'{var}_cat'])
                
                loss += curr_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if grad_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
            # Optimizer step
            optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        
        # Print progress
        sys.stdout.write('\r')
        sys.stdout.write(f"Epoch {epoch:03d} -- Training | {batch_idx/batch_count*100:3.0f}% of batches completed")
        sys.stdout.flush()
    
    # Return average loss
    return total_loss / batch_count


def eval_epoch(model, data_loader, criterion, device, epoch):
    """
    Evaluate the model on the validation set
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for validation data
        criterion: Loss function
        device: Computation device
        epoch: Current epoch number
        
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
            progress_interval = max(1, len(data_loader) // min(50, len(data_loader)))
            if j % progress_interval == 0:
                progress = (j / len(data_loader)) * 100
                sys.stdout.write(f"\rEpoch {epoch:03d} -- Evaluation | {progress:3.0f}% of batches completed")
                sys.stdout.flush() 
            
            # Move data to device
            batch = data_to_device(batch, device)
            
            # Forward pass
            perceptions = model(batch['image'])
            
            # Calculate loss for each perception variable and sum them
            loss = 0
            for i, perception_name in enumerate(['traffic_safety', 'social_safety', 'beautiful']):
                loss += criterion(perceptions[i], batch[perception_name])
            
            # Update total loss and instance count
            total_loss += loss.item()
            num_instances += len(batch['image'])
    
    # Calculate average loss
    avg_loss = total_loss / num_instances
    
    return avg_loss


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance with metrics
    
    Args:
        model: PerceptionModel to evaluate
        data_loader: DataLoader for evaluation data
        device: Computation device
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    correct = [0, 0, 0]  # One counter for each perception variable
    total = 0
    
    # Track confusion matrices for each perception variable
    perception_names = ['traffic_safety', 'social_safety', 'beautiful']
    num_classes = None  # Will be determined from predictions
    
    # Initialize confusion matrices list
    confusion_matrices = None
    
    with torch.no_grad():
        for batch in data_loader:
            batch = data_to_device(batch, device)
            
            # Get predictions
            perceptions = model(batch['image'])
            
            # Initialize confusion matrices if needed
            if confusion_matrices is None:
                num_classes = perceptions[0].size(1)
                confusion_matrices = [
                    np.zeros((num_classes, num_classes), dtype=np.int32)
                    for _ in perception_names
                ]
            
            # Calculate accuracy and update confusion matrices for each perception variable
            for i, name in enumerate(perception_names):
                pred = torch.argmax(perceptions[i], dim=1)
                correct[i] += (pred == batch[name]).sum().item()
                
                # Update confusion matrix
                for j in range(len(pred)):
                    true_label = batch[name][j].item()
                    pred_label = pred[j].item()
                    confusion_matrices[i][true_label][pred_label] += 1
            
            total += len(batch['image'])
    
    # Calculate accuracy for each perception variable
    accuracy = [c / total for c in correct]
    
    # Calculate per-class precision and recall
    precision = []
    recall = []
    f1_score = []
    
    for cm in confusion_matrices:
        # Precision: TP / (TP + FP)
        prec = np.zeros(num_classes)
        for i in range(num_classes):
            if np.sum(cm[:, i]) > 0:
                prec[i] = cm[i, i] / np.sum(cm[:, i])
        precision.append(prec)
        
        # Recall: TP / (TP + FN)
        rec = np.zeros(num_classes)
        for i in range(num_classes):
            if np.sum(cm[i, :]) > 0:
                rec[i] = cm[i, i] / np.sum(cm[i, :])
        recall.append(rec)
        
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        f1 = np.zeros(num_classes)
        for i in range(num_classes):
            if prec[i] + rec[i] > 0:
                f1[i] = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        f1_score.append(f1)
    
    # Build results dictionary
    results = {
        'traffic_safety_acc': accuracy[0],
        'social_safety_acc': accuracy[1],
        'beautiful_acc': accuracy[2],
        'overall_acc': sum(accuracy) / 3,
        'confusion_matrices': confusion_matrices,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }
    
    return results 