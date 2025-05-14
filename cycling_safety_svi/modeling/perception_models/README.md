# Perception Models

This module contains code for training perception models that predict perception variables (traffic safety, social safety, beautiful) from street view images.

## Recent Updates

- **YAML Configuration**: Moved all hyperparameters to YAML config files
- **Weights & Biases Integration**: Added WandB support for experiment tracking
- **Improved Evaluation**: Enhanced metrics reporting with confusion matrices and per-class metrics
- **Category Reduction**: Changed from 5 to 3 categories (low, medium, high) for better class balance and accuracy
- **Code Readability**: Refactored for better structure and readability
- **Parameter Organization**: Better separation of hyperparameters and experiment configurations

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

## Training a Model

The simplest way to train a model is to use the default configuration:

```bash
python main.py
```

This will use the default configuration from `configs/default_config.yaml`.

### Using Best Parameters

To train a model using pre-optimized parameters (skipping hyperparameter optimization):

```bash
python main.py --config configs/three_category_config.yaml --no_optuna
```

Or you can use the provided script:

```bash
./run_best_model.sh
```

### Custom Configuration

You can specify a custom configuration file:

```bash
python main.py --config configs/my_config.yaml
```

You can also override specific parameters:

```bash
python main.py --num_categories 3 --use_wandb
```

### Configuration Options

The following parameters can be configured:

#### Dataset

- `num_categories`: Number of categories (3 or 5)
- `train_ratio`: Proportion of data for training
- `val_ratio`: Proportion of data for validation
- `test_ratio`: Proportion of data for testing
- `seed`: Random seed

#### Training

- `batch_size`: Batch size for training
- `learning_rate`: Base learning rate
- `weight_decay`: Weight decay for regularization
- `patience`: Early stopping patience
- `workers`: Number of workers for data loading
- `num_epochs`: Maximum number of epochs
- `warmup_epochs`: Number of warmup epochs for Optuna trials
- `lr_decay_factor`: Factor for layer-wise learning rate decay
- `scheduler_factor`: Factor for learning rate scheduler
- `scheduler_patience`: Patience for learning rate scheduler
- `grad_clip_value`: Value for gradient clipping

#### Model Architecture

- `hidden_layers`: Number of hidden layers
- `hidden_dims`: List of hidden dimensions
- `dropout_rates`: List of dropout rates
- `freeze_backbone`: Whether to freeze the vision model backbone

#### Hyperparameter Optimization

- `use_optuna`: Whether to use Optuna for hyperparameter optimization
- `n_trials`: Number of Optuna trials
- `trial_epochs`: Number of epochs for each Optuna trial
- `final_epochs`: Number of epochs for final training

You can also bypass Optuna with the `--no_optuna` flag, which will use the parameters defined in the config file directly.

#### Logging and Output

- `use_wandb`: Whether to use Weights & Biases
- `wandb_project`: WandB project name
- `wandb_entity`: WandB entity (username/team)

## Output Files

The training process will generate the following output files in the output directory:

- `TIMESTAMP_perception_output.txt`: Training log
- `TIMESTAMP_loss_history.csv`: Training and validation loss history
- `TIMESTAMP_loss_history.png`: Plot of training and validation loss
- `TIMESTAMP_metrics.json`: Evaluation metrics
- `TIMESTAMP_best_params.json`: Best hyperparameters
- `TIMESTAMP_PerceptionModel_accXXXX.pt`: Trained model
- `TIMESTAMP_confusion_matrix_*.png`: Confusion matrices for each perception variable

## Using WandB

To enable WandB, set `use_wandb: true` in your config file or use the `--use_wandb` flag:

```bash
python main.py --use_wandb
```

You can also set your WandB project and entity in the config file:

```yaml
output:
  use_wandb: true
  wandb_project: "my-project"
  wandb_entity: "my-username"
```

## Understanding the 3-Category System

In the 3-category system, perception ratings are categorized as follows:

- **Low**: Ratings 1-2
- **Medium**: Rating 3
- **High**: Ratings 4-5

This provides a more balanced distribution of classes and focuses on the practical differences between low, medium, and high perception ratings. 