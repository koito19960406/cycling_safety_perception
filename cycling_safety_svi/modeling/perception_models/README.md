# Perception Models

This module contains code for training perception models that predict perception variables (traffic safety, social safety, beautiful) from street view images.

## Recent Updates

- **ConvNextV2 Support**: Added support for the Facebook ConvNextV2 model with feature extraction
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

### Using Scripts (Recommended)

The easiest way to train a model is to use the provided scripts in the `scripts` directory:

```bash
# Run RepVit model
./scripts/run_repvit.sh

# Run ConvNextV2 model
./scripts/run_convnextv2.sh

# Run augmented model
./scripts/run_augmented.sh

# Or use any config file directly
./scripts/run_model.sh your_config.yaml
```

These scripts automatically handle path setup, config copying, and more. See `scripts/README.md` for details.

### Manual Training

Alternatively, you can run the training manually:

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

- `model_type`: Type of model to use ("deit_base" or "convnextv2_tiny")
- `hidden_layers`: Number of hidden layers
- `hidden_dims`: List of hidden dimensions
- `dropout_rates`: List of dropout rates
- `freeze_backbone`: Whether to freeze the vision model backbone
- `backbone_dropout`: Dropout rate for backbone features (only for ConvNextV2)
- `stochastic_depth_rate`: Stochastic depth rate for regularization (only for ConvNextV2)

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

- **Low**: Rating 1
- **Medium**: Ratings 2-3-4
- **High**: Rating 5

This provides a clearer distinction between extreme ratings (1 and 5) and more neutral ratings (2-4), focusing on the most significant differences in perception.

### Model Types

The following model types are supported:

- **deit_base**: The default model using DeiT (Data-efficient Image Transformer) as backbone
- **convnextv2_tiny**: ConvNextV2 model from Facebook AI Research with feature extraction from intermediate layers

To use the ConvNextV2 model with feature extraction:

```bash
./run_convnextv2_model.sh
```

Or manually with:

```bash
python main.py --config configs/your_config.yaml --model_type convnextv2_tiny
``` 