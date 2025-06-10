"""
Main script to run GOLEM-DC model for cycling safety choice modeling

This script:
1. Loads and prepares data (choice data + safety scores + segmentation)
2. Trains GOLEM-DC model with joint optimization
3. Evaluates model performance
4. Visualizes causal relationships
5. Compares with baseline discrete choice model
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime

from golem_dc_model import GOLEMDCModel
from golem_dc_data import GOLEMDCDataLoader
from golem_dc_trainer import GOLEMDCTrainer


def load_baseline_results(baseline_path):
    """
    Load baseline discrete choice model results
    
    Args:
        baseline_path: Path to baseline results file
        
    Returns:
        Dictionary with baseline metrics
    """
    if os.path.exists(baseline_path):
        # Load baseline predictions and compute metrics
        baseline_df = pd.read_csv(baseline_path)
        
        # Assuming the file has columns like 'actual', 'predicted', 'utility_1', 'utility_2'
        if all(col in baseline_df.columns for col in ['CHOICE', 'V_img1', 'V_img2']):
            # Compute probabilities from utilities using logit
            exp_v1 = np.exp(baseline_df['V_img1'])
            exp_v2 = np.exp(baseline_df['V_img2'])
            prob_1 = exp_v1 / (exp_v1 + exp_v2)
            prob_2 = exp_v2 / (exp_v1 + exp_v2)
            
            # Predictions (1-indexed)
            predictions = (prob_2 > prob_1).astype(int) + 1
            actual = baseline_df['CHOICE']
            
            # Accuracy
            accuracy = (predictions == actual).mean()
            
            # Log-likelihood
            log_likelihood = 0
            for i in range(len(baseline_df)):
                if actual.iloc[i] == 1:
                    log_likelihood += np.log(prob_1.iloc[i] + 1e-10)
                else:
                    log_likelihood += np.log(prob_2.iloc[i] + 1e-10)
            
            # Pseudo R²
            null_log_likelihood = len(baseline_df) * np.log(0.5)
            pseudo_r2 = 1 - (log_likelihood / null_log_likelihood)
            
            return {
                'accuracy': accuracy,
                'log_likelihood': log_likelihood,
                'avg_log_likelihood': log_likelihood / len(baseline_df),
                'pseudo_r2': pseudo_r2,
                'n_samples': len(baseline_df)
            }
    
    return None


def main(args):
    """
    Main function to run GOLEM-DC pipeline
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'golem_dc_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print("=" * 80)
    print("GOLEM-DC: Joint Causal Discovery and Choice Modeling")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    data_loader = GOLEMDCDataLoader(
        choice_data_path=args.choice_data,
        safety_scores_path=args.safety_scores,
        segmentation_path=args.segmentation_data,
        selected_seg_features=args.seg_features
    )
    
    # Create data loaders
    train_loader, test_loader, feature_names, scaler = data_loader.create_dataloaders(
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        random_state=args.seed
    )
    
    print(f"   - Number of features: {len(feature_names)}")
    print(f"   - Training samples: {len(train_loader.dataset)}")
    print(f"   - Test samples: {len(test_loader.dataset)}")
    print(f"   - Features: {', '.join(feature_names)}")
    
    # Initialize model with improved features
    print("\n2. Initializing GOLEM-DC model...")
    # Determine segmentation start index (after TT, TL, safety_score)
    segmentation_start_idx = 3 if args.standardize_segmentation else None
    
    model = GOLEMDCModel(
        n_features=len(feature_names),
        hidden_dim=args.hidden_dim,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3,
        standardize=args.standardize,
        segmentation_start_idx=segmentation_start_idx
    )
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    print(f"   - Using device: {device}")
    trainer = GOLEMDCTrainer(model, device=device)
    
    # Train model
    print("\n3. Training model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader if args.use_validation else None,
        n_epochs=args.n_epochs,
        lr=args.learning_rate,
        patience=args.patience,
        verbose=args.verbose
    )
    
    # Evaluate on test set
    print("\n4. Evaluating model...")
    test_metrics = trainer.evaluate(test_loader)
    
    print("\nTest Set Performance:")
    print(f"   - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   - Log-likelihood: {test_metrics['log_likelihood']:.2f}")
    print(f"   - Average log-likelihood: {test_metrics['avg_log_likelihood']:.4f}")
    print(f"   - AIC: {test_metrics['aic']:.2f}")
    print(f"   - BIC: {test_metrics['bic']:.2f}")
    print(f"   - Pseudo R²: {test_metrics['pseudo_r2']:.4f}")
    print(f"   - Number of parameters: {test_metrics['n_parameters']}")
    
    # Load and compare with baseline if available
    if args.baseline_results:
        print("\n5. Comparing with baseline...")
        baseline_metrics = load_baseline_results(args.baseline_results)
        
        if baseline_metrics:
            print("\nBaseline Discrete Choice Model:")
            print(f"   - Accuracy: {baseline_metrics['accuracy']:.4f}")
            print(f"   - Log-likelihood: {baseline_metrics['log_likelihood']:.2f}")
            print(f"   - Average log-likelihood: {baseline_metrics['avg_log_likelihood']:.4f}")
            print(f"   - Pseudo R²: {baseline_metrics['pseudo_r2']:.4f}")
            
            print("\nImprovement over baseline:")
            print(f"   - Accuracy: {(test_metrics['accuracy'] - baseline_metrics['accuracy']):.4f}")
            print(f"   - Log-likelihood: {(test_metrics['log_likelihood'] - baseline_metrics['log_likelihood']):.2f}")
            print(f"   - Pseudo R²: {(test_metrics['pseudo_r2'] - baseline_metrics['pseudo_r2']):.4f}")
            
            # Save comparison
            comparison = {
                'golem_dc': test_metrics,
                'baseline': baseline_metrics,
                'improvement': {
                    'accuracy': test_metrics['accuracy'] - baseline_metrics['accuracy'],
                    'log_likelihood': test_metrics['log_likelihood'] - baseline_metrics['log_likelihood'],
                    'pseudo_r2': test_metrics['pseudo_r2'] - baseline_metrics['pseudo_r2']
                }
            }
            with open(os.path.join(output_dir, 'model_comparison.json'), 'w') as f:
                json.dump(comparison, f, indent=4)
    
    # Visualize results
    print("\n6. Visualizing results...")
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # Plot causal structure
    trainer.plot_causal_structure(
        feature_names=feature_names,
        threshold=args.causal_threshold,
        save_path=os.path.join(output_dir, 'causal_structure.png')
    )
    
    # Save all results
    print("\n7. Saving results...")
    trainer.save_results(output_dir, feature_names, test_metrics)
    
    # Create summary report
    create_summary_report(output_dir, feature_names, test_metrics, trainer.model)
    
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 80)


def create_summary_report(output_dir, feature_names, test_metrics, model):
    """
    Create a summary report of the results
    """
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("GOLEM-DC Model Summary Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"- Number of features: {len(feature_names)}\n")
        f.write(f"- Hidden dimension: {model.utility_layer1.out_features}\n")
        f.write(f"- Lambda 1 (L1 penalty on adjacency): {model.lambda_1}\n")
        f.write(f"- Lambda 2 (DAG penalty): {model.lambda_2}\n")
        f.write(f"- Lambda 3 (L1 penalty weight for utility network): {model.lambda_3}\n")
        f.write(f"- Standardization enabled: {model.standardize}\n\n")
        
        f.write("Features Used:\n")
        for i, feat in enumerate(feature_names):
            f.write(f"  {i+1}. {feat}\n")
        f.write("\n")
        
        f.write("Test Performance:\n")
        for metric, value in test_metrics.items():
            f.write(f"- {metric}: {value}\n")
        f.write("\n")
        
        f.write("Causal Structure Analysis:\n")
        causal_matrix = model.get_causal_matrix()
        threshold = 0.1
        
        # Find significant causal relationships
        edges = []
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                if i != j and abs(causal_matrix[i, j]) > threshold:
                    edges.append((feature_names[i], feature_names[j], causal_matrix[i, j]))
        
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
        
        f.write(f"Significant causal relationships (threshold={threshold}):\n")
        for from_feat, to_feat, strength in edges[:20]:
            f.write(f"  {from_feat} → {to_feat}: {strength:.3f}\n")
        
        if not edges:
            f.write("  No significant causal relationships found.\n")
        
        f.write(f"\nTotal number of causal edges: {len(edges)}\n")
        f.write(f"Graph sparsity: {1 - len(edges) / (len(feature_names) ** 2):.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GOLEM-DC model for cycling safety choice modeling")
    
    # Data arguments
    parser.add_argument('--choice_data', type=str, 
                        default='data/raw/cv_dcm.csv',
                        help='Path to choice data CSV')
    parser.add_argument('--safety_scores', type=str,
                        default='data/processed/predicted_danish/cycling_safety_scores.csv',
                        help='Path to safety scores CSV')
    parser.add_argument('--segmentation_data', type=str,
                        default='data/processed/segmentation_results/pixel_ratios.csv',
                        help='Path to segmentation pixel ratios CSV')
    parser.add_argument('--baseline_results', type=str,
                        default='data/raw/df_choice_with_Vimg.csv',
                        help='Path to baseline model results')
    parser.add_argument('--seg_features', nargs='+', type=str,
                        default=['Road', 'Sidewalk', 'Bike Lane', 'Car', 'Bicycle', 
                                'Bicyclist', 'Building', 'Vegetation'],
                        help='Segmentation features to use')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for utility network')
    parser.add_argument('--lambda_1', type=float, default=0.01,
                        help='L1 penalty weight for adjacency matrix')
    parser.add_argument('--lambda_2', type=float, default=1.0,
                        help='DAG constraint penalty weight')
    parser.add_argument('--lambda_3', type=float, default=0.001,
                        help='L1 penalty weight for utility network')
    
    # Standardization arguments
    parser.add_argument('--standardize', action='store_true', default=True,
                        help='Whether to standardize features')
    parser.add_argument('--standardize_segmentation', action='store_true', default=True,
                        help='Whether to standardize only segmentation features')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion of data for training')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use validation set during training')
    
    # Other arguments
    parser.add_argument('--output_dir', type=str, default='reports/models',
                        help='Output directory for results')
    parser.add_argument('--causal_threshold', type=float, default=0.1,
                        help='Threshold for displaying causal edges')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output during training')
    
    args = parser.parse_args()
    main(args) 