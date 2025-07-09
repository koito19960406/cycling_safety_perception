"""
Model Result Compilation and Analysis

This module contains functions for compiling, comparing, and analyzing results 
from multiple discrete choice models in the cycling safety project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import pickle


class ModelResultCompiler:
    """Class for compiling and analyzing multiple model results"""
    
    def __init__(self, output_dir='reports/models/compiled_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.metrics = {}
        
    def add_model_result(self, model_name, train_results, test_results=None, 
                        model_type='MXL', features_used=None):
        """
        Add a model result to the compilation
        
        Args:
            model_name: Name/identifier for the model
            train_results: Training results (Biogeme results object or metrics dict)
            test_results: Test results (simulation results or metrics dict) 
            model_type: Type of model ('MXL', 'MNL', etc.)
            features_used: List of features used in the model
        """
        self.results[model_name] = {
            'train_results': train_results,
            'test_results': test_results,
            'model_type': model_type,
            'features_used': features_used or [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Extract metrics
        self._extract_metrics(model_name, train_results, test_results, model_type)
        
    def _extract_metrics(self, model_name, train_results, test_results, model_type):
        """Extract standardized metrics from results"""
        
        train_metrics = {}
        test_metrics = {}
        
        # Extract training metrics
        if hasattr(train_results, 'data'):
            # Biogeme results object
            train_metrics = {
                'log_likelihood': train_results.data.logLike,
                'n_parameters': len(train_results.data.betaValues),
                'n_observations': train_results.data.numberOfObservations,
                'pseudo_r2': train_results.data.rhoSquare,
                'AIC': getattr(train_results.data, 'akaike', 
                              2 * len(train_results.data.betaValues) - 2 * train_results.data.logLike),
                'BIC': getattr(train_results.data, 'bayesianInformationCriterion',
                              np.log(train_results.data.numberOfObservations) * len(train_results.data.betaValues) - 2 * train_results.data.logLike)
            }
        elif isinstance(train_results, dict):
            # Already extracted metrics
            train_metrics = train_results.copy()
            
        # Extract test metrics  
        if test_results:
            if isinstance(test_results, dict):
                if 'LL' in test_results:
                    # MXL simulation results
                    test_metrics = {
                        'log_likelihood': test_results['LL'],
                        'pseudo_r2': test_results.get('rho_square', None),
                        'n_observations': None,  # Will be filled if available
                        'n_parameters': train_metrics.get('n_parameters', None)
                    }
                else:
                    # Pre-extracted metrics
                    test_metrics = test_results.copy()
                    
        self.metrics[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'model_type': model_type
        }
        
    def create_comparison_table(self, dataset='train', sort_by='log_likelihood', ascending=False):
        """
        Create comparison table for models
        
        Args:
            dataset: 'train', 'test', or 'both'
            sort_by: Metric to sort by
            ascending: Sort order
            
        Returns:
            DataFrame with model comparison
        """
        if dataset == 'both':
            return self._create_combined_table(sort_by, ascending)
        
        data = []
        for model_name, metrics in self.metrics.items():
            if dataset in metrics and metrics[dataset]:
                row = {'Model': model_name, 'Dataset': dataset.title()}
                row.update(metrics[dataset])
                row['Model_Type'] = metrics['model_type']
                
                # Add feature count if available
                if model_name in self.results and self.results[model_name]['features_used']:
                    row['N_Features'] = len(self.results[model_name]['features_used'])
                
                data.append(row)
                
        df = pd.DataFrame(data)
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
            
        return df
        
    def _create_combined_table(self, sort_by, ascending):
        """Create combined train/test comparison table"""
        train_df = self.create_comparison_table('train', sort_by, ascending)
        test_df = self.create_comparison_table('test', sort_by, ascending)
        
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        return combined_df.sort_values(['Model', 'Dataset'])
        
    def identify_best_models(self, metric='log_likelihood', dataset='test', top_n=5):
        """
        Identify best performing models
        
        Args:
            metric: Metric to use for ranking
            dataset: Dataset to evaluate on ('train' or 'test')
            top_n: Number of top models to return
            
        Returns:
            DataFrame with top models
        """
        df = self.create_comparison_table(dataset, metric, ascending=False)
        return df.head(top_n)
        
    def analyze_feature_importance(self):
        """
        Analyze which features contribute most to model performance
        
        Returns:
            Dictionary with feature analysis
        """
        feature_analysis = {}
        
        for model_name, result_info in self.results.items():
            features = result_info['features_used']
            if model_name in self.metrics and features:
                train_metrics = self.metrics[model_name]['train']
                
                for feature in features:
                    if feature not in feature_analysis:
                        feature_analysis[feature] = {
                            'models_using': [],
                            'avg_log_likelihood': [],
                            'avg_pseudo_r2': []
                        }
                    
                    feature_analysis[feature]['models_using'].append(model_name)
                    if 'log_likelihood' in train_metrics:
                        feature_analysis[feature]['avg_log_likelihood'].append(train_metrics['log_likelihood'])
                    if 'pseudo_r2' in train_metrics:
                        feature_analysis[feature]['avg_pseudo_r2'].append(train_metrics['pseudo_r2'])
        
        # Calculate averages
        for feature, data in feature_analysis.items():
            if data['avg_log_likelihood']:
                data['avg_log_likelihood'] = np.mean(data['avg_log_likelihood'])
            if data['avg_pseudo_r2']:
                data['avg_pseudo_r2'] = np.mean(data['avg_pseudo_r2'])
            data['frequency'] = len(data['models_using'])
            
        return feature_analysis
        
    def plot_model_comparison(self, metrics=['log_likelihood', 'pseudo_r2'], 
                             dataset='train', save_path=None):
        """
        Create visualization comparing models
        
        Args:
            metrics: List of metrics to plot
            dataset: Dataset to plot ('train' or 'test')
            save_path: Path to save plot (optional)
        """
        df = self.create_comparison_table(dataset)
        
        if df.empty:
            print(f"No data available for {dataset} dataset")
            return
            
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 8))
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i]
                
                # Create bar plot
                bars = ax.bar(range(len(df)), df[metric], alpha=0.7)
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} - {dataset.title()} Set')
                ax.set_xticks(range(len(df)))
                ax.set_xticklabels(df['Model'], rotation=45, ha='right')
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if not pd.isna(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            save_path = self.output_dir / f'model_comparison_{dataset}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
    def plot_train_vs_test_performance(self, metric='log_likelihood', save_path=None):
        """
        Plot train vs test performance to identify overfitting
        
        Args:
            metric: Metric to compare
            save_path: Path to save plot (optional)
        """
        train_df = self.create_comparison_table('train')
        test_df = self.create_comparison_table('test')
        
        if train_df.empty or test_df.empty:
            print("Need both train and test results for comparison")
            return
            
        # Merge train and test data
        merged = train_df.merge(test_df, on='Model', suffixes=('_train', '_test'))
        
        if f'{metric}_train' not in merged.columns or f'{metric}_test' not in merged.columns:
            print(f"Metric {metric} not available in both datasets")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.scatter(merged[f'{metric}_train'], merged[f'{metric}_test'], alpha=0.7)
        
        # Add diagonal line (perfect correlation)
        min_val = min(merged[f'{metric}_train'].min(), merged[f'{metric}_test'].min())
        max_val = max(merged[f'{metric}_train'].max(), merged[f'{metric}_test'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')
        
        # Labels and formatting
        plt.xlabel(f'{metric.replace("_", " ").title()} - Train')
        plt.ylabel(f'{metric.replace("_", " ").title()} - Test')
        plt.title(f'Train vs Test Performance: {metric.replace("_", " ").title()}')
        
        # Add model names as labels
        for i, row in merged.iterrows():
            plt.annotate(row['Model'], 
                        (row[f'{metric}_train'], row[f'{metric}_test']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.7)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = self.output_dir / f'train_vs_test_{metric}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        print(f"Plot saved to {save_path}")
        plt.show()
        
    def save_compilation_results(self, filename=None):
        """
        Save compilation results to files
        
        Args:
            filename: Base filename (without extension)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_compilation_{timestamp}"
            
        # Save comparison tables
        train_df = self.create_comparison_table('train')
        test_df = self.create_comparison_table('test')
        combined_df = self.create_comparison_table('both')
        
        train_df.to_csv(self.output_dir / f'{filename}_train.csv', index=False)
        test_df.to_csv(self.output_dir / f'{filename}_test.csv', index=False)
        combined_df.to_csv(self.output_dir / f'{filename}_combined.csv', index=False)
        
        # Save best models analysis
        if not test_df.empty:
            best_models = self.identify_best_models()
            best_models.to_csv(self.output_dir / f'{filename}_best_models.csv', index=False)
            
        # Save feature analysis
        feature_analysis = self.analyze_feature_importance()
        with open(self.output_dir / f'{filename}_feature_analysis.json', 'w') as f:
            json.dump(feature_analysis, f, indent=2, default=str)
            
        # Save metadata
        metadata = {
            'compilation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'n_models': len(self.results),
            'models_included': list(self.results.keys()),
            'output_directory': str(self.output_dir)
        }
        
        with open(self.output_dir / f'{filename}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Compilation results saved to {self.output_dir}")
        print(f"Files created:")
        print(f"  - {filename}_train.csv")
        print(f"  - {filename}_test.csv") 
        print(f"  - {filename}_combined.csv")
        print(f"  - {filename}_best_models.csv")
        print(f"  - {filename}_feature_analysis.json")
        print(f"  - {filename}_metadata.json")
        
    def create_summary_report(self, filename=None):
        """Create a comprehensive summary report"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_summary_report_{timestamp}.txt"
            
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            f.write("MODEL COMPILATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models: {len(self.results)}\n\n")
            
            # Best models section
            f.write("BEST PERFORMING MODELS (by test log-likelihood):\n")
            f.write("-" * 50 + "\n")
            
            test_df = self.create_comparison_table('test')
            if not test_df.empty:
                best_models = test_df.head(5)
                for i, (_, row) in enumerate(best_models.iterrows(), 1):
                    f.write(f"{i}. {row['Model']}\n")
                    f.write(f"   Log-likelihood: {row.get('log_likelihood', 'N/A'):.6f}\n")
                    f.write(f"   Pseudo R²: {row.get('pseudo_r2', 'N/A'):.6f}\n")
                    if 'n_parameters' in row:
                        f.write(f"   Parameters: {row['n_parameters']}\n")
                    f.write("\n")
            else:
                f.write("No test results available\n\n")
                
            # Model types summary
            f.write("MODEL TYPES SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            type_counts = {}
            for model_name, metrics in self.metrics.items():
                model_type = metrics['model_type']
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
                
            for model_type, count in type_counts.items():
                f.write(f"{model_type}: {count} models\n")
            f.write("\n")
            
            # Feature analysis
            f.write("MOST FREQUENTLY USED FEATURES:\n")
            f.write("-" * 50 + "\n")
            
            feature_analysis = self.analyze_feature_importance()
            sorted_features = sorted(feature_analysis.items(), 
                                   key=lambda x: x[1].get('frequency', 0), reverse=True)
            
            for feature, data in sorted_features[:10]:  # Top 10 features
                f.write(f"{feature}: used in {data.get('frequency', 0)} models\n")
                
        print(f"Summary report saved to {report_path}")
        return report_path


def load_model_results_from_pickle(pickle_path):
    """
    Load model results from a pickle file
    
    Args:
        pickle_path: Path to pickle file
        
    Returns:
        Loaded results object
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def compare_nested_models(model_1_results, model_2_results, alpha=0.05):
    """
    Perform likelihood ratio test for nested models
    
    Args:
        model_1_results: Results from more restrictive model
        model_2_results: Results from less restrictive model  
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    ll_1 = model_1_results.data.logLike if hasattr(model_1_results, 'data') else model_1_results['log_likelihood']
    ll_2 = model_2_results.data.logLike if hasattr(model_2_results, 'data') else model_2_results['log_likelihood']
    
    n_params_1 = len(model_1_results.data.betaValues) if hasattr(model_1_results, 'data') else model_1_results['n_parameters']
    n_params_2 = len(model_2_results.data.betaValues) if hasattr(model_2_results, 'data') else model_2_results['n_parameters']
    
    # Likelihood ratio test statistic
    lr_stat = 2 * (ll_2 - ll_1)
    df = n_params_2 - n_params_1
    
    # Critical value from chi-square distribution
    from scipy import stats
    critical_value = stats.chi2.ppf(1 - alpha, df)
    p_value = 1 - stats.chi2.cdf(lr_stat, df)
    
    return {
        'lr_statistic': lr_stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'critical_value': critical_value,
        'significant': lr_stat > critical_value,
        'alpha': alpha
    } 