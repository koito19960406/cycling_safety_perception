"""
Post-Causal Discovery Analysis for GOLEM-DC

This module performs post-hoc analysis of the learned causal structure:
1. Runs structural equation modeling to estimate causal coefficients
2. Performs discrete choice modeling conditioned on causal structure
3. Computes p-values for significance testing
4. Generates LaTeX tables and causal graphs
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import warnings
warnings.filterwarnings('ignore')


class PostCausalAnalysis:
    """
    Post-hoc analysis of GOLEM-DC causal structure
    """
    
    def __init__(self, model_path, data_loader=None):
        """
        Initialize post-causal analysis
        
        Args:
            model_path: Path to saved GOLEM-DC model
            data_loader: Optional data loader for analysis
        """
        # Load model and metadata
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model_state = checkpoint['model_state_dict']
        self.hyperparameters = checkpoint['hyperparameters']
        self.feature_names = checkpoint['feature_names']
        self.n_features = checkpoint['n_features']
        
        # Extract causal adjacency matrix
        self.adjacency_matrix = self._extract_adjacency_matrix()
        
        # Data for analysis
        self.data_loader = data_loader
        self.X_data = None
        self.y_data = None
        
    def _extract_adjacency_matrix(self):
        """Extract causal adjacency matrix from model state"""
        adjacency = self.model_state['adjacency'].numpy()
        # Mask diagonal
        adjacency = adjacency * (1 - np.eye(self.n_features))
        return adjacency
    
    def load_data(self, X, y):
        """
        Load data for analysis
        
        Args:
            X: Features array (n_samples, n_alternatives, n_features)
            y: Choices array (n_samples,)
        """
        self.X_data = X
        self.y_data = y
    
    def structural_equation_modeling(self, threshold=0.1):
        """
        Perform structural equation modeling to estimate causal coefficients
        
        Args:
            threshold: Threshold for considering causal links
            
        Returns:
            Dictionary with SEM results
        """
        # Identify significant causal links
        significant_links = np.abs(self.adjacency_matrix) > threshold
        
        # Results storage
        sem_results = {
            'equations': {},
            'coefficients': {},
            'p_values': {},
            'r_squared': {}
        }
        
        # Flatten data for SEM (use mean across alternatives)
        X_flat = np.mean(self.X_data, axis=1)  # (n_samples, n_features)
        
        # Fit structural equations for each endogenous variable
        for i in range(self.n_features):
            # Find parents of variable i
            parents = np.where(significant_links[:, i])[0]
            
            if len(parents) > 0:
                # Prepare data
                X_parents = X_flat[:, parents]
                y_var = X_flat[:, i]
                
                # Add constant
                X_parents_const = sm.add_constant(X_parents)
                
                # Fit OLS regression
                model = sm.OLS(y_var, X_parents_const)
                results = model.fit()
                
                # Store results
                var_name = self.feature_names[i]
                sem_results['equations'][var_name] = {
                    'parents': [self.feature_names[p] for p in parents],
                    'model': results
                }
                sem_results['coefficients'][var_name] = dict(zip(
                    ['const'] + [self.feature_names[p] for p in parents],
                    results.params
                ))
                sem_results['p_values'][var_name] = dict(zip(
                    ['const'] + [self.feature_names[p] for p in parents],
                    results.pvalues
                ))
                sem_results['r_squared'][var_name] = results.rsquared
        
        return sem_results
    
    def discrete_choice_analysis(self, sem_results):
        """
        Perform discrete choice analysis with causal structure
        
        Args:
            sem_results: Results from structural equation modeling
            
        Returns:
            Discrete choice model results
        """
        # Prepare data for choice modeling
        n_samples = self.X_data.shape[0]
        n_alts = self.X_data.shape[1]
        
        # Create choice-specific data
        choice_data = []
        for i in range(n_samples):
            for j in range(n_alts):
                row = {
                    'sample_id': i,
                    'alt_id': j,
                    'chosen': int(self.y_data[i] == j)
                }
                # Add features
                for k, feat_name in enumerate(self.feature_names):
                    row[feat_name] = self.X_data[i, j, k]
                choice_data.append(row)
        
        df_choice = pd.DataFrame(choice_data)
        
        # Fit MNLogit model
        # Use all features as explanatory variables
        y = df_choice.groupby('sample_id')['chosen'].apply(lambda x: x.values.argmax())
        X_wide = df_choice.pivot(index='sample_id', columns='alt_id', values=self.feature_names)
        
        # Flatten column names
        X_wide.columns = [f'{col[0]}_alt{col[1]}' for col in X_wide.columns]
        
        # Create design matrix for utilities
        n_features = len(self.feature_names)
        X_design = np.zeros((n_samples, n_alts - 1, n_features))
        
        for i in range(n_samples):
            for j in range(1, n_alts):  # Alternative 0 is reference
                for k in range(n_features):
                    X_design[i, j-1, k] = self.X_data[i, j, k] - self.X_data[i, 0, k]
        
        X_design_flat = X_design.reshape(n_samples, -1)
        
        # Fit conditional logit model
        model = sm.MNLogit(y, sm.add_constant(X_design_flat))
        results = model.fit(disp=False)
        
        return results
    
    def generate_latex_tables(self, sem_results, dcm_results, output_dir):
        """
        Generate LaTeX tables for results
        
        Args:
            sem_results: SEM results
            dcm_results: Discrete choice model results
            output_dir: Directory to save LaTeX files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Table 1: Structural Equation Model Results
        latex_sem = "\\begin{table}[htbp]\n"
        latex_sem += "\\centering\n"
        latex_sem += "\\caption{Structural Equation Model Results}\n"
        latex_sem += "\\label{tab:sem_results}\n"
        latex_sem += "\\begin{tabular}{llccc}\n"
        latex_sem += "\\hline\n"
        latex_sem += "Dependent Variable & Independent Variable & Coefficient & Std. Error & p-value \\\\\n"
        latex_sem += "\\hline\n"
        
        for dep_var, eq_results in sem_results['equations'].items():
            model = eq_results['model']
            parents = eq_results['parents']
            
            # Add intercept
            latex_sem += f"{dep_var} & Intercept & {model.params[0]:.3f} & {model.bse[0]:.3f} & {model.pvalues[0]:.3f} \\\\\n"
            
            # Add parent effects
            for i, parent in enumerate(parents):
                coef = model.params[i+1]
                se = model.bse[i+1]
                pval = model.pvalues[i+1]
                sig = ""
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
                
                latex_sem += f" & {parent} & {coef:.3f}{sig} & {se:.3f} & {pval:.3f} \\\\\n"
            
            # Add R-squared
            latex_sem += f" & $R^2$ = {model.rsquared:.3f} & & & \\\\\n"
            latex_sem += "\\hline\n"
        
        latex_sem += "\\end{tabular}\n"
        latex_sem += "\\footnotesize{Note: * p<0.05, ** p<0.01, *** p<0.001}\n"
        latex_sem += "\\end{table}\n"
        
        # Save SEM table
        with open(os.path.join(output_dir, 'sem_results.tex'), 'w') as f:
            f.write(latex_sem)
        
        # Table 2: Discrete Choice Model Results
        latex_dcm = "\\begin{table}[htbp]\n"
        latex_dcm += "\\centering\n"
        latex_dcm += "\\caption{Discrete Choice Model Results}\n"
        latex_dcm += "\\label{tab:dcm_results}\n"
        latex_dcm += "\\begin{tabular}{lccc}\n"
        latex_dcm += "\\hline\n"
        latex_dcm += "Variable & Coefficient & Std. Error & p-value \\\\\n"
        latex_dcm += "\\hline\n"
        
        # Add DCM coefficients
        for i, param_name in enumerate(dcm_results.params.index):
            coef = dcm_results.params[i]
            se = dcm_results.bse[i]
            pval = dcm_results.pvalues[i]
            sig = ""
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            
            latex_dcm += f"{param_name} & {coef:.3f}{sig} & {se:.3f} & {pval:.3f} \\\\\n"
        
        latex_dcm += "\\hline\n"
        latex_dcm += f"Log-Likelihood & {dcm_results.llf:.2f} & & \\\\\n"
        latex_dcm += f"AIC & {dcm_results.aic:.2f} & & \\\\\n"
        latex_dcm += f"BIC & {dcm_results.bic:.2f} & & \\\\\n"
        latex_dcm += f"Pseudo $R^2$ & {dcm_results.prsquared:.3f} & & \\\\\n"
        latex_dcm += "\\hline\n"
        latex_dcm += "\\end{tabular}\n"
        latex_dcm += "\\footnotesize{Note: * p<0.05, ** p<0.01, *** p<0.001}\n"
        latex_dcm += "\\end{table}\n"
        
        # Save DCM table
        with open(os.path.join(output_dir, 'dcm_results.tex'), 'w') as f:
            f.write(latex_dcm)
        
        print(f"LaTeX tables saved to {output_dir}")
    
    def visualize_causal_graph(self, threshold=0.1, output_path=None):
        """
        Visualize the learned causal graph
        
        Args:
            threshold: Threshold for displaying edges
            output_path: Path to save figure
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(self.feature_names):
            G.add_node(i, label=name)
        
        # Add edges
        for i in range(self.n_features):
            for j in range(self.n_features):
                if i != j and abs(self.adjacency_matrix[i, j]) > threshold:
                    weight = self.adjacency_matrix[i, j]
                    G.add_edge(i, j, weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        node_colors = ['lightblue' if i < 3 else 'lightgreen' for i in range(self.n_features)]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
        
        # Draw edges with weights
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        # Normalize weights for visualization
        max_weight = max(abs(w) for w in weights) if weights else 1
        edge_widths = [abs(w) / max_weight * 3 for w in weights]
        edge_colors = ['red' if w < 0 else 'blue' for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                              alpha=0.7, arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {i: self.feature_names[i][:10] + '...' if len(self.feature_names[i]) > 10 
                 else self.feature_names[i] for i in range(self.n_features)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add edge labels with weights
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title("Learned Causal Structure from GOLEM-DC", fontsize=14)
        plt.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Positive effect'),
            Line2D([0], [0], color='red', lw=2, label='Negative effect'),
            plt.scatter([], [], c='lightblue', s=200, label='Core features'),
            plt.scatter([], [], c='lightgreen', s=200, label='Segmentation features')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Causal graph saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
        return G


def main():
    """
    Run post-causal discovery analysis
    """
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Post-causal discovery analysis for GOLEM-DC')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved GOLEM-DC model')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold for causal edges')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'../../reports/models/post_causal_{timestamp}'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = PostCausalAnalysis(args.model_path)
    
    # Load data (same as in run_experiment.py)
    from run_experiment import load_and_prepare_data
    data = load_and_prepare_data()
    
    # Combine all data for analysis
    X_all = torch.cat([
        data['train_dataset'].tensors[0],
        data['val_dataset'].tensors[0],
        data['test_dataset'].tensors[0]
    ], dim=0).numpy()
    
    y_all = torch.cat([
        data['train_dataset'].tensors[1],
        data['val_dataset'].tensors[1],
        data['test_dataset'].tensors[1]
    ], dim=0).numpy()
    
    analyzer.load_data(X_all, y_all)
    
    # Run structural equation modeling
    print("Running structural equation modeling...")
    sem_results = analyzer.structural_equation_modeling(threshold=args.threshold)
    
    # Run discrete choice analysis
    print("Running discrete choice analysis...")
    dcm_results = analyzer.discrete_choice_analysis(sem_results)
    
    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    analyzer.generate_latex_tables(sem_results, dcm_results, args.output_dir)
    
    # Visualize causal graph
    print("Visualizing causal graph...")
    graph_path = os.path.join(args.output_dir, 'causal_graph.png')
    analyzer.visualize_causal_graph(threshold=args.threshold, output_path=graph_path)
    
    # Save adjacency matrix
    adj_df = pd.DataFrame(
        analyzer.adjacency_matrix,
        index=analyzer.feature_names,
        columns=analyzer.feature_names
    )
    adj_df.to_csv(os.path.join(args.output_dir, 'adjacency_matrix.csv'))
    
    # Create summary report
    with open(os.path.join(args.output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Post-Causal Discovery Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"- Number of features: {analyzer.n_features}\n")
        f.write(f"- Hyperparameters: {analyzer.hyperparameters}\n\n")
        
        f.write("Structural Equation Model Results:\n")
        for var, r2 in sem_results['r_squared'].items():
            f.write(f"- {var}: R² = {r2:.3f}\n")
        
        f.write(f"\nDiscrete Choice Model Results:\n")
        f.write(f"- Log-likelihood: {dcm_results.llf:.2f}\n")
        f.write(f"- AIC: {dcm_results.aic:.2f}\n")
        f.write(f"- BIC: {dcm_results.bic:.2f}\n")
        f.write(f"- Pseudo R²: {dcm_results.prsquared:.3f}\n")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 