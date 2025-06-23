#!/usr/bin/env python3
"""
Correlation Analysis: Traffic Safety Scores vs Segmentation Results

This script performs correlation analysis between traffic safety scores and 
segmentation results (pixel ratios) as specified in the project requirements.

Key outputs:
- Correlation matrix and analysis
- Statistical significance tests
- Visualizations of relationships
- LaTeX report with results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load cycling safety scores and segmentation results data.
    
    Returns:
        tuple: (safety_df, segmentation_df, merged_df)
    """
    logger.info("Loading data files...")
    
    # Load cycling safety scores
    safety_path = "data/processed/predicted_danish/cycling_safety_scores.csv"
    safety_df = pd.read_csv(safety_path)
    logger.info(f"Loaded safety scores: {len(safety_df)} records")
    
    # Load segmentation results
    seg_path = "data/processed/segmentation_results/pixel_ratios.csv"
    segmentation_df = pd.read_csv(seg_path)
    logger.info(f"Loaded segmentation results: {len(segmentation_df)} records")
    
    # Extract image name without extension for matching
    safety_df['image_key'] = safety_df['image_name'].str.replace('.jpg', '')
    
    # Merge datasets on image identifier
    merged_df = pd.merge(
        safety_df[['image_key', 'safety_score']], 
        segmentation_df, 
        left_on='image_key', 
        right_on='filename_key', 
        how='inner'
    )
    
    logger.info(f"Merged dataset: {len(merged_df)} records")
    logger.info(f"Missing values in safety_score: {merged_df['safety_score'].isna().sum()}")
    
    return safety_df, segmentation_df, merged_df

def get_segmentation_features(merged_df):
    """
    Extract segmentation feature columns from the merged dataset.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset
        
    Returns:
        list: List of segmentation feature column names
    """
    # Get all columns except the key columns
    exclude_cols = ['image_key', 'safety_score', 'filename_key']
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
    
    # Filter out columns with all zeros (features not present in any images)
    active_features = []
    for col in feature_cols:
        if merged_df[col].sum() > 0:  # Has some non-zero values
            active_features.append(col)
    
    logger.info(f"Found {len(active_features)} active segmentation features out of {len(feature_cols)} total")
    return active_features

def calculate_correlations(merged_df, features):
    """
    Calculate correlations between safety scores and segmentation features.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset
        features (list): List of feature column names
        
    Returns:
        tuple: (pearson_results, spearman_results)
    """
    logger.info("Calculating correlations...")
    
    safety_scores = merged_df['safety_score']
    
    pearson_results = []
    spearman_results = []
    
    for feature in features:
        feature_values = merged_df[feature]
        
        # Skip if feature has no variance
        if feature_values.std() == 0:
            continue
            
        # Pearson correlation
        try:
            pearson_r, pearson_p = pearsonr(safety_scores, feature_values)
            pearson_results.append({
                'feature': feature,
                'correlation': pearson_r,
                'p_value': pearson_p,
                'significant': pearson_p < 0.05
            })
        except:
            logger.warning(f"Could not calculate Pearson correlation for {feature}")
        
        # Spearman correlation  
        try:
            spearman_r, spearman_p = spearmanr(safety_scores, feature_values)
            spearman_results.append({
                'feature': feature,
                'correlation': spearman_r,
                'p_value': spearman_p,
                'significant': spearman_p < 0.05
            })
        except:
            logger.warning(f"Could not calculate Spearman correlation for {feature}")
    
    # Convert to DataFrames for easier handling
    pearson_df = pd.DataFrame(pearson_results)
    spearman_df = pd.DataFrame(spearman_results)
    
    # Sort by absolute correlation value
    if not pearson_df.empty:
        pearson_df = pearson_df.reindex(pearson_df['correlation'].abs().sort_values(ascending=False).index)
    if not spearman_df.empty:
        spearman_df = spearman_df.reindex(spearman_df['correlation'].abs().sort_values(ascending=False).index)
    
    logger.info(f"Calculated {len(pearson_df)} Pearson correlations")
    logger.info(f"Calculated {len(spearman_df)} Spearman correlations")
    
    return pearson_df, spearman_df

def create_correlation_matrix(merged_df, features, method='pearson'):
    """
    Create correlation matrix for top features.
    
    Args:
        merged_df (pd.DataFrame): Merged dataset
        features (list): List of feature column names
        method (str): Correlation method ('pearson' or 'spearman')
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    # Select features with sufficient variance
    selected_features = []
    for feature in features:
        if merged_df[feature].std() > 0:
            selected_features.append(feature)
    
    # Limit to top features to keep matrix readable
    if len(selected_features) > 20:
        # Calculate correlations with safety score first
        corr_with_safety = []
        for feature in selected_features:
            if method == 'pearson':
                corr, _ = pearsonr(merged_df['safety_score'], merged_df[feature])
            else:
                corr, _ = spearmanr(merged_df['safety_score'], merged_df[feature])
            corr_with_safety.append((feature, abs(corr)))
        
        # Select top 20 features
        corr_with_safety.sort(key=lambda x: x[1], reverse=True)
        selected_features = [x[0] for x in corr_with_safety[:20]]
    
    # Include safety score in the matrix
    matrix_data = merged_df[['safety_score'] + selected_features]
    
    if method == 'pearson':
        corr_matrix = matrix_data.corr(method='pearson')
    else:
        corr_matrix = matrix_data.corr(method='spearman')
    
    return corr_matrix

def create_visualizations(pearson_df, spearman_df, merged_df, features):
    """
    Create correlation visualizations.
    
    Args:
        pearson_df (pd.DataFrame): Pearson correlation results
        spearman_df (pd.DataFrame): Spearman correlation results
        merged_df (pd.DataFrame): Merged dataset
        features (list): List of feature column names
    """
    logger.info("Creating visualizations...")
    
    # Ensure output directory exists
    os.makedirs('reports/figures', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Top correlations bar plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Pearson top 15
    if not pearson_df.empty:
        top_pearson = pearson_df.head(15)
        colors_p = ['red' if p < 0.05 else 'lightcoral' for p in top_pearson['p_value']]
        bars1 = ax1.barh(range(len(top_pearson)), top_pearson['correlation'], color=colors_p)
        ax1.set_yticks(range(len(top_pearson)))
        ax1.set_yticklabels(top_pearson['feature'], fontsize=8)
        ax1.set_xlabel('Pearson Correlation Coefficient')
        ax1.set_title('Top 15 Pearson Correlations with Safety Score\n(Red = p < 0.05, Light Red = p ≥ 0.05)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Spearman top 15
    if not spearman_df.empty:
        top_spearman = spearman_df.head(15)
        colors_s = ['blue' if p < 0.05 else 'lightblue' for p in top_spearman['p_value']]
        bars2 = ax2.barh(range(len(top_spearman)), top_spearman['correlation'], color=colors_s)
        ax2.set_yticks(range(len(top_spearman)))
        ax2.set_yticklabels(top_spearman['feature'], fontsize=8)
        ax2.set_xlabel('Spearman Correlation Coefficient')
        ax2.set_title('Top 15 Spearman Correlations with Safety Score\n(Blue = p < 0.05, Light Blue = p ≥ 0.05)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation matrix heatmap
    corr_matrix = create_correlation_matrix(merged_df, features, method='pearson')
    
    # Save correlation matrix to CSV and TXT
    os.makedirs('data/processed', exist_ok=True)
    corr_matrix.to_csv('data/processed/correlation_matrix.csv')
    corr_matrix.to_string(buf=open('data/processed/correlation_matrix.txt', 'w'))
    logger.info("Correlation matrix saved to data/processed/correlation_matrix.csv and correlation_matrix.txt")
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={"shrink": .8},
                mask=mask, annot_kws={'size': 8})
    plt.title('Correlation Matrix: Safety Score vs Top Segmentation Features', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Scatter plots for top correlations
    if not pearson_df.empty:
        top_5_features = pearson_df.head(5)['feature'].tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_5_features):
            if i < 5:  # We have 6 subplots but only 5 features
                ax = axes[i]
                x = merged_df[feature]
                y = merged_df['safety_score']
                
                # Create scatter plot
                ax.scatter(x, y, alpha=0.6, s=20)
                
                # Add regression line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                ax.plot(x, p(x), "r--", alpha=0.8)
                
                # Get correlation info
                corr_info = pearson_df[pearson_df['feature'] == feature].iloc[0]
                
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel('Safety Score', fontsize=10)
                ax.set_title(f'{feature}\nr = {corr_info["correlation"]:.3f}, p = {corr_info["p_value"]:.3f}', 
                           fontsize=9)
                ax.grid(True, alpha=0.3)
        
        # Remove the last empty subplot
        fig.delaxes(axes[5])
        
        plt.suptitle('Scatter Plots: Safety Score vs Top 5 Correlated Features', fontsize=14)
        plt.tight_layout()
        plt.savefig('reports/figures/scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Safety score distribution
    ax1.hist(merged_df['safety_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Safety Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Safety Scores')
    ax1.grid(True, alpha=0.3)
    
    # Correlation coefficients distribution
    if not pearson_df.empty:
        ax2.hist(pearson_df['correlation'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Pearson Correlation Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Correlation Coefficients')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('reports/figures/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to reports/figures/")

def generate_latex_report(pearson_df, spearman_df, merged_df):
    """
    Generate LaTeX report with correlation analysis results.
    
    Args:
        pearson_df (pd.DataFrame): Pearson correlation results
        spearman_df (pd.DataFrame): Spearman correlation results
        merged_df (pd.DataFrame): Merged dataset
    """
    logger.info("Generating LaTeX report...")
    
    # Ensure output directory exists
    os.makedirs('reports/models', exist_ok=True)
    
    latex_content = r"""
\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{hyperref}

\geometry{margin=1in}

\title{Correlation Analysis: Traffic Safety Scores vs Segmentation Results}
\author{Cycling Safety Analysis}
\date{\today}

\begin{document}

\maketitle

\section{Executive Summary}

This report presents the correlation analysis between traffic safety scores and segmentation results (pixel ratios) for cycling infrastructure. The analysis examines the relationship between perceived safety and various visual elements captured through image segmentation.

\section{Data Overview}

\subsection{Dataset Statistics}
\begin{itemize}
"""
    
    # Add data statistics
    latex_content += f"""
    \\item Total images analyzed: {len(merged_df)}
    \\item Safety score range: [{merged_df['safety_score'].min():.3f}, {merged_df['safety_score'].max():.3f}]
    \\item Safety score mean: {merged_df['safety_score'].mean():.3f} (SD: {merged_df['safety_score'].std():.3f})
    \\item Number of segmentation features: {len([col for col in merged_df.columns if col not in ['image_key', 'safety_score', 'filename_key']])}
"""
    
    if not pearson_df.empty:
        sig_pearson = len(pearson_df[pearson_df['significant']])
        latex_content += f"    \\item Significant Pearson correlations (p < 0.05): {sig_pearson}/{len(pearson_df)}\n"
    
    if not spearman_df.empty:
        sig_spearman = len(spearman_df[spearman_df['significant']])
        latex_content += f"    \\item Significant Spearman correlations (p < 0.05): {sig_spearman}/{len(spearman_df)}\n"
    
    latex_content += """
\\end{itemize}

\\section{Correlation Results}

\\subsection{Pearson Correlation Analysis}

The Pearson correlation analysis examines linear relationships between safety scores and segmentation features.

"""
    
    # Add top Pearson correlations table
    if not pearson_df.empty:
        latex_content += """
\\begin{longtable}{l r r l}
\\toprule
Feature & Correlation & p-value & Significant \\\\
\\midrule
"""
        
        for _, row in pearson_df.head(20).iterrows():
            sig_mark = "Yes" if row['significant'] else "No"
            feature_name = row['feature'].replace('_', '\\_')
            latex_content += f"{feature_name} & {row['correlation']:.4f} & {row['p_value']:.4f} & {sig_mark} \\\\\n"
        
        latex_content += """
\\bottomrule
\\caption{Top 20 Pearson Correlations with Safety Scores}
\\end{longtable}

"""
    
    # Add Spearman correlations
    latex_content += """
\\subsection{Spearman Correlation Analysis}

The Spearman correlation analysis examines monotonic relationships between safety scores and segmentation features.

"""
    
    if not spearman_df.empty:
        latex_content += """
\\begin{longtable}{l r r l}
\\toprule
Feature & Correlation & p-value & Significant \\\\
\\midrule
"""
        
        for _, row in spearman_df.head(20).iterrows():
            sig_mark = "Yes" if row['significant'] else "No"
            feature_name = row['feature'].replace('_', '\\_')
            latex_content += f"{feature_name} & {row['correlation']:.4f} & {row['p_value']:.4f} & {sig_mark} \\\\\n"
        
        latex_content += """
\\bottomrule
\\caption{Top 20 Spearman Correlations with Safety Scores}
\\end{longtable}

"""
    
    # Add key findings section
    latex_content += """
\\section{Key Findings}

\\subsection{Strong Positive Correlations}
"""
    
    if not pearson_df.empty:
        strong_positive = pearson_df[(pearson_df['correlation'] > 0.3) & (pearson_df['significant'])]
        if not strong_positive.empty:
            latex_content += "The following features show strong positive correlations with safety scores:\n\\begin{itemize}\n"
            for _, row in strong_positive.iterrows():
                feature_name = row['feature'].replace('_', '\\_')
                latex_content += f"\\item {feature_name}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f}\n"
            latex_content += "\\end{itemize}\n\n"
        else:
            latex_content += "No features showed strong positive correlations (r > 0.3, p < 0.05) with safety scores.\n\n"
    
    latex_content += """
\\subsection{Strong Negative Correlations}
"""
    
    if not pearson_df.empty:
        strong_negative = pearson_df[(pearson_df['correlation'] < -0.3) & (pearson_df['significant'])]
        if not strong_negative.empty:
            latex_content += "The following features show strong negative correlations with safety scores:\n\\begin{itemize}\n"
            for _, row in strong_negative.iterrows():
                feature_name = row['feature'].replace('_', '\\_')
                latex_content += f"\\item {feature_name}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f}\n"
            latex_content += "\\end{itemize}\n\n"
        else:
            latex_content += "No features showed strong negative correlations (r < -0.3, p < 0.05) with safety scores.\n\n"
    
    # Add figures
    latex_content += """
\\section{Visualizations}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{../figures/correlation_rankings.png}
\\caption{Ranking of correlations between safety scores and segmentation features}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{../figures/correlation_matrix.png}
\\caption{Correlation matrix showing relationships between safety scores and top segmentation features}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{../figures/scatter_plots.png}
\\caption{Scatter plots showing relationships between safety scores and top correlated features}
\\end{figure}

\\begin{figure}[H]
\\centering
\\includegraphics[width=\\textwidth]{../figures/distributions.png}
\\caption{Distribution of safety scores and correlation coefficients}
\\end{figure}

\\section{Methodology}

\\subsection{Data Preparation}
\\begin{itemize}
\\item Merged cycling safety scores with segmentation pixel ratios based on image filename
\\item Excluded segmentation features with zero variance (no occurrences across images)
\\item Applied both Pearson and Spearman correlation methods
\\end{itemize}

\\subsection{Statistical Analysis}
\\begin{itemize}
\\item Pearson correlation: Measures linear relationships
\\item Spearman correlation: Measures monotonic relationships
\\item Significance threshold: p < 0.05
\\item Multiple comparison correction: Not applied (exploratory analysis)
\\end{itemize}

\\section{Limitations}
\\begin{itemize}
\\item Correlation does not imply causation
\\item Some segmentation features may be sparse across the dataset
\\item Multiple testing may inflate Type I error rates
\\item Linear and monotonic assumptions may not capture complex relationships
\\end{itemize}

\\section{Recommendations}
\\begin{itemize}
\\item Further investigate features with strong correlations through qualitative analysis
\\item Consider non-linear modeling approaches for complex relationships
\\item Validate findings with additional datasets
\\item Apply multiple comparison corrections for confirmatory analysis
\\end{itemize}

\\end{document}
"""
    
    # Save LaTeX report
    with open('reports/models/correlation_analysis.tex', 'w') as f:
        f.write(latex_content)
    
    logger.info("LaTeX report saved to reports/models/correlation_analysis.tex")

def save_results_csv(pearson_df, spearman_df, merged_df):
    """
    Save correlation results to CSV files for further analysis.
    
    Args:
        pearson_df (pd.DataFrame): Pearson correlation results
        spearman_df (pd.DataFrame): Spearman correlation results
        merged_df (pd.DataFrame): Merged dataset
    """
    logger.info("Saving results to CSV files...")
    
    # Ensure output directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save correlation results
    if not pearson_df.empty:
        pearson_df.to_csv('data/processed/pearson_correlations.csv', index=False)
    
    if not spearman_df.empty:
        spearman_df.to_csv('data/processed/spearman_correlations.csv', index=False)
    
    # Save merged dataset for further analysis
    merged_df.to_csv('data/processed/merged_safety_segmentation.csv', index=False)
    
    logger.info("Results saved to data/processed/")

def main():
    """
    Main function to run the correlation analysis.
    """
    logger.info("Starting correlation analysis...")
    
    try:
        # Load and prepare data
        safety_df, segmentation_df, merged_df = load_and_prepare_data()
        
        if merged_df.empty:
            logger.error("No matching records found between safety scores and segmentation data")
            return
        
        # Get segmentation features
        features = get_segmentation_features(merged_df)
        
        if not features:
            logger.error("No active segmentation features found")
            return
        
        # Calculate correlations
        pearson_df, spearman_df = calculate_correlations(merged_df, features)
        
        # Create visualizations
        create_visualizations(pearson_df, spearman_df, merged_df, features)
        
        # Generate LaTeX report
        generate_latex_report(pearson_df, spearman_df, merged_df)
        
        # Save CSV results
        save_results_csv(pearson_df, spearman_df, merged_df)
        
        # Print summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to:")
        logger.info(f"  - reports/figures/ (visualizations)")
        logger.info(f"  - reports/models/correlation_analysis.tex (LaTeX report)")
        logger.info(f"  - data/processed/ (CSV results)")
        
        if not pearson_df.empty:
            logger.info(f"\nTop 5 Pearson correlations:")
            for _, row in pearson_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: r = {row['correlation']:.3f}, p = {row['p_value']:.4f}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 