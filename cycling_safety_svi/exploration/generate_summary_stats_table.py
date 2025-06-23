import pandas as pd
import os

def generate_summary_statistics():
    """
    Reads the cv_dcm.csv file, calculates summary statistics,
    and saves them as a LaTeX table.
    """
    # Define file paths
    data_path = 'data/raw/cv_dcm.csv'
    output_dir = 'reports/models'
    output_path = os.path.join(output_dir, 'summary_statistics.tex')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the data
    df = pd.read_csv(data_path)

    # Calculate statistics
    num_participants = df['RID'].nunique()
    num_choices = len(df)
    unique_images = pd.unique(df[['IMG1', 'IMG2']].values.ravel('K'))
    num_unique_images = len(unique_images)
    
    # Assuming 'train' column is 1 for training and 0 for testing
    num_train = int(df['train'].sum())
    num_test = num_choices - num_train

    # Create LaTeX table
    latex_table = f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\begin{{document}}
\\begin{{table}}[ht]
\\centering
\\caption{{Summary Statistics of the Dataset}}
\\label{{tab:summary_stats}}
\\begin{{tabular}}{{lr}}
\\toprule
Metric & \\multicolumn{{1}}{{c}}{{Value}} \\\\
\\midrule
Number of participants & {num_participants:,} \\\\
Number of choices & {num_choices:,} \\\\
Number of unique images & {num_unique_images:,} \\\\
\\midrule
Number of training samples & {num_train:,} \\\\
Number of test samples & {num_test:,} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
\\end{{document}}
"""

    # Save the LaTeX table to a file
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX summary statistics table saved to {output_path}")

if __name__ == '__main__':
    generate_summary_statistics() 