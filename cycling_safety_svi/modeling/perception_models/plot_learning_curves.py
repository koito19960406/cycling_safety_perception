import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


def plot_learning_curves(loss_file, output_dir=None, show_plot=True):
    """
    Plot learning curves from a CSV file containing loss history
    
    Args:
        loss_file: Path to the CSV file with loss history
        output_dir: Directory to save the plot (if None, uses same directory as loss_file)
        show_plot: Whether to show the plot (default: True)
    """
    # Load loss history
    loss_df = pd.read_csv(loss_file)
    
    # Extract data
    epochs = loss_df['epoch']
    train_loss = loss_df['train_loss']
    test_loss = loss_df['test_loss']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, test_loss, 'r-', label='Testing Loss')
    
    # Find best epoch
    best_epoch = test_loss.idxmin() + 1
    min_test_loss = test_loss.min()
    
    # Highlight best epoch
    plt.axvline(x=best_epoch, color='g', linestyle='--', 
                label=f'Best Epoch ({best_epoch}, Loss={min_test_loss:.4f})')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set y-axis limits with some padding
    y_min = min(train_loss.min(), test_loss.min()) * 0.9
    y_max = max(train_loss[:10].max(), test_loss[:10].max()) * 1.1  # Use first 10 epochs to avoid outliers
    plt.ylim(y_min, y_max)
    
    # Save figure if output_dir is provided
    if output_dir is None:
        # Use same directory as loss_file
        output_dir = os.path.dirname(loss_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file name without extension
    loss_file_name = os.path.basename(loss_file)
    loss_file_stem = os.path.splitext(loss_file_name)[0]
    
    # Save figure
    output_file = os.path.join(output_dir, f"{loss_file_stem}_plot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()


def plot_multiple_curves(loss_files, labels=None, output_file=None, show_plot=True):
    """
    Plot multiple learning curves for comparison
    
    Args:
        loss_files: List of paths to CSV files with loss history
        labels: List of labels for each curve (default: None, uses filenames)
        output_file: Path to save the combined plot
        show_plot: Whether to show the plot (default: True)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Generate labels if not provided
    if labels is None:
        labels = [os.path.basename(os.path.splitext(f)[0]) for f in loss_files]
    
    # Plot each curve
    for i, loss_file in enumerate(loss_files):
        # Load loss history
        loss_df = pd.read_csv(loss_file)
        
        # Extract data
        epochs = loss_df['epoch']
        train_loss = loss_df['train_loss']
        test_loss = loss_df['test_loss']
        
        # Plot with different colors
        plt.plot(epochs, test_loss, marker='.', markersize=5, label=f"{labels[i]} - Testing")
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comparison of Testing Loss for Different Models')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save figure if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {output_file}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from loss history")
    
    # File selection
    parser.add_argument("--loss_file", type=str, help="Path to a single loss history CSV file")
    parser.add_argument("--dir", type=str, help="Directory containing loss history CSV files")
    parser.add_argument("--pattern", type=str, default="*loss_history.csv", 
                        help="Pattern to match loss history files (default: *loss_history.csv)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, help="Directory to save plots")
    parser.add_argument("--combined", action="store_true", 
                        help="Create a combined plot of all curves")
    parser.add_argument("--combined_output", type=str, default="combined_loss_plot.png", 
                        help="Filename for combined plot (default: combined_loss_plot.png)")
    parser.add_argument("--no_show", action="store_true", 
                        help="Don't show plots, just save them")
    
    args = parser.parse_args()
    
    # Check if we have a single file or a directory
    if args.loss_file:
        # Plot a single file
        plot_learning_curves(args.loss_file, args.output_dir, not args.no_show)
    
    elif args.dir:
        # Plot all matching files in the directory
        loss_files = list(Path(args.dir).glob(args.pattern))
        
        if not loss_files:
            print(f"No files matching pattern '{args.pattern}' found in {args.dir}")
            exit(1)
        
        # Plot individual files
        for loss_file in loss_files:
            plot_learning_curves(str(loss_file), args.output_dir, not args.no_show)
        
        # Plot combined if requested
        if args.combined:
            output_file = os.path.join(args.output_dir, args.combined_output) if args.output_dir else args.combined_output
            plot_multiple_curves(loss_files, output_file=output_file, show_plot=not args.no_show)
    
    else:
        print("Error: Either --loss_file or --dir must be provided.")
        parser.print_help() 