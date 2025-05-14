import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from skimage import io
from transformers import AutoImageProcessor

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from perception_model import PerceptionModel
from perception_dataset import PerceptionDataset, data_to_device
from torch.utils.data import DataLoader

# Load image processor
image_processor = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
image_processor.size['height'] = 384
image_processor.size['width'] = 384
image_processor.do_center_crop = False


def predict_single_image(model, image_path, device):
    """
    Make prediction for a single image
    
    Args:
        model: PerceptionModel instance
        image_path: Path to the image
        device: Computation device
        
    Returns:
        Dictionary with predictions for each perception variable
    """
    # Load and preprocess image
    image = io.imread(image_path)
    
    # Ensure image has 3 channels (RGB)
    if image.ndim == 2:
        image = np.atleast_3d(image)
        image = np.tile(image, 3)
    
    # Transform image
    image_tensor = image_processor(image, return_tensors="pt").pixel_values[0]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        perceptions = model(image_tensor)
        predictions = [torch.argmax(p, dim=1).item() + 1 for p in perceptions]
    
    # Return results
    return {
        'image_path': image_path,
        'traffic_safety': predictions[0],
        'social_safety': predictions[1],
        'beautiful': predictions[2]
    }


def predict_batch(model, data_loader, device):
    """
    Make predictions for a batch of images
    
    Args:
        model: PerceptionModel instance
        data_loader: DataLoader with images
        device: Computation device
        
    Returns:
        DataFrame with predictions
    """
    model.eval()
    results = []
    
    # Prediction variables
    perception_names = ['traffic_safety', 'social_safety', 'beautiful']
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Move data to device
            batch = data_to_device(batch, device)
            
            # Get model predictions
            perceptions = model(batch['image'])
            predictions = [torch.argmax(p, dim=1) + 1 for p in perceptions]  # +1 to get 1-5 scale
            
            # Store results
            for i in range(len(batch['image'])):
                result = {'img_name': batch['img_name'][i]}
                
                # Add predictions
                for j, name in enumerate(perception_names):
                    result[name] = predictions[j][i].item()
                
                results.append(result)
    
    # Convert to DataFrame
    return pd.DataFrame(results)


def main(args):
    """
    Main function for making predictions
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = PerceptionModel(num_classes=3, ordinal_levels=args.num_categories)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {args.model_path}")
    
    if args.image_path:
        # Predict for a single image
        result = predict_single_image(model, args.image_path, device)
        print("\nPredictions:")
        for key, value in result.items():
            if key != 'image_path':
                print(f"{key}: {value}")
    
    elif args.image_dir and args.output_file:
        # Get all image files in directory
        image_files = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Create dataset and dataloader
        results = []
        for image_path in tqdm(image_files, desc="Predicting"):
            result = predict_single_image(model, image_path, device)
            # Extract just the filename from the path
            result['img_name'] = os.path.basename(image_path)
            results.append(result)
        
        # Convert to DataFrame and save to CSV
        df_results = pd.DataFrame(results)
        df_results = df_results.drop(columns=['image_path'])
        df_results.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    
    elif args.data_file and args.output_file:
        # Create dataset and dataloader
        dataset = PerceptionDataset(
            data_file=args.data_file,
            img_path=args.image_dir,
            set_type='all',
            transform=True,
            num_categories=args.num_categories
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers
        )
        
        # Make predictions
        df_results = predict_batch(model, dataloader, device)
        
        # Save to CSV
        df_results.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    
    else:
        print("Error: Must provide either an image path for single prediction, or image directory and output file for batch prediction.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with a trained perception model")
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    
    # Input options
    parser.add_argument("--image_path", type=str,
                        help="Path to a single image for prediction")
    parser.add_argument("--image_dir", type=str,
                        help="Directory containing images for batch prediction")
    parser.add_argument("--data_file", type=str,
                        help="Path to the data file with perception ratings")
    
    # Output options
    parser.add_argument("--output_file", type=str,
                        help="Path to save prediction results (CSV)")
    
    # Additional options
    parser.add_argument("--num_categories", type=int, default=5,
                        help="Number of ordinal categories for perception ratings (3 or 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for inference if available")
    
    args = parser.parse_args()
    main(args) 