import torch
import os
import sys
import argparse
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from perception_model import SinglePerceptionModel
from perception_dataset import image_processor


def load_perception_models(model_paths, device, num_categories=5):
    """
    Load the perception models from the specified paths
    
    Args:
        model_paths (dict): Dictionary mapping perception types to model paths
        device (torch.device): Device to load the models on
        num_categories (int): Number of ordinal categories
        
    Returns:
        dict: Dictionary mapping perception types to loaded models
    """
    models = {}
    
    for perception_type, model_path in model_paths.items():
        print(f"Loading {perception_type} model from {model_path}")
        
        # Create model
        model = SinglePerceptionModel(
            perception_type=perception_type,
            ordinal_levels=num_categories
        ).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Set to evaluation mode
        model.eval()
        
        # Store in dictionary
        models[perception_type] = model
    
    return models


def predict_perceptions(image_paths, models, device, batch_size=16):
    """
    Predict perceptions for the given images
    
    Args:
        image_paths (list): List of image paths
        models (dict): Dictionary mapping perception types to models
        device (torch.device): Device to run inference on
        batch_size (int): Batch size for inference
        
    Returns:
        pandas.DataFrame: DataFrame with image paths and predicted perceptions
    """
    # Initialize results
    results = {
        'image_path': []
    }
    
    # Add columns for each perception type
    for perception_type in models.keys():
        results[perception_type] = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting perceptions"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Load and preprocess images
        for img_path in batch_paths:
            try:
                image = io.imread(img_path)
                
                # Ensure image has 3 channels (RGB)
                if image.ndim == 2:
                    image = np.atleast_3d(image)
                    image = np.tile(image, 3)
                
                # Preprocess image
                processed_image = image_processor(image, return_tensors="pt").pixel_values[0]
                batch_images.append(processed_image)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into a batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Get predictions from each model
        for perception_type, model in models.items():
            with torch.no_grad():
                outputs = model(batch_tensor)
                predictions = torch.argmax(outputs, dim=1) + 1  # +1 to get 1-5 scale
                
                # Add predictions to results
                for img_path, pred in zip(batch_paths, predictions.cpu().numpy()):
                    if img_path not in results['image_path']:
                        results['image_path'].append(img_path)
                        
                        # Initialize other perception values to NaN
                        for other_type in models.keys():
                            if other_type != perception_type:
                                results[other_type].append(np.nan)
                    
                    # Find the index of this image path
                    idx = results['image_path'].index(img_path)
                    
                    # Update the prediction
                    if idx < len(results[perception_type]):
                        results[perception_type][idx] = int(pred)
                    else:
                        # Expand the list if needed
                        results[perception_type].append(int(pred))
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Make sure all perception columns are filled (for images that failed for some models)
    for perception_type in models.keys():
        df[perception_type] = df[perception_type].fillna(-1).astype(int)
    
    return df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict perceptions from images using separate models')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images to predict')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save predictions CSV')
    parser.add_argument('--models_dir', type=str, default='output', help='Directory containing trained models')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--num_categories', type=int, default=5, help='Number of ordinal categories (3 or 5)')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (cuda or cpu)')
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Find model paths for each perception type
    perception_types = ['traffic_safety', 'social_safety', 'beautiful']
    model_paths = {}
    
    for perception_type in perception_types:
        # Find the most recent model file for this perception type
        model_pattern = os.path.join(args.models_dir, f'*_{perception_type}_model_*.pt')
        model_files = sorted(glob.glob(model_pattern), reverse=True)
        
        if not model_files:
            print(f"Error: No model found for {perception_type}. Check --models_dir or train models first.")
            return
        
        # Use the most recent model (sorted by timestamp)
        model_paths[perception_type] = model_files[0]
    
    # Load models
    models = load_perception_models(model_paths, device, args.num_categories)
    
    # Find all images in the directory
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    
    for ext in img_extensions:
        image_paths.extend(glob.glob(os.path.join(args.img_dir, ext)))
    
    if not image_paths:
        print(f"Error: No images found in {args.img_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Predict perceptions
    predictions = predict_perceptions(image_paths, models, device, args.batch_size)
    
    # Extract image IDs from paths
    predictions['image_id'] = predictions['image_path'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )
    
    # Reorder columns
    columns = ['image_id', 'image_path'] + perception_types
    predictions = predictions[columns]
    
    # Save predictions to CSV
    predictions.to_csv(args.output_file, index=False)
    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main() 