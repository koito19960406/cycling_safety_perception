#!/usr/bin/env python3
"""
Script to apply cycling safety perception model to images and save results.
"""

import os
import sys
import argparse
import torch
import pandas as pd
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm import tqdm

# Add the submodule to the path
current_dir = 'cycling_safety_svi/cycling_safety_subjective_learning_pairwise'
sys.path.append(current_dir)

from nets.cnn import CNN  # noqa: E402


class SingleImageDataset(Dataset):
    """Dataset for processing single images for safety prediction."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'image_name': os.path.basename(image_path)
        }


def load_model(model_path, backbone='alex', model_type='rsscnn', 
               device='cuda'):
    """Load the trained cycling safety model."""
    
    # Define models available
    backbones = {
        'alex': models.alexnet,
        'vgg': models.vgg19,
        'dense': models.densenet121,
        'resnet': models.resnet50,
    }
    
    # Initialize model
    net = CNN(
        backbone=backbones[backbone],
        model=model_type,
        finetune=False,
        num_classes=3,  # Assuming ties are included
    )
    
    # Load weights
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    return net


def predict_safety_scores(model, dataloader, device):
    """Predict safety scores for images."""
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            images = batch['image'].to(device)
            image_names = batch['image_name']
            
            # For ranking model, we need to process each image individually
            # We'll use the same image as both left and right input to get 
            # individual scores
            if (hasattr(model, 'model') and 
                    model.model in ['rcnn', 'rsscnn']):
                # Get ranking scores (individual safety scores)
                forward_dict = model(images, images)  # Self-pairing
                
                if 'left' in forward_dict:
                    output = forward_dict['left']['output']
                    safety_scores = output.squeeze().cpu().numpy()
                    
                    # Handle both single images and batches
                    if safety_scores.ndim == 0:
                        safety_scores = [safety_scores.item()]
                    elif safety_scores.ndim == 1:
                        safety_scores = safety_scores.tolist()
                    
                    for name, score in zip(image_names, safety_scores):
                        results.append({
                            'image_name': name,
                            'safety_score': score
                        })
            
            else:
                raise ValueError(f"Unsupported model type: {model.model}")
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Apply cycling safety model to images')
    parser.add_argument(
        '--images_dir', 
        default='/srv/shared/bicycle_project_roos/images_scaled',
        help='Directory containing images to process')
    
    model_default = ('cycling_safety_svi/cycling_safety_subjective_'
                     'learning_pairwise/models/vgg_syn+ber.pt')
    parser.add_argument(
        '--model_path', 
        default=model_default,
        help='Path to the trained model (.pt file)')
    
    parser.add_argument(
        '--output_dir',
        default='data/processed/predicted_danish',
        help='Directory to save results CSV')
    
    parser.add_argument(
        '--backbone', 
        default='vgg',
        choices=['alex', 'vgg', 'dense', 'resnet'],
        help='CNN backbone architecture')
    
    parser.add_argument(
        '--model_type',
        default='rsscnn',
        choices=['rcnn', 'sscnn', 'rsscnn'],
        help='Model type')
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for processing')
    
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please place your .pt model file in the models directory:")
        model_dir = ("  cycling_safety_svi/cycling_safety_subjective_"
                     "learning_pairwise/models/")
        print(model_dir)
        return 1
    
    # Get list of images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(args.images_dir, ext)))
        image_paths.extend(glob(os.path.join(args.images_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {args.images_dir}")
        return 1
    
    print(f"Found {len(image_paths)} images to process")
    
    # Define transforms (same as used in training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Create dataset and dataloader
    dataset = SingleImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=4)
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, 
                       backbone=args.backbone, 
                       model_type=args.model_type, 
                       device=device)
    
    # Predict safety scores
    print("Predicting safety scores...")
    results_df = predict_safety_scores(model, dataloader, device)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'cycling_safety_scores.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results_df)} images")
    score_min = results_df['safety_score'].min()
    score_max = results_df['safety_score'].max()
    print(f"Safety score range: {score_min:.3f} to {score_max:.3f}")
    
    return 0


if __name__ == '__main__':
    exit(main()) 