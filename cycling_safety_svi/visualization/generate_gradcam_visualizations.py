#!/usr/bin/env python3
"""
Script to generate Grad-CAM visualizations for cycling safety perception model.
This helps visualize what parts of images the model focuses on for safety predictions.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Add the submodule to the path
current_dir = 'cycling_safety_svi/cycling_safety_subjective_learning_pairwise'
sys.path.append(current_dir)

from nets.cnn import CNN  # noqa: E402


class GradCAM:
    """Grad-CAM implementation for CNN models"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer"""
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            # Try to find the last conv layer automatically
            conv_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    conv_layers.append((name, module))
            
            if conv_layers:
                self.target_layer_name, target_layer = conv_layers[-1]
                print(f"Using last conv layer: {self.target_layer_name}")
            else:
                raise ValueError(f"Could not find target layer: {self.target_layer_name}")
        
        target_layer.register_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate class activation map for the input image"""
        
        # Forward pass
        self.model.eval()
        output = self.model(input_image, input_image)  # Self-pairing for ranking models
        
        # Get the output tensor based on model type
        if isinstance(output, dict) and 'left' in output:
            logits = output['left']['output']
        else:
            logits = output
        
        # If no class index specified, use the predicted class
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        loss = logits[0, class_idx] if logits.dim() > 1 else logits[class_idx]
        loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


class SafetyImageDataset(Dataset):
    """Dataset for processing images for Grad-CAM visualization."""
    
    def __init__(self, image_paths, transform=None, original_transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.original_transform = original_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Keep original for visualization
        original = self.original_transform(image) if self.original_transform else image
        
        # Transform for model
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'original': original,
            'image_name': os.path.basename(image_path),
            'image_path': image_path
        }


def load_model(model_path, backbone='vgg', model_type='rsscnn', device='cuda'):
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


def get_target_layer_name(model, backbone):
    """Get the appropriate target layer name based on the backbone."""
    
    layer_mappings = {
        'vgg': 'backbone.features.35',  # Last conv layer in VGG19
        'resnet': 'backbone.layer4.2.conv2',  # Last conv in ResNet50 
        'alex': 'backbone.features.12',  # Last conv in AlexNet
        'dense': 'backbone.features.denseblock4.denselayer16.conv2'  # Last conv in DenseNet121
    }
    
    target_layer = layer_mappings.get(backbone)
    if target_layer is None:
        # Fallback: find last conv layer
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(name)
        target_layer = conv_layers[-1] if conv_layers else None
    
    return target_layer


def overlay_heatmap(image, heatmap, alpha=0.5, is_normalized=False):
    """Overlay heatmap on image with specified alpha blending."""
    
    # Convert image to numpy if it's a tensor
    if torch.is_tensor(image):
        if image.dim() == 4:
            image = image.squeeze(0)
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image (assuming ImageNet normalization was used)
    if is_normalized:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    
    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap to heatmap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # Remove alpha channel
    
    # Blend images
    overlayed = image + alpha * heatmap_colored
    
    return overlayed

def process_images_gradcam(model, dataloader, device, output_dir, backbone='vgg'):
    """Process images and generate Grad-CAM visualizations, saving images directly without plt."""
    import cv2
    from PIL import Image

    # Get target layer name
    target_layer = get_target_layer_name(model, backbone)
    if target_layer is None:
        raise ValueError(f"Could not determine target layer for backbone: {backbone}")

    print(f"Using target layer: {target_layer}")

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Create output subdirectories
    gradcam_dir = os.path.join(output_dir, 'gradcam_visualizations')
    heatmaps_dir = os.path.join(gradcam_dir, 'heatmaps')
    overlays_dir = os.path.join(gradcam_dir, 'overlays')

    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    results = []

    for batch in tqdm(dataloader, desc="Generating Grad-CAM visualizations"):
        images = batch['image'].to(device)
        originals = batch['original']
        image_names = batch['image_name']

        for i in range(len(images)):
            try:
                # Generate CAM for single image
                single_image = images[i:i+1]  # Keep batch dimension

                # Enable gradients for this computation
                single_image.requires_grad_(True)

                cam = gradcam.generate_cam(single_image)

                # Get original image
                original = originals[i]
                image_name = image_names[i]

                # Save heatmap as image (apply colormap and save as PNG)
                heatmap_resized = cv2.resize(cam, (original.shape[2], original.shape[1]))
                heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # shape (H, W, 3), float [0,1]
                heatmap_colored_uint8 = np.uint8(255 * heatmap_colored)
                heatmap_img = Image.fromarray(heatmap_colored_uint8)
                heatmap_path = os.path.join(heatmaps_dir, f'heatmap_{image_name}')
                heatmap_img.save(heatmap_path)

                # Create overlay
                overlayed = overlay_heatmap(original, cam, alpha=0.3)
                # overlayed is float [0,1], convert to uint8
                overlayed_uint8 = np.uint8(255 * np.clip(overlayed, 0, 1))
                overlay_img = Image.fromarray(overlayed_uint8)
                overlay_path = os.path.join(overlays_dir, f'overlay_{image_name}')
                overlay_img.save(overlay_path)

                # Calculate some statistics about the heatmap
                cam_stats = {
                    'image_name': image_name,
                    'cam_mean': float(np.mean(cam)),
                    'cam_std': float(np.std(cam)),
                    'cam_max': float(np.max(cam)),
                    'cam_min': float(np.min(cam)),
                    'heatmap_path': heatmap_path,
                    'overlay_path': overlay_path
                }

                results.append(cam_stats)

            except Exception as e:
                print(f"Error processing {image_names[i]}: {str(e)}")
                continue

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for cycling safety model')
    
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
        default='/home/kiito/cycling_safety_svi/data/processed',
        help='Directory to save Grad-CAM visualizations')
    
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
        default=16,  # Smaller batch size for Grad-CAM
        help='Batch size for processing')
    
    parser.add_argument(
        '--max_images',
        type=int,
        default=1000000,
        help='Maximum number of images to process (for testing)')
    
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
    
    # Limit number of images if specified
    if args.max_images and len(image_paths) > args.max_images:
        image_paths = image_paths[:args.max_images]
        print(f"Limited to {args.max_images} images for processing")
    
    print(f"Found {len(image_paths)} images to process")
    
    # Define transforms
    model_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    original_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = SafetyImageDataset(image_paths, 
                               transform=model_transform,
                               original_transform=original_transform)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=8)  # Reduced for Grad-CAM
    
    # Load model
    print("Loading model...")
    model = load_model(args.model_path, 
                       backbone=args.backbone, 
                       model_type=args.model_type, 
                       device=device)
    
    # Generate Grad-CAM visualizations
    print("Generating Grad-CAM visualizations...")
    results_df = process_images_gradcam(model, dataloader, device, 
                                      args.output_dir, args.backbone)
    
    # Save results summary
    results_csv = os.path.join(args.output_dir, 'gradcam_visualizations', 
                              'gradcam_results.csv')
    results_df.to_csv(results_csv, index=False)
    
    print(f"Grad-CAM visualizations saved to: {args.output_dir}/gradcam_visualizations/")
    print(f"Results summary saved to: {results_csv}")
    print(f"Processed {len(results_df)} images successfully")
    
    # Print some statistics
    if len(results_df) > 0:
        print(f"Average CAM intensity: {results_df['cam_mean'].mean():.3f}")
        print(f"Average CAM max: {results_df['cam_max'].mean():.3f}")
    
    return 0


if __name__ == '__main__':
    exit(main())