import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import io
from transformers import AutoImageProcessor
from transformers.utils.logging import set_verbosity_error
import logging
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set verbosity of transformers to error
set_verbosity_error()

# Load image processor
image_processor = AutoImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
image_processor.size['height'] = 384
image_processor.size['width'] = 384
image_processor.do_center_crop = False

class PerceptionDataset(torch.utils.data.Dataset):
    """
    Dataset class for perception prediction
    """
    
    def __init__(self, data_file, img_path, set_type='train', transform=True, num_categories=3, seed=42, 
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, model_type="deit_base", use_z_score=False):
        """
        Initialize the dataset
        
        Args:
            data_file: Path to CSV file with perception ratings
            img_path: Path to directory containing images
            set_type: 'train', 'val', or 'test'
            transform: Whether to apply data augmentation (for training only)
            num_categories: Number of ordinal categories (3 or 5)
            seed: Random seed for reproducibility
            train_ratio: Proportion of data to use for training
            val_ratio: Proportion of data to use for validation
            test_ratio: Proportion of data to use for testing
            model_type: Type of model being used (affects image size and transforms)
            use_z_score: Whether to apply z-score standardization instead of normalization
        """
        super(PerceptionDataset, self).__init__()
        
        # Store parameters
        self.img_path = Path(img_path)
        self.set_type = set_type
        self.transform = transform
        self.num_categories = num_categories
        self.model_type = model_type
        self.use_z_score = use_z_score
        
        # Load annotations
        self.annotations = pd.read_csv(data_file)
        
        # Create train/val/test split based on seed
        np.random.seed(seed)
        indices = np.random.permutation(len(self.annotations))
        
        # Calculate split indices
        train_end = int(train_ratio * len(indices))
        val_end = train_end + int(val_ratio * len(indices))
        
        # Split based on set_type
        if set_type == 'train':
            self.annotations = self.annotations.iloc[indices[:train_end]]
        elif set_type == 'val':
            self.annotations = self.annotations.iloc[indices[train_end:val_end]]
        elif set_type == 'test':
            self.annotations = self.annotations.iloc[indices[val_end:]]
        else:
            raise ValueError(f"Invalid set_type: {set_type}. Must be 'train', 'val', or 'test'")
        
        # Reset index
        self.annotations.reset_index(drop=True, inplace=True)
        
        # Set image size based on model type
        if model_type == "efficientvit_m1":
            # EfficientViT M1 expects 224x224 images
            self.img_size = (224, 224)
            # Get model-specific transforms from timm
            efficientvit_model = timm.create_model('efficientvit_m1.r224_in1k', pretrained=True)
            data_config = timm.data.resolve_model_data_config(efficientvit_model)
            # Extract normalization params
            norm_mean = data_config.get('mean', [0.485, 0.456, 0.406])
            norm_std = data_config.get('std', [0.229, 0.224, 0.225])
        else:
            # Default size for other models (DeiT, RepVit, ConvNextV2)
            self.img_size = (384, 384)
            # Standard ImageNet normalization
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]
        
        # Define image transforms
        if transform and set_type == 'train':
            # Common augmentation transforms
            aug_transforms = [
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.1),  # Occasionally flip vertically (some urban scenes may be similar upside down)
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # More aggressive affine transforms
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Add perspective transforms
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # More aggressive color jitter
                transforms.RandomGrayscale(p=0.1),  # Occasionally convert to grayscale
                transforms.ToTensor(),
            ]
            
            # Add normalization or z-score standardization
            if self.use_z_score:
                # Use Z-score standardization (normalize each image individually)
                aug_transforms.append(transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7)))
            else:
                # Use standard normalization with fixed mean and std
                aug_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
            
            # Add random erasing after normalization
            aug_transforms.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)))  # Randomly erase parts of the image
            
            self.transform_fn = transforms.Compose(aug_transforms)
        else:
            # No augmentation for validation and test sets
            norm_transforms = [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
            
            # Add normalization or z-score standardization
            if self.use_z_score:
                # Use Z-score standardization (normalize each image individually)
                norm_transforms.append(transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-7)))
            else:
                # Use standard normalization with fixed mean and std
                norm_transforms.append(transforms.Normalize(mean=norm_mean, std=norm_std))
                
            self.transform_fn = transforms.Compose(norm_transforms)
        
        # Categorize perception ratings
        self._categorize_ratings()
        
        # Print statistics about class distribution
        if set_type == 'train':
            self._print_class_distribution()
            
        # Log whether using z-score standardization
        if self.use_z_score:
            logger.info(f"Using Z-score standardization for {set_type} set")
        else:
            logger.info(f"Using standard normalization for {set_type} set with mean={norm_mean}, std={norm_std}")
    
    def _categorize_ratings(self):
        """Categorize continuous ratings into ordinal categories"""
        perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
        
        for var in perception_vars:
            if self.num_categories == 3:
                # 3 categories: Low (1-2), Medium (3), High (4-5)
                bins = [0, 2.5, 3.5, 5.1]
                labels = [0, 1, 2]  # 0-indexed (low, medium, high)
                
                # Add semantic labels for better interpretability
                label_map = {0: 'low', 1: 'medium', 2: 'high'}
            elif self.num_categories == 5:
                # 5 categories: 1, 2, 3, 4, 5
                bins = [0, 1.5, 2.5, 3.5, 4.5, 5.1]
                labels = [0, 1, 2, 3, 4]  # 0-indexed
                
                # Add semantic labels for better interpretability
                label_map = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
            else:
                raise ValueError(f"Invalid num_categories: {self.num_categories}. Must be 3 or 5")
            
            # Create categorical variable
            self.annotations[f'{var}_cat'] = pd.cut(
                self.annotations[var], 
                bins=bins, 
                labels=labels
            ).astype(int)
            
            # Add string label for display purposes (not used in training)
            self.annotations[f'{var}_label'] = self.annotations[f'{var}_cat'].map(label_map)
    
    def _print_class_distribution(self):
        """Print the distribution of classes for each perception variable"""
        perception_vars = ['traffic_safety', 'social_safety', 'beautiful']
        
        logger.info(f"Class distribution for {self.set_type} set (num_categories={self.num_categories}):")
        for var in perception_vars:
            counts = self.annotations[f'{var}_cat'].value_counts().sort_index()
            total = len(self.annotations)
            percentages = counts / total * 100
            
            logger.info(f"  {var}:")
            for idx, count in enumerate(counts):
                if self.num_categories == 3:
                    label = ['low', 'medium', 'high'][idx]
                else:
                    label = str(idx + 1)
                logger.info(f"    {label}: {count} samples ({percentages[idx]:.1f}%)")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
        Get a sample from the dataset
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Dictionary containing image and labels
        """
        row = self.annotations.iloc[index]
        
        if self.set_type == 'train':
            # For training set, use the augmented image path
            image_path = os.path.join(self.img_path, row['imageid'] + '.jpg')
        else:
            # For validation and test sets, use the original image path
            image_path = os.path.join(self.img_path, row['imageid'] + '.jpg')
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # If image loading fails, print error and use a fallback image
            print(f"Error loading image {image_path}: {e}")
            # Return a black image of the same size as a fallback
            image = Image.new('RGB', (384, 384), (0, 0, 0))
        
        # Apply transforms
        if self.transform_fn:
            image = self.transform_fn(image)
        
        # Get labels based on number of categories
        if self.num_categories == 3:  # Low, Medium, High
            traffic_safety = self.annotations.iloc[index]['traffic_safety_cat']
            social_safety = self.annotations.iloc[index]['social_safety_cat']
            beautiful = self.annotations.iloc[index]['beautiful_cat']
        elif self.num_categories == 5:  # Original 5 levels
            traffic_safety = row['traffic_safety_cat'] - 1  # Convert from 1-5 to 0-4
            social_safety = row['social_safety_cat'] - 1
            beautiful = row['beautiful_cat'] - 1
        else:  # Binary classification (low/high)
            traffic_safety = 1 if row['traffic_safety_cat'] >= 3 else 0
            social_safety = 1 if row['social_safety_cat'] >= 3 else 0
            beautiful = 1 if row['beautiful_cat'] >= 3 else 0
            
        # Create sample dictionary
        sample = {
            'image': image,
            'traffic_safety': traffic_safety,
            'social_safety': social_safety,
            'beautiful': beautiful,
            'traffic_safety_cat': traffic_safety,  # Add for consistency with training loop
            'social_safety_cat': social_safety,    # Add for consistency with training loop
            'beautiful_cat': beautiful,           # Already included for compatibility with training loop
            'image_path': image_path,
            'image_name': row['imageid'],
            'id': row['id'] if 'id' in row else index
        }
        
        return sample

def data_to_device(batch, device):
    """
    Transfer batch data to the specified device
    
    Args:
        batch: Batch data (dictionary)
        device: Device to transfer to
        
    Returns:
        Batch data on the device
    """
    batch['image'] = batch['image'].to(device)
    batch['traffic_safety'] = batch['traffic_safety'].to(device)
    batch['social_safety'] = batch['social_safety'].to(device)
    batch['beautiful'] = batch['beautiful'].to(device)
    
    # Also transfer categorical variables if they exist
    if 'traffic_safety_cat' in batch:
        batch['traffic_safety_cat'] = batch['traffic_safety_cat'].to(device)
    if 'social_safety_cat' in batch:
        batch['social_safety_cat'] = batch['social_safety_cat'].to(device)
    if 'beautiful_cat' in batch:
        batch['beautiful_cat'] = batch['beautiful_cat'].to(device)
    
    return batch 