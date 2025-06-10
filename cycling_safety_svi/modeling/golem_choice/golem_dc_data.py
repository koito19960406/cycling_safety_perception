"""
Data loading and preprocessing for GOLEM-DC model

This module handles:
1. Loading choice data (travel time, traffic lights, choices)
2. Loading predicted safety scores
3. Loading segmentation pixel ratios
4. Combining all features into proper format for GOLEM-DC
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os


class ChoiceDataset(Dataset):
    """
    PyTorch Dataset for discrete choice data with image features
    """
    
    def __init__(self, choice_data, features_alt1, features_alt2, choices, choice_sets=None):
        """
        Initialize dataset
        
        Args:
            choice_data: DataFrame with choice data
            features_alt1: Features for alternative 1 (n_samples, n_features)
            features_alt2: Features for alternative 2 (n_samples, n_features)
            choices: Chosen alternatives (0 or 1)
            choice_sets: Availability of alternatives (optional)
        """
        self.choice_data = choice_data
        self.features_alt1 = torch.FloatTensor(features_alt1)
        self.features_alt2 = torch.FloatTensor(features_alt2)
        self.choices = torch.LongTensor(choices - 1)  # Convert to 0-indexed
        
        # Stack features for both alternatives
        self.features = torch.stack([self.features_alt1, self.features_alt2], dim=1)
        
        # Default choice sets (both alternatives available)
        if choice_sets is None:
            self.choice_sets = torch.ones(len(choices), 2)
        else:
            self.choice_sets = torch.FloatTensor(choice_sets)
            
    def __len__(self):
        return len(self.choices)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'choice': self.choices[idx],
            'choice_set': self.choice_sets[idx]
        }


class GOLEMDCDataLoader:
    """
    Data loader for GOLEM-DC model that combines all features
    """
    
    def __init__(self, choice_data_path, safety_scores_path, 
                 segmentation_path=None, selected_seg_features=None):
        """
        Initialize data loader
        
        Args:
            choice_data_path: Path to choice data CSV
            safety_scores_path: Path to safety scores CSV
            segmentation_path: Path to segmentation pixel ratios CSV (optional)
            selected_seg_features: List of segmentation features to use (optional)
        """
        self.choice_data_path = choice_data_path
        self.safety_scores_path = safety_scores_path
        self.segmentation_path = segmentation_path
        self.selected_seg_features = selected_seg_features
        
        # Load all data
        self._load_data()
        
    def _load_data(self):
        """Load all data files"""
        # Load choice data
        self.choice_data = pd.read_csv(self.choice_data_path)
        
        # Load safety scores
        self.safety_scores = pd.read_csv(self.safety_scores_path)
        self.safety_dict = dict(zip(self.safety_scores['image_name'], 
                                   self.safety_scores['safety_score']))
        
        # Load segmentation data if provided
        if self.segmentation_path:
            # Read only the first few rows to get column names
            seg_sample = pd.read_csv(self.segmentation_path, nrows=5)
            
            # If no features selected, use the most relevant ones for cycling safety
            if self.selected_seg_features is None:
                self.selected_seg_features = [
                    'Road', 'Sidewalk', 'Bike Lane', 'Car', 'Person',
                    'Bicycle', 'Bicyclist', 'Building', 'Vegetation',
                    'Traffic Light', 'Traffic Sign (Front)'
                ]
            
            # Filter to available features
            available_features = [f for f in self.selected_seg_features if f in seg_sample.columns]
            
            # Read only selected columns
            cols_to_read = ['filename_key'] + available_features
            self.segmentation_data = pd.read_csv(self.segmentation_path, usecols=cols_to_read)
            # add ".jpg" to filenames if not present
            self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].apply(
                lambda x: x if x.endswith('.jpg') else f"{x}.jpg"
            )
            
            # Create dictionary for quick lookup
            self.seg_dict = {}
            for _, row in self.segmentation_data.iterrows():
                self.seg_dict[row['filename_key']] = row[available_features].values
        else:
            self.seg_dict = None
            self.selected_seg_features = []
            
    def prepare_features(self, normalize=True):
        """
        Prepare feature matrices for both alternatives
        
        Args:
            normalize: Whether to normalize features
            
        Returns:
            features_alt1: Features for alternative 1
            features_alt2: Features for alternative 2
            feature_names: List of feature names
            scaler: Fitted StandardScaler (if normalize=True)
        """
        feature_names = []
        features_list_alt1 = []
        features_list_alt2 = []
        
        # Traditional features (normalized)
        features_list_alt1.append(self.choice_data['TT1'].values / 10.0)  # Travel time
        features_list_alt2.append(self.choice_data['TT2'].values / 10.0)
        feature_names.append('travel_time_norm')
        
        features_list_alt1.append(self.choice_data['TL1'].values / 3.0)  # Traffic lights
        features_list_alt2.append(self.choice_data['TL2'].values / 3.0)
        feature_names.append('traffic_lights_norm')
        
        # Safety scores
        safety_alt1 = []
        safety_alt2 = []
        
        for _, row in self.choice_data.iterrows():
            # Get safety score for each alternative's image
            safety_alt1.append(self.safety_dict.get(row['IMG1'], 0.0))
            safety_alt2.append(self.safety_dict.get(row['IMG2'], 0.0))
            
        features_list_alt1.append(np.array(safety_alt1))
        features_list_alt2.append(np.array(safety_alt2))
        feature_names.append('safety_score')
        
        # Segmentation features if available
        if self.seg_dict is not None:
            for feat_name in self.selected_seg_features:
                seg_alt1 = []
                seg_alt2 = []
                
                for _, row in self.choice_data.iterrows():
                    # Get segmentation features for each image
                    seg_vec1 = self.seg_dict.get(row['IMG1'], np.zeros(len(self.selected_seg_features)))
                    seg_vec2 = self.seg_dict.get(row['IMG2'], np.zeros(len(self.selected_seg_features)))
                    
                    # Find index of current feature
                    idx = self.selected_seg_features.index(feat_name)
                    seg_alt1.append(seg_vec1[idx] if len(seg_vec1) > idx else 0.0)
                    seg_alt2.append(seg_vec2[idx] if len(seg_vec2) > idx else 0.0)
                    
                features_list_alt1.append(np.array(seg_alt1))
                features_list_alt2.append(np.array(seg_alt2))
                feature_names.append(f'seg_{feat_name.lower().replace(" ", "_")}')
        
        # Stack features
        features_alt1 = np.column_stack(features_list_alt1)
        features_alt2 = np.column_stack(features_list_alt2)
        
        # Normalize if requested
        scaler = None
        if normalize:
            # Combine both alternatives for fitting scaler
            all_features = np.vstack([features_alt1, features_alt2])
            scaler = StandardScaler()
            scaler.fit(all_features)
            
            # Transform separately
            features_alt1 = scaler.transform(features_alt1)
            features_alt2 = scaler.transform(features_alt2)
            
        return features_alt1, features_alt2, feature_names, scaler
    
    def create_datasets(self, train_ratio=0.8, random_state=42):
        """
        Create train and test datasets
        
        Args:
            train_ratio: Proportion of data for training
            random_state: Random seed
            
        Returns:
            train_dataset: Training dataset
            test_dataset: Test dataset
            feature_names: List of feature names
            scaler: Fitted StandardScaler
        """
        # Prepare features
        features_alt1, features_alt2, feature_names, scaler = self.prepare_features()
        
        # Get choices
        choices = self.choice_data['CHOICE'].values
        
        # Create train/test split based on train column if available
        if 'train' in self.choice_data.columns:
            train_mask = self.choice_data['train'] == 1
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(~train_mask)[0]
        else:
            # Random split
            n_samples = len(choices)
            indices = np.arange(n_samples)
            np.random.seed(random_state)
            np.random.shuffle(indices)
            
            n_train = int(n_samples * train_ratio)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = ChoiceDataset(
            self.choice_data.iloc[train_indices],
            features_alt1[train_indices],
            features_alt2[train_indices],
            choices[train_indices]
        )
        
        test_dataset = ChoiceDataset(
            self.choice_data.iloc[test_indices],
            features_alt1[test_indices],
            features_alt2[test_indices],
            choices[test_indices]
        )
        
        return train_dataset, test_dataset, feature_names, scaler
    
    def create_dataloaders(self, batch_size=32, train_ratio=0.8, random_state=42):
        """
        Create PyTorch DataLoaders for training and testing
        
        Args:
            batch_size: Batch size for DataLoaders
            train_ratio: Proportion of data for training
            random_state: Random seed
            
        Returns:
            train_loader: Training DataLoader
            test_loader: Test DataLoader
            feature_names: List of feature names
            scaler: Fitted StandardScaler
        """
        train_dataset, test_dataset, feature_names, scaler = self.create_datasets(
            train_ratio, random_state
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, test_loader, feature_names, scaler 