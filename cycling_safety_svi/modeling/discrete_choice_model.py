import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import biogeme.messaging as msg
from biogeme.expressions import Beta, Variable, log, bioDraws, exp, MonteCarlo

from ImageChoicedata_preprocessing import ImageChoiceDataset
from cvdcm import cvdcm_model


class FeatureExtractor:
    """Extract features from images using pre-trained model"""
    
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = cvdcm_model(path_pretrained_model=model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, image_path):
        """Extract features from a single image"""
        from ImageChoicedata_preprocessing import image_processor
        
        # Load and process image
        from skimage import io
        image = io.imread(image_path)
        if image.ndim == 2:
            image = np.atleast_3d(image)
            image = np.tile(image, 3)
        
        image = image_processor(image, return_tensors="pt").pixel_values[0]
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature_map, _ = self.model.return_featuremap(image)
            
        return feature_map.cpu().numpy().flatten()
    
    def process_dataset(self, data_path, img_dir):
        """Process all images in the dataset and extract features"""
        data = pd.read_csv(data_path)
        
        features = {}
        for img_name in pd.concat([data['IMG1'], data['IMG2']]).unique():
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                features[img_name] = self.extract_features(img_path)
        
        return features


class ChoiceModel:
    """Discrete choice models with different feature combinations"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure Biogeme messaging
        logger = msg.bioMessage()
        logger.setWarning()
        logger.setDisplayLevel(msg.QUIET)
    
    def prepare_data(self, data_path, image_features, perception_data_path=None, 
                     segmentation_data_path=None):
        """Prepare data for Biogeme model"""
        # Load choice data
        choice_data = pd.read_csv(data_path)
        
        # Add image features
        for i, row in choice_data.iterrows():
            for j, img in enumerate(['IMG1', 'IMG2']):
                img_name = row[img]
                if img_name in image_features:
                    for k, feat_val in enumerate(image_features[img_name]):
                        choice_data.loc[i, f'feat_{j+1}_{k}'] = feat_val
        
        # Add perception data if available
        if perception_data_path:
            perception_data = pd.read_csv(perception_data_path)
            perception_data = perception_data.groupby('imageid').mean().reset_index()
            
            perception_dict = {row['imageid']: row for _, row in perception_data.iterrows()}
            
            for i, row in choice_data.iterrows():
                for j, img in enumerate(['IMG1', 'IMG2']):
                    img_name = row[img]
                    img_id = img_name.split('.')[0]  # Assuming image ID is the name without extension
                    
                    if img_id in perception_dict:
                        choice_data.loc[i, f'traffic_safety_{j+1}'] = perception_dict[img_id]['traffic_safety']
                        choice_data.loc[i, f'social_safety_{j+1}'] = perception_dict[img_id]['social_safety']
                        choice_data.loc[i, f'beauty_{j+1}'] = perception_dict[img_id]['beauty']
        
        # Add segmentation data if available
        if segmentation_data_path:
            segmentation_data = pd.read_csv(segmentation_data_path)
            
            segmentation_dict = {row['image_id']: row for _, row in segmentation_data.iterrows()}
            
            for i, row in choice_data.iterrows():
                for j, img in enumerate(['IMG1', 'IMG2']):
                    img_name = row[img]
                    img_id = img_name.split('.')[0]  # Assuming image ID is the name without extension
                    
                    if img_id in segmentation_dict:
                        for seg_class in [c for c in segmentation_data.columns if c != 'image_id']:
                            choice_data.loc[i, f'{seg_class}_{j+1}'] = segmentation_dict[img_id][seg_class]
        
        return choice_data
    
    def run_model(self, data, model_type='base', num_dimensions=5):
        """Run a specific type of choice model"""
        # Create Biogeme database
        database = db.Database('choice_model', data)
        
        # Define variables
        Choice = Variable('CHOICE')
        TL1 = Variable('TL1')
        TL2 = Variable('TL2')
        TT1 = Variable('TT1')
        TT2 = Variable('TT2')
        
        # Create availability variables (both alternatives are always available)
        av1 = 1
        av2 = 1
        
        # Define parameters for base attributes
        beta_tl = Beta('beta_tl', -0.70, None, None, 0)
        beta_tt = Beta('beta_tt', -0.95, None, None, 0)
        
        # Define utility functions based on model type
        if model_type == 'base':
            # Base model with only traffic lights and travel time
            V1 = beta_tl * TL1 / 3 + beta_tt * TT1 / 10
            V2 = beta_tl * TL2 / 3 + beta_tt * TT2 / 10
            
        elif model_type == 'perception_only':
            # Add perception variables
            traffic_safety_1 = Variable('traffic_safety_1')
            traffic_safety_2 = Variable('traffic_safety_2')
            social_safety_1 = Variable('social_safety_1')
            social_safety_2 = Variable('social_safety_2')
            beauty_1 = Variable('beauty_1')
            beauty_2 = Variable('beauty_2')
            
            beta_traffic = Beta('beta_traffic', 0.5, None, None, 0)
            beta_social = Beta('beta_social', 0.5, None, None, 0)
            beta_beauty = Beta('beta_beauty', 0.5, None, None, 0)
            
            V1 = (beta_tl * TL1 / 3 + beta_tt * TT1 / 10 + 
                  beta_traffic * traffic_safety_1 + 
                  beta_social * social_safety_1 + 
                  beta_beauty * beauty_1)
            
            V2 = (beta_tl * TL2 / 3 + beta_tt * TT2 / 10 + 
                  beta_traffic * traffic_safety_2 + 
                  beta_social * social_safety_2 + 
                  beta_beauty * beauty_2)
            
        elif model_type == 'segmentation_only':
            # Add segmentation variables
            # We'll use a dynamic approach here based on available segmentation classes
            seg_vars = [col.split('_')[0] for col in data.columns if col.endswith('_1') 
                       and col not in ['TL1', 'TT1', 'traffic_safety_1', 'social_safety_1', 'beauty_1']]
            
            seg_betas = {}
            seg_terms_1 = []
            seg_terms_2 = []
            
            for seg_class in seg_vars:
                var_1 = Variable(f'{seg_class}_1')
                var_2 = Variable(f'{seg_class}_2')
                beta_name = f'beta_{seg_class}'
                seg_betas[beta_name] = Beta(beta_name, 0.1, None, None, 0)
                
                seg_terms_1.append(seg_betas[beta_name] * var_1)
                seg_terms_2.append(seg_betas[beta_name] * var_2)
            
            V1 = beta_tl * TL1 / 3 + beta_tt * TT1 / 10 + sum(seg_terms_1)
            V2 = beta_tl * TL2 / 3 + beta_tt * TT2 / 10 + sum(seg_terms_2)
            
        elif model_type == 'perception_and_segmentation':
            # Add both perception and segmentation variables
            # Perception variables
            traffic_safety_1 = Variable('traffic_safety_1')
            traffic_safety_2 = Variable('traffic_safety_2')
            social_safety_1 = Variable('social_safety_1')
            social_safety_2 = Variable('social_safety_2')
            beauty_1 = Variable('beauty_1')
            beauty_2 = Variable('beauty_2')
            
            beta_traffic = Beta('beta_traffic', 0.5, None, None, 0)
            beta_social = Beta('beta_social', 0.5, None, None, 0)
            beta_beauty = Beta('beta_beauty', 0.5, None, None, 0)
            
            # Segmentation variables
            seg_vars = [col.split('_')[0] for col in data.columns if col.endswith('_1') 
                       and col not in ['TL1', 'TT1', 'traffic_safety_1', 'social_safety_1', 'beauty_1']]
            
            seg_betas = {}
            seg_terms_1 = []
            seg_terms_2 = []
            
            for seg_class in seg_vars:
                var_1 = Variable(f'{seg_class}_1')
                var_2 = Variable(f'{seg_class}_2')
                beta_name = f'beta_{seg_class}'
                seg_betas[beta_name] = Beta(beta_name, 0.1, None, None, 0)
                
                seg_terms_1.append(seg_betas[beta_name] * var_1)
                seg_terms_2.append(seg_betas[beta_name] * var_2)
            
            V1 = (beta_tl * TL1 / 3 + beta_tt * TT1 / 10 + 
                  beta_traffic * traffic_safety_1 + 
                  beta_social * social_safety_1 + 
                  beta_beauty * beauty_1 + 
                  sum(seg_terms_1))
            
            V2 = (beta_tl * TL2 / 3 + beta_tt * TT2 / 10 + 
                  beta_traffic * traffic_safety_2 + 
                  beta_social * social_safety_2 + 
                  beta_beauty * beauty_2 + 
                  sum(seg_terms_2))
            
        elif model_type == 'image_features':
            # Use deep image features
            image_terms_1 = []
            image_terms_2 = []
            
            # Use top n feature dimensions
            for i in range(num_dimensions):
                feat_1 = Variable(f'feat_1_{i}')
                feat_2 = Variable(f'feat_2_{i}')
                beta_name = f'beta_feat_{i}'
                beta_feat = Beta(beta_name, 0.1, None, None, 0)
                
                image_terms_1.append(beta_feat * feat_1)
                image_terms_2.append(beta_feat * feat_2)
            
            V1 = beta_tl * TL1 / 3 + beta_tt * TT1 / 10 + sum(image_terms_1)
            V2 = beta_tl * TL2 / 3 + beta_tt * TT2 / 10 + sum(image_terms_2)
            
        elif model_type == 'full_model':
            # Combine all features
            # Perception variables
            traffic_safety_1 = Variable('traffic_safety_1')
            traffic_safety_2 = Variable('traffic_safety_2')
            social_safety_1 = Variable('social_safety_1')
            social_safety_2 = Variable('social_safety_2')
            beauty_1 = Variable('beauty_1')
            beauty_2 = Variable('beauty_2')
            
            beta_traffic = Beta('beta_traffic', 0.5, None, None, 0)
            beta_social = Beta('beta_social', 0.5, None, None, 0)
            beta_beauty = Beta('beta_beauty', 0.5, None, None, 0)
            
            # Segmentation variables
            seg_vars = [col.split('_')[0] for col in data.columns if col.endswith('_1') 
                       and col not in ['TL1', 'TT1', 'traffic_safety_1', 'social_safety_1', 'beauty_1']]
            
            seg_betas = {}
            seg_terms_1 = []
            seg_terms_2 = []
            
            for seg_class in seg_vars:
                var_1 = Variable(f'{seg_class}_1')
                var_2 = Variable(f'{seg_class}_2')
                beta_name = f'beta_{seg_class}'
                seg_betas[beta_name] = Beta(beta_name, 0.1, None, None, 0)
                
                seg_terms_1.append(seg_betas[beta_name] * var_1)
                seg_terms_2.append(seg_betas[beta_name] * var_2)
            
            # Image features
            image_terms_1 = []
            image_terms_2 = []
            
            for i in range(num_dimensions):
                feat_1 = Variable(f'feat_1_{i}')
                feat_2 = Variable(f'feat_2_{i}')
                beta_name = f'beta_feat_{i}'
                beta_feat = Beta(beta_name, 0.1, None, None, 0)
                
                image_terms_1.append(beta_feat * feat_1)
                image_terms_2.append(beta_feat * feat_2)
            
            V1 = (beta_tl * TL1 / 3 + beta_tt * TT1 / 10 + 
                  beta_traffic * traffic_safety_1 + 
                  beta_social * social_safety_1 + 
                  beta_beauty * beauty_1 + 
                  sum(seg_terms_1) + 
                  sum(image_terms_1))
            
            V2 = (beta_tl * TL2 / 3 + beta_tt * TT2 / 10 + 
                  beta_traffic * traffic_safety_2 + 
                  beta_social * social_safety_2 + 
                  beta_beauty * beauty_2 + 
                  sum(seg_terms_2) + 
                  sum(image_terms_2))
        
        # Define the binary logit model
        logit = models.logit(V1, V2, None, Choice)
        
        # Create and estimate the model
        biogeme = bio.BIOGEME(database, logit)
        biogeme.modelName = f"choice_model_{model_type}"
        
        # Estimate the parameters
        results = biogeme.estimate()
        
        # Save results
        results_file = os.path.join(self.output_dir, f"{model_type}_results.html")
        results.writeLaTeX()
        results.writePickle()
        
        # Return results summary
        return results


def run_choice_models(data_path, img_dir, model_path, 
                      perception_data_path=None, segmentation_data_path=None,
                      output_dir='results'):
    """Run all choice models with different feature combinations"""
    
    # Extract features from images
    print("Extracting features from images...")
    feature_extractor = FeatureExtractor(model_path)
    image_features = feature_extractor.process_dataset(data_path, img_dir)
    
    # Initialize choice model
    choice_model = ChoiceModel(output_dir)
    
    # Prepare data
    print("Preparing data for choice modeling...")
    data = choice_model.prepare_data(data_path, image_features,
                                     perception_data_path, segmentation_data_path)
    
    # Run models
    results = {}
    
    print("Running base model...")
    results['base'] = choice_model.run_model(data, 'base')
    
    if perception_data_path:
        print("Running perception-only model...")
        results['perception_only'] = choice_model.run_model(data, 'perception_only')
    
    if segmentation_data_path:
        print("Running segmentation-only model...")
        results['segmentation_only'] = choice_model.run_model(data, 'segmentation_only')
    
    if perception_data_path and segmentation_data_path:
        print("Running combined perception and segmentation model...")
        results['perception_and_segmentation'] = choice_model.run_model(data, 'perception_and_segmentation')
    
    print("Running image features model...")
    results['image_features'] = choice_model.run_model(data, 'image_features', num_dimensions=10)
    
    if perception_data_path and segmentation_data_path:
        print("Running full model...")
        results['full_model'] = choice_model.run_model(data, 'full_model', num_dimensions=10)
    
    # Compare models
    print("Comparing models...")
    compare_results = {}
    for model_name, result in results.items():
        compare_results[model_name] = {
            'log_likelihood': result.data.logLike,
            'rho_square': result.data.rhoSquare,
            'num_parameters': len(result.data.betaValues)
        }
    
    comparison_df = pd.DataFrame(compare_results).T
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))
    
    return results, comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run discrete choice models with different feature combinations')
    parser.add_argument('--data_path', type=str, required=True, help='Path to choice data CSV')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained CVDCM model')
    parser.add_argument('--perception_data', type=str, help='Path to perception data CSV')
    parser.add_argument('--segmentation_data', type=str, help='Path to segmentation data CSV')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    results, comparison = run_choice_models(
        args.data_path, 
        args.img_dir,
        args.model_path,
        args.perception_data,
        args.segmentation_data,
        args.output_dir
    )
    
    print("\nModel comparison:")
    print(comparison)
    
    print(f"\nResults saved to {args.output_dir}") 