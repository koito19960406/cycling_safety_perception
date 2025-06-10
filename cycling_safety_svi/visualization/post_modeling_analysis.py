"""
Post-Modeling Analysis Script

This script generates 8 visualizations comparing the stepwise_best and stepwise_wo_safety models:
1. Figure 1: Top 5 images where stepwise_best utilities > stepwise_wo_safety utilities
2. Figure 2: Top 5 images where stepwise_wo_safety utilities > stepwise_best utilities  
3. Figure 3: Top 5 image pairs with wrong choice using stepwise_wo_safety but correct with stepwise_best
4. Figure 4: Histogram of predicted safety scores grouped by wegtype (road type)
5. Figure 5: Histogram of predicted safety scores grouped by buildenvironment (land use)
6. Figure 6: Scatter plots between predicted safety scores, utilities, and segmentation features
7. Figure 7: Grid of images sorted by predicted safety scores, grouped by wegtype
8. Figure 8: Grid of images sorted by predicted safety scores, grouped by buildenvironment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import seaborn as sns
from PIL import Image
import os
from scipy.stats import rankdata


class PostModelingAnalyzer:
    """Analyzes and visualizes results from stepwise choice models"""
    
    def __init__(self, comparison_file_path='data/processed/model_results/df_choice_with_Vimg_comparison.csv'):
        """
        Initialize analyzer with comparison data
        
        Args:
            comparison_file_path: Path to the model comparison CSV file
        """
        self.comparison_file_path = comparison_file_path
        self.output_dir = Path('reports/figures/post_modeling_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define additional data file paths
        self.design_file_path = 'data/raw/main_design.csv'
        self.safety_scores_path = 'data/processed/predicted_danish/cycling_safety_scores.csv'
        self.pixel_ratios_path = 'data/processed/segmentation_results/pixel_ratios.csv'
        self.scaled_images_dir = '/srv/shared/bicycle_project_roos/images_scaled/'
        
        # Load all data
        self.load_data()
        
    def load_data(self):
        """Load and prepare comparison data and additional datasets"""
        print(f"Loading comparison data from: {self.comparison_file_path}")
        
        try:
            # Load main comparison data
            self.data = pd.read_csv(self.comparison_file_path)
            print(f"Loaded {len(self.data)} choice observations")
            
            # Calculate utility differences
            self.data['utility_diff_img1'] = self.data['V1_stepwise_best'] - self.data['V1_stepwise_wo_safety']
            self.data['utility_diff_img2'] = self.data['V2_stepwise_best'] - self.data['V2_stepwise_wo_safety']
            
            # Calculate prediction gaps for each choice situation
            self.data['prob_chosen_stepwise_best'] = np.where(
                self.data['CHOICE'] == 1,
                self.data['prob1_stepwise_best'],
                self.data['prob2_stepwise_best']
            )
            
            self.data['prob_chosen_stepwise_wo_safety'] = np.where(
                self.data['CHOICE'] == 1,
                self.data['prob1_stepwise_wo_safety'],
                self.data['prob2_stepwise_wo_safety']
            )
            
            # Calculate prediction gaps (actual choice probability vs 1.0)
            self.data['prediction_gap_stepwise_best'] = 1.0 - self.data['prob_chosen_stepwise_best']
            self.data['prediction_gap_stepwise_wo_safety'] = 1.0 - self.data['prob_chosen_stepwise_wo_safety']
            
            # Load additional datasets
            self.load_additional_datasets()
            
            print("✓ Data preprocessing completed")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Comparison file not found: {self.comparison_file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")
    
    def load_additional_datasets(self):
        """Load additional datasets for extended analysis"""
        
        # Load design data with wegtype and buildenvironment
        try:
            self.design_data = pd.read_csv(self.design_file_path)
            print(f"✓ Loaded design data with {len(self.design_data)} records")
        except FileNotFoundError:
            print(f"Warning: Design file not found: {self.design_file_path}")
            self.design_data = None
        
        # Load safety scores
        try:
            self.safety_scores = pd.read_csv(self.safety_scores_path)
            print(f"✓ Loaded safety scores for {len(self.safety_scores)} images")
        except FileNotFoundError:
            print(f"Warning: Safety scores file not found: {self.safety_scores_path}")
            self.safety_scores = None
        
        # Load pixel ratios
        try:
            self.pixel_ratios = pd.read_csv(self.pixel_ratios_path)
            self.pixel_ratios["filename_key"] = self.pixel_ratios["filename_key"] + ".jpg"
            print(f"✓ Loaded pixel ratios for {len(self.pixel_ratios)} images")
        except FileNotFoundError:
            print(f"Warning: Pixel ratios file not found: {self.pixel_ratios_path}")
            self.pixel_ratios = None
    
    def get_unique_image_utilities(self):
        """Get utility values for each unique image with percentile conversion"""
        
        # Combine IMG1 and IMG2 data
        img1_data = self.data[['IMG1', 'V1_stepwise_best', 'V1_stepwise_wo_safety']].copy()
        img1_data.columns = ['image_name', 'utility_stepwise_best', 'utility_stepwise_wo_safety']
        
        img2_data = self.data[['IMG2', 'V2_stepwise_best', 'V2_stepwise_wo_safety']].copy()
        img2_data.columns = ['image_name', 'utility_stepwise_best', 'utility_stepwise_wo_safety']
        
        # Combine and remove duplicates (taking first occurrence)
        all_utilities = pd.concat([img1_data, img2_data], ignore_index=True)
        unique_utilities = all_utilities.drop_duplicates(subset='image_name', keep='first').copy()
        
        # Convert utilities to percentiles
        unique_utilities.loc[:, 'percentile_stepwise_best'] = rankdata(unique_utilities['utility_stepwise_best'], method='average') / len(unique_utilities) * 100
        unique_utilities.loc[:, 'percentile_stepwise_wo_safety'] = rankdata(unique_utilities['utility_stepwise_wo_safety'], method='average') / len(unique_utilities) * 100
        
        # Calculate percentile differences
        unique_utilities.loc[:, 'percentile_diff'] = (
            unique_utilities['percentile_stepwise_best'] - unique_utilities['percentile_stepwise_wo_safety']
        )
        
        # Keep raw utility differences for backward compatibility
        unique_utilities.loc[:, 'utility_diff'] = (
            unique_utilities['utility_stepwise_best'] - unique_utilities['utility_stepwise_wo_safety']
        )
        
        return unique_utilities
    
    def load_and_resize_image(self, image_name, target_size=(150, 150), use_blend=False, use_segmented=False):
        """
        Load and resize an image for display
        
        Args:
            image_name: Name of the image file
            target_size: Target size (width, height) for resizing
            use_blend: If True, look for _blend.png version in scaled images directory
            use_segmented: If True, look for _blend.png version in segmented images directory
            
        Returns:
            PIL Image object or None if image not found
        """
        # If use_segmented is True, look in segmented images directory first
        if use_segmented:
            # Remove extension and add _blend.png
            base_name = Path(image_name).stem
            blend_name = f"{base_name}_blend.png"
            segmented_path = Path('data/processed/segmented_images') / blend_name
            
            if segmented_path.exists():
                try:
                    img = Image.open(segmented_path)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    return img
                except Exception as e:
                    print(f"Warning: Could not load segmented blend image {blend_name}: {e}")
        
        # If use_blend is True, modify image name and look in scaled directory first
        if use_blend:
            # Remove extension and add _blend.png
            base_name = Path(image_name).stem
            blend_name = f"{base_name}_blend.png"
            blend_path = Path(self.scaled_images_dir) / blend_name
            
            if blend_path.exists():
                try:
                    img = Image.open(blend_path)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    return img
                except Exception as e:
                    print(f"Warning: Could not load blend image {blend_name}: {e}")
        
        # Common image directories to search
        image_dirs = [
            self.scaled_images_dir,
            'data/raw/images',
            'data/processed/images', 
            'data/images',
            'images'
        ]
        
        for img_dir in image_dirs:
            img_path = Path(img_dir) / image_name
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                    return img
                except Exception as e:
                    print(f"Warning: Could not load image {image_name}: {e}")
                    continue
        
        print(f"Warning: Image not found: {image_name}")
        return None
    
    def create_figure_1(self):
        """
        Figure 1: Top 5 images with biggest gaps where stepwise_best > stepwise_w/o_safety
        """
        print("Creating Figure 1: Best > Best w/o Safety")
        
        # Get unique image utilities
        unique_utilities = self.get_unique_image_utilities()
        
        # Filter for positive percentile differences (stepwise_best > stepwise_wo_safety)
        positive_diffs = unique_utilities[unique_utilities['percentile_diff'] > 0]
        
        # Get top 5
        top_5 = positive_diffs.nlargest(5, 'percentile_diff')
        
        # Create figure with image and combined bar plot
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        fig.suptitle('Top 5 Images: Best > Best w/o Safety\n(Percentile Differences)', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            # Load image
            img = self.load_and_resize_image(row['image_name'])
            
            # Display image
            if img is not None:
                axes[i, 0].imshow(img)
            else:
                axes[i, 0].text(0.5, 0.5, "Image not found", 
                               ha='center', va='center', transform=axes[i, 0].transAxes, fontsize=16)
            
            axes[i, 0].axis('off')
            
            # Combined bar plot using percentiles
            models = ['Best', 'Best w/o Safety']
            percentiles = [row['percentile_stepwise_best'], row['percentile_stepwise_wo_safety']]
            colors = ['red', 'gray']
            
            bars = axes[i, 1].bar(models, percentiles, color=colors, alpha=0.8)
            axes[i, 1].set_ylabel('Percentile', fontsize=16)
            axes[i, 1].tick_params(axis='x', labelsize=14)
            axes[i, 1].tick_params(axis='y', labelsize=14)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, percentile in zip(bars, percentiles):
                height = bar.get_height()
                axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{percentile:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Add difference annotation
            diff = row['percentile_diff']
            axes[i, 1].text(0.5, 0.95, f"Difference: +{diff:.1f}%", 
                           transform=axes[i, 1].transAxes, ha='center', va='top',
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        fig_path = self.output_dir / 'figure_1_stepwise_best_greater.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 1 saved to: {fig_path}")
        plt.close()
        
        return top_5
    
    def create_figure_2(self):
        """
        Figure 2: Top 5 images with biggest gaps where stepwise_w/o_safety > stepwise_best
        """
        print("Creating Figure 2: Best w/o Safety > Best")
        
        # Get unique image utilities
        unique_utilities = self.get_unique_image_utilities()
        
        # Filter for negative percentile differences (stepwise_wo_safety > stepwise_best)
        negative_diffs = unique_utilities[unique_utilities['percentile_diff'] < 0]
        
        # Get top 5 (most negative = largest positive difference in favor of wo_safety)
        top_5 = negative_diffs.nsmallest(5, 'percentile_diff')
        
        # Create figure with image and combined bar plot
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))
        fig.suptitle('Top 5 Images: Best w/o Safety > Best\n(Percentile Differences)', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        for i, (_, row) in enumerate(top_5.iterrows()):
            # Load image
            img = self.load_and_resize_image(row['image_name'])
            
            # Display image
            if img is not None:
                axes[i, 0].imshow(img)
            else:
                axes[i, 0].text(0.5, 0.5, "Image not found", 
                               ha='center', va='center', transform=axes[i, 0].transAxes, fontsize=16)
            
            axes[i, 0].axis('off')
            
            # Combined bar plot using percentiles
            models = ['Best', 'Best w/o Safety']
            percentiles = [row['percentile_stepwise_best'], row['percentile_stepwise_wo_safety']]
            colors = ['red', 'gray']
            
            bars = axes[i, 1].bar(models, percentiles, color=colors, alpha=0.8)
            axes[i, 1].set_ylabel('Percentile', fontsize=16)
            axes[i, 1].tick_params(axis='x', labelsize=14)
            axes[i, 1].tick_params(axis='y', labelsize=14)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, percentile in zip(bars, percentiles):
                height = bar.get_height()
                axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{percentile:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            # Add difference annotation (show as negative for wo_safety advantage)
            diff = row['percentile_diff']
            axes[i, 1].text(0.5, 0.95, f"Difference: {diff:.1f}%", 
                           transform=axes[i, 1].transAxes, ha='center', va='top',
                           fontsize=16, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        fig_path = self.output_dir / 'figure_2_stepwise_wo_safety_greater.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 2 saved to: {fig_path}")
        plt.close()
        
        return top_5
    
    def create_figure_3(self):
        """
        Figure 3: Top 5 image pairs with wrong choice using stepwise_w/o_safety but correct with stepwise_best
        """
        print("Creating Figure 3: Wrong Choice with w/o_safety, Correct with best")
        
        # Find situations where stepwise_wo_safety was wrong but stepwise_best was correct
        wrong_wo_safety_correct_best = self.data[
            (self.data['correct_stepwise_wo_safety'] == False) & 
            (self.data['correct_stepwise_best'] == True)
        ].copy()
        
        if len(wrong_wo_safety_correct_best) == 0:
            print("Warning: No cases found where stepwise_wo_safety was wrong but stepwise_best was correct")
            return pd.DataFrame()
        
        # Calculate improvement in probability for chosen option
        wrong_wo_safety_correct_best.loc[:, 'prob_improvement'] = (
            wrong_wo_safety_correct_best['prob_chosen_stepwise_best'] - 
            wrong_wo_safety_correct_best['prob_chosen_stepwise_wo_safety']
        )
        
        # Get top 5 biggest improvements
        top_5_improvements = wrong_wo_safety_correct_best.nlargest(5, 'prob_improvement')
        
        # Create figure with 5 rows and 4 columns
        fig, axes = plt.subplots(5, 4, figsize=(20, 24))
        fig.suptitle('Top 5 Image Pairs: Wrong Choice with Best w/o Safety, Correct with Best\n(Probability Improvements)', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        for i, (_, row) in enumerate(top_5_improvements.iterrows()):
            chosen_img = row['IMG1'] if row['CHOICE'] == 1 else row['IMG2']
            not_chosen_img = row['IMG2'] if row['CHOICE'] == 1 else row['IMG1']
            
            # Probabilities for chosen image
            prob_chosen_stepwise_best = row['prob_chosen_stepwise_best']
            prob_chosen_stepwise_wo_safety = row['prob_chosen_stepwise_wo_safety']
            
            # Probabilities for not chosen image  
            prob_not_chosen_stepwise_best = 1 - prob_chosen_stepwise_best
            prob_not_chosen_stepwise_wo_safety = 1 - prob_chosen_stepwise_wo_safety
            
            # Column 1: Chosen image with probabilities
            img_chosen = self.load_and_resize_image(chosen_img)
            if img_chosen is not None:
                axes[i, 0].imshow(img_chosen)
            else:
                axes[i, 0].text(0.5, 0.5, f"Image not found:\n{chosen_img}", 
                               ha='center', va='center', transform=axes[i, 0].transAxes)
            
            axes[i, 0].set_title(f"CHOSEN", fontsize=16, color='green', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Add text box below chosen image
            prob_improvement = row['prob_improvement']
            axes[i, 0].text(0.5, -0.1, 
                           f"Probability Improvement:\n{prob_improvement:.3f}\n\nBest w/o Safety: WRONG\nBest: CORRECT", 
                           transform=axes[i, 0].transAxes, ha='center', va='top',
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
            
            # Column 2: Probabilities for chosen image
            model_names = ['Best', 'Best w/o Safety']
            probs_chosen = [prob_chosen_stepwise_best, prob_chosen_stepwise_wo_safety]
            colors = ['red', 'gray']
            
            bars = axes[i, 1].bar(model_names, probs_chosen, color=colors, alpha=0.8)
            axes[i, 1].set_title(f"Chosen Image Probabilities", fontsize=16, fontweight='bold')
            axes[i, 1].set_ylabel('Probability', fontsize=14)
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].tick_params(axis='x', rotation=45, labelsize=13)
            axes[i, 1].tick_params(axis='y', labelsize=13)
            axes[i, 1].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs_chosen):
                height = bar.get_height()
                axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
            
            # Column 3: Not chosen image with probabilities
            img_not_chosen = self.load_and_resize_image(not_chosen_img)
            if img_not_chosen is not None:
                axes[i, 2].imshow(img_not_chosen)
            else:
                axes[i, 2].text(0.5, 0.5, f"Image not found:\n{not_chosen_img}", 
                               ha='center', va='center', transform=axes[i, 2].transAxes)
            
            axes[i, 2].set_title(f"NOT CHOSEN", fontsize=16, color='red', fontweight='bold')
            axes[i, 2].axis('off')
            
            # Column 4: Probabilities for not chosen image
            probs_not_chosen = [prob_not_chosen_stepwise_best, prob_not_chosen_stepwise_wo_safety]
            
            bars = axes[i, 3].bar(model_names, probs_not_chosen, color=colors, alpha=0.8)
            axes[i, 3].set_title(f"Not Chosen Image Probabilities", fontsize=16, fontweight='bold')
            axes[i, 3].set_ylabel('Probability', fontsize=14)
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].tick_params(axis='x', rotation=45, labelsize=13)
            axes[i, 3].tick_params(axis='y', labelsize=13)
            axes[i, 3].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs_not_chosen):
                height = bar.get_height()
                axes[i, 3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        fig_path = self.output_dir / 'figure_3_model_corrections.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 3 saved to: {fig_path}")
        plt.close()
        
        return top_5_improvements
    
    def create_figure_4(self):
        """
        Figure 4: Histogram of predicted safety scores grouped/colored by wegtype (road type)
        """
        print("Creating Figure 4: Safety Scores by Road Type")
        
        if self.safety_scores is None or self.design_data is None:
            print("Warning: Cannot create Figure 4 - missing safety scores or design data")
            return None
        
        # Translation dictionary for road types
        wegtype_translation = {
            'Fietspad vrijliggend': 'Separated cycle path',
            'Solitair fietspad': 'Separated cycle path',
            'Normale weg': 'Normal road',
            'Fietsstrook': 'Painted cycle path',
            'Wijkontsluitingsweg': 'Access road',
            'Hoofdweg': 'Main road'
        }
        
        # Merge safety scores with design data
        # First get all unique images from design data
        img1_design = self.design_data[['alt1_imageid', 'alt1_wegtype']].rename(
            columns={'alt1_imageid': 'image_name', 'alt1_wegtype': 'wegtype'})
        img2_design = self.design_data[['alt2_imageid', 'alt2_wegtype']].rename(
            columns={'alt2_imageid': 'image_name', 'alt2_wegtype': 'wegtype'})
        # add ".jpg" to image names if not already present
        img1_design['image_name'] = img1_design['image_name'].astype(str) + '.jpg'
        img2_design['image_name'] = img2_design['image_name'].astype(str) + '.jpg'
        all_design = pd.concat([img1_design, img2_design], ignore_index=True)
        all_design = all_design.drop_duplicates(subset='image_name', keep='first')
        
        # Translate wegtype to English
        all_design['wegtype_en'] = all_design['wegtype'].map(wegtype_translation).fillna(all_design['wegtype'])
        
        # Ensure image_name is string type in both datasets
        all_design['image_name'] = all_design['image_name'].astype(str)
        safety_scores_copy = self.safety_scores.copy()
        safety_scores_copy['image_name'] = safety_scores_copy['image_name'].astype(str)
        
        # Merge with safety scores
        merged_data = safety_scores_copy.merge(all_design, on='image_name', how='inner')
        
        if len(merged_data) == 0:
            print("Warning: No matching images found between safety scores and design data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get unique wegtypes and create distinctive color palette
        unique_wegtypes = sorted(merged_data['wegtype_en'].unique())
        # Use a more distinctive color palette
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']
        
        # Create histogram for each wegtype with density normalization
        for i, wegtype in enumerate(unique_wegtypes):
            data_subset = merged_data[merged_data['wegtype_en'] == wegtype]
            color = colors[i % len(colors)]
            ax.hist(data_subset['safety_score'], bins=20, alpha=0.7, density=True,
                   label=f'{wegtype} (n={len(data_subset)})', color=color)
        
        ax.set_xlabel('Predicted Safety Score', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title('Distribution of Predicted Safety Scores by Road Type', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'figure_4_safety_scores_by_wegtype.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 4 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_figure_5(self):
        """
        Figure 5: Histogram of predicted safety scores grouped/colored by buildenvironment (land use)
        """
        print("Creating Figure 5: Safety Scores by Land Use")
        
        if self.safety_scores is None or self.design_data is None:
            print("Warning: Cannot create Figure 5 - missing safety scores or design data")
            return None
        
        # Translation dictionary for land use types
        buildenvironment_translation = {
            'Industrieterrein': 'Industrial',
            'Woongebied': 'Residential'
        }
        
        # Merge safety scores with design data
        # First get all unique images from design data
        img1_design = self.design_data[['alt1_imageid', 'alt1_buildenvironment']].rename(
            columns={'alt1_imageid': 'image_name', 'alt1_buildenvironment': 'buildenvironment'})
        img2_design = self.design_data[['alt2_imageid', 'alt2_buildenvironment']].rename(
            columns={'alt2_imageid': 'image_name', 'alt2_buildenvironment': 'buildenvironment'})
        # add ".jpg" to image names if not already present
        img1_design['image_name'] = img1_design['image_name'].astype(str) + '.jpg'
        img2_design['image_name'] = img2_design['image_name'].astype(str) + '.jpg'
        
        all_design = pd.concat([img1_design, img2_design], ignore_index=True)
        all_design = all_design.drop_duplicates(subset='image_name', keep='first')
        
        # Translate buildenvironment to English
        all_design['buildenvironment_en'] = all_design['buildenvironment'].map(buildenvironment_translation).fillna(all_design['buildenvironment'])
        
        # Ensure image_name is string type in both datasets
        all_design['image_name'] = all_design['image_name'].astype(str)
        safety_scores_copy = self.safety_scores.copy()
        safety_scores_copy['image_name'] = safety_scores_copy['image_name'].astype(str)
        
        # Merge with safety scores
        merged_data = safety_scores_copy.merge(all_design, on='image_name', how='inner')
        
        if len(merged_data) == 0:
            print("Warning: No matching images found between safety scores and design data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get unique buildenvironments and create distinctive color palette
        unique_envs = sorted(merged_data['buildenvironment_en'].unique())
        # Use a more distinctive color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Create histogram for each buildenvironment with density normalization
        for i, env in enumerate(unique_envs):
            data_subset = merged_data[merged_data['buildenvironment_en'] == env]
            color = colors[i % len(colors)]
            ax.hist(data_subset['safety_score'], bins=20, alpha=0.7, density=True,
                   label=f'{env} (n={len(data_subset)})', color=color)
        
        ax.set_xlabel('Predicted Safety Score', fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        ax.set_title('Distribution of Predicted Safety Scores by Land Use', fontsize=18, fontweight='bold')
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'figure_5_safety_scores_by_buildenvironment.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 5 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_figure_6(self):
        """
        Figure 6: Scatter plots between predicted safety scores, utilities, Car, Terrain, Bike Lane, Road, Vegetation
        """
        print("Creating Figure 6: Scatter Plot Matrix")
        
        if self.safety_scores is None or self.pixel_ratios is None:
            print("Warning: Cannot create Figure 6 - missing safety scores or pixel ratios")
            return None
        
        # Merge data for unique images
        unique_utilities = self.get_unique_image_utilities()
        
        # Ensure string types for merging
        unique_utilities['image_name'] = unique_utilities['image_name'].astype(str)
        safety_scores_copy = self.safety_scores.copy()
        safety_scores_copy['image_name'] = safety_scores_copy['image_name'].astype(str)
        pixel_ratios_copy = self.pixel_ratios.copy()
        pixel_ratios_copy['filename_key'] = pixel_ratios_copy['filename_key'].astype(str)
        
        # Merge with safety scores
        merged_data = unique_utilities.merge(safety_scores_copy, on='image_name', how='inner')
        
        # Merge with pixel ratios
        merged_data = merged_data.merge(pixel_ratios_copy, left_on='image_name', right_on='filename_key', how='inner')
        
        if len(merged_data) == 0:
            print("Warning: No matching data found for scatter plots")
            return None
        
        # Select variables for scatter plots (excluding utility_stepwise_wo_safety)
        variables = ['safety_score', 'utility_stepwise_best', 'Car', 'Terrain', 'Bike Lane', 'Road', 'Vegetation']
        
        # Filter variables that exist in the data
        available_vars = [var for var in variables if var in merged_data.columns]
        
        if len(available_vars) < 2:
            print(f"Warning: Not enough variables available for scatter plots. Available: {available_vars}")
            return None
        
        # Create scatter plot matrix
        n_vars = len(available_vars)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(20, 20))
        fig.suptitle('Scatter Plot Matrix: Safety Scores, Utilities, and Segmentation Features', fontsize=20, fontweight='bold', y=0.98)
        
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i == j:
                    # Diagonal: histogram
                    axes[i, j].hist(merged_data[var1], bins=20, alpha=0.7, color='skyblue')
                    axes[i, j].set_title(var1, fontsize=14, fontweight='bold')
                    axes[i, j].tick_params(axis='both', labelsize=12)
                else:
                    # Off-diagonal: scatter plot
                    axes[i, j].scatter(merged_data[var2], merged_data[var1], alpha=0.6, s=25)
                    
                    # Calculate and display correlation
                    corr = merged_data[var1].corr(merged_data[var2])
                    axes[i, j].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i, j].transAxes, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                                   fontsize=13, fontweight='bold')
                    axes[i, j].tick_params(axis='both', labelsize=12)
                
                if i == n_vars - 1:
                    axes[i, j].set_xlabel(var2, fontsize=14)
                if j == 0:
                    axes[i, j].set_ylabel(var1, fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'figure_6_scatter_plot_matrix.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 6 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_figure_7(self):
        """
        Figure 7: Grid of images sorted by predicted safety scores, grouped by wegtype
        """
        print("Creating Figure 7: Image Grid by Road Type and Safety Scores")
        
        if self.safety_scores is None or self.design_data is None:
            print("Warning: Cannot create Figure 7 - missing safety scores or design data")
            return None
        
        # Translation dictionary for road types
        wegtype_translation = {
            'Fietspad vrijliggend': 'Separated cycle path',
            'Solitair fietspad': 'Separated cycle path',
            'Normale weg': 'Normal road',
            'Fietsstrook': 'Painted cycle path',
            'Wijkontsluitingsweg': 'Access road',
            'Hoofdweg': 'Main road'
        }
        
        # Merge safety scores with design data
        img1_design = self.design_data[['alt1_imageid', 'alt1_wegtype']].rename(
            columns={'alt1_imageid': 'image_name', 'alt1_wegtype': 'wegtype'})
        img2_design = self.design_data[['alt2_imageid', 'alt2_wegtype']].rename(
            columns={'alt2_imageid': 'image_name', 'alt2_wegtype': 'wegtype'})
        # add ".jpg" to image names if not already present
        img1_design['image_name'] = img1_design['image_name'].astype(str) + '.jpg'
        img2_design['image_name'] = img2_design['image_name'].astype(str) + '.jpg'
        
        all_design = pd.concat([img1_design, img2_design], ignore_index=True)
        all_design = all_design.drop_duplicates(subset='image_name', keep='first')
        
        # Translate wegtype to English
        all_design['wegtype_en'] = all_design['wegtype'].map(wegtype_translation).fillna(all_design['wegtype'])
        
        # Ensure string types for merging
        all_design['image_name'] = all_design['image_name'].astype(str)
        safety_scores_copy = self.safety_scores.copy()
        safety_scores_copy['image_name'] = safety_scores_copy['image_name'].astype(str)
        
        merged_data = safety_scores_copy.merge(all_design, on='image_name', how='inner')
        
        if len(merged_data) == 0:
            print("Warning: No matching images found for Figure 7")
            return None
        
        # Get unique wegtypes and sort by safety scores within each
        unique_wegtypes = sorted(merged_data['wegtype_en'].unique())
        n_wegtypes = len(unique_wegtypes)
        n_cols = 5
        
        # Create figure with one extra column for labels
        fig, axes = plt.subplots(n_wegtypes, n_cols + 1, figsize=(18, 3*n_wegtypes))
        if n_wegtypes == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Image Grid: Representative Samples by Safety Score Quantiles within Road Types', fontsize=20, fontweight='bold', y=0.98)
        
        for row, wegtype in enumerate(unique_wegtypes):
            wegtype_data = merged_data[merged_data['wegtype_en'] == wegtype].sort_values('safety_score')
            
            # First column: wegtype label
            axes[row, 0].text(0.5, 0.5, wegtype, ha='center', va='center', 
                             transform=axes[row, 0].transAxes, fontsize=16, fontweight='bold', rotation=0)
            axes[row, 0].axis('off')
            
            # Divide into quantiles and sample randomly from each
            if len(wegtype_data) >= n_cols:
                quantiles = np.linspace(0, 1, n_cols + 1)
                for col in range(n_cols):
                    q_start = quantiles[col]
                    q_end = quantiles[col + 1]
                    start_idx = int(q_start * len(wegtype_data))
                    end_idx = int(q_end * len(wegtype_data))
                    if end_idx == start_idx:
                        end_idx = start_idx + 1
                    quantile_data = wegtype_data.iloc[start_idx:end_idx]
                    
                    if len(quantile_data) > 0:
                        # Randomly sample one image from this quantile
                        sampled_row = quantile_data.sample(n=1).iloc[0]
                        img_name = sampled_row['image_name']
                        safety_score = sampled_row['safety_score']
                        
                        # Load image (use segmented blend version)
                        img = self.load_and_resize_image(img_name, target_size=(120, 120), use_segmented=True)
                        
                        if img is not None:
                            axes[row, col + 1].imshow(img)
                        else:
                            axes[row, col + 1].text(0.5, 0.5, 'Not found', ha='center', va='center', 
                                                  transform=axes[row, col + 1].transAxes, fontsize=12)
                        
                        axes[row, col + 1].set_title(f'{safety_score:.3f}', fontsize=12, fontweight='bold')
                        axes[row, col + 1].axis('off')
                    else:
                        axes[row, col + 1].axis('off')
            else:
                # If not enough data, just show available images
                for col in range(n_cols):
                    if col < len(wegtype_data):
                        img_name = wegtype_data.iloc[col]['image_name']
                        safety_score = wegtype_data.iloc[col]['safety_score']
                        
                        # Load image (use segmented blend version)
                        img = self.load_and_resize_image(img_name, target_size=(120, 120), use_segmented=True)
                        
                        if img is not None:
                            axes[row, col + 1].imshow(img)
                        else:
                            axes[row, col + 1].text(0.5, 0.5, 'Not found', ha='center', va='center', 
                                                  transform=axes[row, col + 1].transAxes, fontsize=12)
                        
                        axes[row, col + 1].set_title(f'{safety_score:.3f}', fontsize=12, fontweight='bold')
                        axes[row, col + 1].axis('off')
                    else:
                        axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'figure_7_image_grid_by_wegtype.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 7 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_figure_8(self):
        """
        Figure 8: Grid of images sorted by predicted safety scores, grouped by buildenvironment
        """
        print("Creating Figure 8: Image Grid by Land Use and Safety Scores")
        
        if self.safety_scores is None or self.design_data is None:
            print("Warning: Cannot create Figure 8 - missing safety scores or design data")
            return None
        
        # Translation dictionary for land use types
        buildenvironment_translation = {
            'Industrieterrein': 'Industrial',
            'Woongebied': 'Residential'
        }
        
        # Merge safety scores with design data
        img1_design = self.design_data[['alt1_imageid', 'alt1_buildenvironment']].rename(
            columns={'alt1_imageid': 'image_name', 'alt1_buildenvironment': 'buildenvironment'})
        img2_design = self.design_data[['alt2_imageid', 'alt2_buildenvironment']].rename(
            columns={'alt2_imageid': 'image_name', 'alt2_buildenvironment': 'buildenvironment'})
        # add ".jpg" to image names if not already present
        img1_design['image_name'] = img1_design['image_name'].astype(str) + '.jpg'
        img2_design['image_name'] = img2_design['image_name'].astype(str) + '.jpg'
        
        all_design = pd.concat([img1_design, img2_design], ignore_index=True)
        all_design = all_design.drop_duplicates(subset='image_name', keep='first')
        
        # Translate buildenvironment to English
        all_design['buildenvironment_en'] = all_design['buildenvironment'].map(buildenvironment_translation).fillna(all_design['buildenvironment'])
        
        # Ensure string types for merging
        all_design['image_name'] = all_design['image_name'].astype(str)
        safety_scores_copy = self.safety_scores.copy()
        safety_scores_copy['image_name'] = safety_scores_copy['image_name'].astype(str)
        
        merged_data = safety_scores_copy.merge(all_design, on='image_name', how='inner')
        
        if len(merged_data) == 0:
            print("Warning: No matching images found for Figure 8")
            return None
        
        # Get unique buildenvironments and sort by safety scores within each
        unique_envs = sorted(merged_data['buildenvironment_en'].unique())
        n_envs = len(unique_envs)
        n_cols = 5
        
        # Create figure with one extra column for labels
        fig, axes = plt.subplots(n_envs, n_cols + 1, figsize=(18, 3*n_envs))
        if n_envs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Image Grid: Representative Samples by Safety Score Quantiles within Land Use Types', fontsize=20, fontweight='bold', y=0.98)
        
        for row, env in enumerate(unique_envs):
            env_data = merged_data[merged_data['buildenvironment_en'] == env].sort_values('safety_score')
            
            # First column: environment label
            axes[row, 0].text(0.5, 0.5, env, ha='center', va='center', 
                             transform=axes[row, 0].transAxes, fontsize=16, fontweight='bold', rotation=0)
            axes[row, 0].axis('off')
            
            # Divide into quantiles and sample randomly from each
            if len(env_data) >= n_cols:
                quantiles = np.linspace(0, 1, n_cols + 1)
                for col in range(n_cols):
                    q_start = quantiles[col]
                    q_end = quantiles[col + 1]
                    start_idx = int(q_start * len(env_data))
                    end_idx = int(q_end * len(env_data))
                    if end_idx == start_idx:
                        end_idx = start_idx + 1
                    quantile_data = env_data.iloc[start_idx:end_idx]
                    
                    if len(quantile_data) > 0:
                        # Randomly sample one image from this quantile
                        sampled_row = quantile_data.sample(n=1).iloc[0]
                        img_name = sampled_row['image_name']
                        safety_score = sampled_row['safety_score']
                        
                        # Load image (use segmented blend version)
                        img = self.load_and_resize_image(img_name, target_size=(120, 120), use_segmented=True)
                        
                        if img is not None:
                            axes[row, col + 1].imshow(img)
                        else:
                            axes[row, col + 1].text(0.5, 0.5, 'Not found', ha='center', va='center', 
                                                  transform=axes[row, col + 1].transAxes, fontsize=12)
                        
                        axes[row, col + 1].set_title(f'{safety_score:.3f}', fontsize=12, fontweight='bold')
                        axes[row, col + 1].axis('off')
                    else:
                        axes[row, col + 1].axis('off')
            else:
                # If not enough data, just show available images
                for col in range(n_cols):
                    if col < len(env_data):
                        img_name = env_data.iloc[col]['image_name']
                        safety_score = env_data.iloc[col]['safety_score']
                        
                        # Load image (use segmented blend version)
                        img = self.load_and_resize_image(img_name, target_size=(120, 120), use_segmented=True)
                        
                        if img is not None:
                            axes[row, col + 1].imshow(img)
                        else:
                            axes[row, col + 1].text(0.5, 0.5, 'Not found', ha='center', va='center', 
                                                  transform=axes[row, col + 1].transAxes, fontsize=12)
                        
                        axes[row, col + 1].set_title(f'{safety_score:.3f}', fontsize=12, fontweight='bold')
                        axes[row, col + 1].axis('off')
                    else:
                        axes[row, col + 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.output_dir / 'figure_8_image_grid_by_buildenvironment.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 8 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_summary_statistics(self):
        """Create and save summary statistics"""
        
        print("Creating summary statistics...")
        
        # Get unique image utilities
        unique_utilities = self.get_unique_image_utilities()
        
        stats = {
            'total_images': len(unique_utilities),
            'total_choice_situations': len(self.data),
            'stepwise_best_higher_count': len(unique_utilities[unique_utilities['utility_diff'] > 0]),
            'stepwise_wo_safety_higher_count': len(unique_utilities[unique_utilities['utility_diff'] < 0]),
            'equal_utilities_count': len(unique_utilities[unique_utilities['utility_diff'] == 0]),
            'mean_utility_diff': unique_utilities['utility_diff'].mean(),
            'std_utility_diff': unique_utilities['utility_diff'].std(),
            'max_positive_diff': unique_utilities['utility_diff'].max(),
            'max_negative_diff': unique_utilities['utility_diff'].min(),
            'stepwise_best_accuracy': self.data['correct_stepwise_best'].mean(),
            'stepwise_wo_safety_accuracy': self.data['correct_stepwise_wo_safety'].mean(),
            'mean_prediction_gap_stepwise_best': self.data['prediction_gap_stepwise_best'].mean(),
            'mean_prediction_gap_stepwise_wo_safety': self.data['prediction_gap_stepwise_wo_safety'].mean()
        }
        
        # Save statistics
        stats_path = self.output_dir / 'summary_statistics.txt'
        with open(stats_path, 'w') as f:
            f.write("POST-MODELING ANALYSIS SUMMARY STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("UTILITY DIFFERENCES:\n")
            f.write(f"  Total unique images: {stats['total_images']}\n")
            f.write(f"  Images where Stepwise Best > Stepwise wo Safety: {stats['stepwise_best_higher_count']}\n")
            f.write(f"  Images where Stepwise wo Safety > Stepwise Best: {stats['stepwise_wo_safety_higher_count']}\n")
            f.write(f"  Images with equal utilities: {stats['equal_utilities_count']}\n")
            f.write(f"  Mean utility difference: {stats['mean_utility_diff']:.4f}\n")
            f.write(f"  Std utility difference: {stats['std_utility_diff']:.4f}\n")
            f.write(f"  Max positive difference: {stats['max_positive_diff']:.4f}\n")
            f.write(f"  Max negative difference: {stats['max_negative_diff']:.4f}\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write(f"  Total choice situations: {stats['total_choice_situations']}\n")
            f.write(f"  Stepwise Best accuracy: {stats['stepwise_best_accuracy']:.3f}\n")
            f.write(f"  Stepwise wo Safety accuracy: {stats['stepwise_wo_safety_accuracy']:.3f}\n")
            f.write(f"  Mean prediction gap (Stepwise Best): {stats['mean_prediction_gap_stepwise_best']:.3f}\n")
            f.write(f"  Mean prediction gap (Stepwise wo Safety): {stats['mean_prediction_gap_stepwise_wo_safety']:.3f}\n")
        
        print(f"Summary statistics saved to: {stats_path}")
        
        return stats
    
    def run_all_analyses(self):
        """Run all analyses and create all figures"""
        
        print("=== POST-MODELING ANALYSIS ===")
        print(f"Output directory: {self.output_dir}")
        
        # Create all figures
        results = {}
        
        # Figures 1-3: Utility comparisons and model corrections
        results['top_5_best_higher'] = self.create_figure_1()
        results['top_5_wo_safety_higher'] = self.create_figure_2()
        results['model_corrections'] = self.create_figure_3()
        
        # Figures 4-5: Safety score distributions
        results['safety_by_wegtype'] = self.create_figure_4()
        results['safety_by_buildenvironment'] = self.create_figure_5()
        
        # Figure 6: Scatter plot matrix
        results['scatter_matrix'] = self.create_figure_6()
        
        # Figures 7-8: Image grids
        results['image_grid_wegtype'] = self.create_figure_7()
        results['image_grid_buildenvironment'] = self.create_figure_8()
        
        # Create summary statistics
        results['statistics'] = self.create_summary_statistics()
        
        print("\n=== ANALYSIS COMPLETED ===")
        print(f"All 8 figures and statistics saved to: {self.output_dir}")
        
        return results


def main():
    """Main function to run post-modeling analysis"""
    
    # Check if comparison file exists
    comparison_file = 'data/processed/model_results/df_choice_with_Vimg_comparison.csv'
    
    if not Path(comparison_file).exists():
        print(f"Error: Comparison file not found: {comparison_file}")
        print("Please run the choice model benchmark script first to generate the comparison data.")
        return
    
    # Run analysis
    analyzer = PostModelingAnalyzer(comparison_file)
    results = analyzer.run_all_analyses()
    
    print("\n✓ Post-modeling analysis completed successfully!")


if __name__ == "__main__":
    main() 