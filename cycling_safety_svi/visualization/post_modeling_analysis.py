"""
Post-Modeling Analysis Script

This script generates 10 visualizations comparing the stepwise_best and stepwise_wo_safety models:
1. Figure 1: Top 5 images where stepwise_best utilities > stepwise_wo_safety utilities
2. Figure 2: Top 5 images where stepwise_wo_safety utilities > stepwise_best utilities  
3. Figure 3: Top 5 image pairs with wrong choice using stepwise_wo_safety but correct with stepwise_best
4. Figure 4: Histogram of predicted safety scores grouped by wegtype (road type)
5. Figure 5: Histogram of predicted safety scores grouped by buildenvironment (land use)
6. Figure 6: Scatter plots between predicted safety scores, utilities, and segmentation features
7. Figure 7: Grid of images sorted by predicted safety scores, grouped by wegtype
8. Figure 8: Grid of images sorted by predicted safety scores, grouped by buildenvironment
9. Figure 9: 3D scatter plot of safety score, utility, and vegetation for eye-catching visualization.
10. Figure 10: 4D scatter plot of vegetation, terrain, and safety, with utility as dot size.
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
        self.gradcam_overlays_dir = Path('data/processed/gradcam_visualizations/overlays/')
        
        # Load all data
        self.load_data()
        
        # Check for Grad-CAM data
        self.has_gradcam_data = self.gradcam_overlays_dir.exists()
        if not self.has_gradcam_data:
            print(f"Warning: Grad-CAM overlay directory not found at {self.gradcam_overlays_dir}")
            print("Figures 7 and 8 will not include Grad-CAM images.")
        else:
            print(f"✓ Found Grad-CAM overlays at: {self.gradcam_overlays_dir}")
        
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
            blend_path = Path("data/processed/segmented_images") / blend_name
            
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
    
    def load_gradcam_overlay(self, image_name, target_size=(120, 120)):
        """
        Load a Grad-CAM overlay image.
        """
        overlay_name = f"overlay_{image_name}"
        overlay_path = Path(self.gradcam_overlays_dir) / overlay_name
        
        if overlay_path.exists():
            try:
                img = Image.open(overlay_path).convert('RGB')
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                return img
            except Exception as e:
                print(f"Warning: Could not load Grad-CAM overlay {overlay_name}: {e}")
                return None
        
        return None
    
    def create_figure_1(self):
        """
        Figure 1: Combined comparison of top 5 images where models differ most
        Left side: stepwise_best > stepwise_w/o_safety
        Right side: stepwise_w/o_safety > stepwise_best
        """
        print("Creating Figure 1: Combined Model Comparison")
        
        # Get unique image utilities
        unique_utilities = self.get_unique_image_utilities()
        
        # Filter for positive and negative percentile differences
        positive_diffs = unique_utilities[unique_utilities['percentile_diff'] > 0]
        negative_diffs = unique_utilities[unique_utilities['percentile_diff'] < 0]
        
        # Get top 5 for each case
        top_5_positive = positive_diffs.nlargest(5, 'percentile_diff')
        top_5_negative = negative_diffs.nsmallest(5, 'percentile_diff')
        
        # Create figure with 5 rows and 4 columns (2 images + 2 bar plots per row)
        fig, axes = plt.subplots(5, 4, figsize=(20, 28), gridspec_kw={'width_ratios': [3, 2, 3, 2]})
        fig.suptitle('Model Comparison: Images with Largest Utility Differences\n' + 
                     'Left: Higher with Safety Model | Right: Higher without Safety Model', 
                     fontsize=22, fontweight='bold', y=0.96)
        
        # Column headers
        axes[0, 0].text(0.5, 1.1, 'HIGHER WITH SAFETY MODEL', 
                       transform=axes[0, 0].transAxes, ha='center', va='bottom',
                       fontsize=18, fontweight='bold', color='darkgreen')
        axes[0, 1].text(0.5, 1.1, 'Percentile Comparison', 
                       transform=axes[0, 1].transAxes, ha='center', va='bottom',
                       fontsize=18, fontweight='bold', color='darkgreen')
        axes[0, 2].text(0.5, 1.1, 'HIGHER WITHOUT SAFETY MODEL', 
                       transform=axes[0, 2].transAxes, ha='center', va='bottom',
                       fontsize=18, fontweight='bold', color='darkred')
        axes[0, 3].text(0.5, 1.1, 'Percentile Comparison', 
                       transform=axes[0, 3].transAxes, ha='center', va='bottom',
                       fontsize=18, fontweight='bold', color='darkred')
        
        for i in range(5):
            # Left side: Positive differences (safety model performs better)
            if i < len(top_5_positive):
                row_pos = top_5_positive.iloc[i]
                
                # Load and display image
                img = self.load_and_resize_image(row_pos['image_name'], target_size=(240, 160))
                if img is not None:
                    axes[i, 0].imshow(img)
                else:
                    axes[i, 0].text(0.5, 0.5, "Image not found", 
                                   ha='center', va='center', transform=axes[i, 0].transAxes, fontsize=14)
                
                axes[i, 0].axis('off')
                
                # Bar plot for positive case
                models = ['With Safety', 'Without Safety']
                percentiles = [row_pos['percentile_stepwise_best'], row_pos['percentile_stepwise_wo_safety']]
                colors = ['darkgreen', 'lightgray']
                
                bars = axes[i, 1].bar(models, percentiles, color=colors, alpha=0.8)
                axes[i, 1].set_ylabel('Percentile', fontsize=14)
                axes[i, 1].tick_params(axis='x', labelsize=12, rotation=45)
                axes[i, 1].tick_params(axis='y', labelsize=12)
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, percentile in zip(bars, percentiles):
                    height = bar.get_height()
                    axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{percentile:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Add difference annotation
                diff = row_pos['percentile_diff']
                axes[i, 1].text(0.5, 0.95, f"+{diff:.1f}%", 
                               transform=axes[i, 1].transAxes, ha='center', va='top',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
            else:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                
            # Right side: Negative differences (without safety model performs better)
            if i < len(top_5_negative):
                row_neg = top_5_negative.iloc[i]
                
                # Load and display image
                img = self.load_and_resize_image(row_neg['image_name'], target_size=(240, 160))
                if img is not None:
                    axes[i, 2].imshow(img)
                else:
                    axes[i, 2].text(0.5, 0.5, "Image not found", 
                                   ha='center', va='center', transform=axes[i, 2].transAxes, fontsize=14)
                
                axes[i, 2].axis('off')
                
                # Bar plot for negative case
                models = ['With Safety', 'Without Safety']
                percentiles = [row_neg['percentile_stepwise_best'], row_neg['percentile_stepwise_wo_safety']]
                colors = ['lightgray', 'darkred']
                
                bars = axes[i, 3].bar(models, percentiles, color=colors, alpha=0.8)
                axes[i, 3].set_ylabel('Percentile', fontsize=14)
                axes[i, 3].tick_params(axis='x', labelsize=12, rotation=45)
                axes[i, 3].tick_params(axis='y', labelsize=12)
                axes[i, 3].grid(True, alpha=0.3)
                axes[i, 3].set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, percentile in zip(bars, percentiles):
                    height = bar.get_height()
                    axes[i, 3].text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{percentile:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                # Add difference annotation
                diff = row_neg['percentile_diff']
                axes[i, 3].text(0.5, 0.95, f"{diff:.1f}%", 
                               transform=axes[i, 3].transAxes, ha='center', va='top',
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
            else:
                axes[i, 2].axis('off')
                axes[i, 3].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        fig_path = self.output_dir / 'figure_1_combined_model_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure 1 saved to: {fig_path}")
        plt.close()
        
        return {'top_5_positive': top_5_positive, 'top_5_negative': top_5_negative}
    
    def create_figure_2(self):
        """
        Figure 2: Placeholder - merged into Figure 1
        This method is kept for backward compatibility but does nothing
        """
        print("Figure 2 has been merged into Figure 1")
        return None
    
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
        fig, axes = plt.subplots(5, 4, figsize=(20, 28), gridspec_kw={'width_ratios': [3, 2, 3, 2]})
        fig.suptitle('Top 5 Image Pairs: Wrong Choice without Percevied Safety, Correct with Perceived Safety\n(Probability Improvements)', 
                     fontsize=20, fontweight='bold', y=0.96)
        
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
            img_chosen = self.load_and_resize_image(chosen_img, target_size=(250, 200))
            if img_chosen is not None:
                axes[i, 0].imshow(img_chosen)
            else:
                axes[i, 0].text(0.5, 0.5, f"Image not found:\n{chosen_img}", 
                               ha='center', va='center', transform=axes[i, 0].transAxes)
            
            axes[i, 0].set_title(f"CHOSEN", fontsize=16, color='green', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Column 2: Probabilities for chosen image
            model_names = ['With Safety', 'Without Safety']
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
            img_not_chosen = self.load_and_resize_image(not_chosen_img, target_size=(250, 200))
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
        
        # Translation dictionary for land use types (expanded to match Figure 8)
        buildenvironment_translation = {
            'Hoofdweg': 'Main road',
            'Industriet': 'Industrial',
            'Recreatie': 'Recreation',
            'Woongebied': 'Residential',
            'Wijkontslu': 'Access road',
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
        
        # Translate buildenvironment to English (apply translation as in Figure 8)
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
        Figure 6: Scatter plots between predicted safety scores, utilities, Car, Terrain, Bike Lane, Road, Vegetation.
        Also saves the correlation matrix to a text file.
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
        
        # Compute correlation matrix for the selected variables
        corr_matrix = merged_data[available_vars].corr()
        
        # Save correlation matrix to a text file
        corr_txt_path = self.output_dir / 'figure_6_correlation_matrix.txt'
        with open(corr_txt_path, 'w') as f:
            f.write("Correlation matrix for Figure 6 variables:\n")
            f.write(corr_matrix.to_string(float_format="%.3f"))
        print(f"Correlation matrix saved to: {corr_txt_path}")
        
        # Create scatter plot matrix
        n_vars = len(available_vars)
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(20, 20))
        fig.suptitle('Scatter Plot Matrix: Safety Scores, Utilities, and Segmentation Features', fontsize=24, fontweight='bold', y=0.98)
        
        for i, var1 in enumerate(available_vars):
            for j, var2 in enumerate(available_vars):
                if i == j:
                    # Diagonal: histogram
                    axes[i, j].hist(merged_data[var1], bins=20, alpha=0.7, color='skyblue')
                    axes[i, j].set_title(var1, fontsize=18, fontweight='bold')
                    axes[i, j].tick_params(axis='both', labelsize=14)
                else:
                    # Off-diagonal: scatter plot
                    axes[i, j].scatter(merged_data[var2], merged_data[var1], alpha=0.6, s=25)
                    
                    # Calculate and display correlation
                    corr = merged_data[var1].corr(merged_data[var2])
                    axes[i, j].text(0.05, 0.95, f'r={corr:.3f}', transform=axes[i, j].transAxes, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                                   fontsize=16, fontweight='bold')
                    axes[i, j].tick_params(axis='both', labelsize=14)
                
                if i == n_vars - 1:
                    axes[i, j].set_xlabel(var2, fontsize=18)
                if j == 0:
                    axes[i, j].set_ylabel(var1, fontsize=18)
        
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
            'Fietspad vrijliggend': 'Separated\ncycle path',
            'Solitair fietspad': 'Separated\ncycle path',
            'Normale weg': 'Normal road',
            'Fietsstrook': 'Painted\ncycle path',
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
        
        if self.has_gradcam_data:
            self._create_image_grid_figure_with_cam(
                merged_data=merged_data,
                grouping_col='wegtype_en',
                unique_groups=unique_wegtypes,
                n_groups=n_wegtypes,
                figure_num=7,
                title='Examples of Images by Safety Score Quantiles within Road Types'
            )
        else:
            self._create_image_grid_figure_without_cam(
                merged_data=merged_data,
                grouping_col='wegtype_en',
                unique_groups=unique_wegtypes,
                n_groups=n_wegtypes,
                figure_num=7,
                title='Examples of Images by Safety Score Quantiles within Road Types'
            )
        
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
            'Hoofdweg': 'Main road',
            'Industriet': 'Industrial',
            'Recreatie': 'Recreation',
            'Woongebied': 'Residential',
            'Wijkontslu': 'Access road',
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

        if self.has_gradcam_data:
            self._create_image_grid_figure_with_cam(
                merged_data=merged_data,
                grouping_col='buildenvironment_en',
                unique_groups=unique_envs,
                n_groups=n_envs,
                figure_num=8,
                title='Examples of Images by Safety Score Quantiles within Land Use Types'
            )
        else:
            self._create_image_grid_figure_without_cam(
                merged_data=merged_data,
                grouping_col='buildenvironment_en',
                unique_groups=unique_envs,
                n_groups=n_envs,
                figure_num=8,
                title='Examples of Images by Safety Score Quantiles within Land Use Types'
            )
        
        return merged_data
    
    def create_figure_9_3d_scatter(self):
        """
        Figure 9: 3D scatter plot of safety score, utility, and vegetation for eye-catching visualization.
        """
        print("Creating Figure 9: 3D Scatter Plot")

        if self.safety_scores is None or self.pixel_ratios is None:
            print("Warning: Cannot create Figure 9 - missing safety scores or pixel ratios")
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

        if len(merged_data) == 0 or 'Vegetation' not in merged_data.columns:
            print("Warning: No matching data or 'Vegetation' column found for 3D scatter plot")
            return None

        # Select variables for the plot
        x = merged_data['safety_score']
        y = merged_data['utility_stepwise_best']
        z = merged_data['Vegetation']
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        ax.scatter(x, y, z, c='purple', marker='o', s=25, alpha=0.6, depthshade=True)
        
        # Remove axes, panes, and text
        ax.set_axis_off()
        
        # Set transparent background for the figure and axes
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Set a nice perspective
        ax.view_init(elev=25, azim=135)
        
        # Save figure
        fig_path = self.output_dir / 'figure_9_3d_scatter_eyecatcher.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
        print(f"Figure 9 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def create_figure_10_4d_scatter(self):
        """
        Figure 10: 4D scatter plot of vegetation, terrain, and safety, with utility as dot size.
        """
        print("Creating Figure 10: 4D Scatter Plot")

        if self.safety_scores is None or self.pixel_ratios is None:
            print("Warning: Cannot create Figure 10 - missing safety scores or pixel ratios")
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

        required_cols = ['Vegetation', 'Terrain', 'safety_score', 'utility_stepwise_best']
        if not all(col in merged_data.columns for col in required_cols):
            print(f"Warning: Missing one or more required columns for Figure 10: {required_cols}")
            return None

        # Select variables for the plot
        x = merged_data['Vegetation']
        y = merged_data['Terrain']
        z = merged_data['safety_score']
        size = merged_data['utility_stepwise_best']
        
        # Scale utility to be used as size (s must be non-negative)
        # We'll normalize it to a range, e.g., 10 to 500
        size_scaled = np.interp(size, (size.min(), size.max()), (10, 500))

        # Create figure
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create scatter plot
        ax.scatter(x, y, z, s=size_scaled, c='purple', marker='o', alpha=0.6, depthshade=True)
        
        # Remove axes, panes, and text
        ax.set_axis_off()
        
        # Set transparent background for the figure and axes
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        # Set a nice perspective
        ax.view_init(elev=30, azim=45)
        
        # Save figure
        fig_path = self.output_dir / 'figure_10_4d_scatter_eyecatcher.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)
        print(f"Figure 10 saved to: {fig_path}")
        plt.close()
        
        return merged_data
    
    def _create_image_grid_figure_with_cam(self, merged_data, grouping_col, unique_groups, n_groups, figure_num, title):
        """Helper function to create image grid with Grad-CAM overlays."""
        n_cols = 3
        n_image_cols = n_cols * 2
        
        fig, axes = plt.subplots(n_groups, n_image_cols + 1, figsize=(18, 3 * n_groups), 
                                 gridspec_kw={'width_ratios': [1.5] + [1]*n_image_cols})
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        # fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        col_names = ["Minimum", "Median", "Maximum"]

        for row, group_val in enumerate(unique_groups):
            group_data = merged_data[merged_data[grouping_col] == group_val].sort_values('safety_score')
            
            axes[row, 0].text(0.5, 0.5, group_val, ha='center', va='center', 
                             transform=axes[row, 0].transAxes, fontsize=26, fontweight='bold', rotation=0, wrap=True)
            axes[row, 0].axis('off')
            
            if len(group_data) >= n_cols:
                images_to_plot = [
                    group_data.iloc[0],  # Minimum
                    group_data.iloc[len(group_data) // 2],  # Median
                    group_data.iloc[-1]  # Maximum
                ]
                
                for col in range(n_cols):
                    ax_segmented = axes[row, 1 + 2 * col]
                    ax_gradcam = axes[row, 1 + 2 * col + 1]
                    
                    if row == 0:
                        ax_segmented.set_title(f"Segmented\n({col_names[col]})", fontsize=20)
                        ax_gradcam.set_title(f"Grad-CAM\n({col_names[col]})", fontsize=20)

                    sampled = images_to_plot[col]
                    img_name, score = sampled['image_name'], sampled['safety_score']
                        
                    seg_img = self.load_and_resize_image(img_name, use_segmented=True)
                    cam_img = self.load_gradcam_overlay(img_name)
                    
                    ax_segmented.imshow(seg_img if seg_img else self._create_placeholder("Not Found"))
                    ax_gradcam.imshow(cam_img if cam_img else self._create_placeholder("No CAM"))
                    
                    ax_segmented.text(0.05, 0.95, f'{score:.3f}', transform=ax_segmented.transAxes, 
                                      ha='left', va='top', color='white', fontsize=12,
                                      bbox=dict(facecolor='black', alpha=0.5, pad=2))
                    
                    ax_segmented.axis('off')
                    ax_gradcam.axis('off')
            else:
                for col in range(n_image_cols):
                    axes[row, 1+col].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig_path = self.output_dir / f'figure_{figure_num}_image_grid_by_{grouping_col}_with_cam.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure {figure_num} with CAM saved to: {fig_path}")
        plt.close()

    def _create_image_grid_figure_without_cam(self, merged_data, grouping_col, unique_groups, n_groups, figure_num, title):
        """Helper function to create image grid without Grad-CAM overlays (original logic)."""
        n_cols = 3
        fig, axes = plt.subplots(n_groups, n_cols + 1, figsize=(12, 3 * n_groups),
                                 gridspec_kw={'width_ratios': [1.5] + [1]*n_cols})
        if n_groups == 1:
            axes = axes.reshape(1, -1)
        
        # fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        col_names = ["Minimum", "Median", "Maximum"]

        for row, group_val in enumerate(unique_groups):
            group_data = merged_data[merged_data[grouping_col] == group_val].sort_values('safety_score')
            
            axes[row, 0].text(0.5, 0.5, group_val, ha='center', va='center', 
                             transform=axes[row, 0].transAxes, fontsize=16, fontweight='bold', rotation=0, wrap=True)
            axes[row, 0].axis('off')
            
            if len(group_data) >= n_cols:
                images_to_plot = [
                    group_data.iloc[0],  # Minimum
                    group_data.iloc[len(group_data) // 2],  # Median
                    group_data.iloc[-1]  # Maximum
                ]

                for col in range(n_cols):
                    ax = axes[row, col + 1]

                    if row == 0:
                        ax.set_title(col_names[col], fontsize=14, fontweight='bold')

                    sampled = images_to_plot[col]
                    img_name, score = sampled['image_name'], sampled['safety_score']
                    
                    img = self.load_and_resize_image(img_name, use_segmented=True)
                    ax.imshow(img if img else self._create_placeholder("Not Found"))
                    
                    ax.text(0.5, -0.15, f'{score:.3f}', transform=ax.transAxes, ha='center', fontsize=12)
                    
                    ax.axis('off')
            else:
                for col in range(n_cols):
                    axes[row, col+1].axis('off')
        
        plt.tight_layout()
        fig_path = self.output_dir / f'figure_{figure_num}_image_grid_by_{grouping_col}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure {figure_num} saved to: {fig_path}")
        plt.close()

    def _create_placeholder(self, text, size=(120, 120)):
        """Creates a placeholder image with text."""
        img = Image.new('RGB', size, color='lightgray')
        # No drawing on the image to avoid dependency on Pillow's ImageDraw
        return img

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
        results['combined_model_comparison'] = self.create_figure_1()
        # Figure 2 is now merged into Figure 1, so we skip it
        results['model_corrections'] = self.create_figure_3()
        
        # Figures 4-5: Safety score distributions
        results['safety_by_wegtype'] = self.create_figure_4()
        results['safety_by_buildenvironment'] = self.create_figure_5()
        
        # Figure 6: Scatter plot matrix
        results['scatter_matrix'] = self.create_figure_6()
        
        # Figures 7-8: Image grids
        results['image_grid_wegtype'] = self.create_figure_7()
        results['image_grid_buildenvironment'] = self.create_figure_8()
        
        # Figure 9: 3D scatter plot
        results['3d_scatter'] = self.create_figure_9_3d_scatter()
        
        # Figure 10: 4D scatter plot
        results['4d_scatter'] = self.create_figure_10_4d_scatter()
        
        # Create summary statistics
        results['statistics'] = self.create_summary_statistics()
        
        print("\n=== ANALYSIS COMPLETED ===")
        print(f"All 10 figures and statistics saved to: {self.output_dir}")
        print("Note: Figures 1 and 2 have been merged into a single combined comparison figure")
        
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