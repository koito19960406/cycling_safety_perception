"""
Safety-Demographics Interaction Model (Enhanced Version)

This script extends the best choice model with safety * demographics interaction effects.
It includes age, gender, education, income, and cycling incident variables as well as 
segmentation features.

Usage:
    python safety_demographics_interaction_model.py
"""

import os
import pandas as pd
import numpy as np
import sqlite3
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
import logging
from biogeme.expressions import Beta, Variable, log, exp
from pathlib import Path
from datetime import datetime
import json


class SafetyDemographicsInteractionModelV2:
    """Extends choice model with safety * demographics interaction effects using merged data"""
    
    def __init__(self, output_dir='reports/models/demographics_interaction'):
        """Initialize the demographics interaction model"""
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"safety_demographics_v2_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Safety-Demographics Interaction Model V2 (Enhanced)")
        print(f"Output directory: {self.output_dir}")
        
        # Configure logging
        logger = logging.getLogger('biogeme')
        logger.setLevel(logging.WARNING)
        
        # Age and gender mappings (from database analysis)
        self.age_mapping = {1: '18-30', 2: '31-45', 3: '46-60', 4: '60+', 5: 'other'}
        self.gender_mapping = {1: 'male', 2: 'female', 3: 'other'}
        
        # Additional demographic mappings (based on actual database values)
        # Note: These will be updated based on actual data distributions
        self.education_mapping = {}
        self.income_mapping = {}
        self.cycling_incident_mapping = {}
        self.cycler_mapping = {}
        self.cyclinglike_mapping = {}
        self.cyclingunsafe_mapping = {}
        self.biketype_mapping = {}
        self.work_mapping = {}
        self.car_mapping = {}
        
        # Manual mapping from parameter names to actual segmentation feature names
        self.feature_name_mapping = {
            'barrier': 'Barrier',
            'bicycle': 'Bicycle',
            'bike lane': 'Bike Lane',
            'billboard': 'Billboard',
            'boat': 'Boat',
            'bridge': 'Bridge',
            'building': 'Building',
            'bus': 'Bus',
            'car': 'Car',
            'curb': 'Curb',
            'fence': 'Fence',
            'guard rail': 'Guard Rail',
            'lane marking   crosswalk': 'Lane Marking - Crosswalk',
            'lane marking   general': 'Lane Marking - General',
            'on rails': 'On Rails',
            'other vehicle': 'Other Vehicle',
            'parking': 'Parking',
            'pedestrian area': 'Pedestrian Area',
            'person': 'Person',
            'pole': 'Pole',
            'rail track': 'Rail Track',
            'road': 'Road',
            'sand': 'Sand',
            'sidewalk': 'Sidewalk',
            'sky': 'Sky',
            'snow': 'Snow',
            'terrain': 'Terrain',
            'traffic sign (front)': 'Traffic Sign (Front)',
            'trash can': 'Trash Can',
            'truck': 'Truck',
            'tunnel': 'Tunnel',
            'utility pole': 'Utility Pole',
            'vegetation': 'Vegetation',
            'wall': 'Wall',
            'water': 'Water'
        }
        
    def load_and_prepare_data(self, 
                             cv_dcm_path='data/raw/cv_dcm.csv',
                             database_path='data/raw/database_2024_10_07_135133.db',
                             safety_scores_path='data/processed/predicted_danish/cycling_safety_scores.csv',
                             segmentation_path='data/processed/segmentation_results/pixel_ratios.csv'):
        """Load and prepare all datasets with enhanced demographics and segmentation"""
        
        print("Loading choice data and enhanced demographics...")
        
        # Load choice data
        self.choice_data = pd.read_csv(cv_dcm_path)
        print(f"Loaded choice data: {len(self.choice_data)} observations")
        
        # Load enhanced demographics from database
        self._load_enhanced_demographics(database_path)
        
        # Load safety scores
        self.safety_scores = pd.read_csv(safety_scores_path)
        self.safety_scores['image_name'] = self.safety_scores['image_name'].str.strip()
        print(f"Loaded safety scores: {len(self.safety_scores)} images")
        
        # Load segmentation data
        print("Loading segmentation data...")
        try:
            seg_chunks = []
            for chunk in pd.read_csv(segmentation_path, chunksize=1000):
                seg_chunks.append(chunk)
            self.segmentation_data = pd.concat(seg_chunks, ignore_index=True)
            self.segmentation_data['filename_key'] = self.segmentation_data['filename_key'].str.strip()
            print(f"Loaded segmentation data: {len(self.segmentation_data)} images")
        except Exception as e:
            print(f"Warning: Could not load segmentation data: {e}")
            self.segmentation_data = None
        
        # Merge all datasets
        self._merge_all_datasets()
        
        # Process demographics
        self._process_demographics()
        
        print(f"Final dataset: {len(self.merged_data)} observations")
        
    def _load_enhanced_demographics(self, database_path):
        """Load enhanced demographics from database including education, income, and cycling incident"""
        
        print("Loading enhanced demographics from database...")
        
        # Connect to database
        conn = sqlite3.connect(database_path)
        
        # Get enhanced demographics with set_id
        self.demographics = pd.read_sql_query('''
            SELECT respondent_id, set_id, age, gender, education, income, 
                   cyclingincident, cycler, cyclinglike, cyclingunsafe, 
                   biketype, work, car
            FROM Response 
            WHERE age IS NOT NULL AND gender IS NOT NULL
        ''', conn)
        
        conn.close()
        
        print(f"Demographics data: {len(self.demographics)} participants")
        print(f"Set_id range in demographics: {self.demographics['set_id'].min()} to {self.demographics['set_id'].max()}")
        
        # Print distributions and create mappings based on actual data
        print("\nAdditional demographic distributions:")
        
        if 'education' in self.demographics.columns:
            education_counts = self.demographics['education'].value_counts().sort_index()
            print("Education distribution:")
            print(education_counts)
            
            # Create education mapping based on observed values
            unique_education = sorted([x for x in self.demographics['education'].unique() if pd.notna(x)])
            education_labels = ['primary', 'secondary', 'vocational', 'bachelor', 'master', 'phd', 'doctorate', 'other1', 'other2', 'other3', 'other4', 'other5', 'other6', 'other7']
            self.education_mapping = {val: education_labels[i] if i < len(education_labels) else f'education_{int(val)}' 
                                    for i, val in enumerate(unique_education)}
            print(f"Education mapping: {self.education_mapping}")
        
        if 'income' in self.demographics.columns:
            income_counts = self.demographics['income'].value_counts().sort_index()
            print("Income distribution:")
            print(income_counts)
            
            # Create income mapping based on observed values
            unique_income = sorted([x for x in self.demographics['income'].unique() if pd.notna(x)])
            income_labels = ['very_low', 'low', 'medium_low', 'medium', 'medium_high', 'high', 'very_high', 'highest']
            self.income_mapping = {val: income_labels[i] if i < len(income_labels) else f'income_{int(val)}' 
                                 for i, val in enumerate(unique_income)}
            print(f"Income mapping: {self.income_mapping}")
        
        if 'cyclingincident' in self.demographics.columns:
            incident_counts = self.demographics['cyclingincident'].value_counts().sort_index()
            print("Cycling incident distribution:")
            print(incident_counts)
            
            # Create cycling incident mapping based on observed values
            unique_incident = sorted([x for x in self.demographics['cyclingincident'].unique() if pd.notna(x)])
            incident_labels = ['never', 'once', 'multiple_times']
            self.cycling_incident_mapping = {val: incident_labels[i] if i < len(incident_labels) else f'incident_{int(val)}' 
                                           for i, val in enumerate(unique_incident)}
            print(f"Cycling incident mapping: {self.cycling_incident_mapping}")

        # Mappings for new variables
        if 'cycler' in self.demographics.columns:
            cycler_counts = self.demographics['cycler'].value_counts().sort_index()
            print("Cycler distribution:")
            print(cycler_counts)
            unique_cycler = sorted([x for x in self.demographics['cycler'].unique() if pd.notna(x)])
            cycler_labels = ['daily', '4-6_days_week', '1-3_days_week', 'monthly', 'less_than_monthly', 'only_on_holidays', 'never']
            self.cycler_mapping = {val: cycler_labels[i] if i < len(cycler_labels) else f'cycler_{int(val)}' for i, val in enumerate(unique_cycler)}
            print(f"Cycler mapping: {self.cycler_mapping}")

        if 'cyclinglike' in self.demographics.columns:
            cyclinglike_counts = self.demographics['cyclinglike'].value_counts().sort_index()
            print("Cyclinglike distribution:")
            print(cyclinglike_counts)
            unique_cyclinglike = sorted([x for x in self.demographics['cyclinglike'].unique() if pd.notna(x)])
            cyclinglike_labels = ['yes', 'no']
            self.cyclinglike_mapping = {val: cyclinglike_labels[i] if i < len(cyclinglike_labels) else f'cyclinglike_{int(val)}' for i, val in enumerate(unique_cyclinglike)}
            print(f"Cyclinglike mapping: {self.cyclinglike_mapping}")

        if 'cyclingunsafe' in self.demographics.columns:
            cyclingunsafe_counts = self.demographics['cyclingunsafe'].value_counts().sort_index()
            print("Cyclingunsafe distribution:")
            print(cyclingunsafe_counts)
            unique_cyclingunsafe = sorted([x for x in self.demographics['cyclingunsafe'].unique() if pd.notna(x)])
            cyclingunsafe_labels = ['never', 'sometimes', 'often']
            self.cyclingunsafe_mapping = {val: cyclingunsafe_labels[i] if i < len(cyclingunsafe_labels) else f'cyclingunsafe_{int(val)}' for i, val in enumerate(unique_cyclingunsafe)}
            print(f"Cyclingunsafe mapping: {self.cyclingunsafe_mapping}")

        if 'biketype' in self.demographics.columns:
            biketype_counts = self.demographics['biketype'].value_counts().sort_index()
            print("Biketype distribution:")
            print(biketype_counts)
            unique_biketype = sorted([x for x in self.demographics['biketype'].unique() if pd.notna(x)])
            biketype_labels = ['regular_bike', 'e-bike', 'sport_bike', 'cargo_bike', 'other']
            self.biketype_mapping = {val: biketype_labels[i] if i < len(biketype_labels) else f'biketype_{int(val)}' for i, val in enumerate(unique_biketype)}
            print(f"Biketype mapping: {self.biketype_mapping}")

        if 'work' in self.demographics.columns:
            work_counts = self.demographics['work'].value_counts().sort_index()
            print("Work distribution:")
            print(work_counts)
            unique_work = sorted([x for x in self.demographics['work'].unique() if pd.notna(x)])
            work_labels = ['full-time', 'part-time', 'student', 'unemployed', 'retired', 'other']
            self.work_mapping = {val: work_labels[i] if i < len(work_labels) else f'work_{int(val)}' for i, val in enumerate(unique_work)}
            print(f"Work mapping: {self.work_mapping}")

        if 'car' in self.demographics.columns:
            car_counts = self.demographics['car'].value_counts().sort_index()
            print("Car distribution:")
            print(car_counts)
            unique_car = sorted([x for x in self.demographics['car'].unique() if pd.notna(x)])
            car_labels = ['no_car', 'one_car', 'multiple_cars', 'company_car']
            self.car_mapping = {val: car_labels[i] if i < len(car_labels) else f'car_{int(val)}' for i, val in enumerate(unique_car)}
            print(f"Car mapping: {self.car_mapping}")
    
    def _merge_all_datasets(self):
        """Merge choice data, demographics, safety scores, and segmentation data"""
        
        print("Merging all datasets...")
        
        # Start with choice data
        merged_data = self.choice_data.copy()
        
        # Merge with demographics
        merged_data = merged_data.merge(self.demographics, left_on='RID', right_on='set_id', how='inner')
        print(f"After demographics merge: {len(merged_data)} observations")
        
        # Add safety scores for both alternatives
        safety_dict = dict(zip(self.safety_scores['image_name'], self.safety_scores['safety_score']))
        merged_data['safety_score_1'] = merged_data['IMG1'].map(safety_dict)
        merged_data['safety_score_2'] = merged_data['IMG2'].map(safety_dict)
        
        # Fill missing safety scores with mean
        mean_safety = self.safety_scores['safety_score'].mean()
        merged_data['safety_score_1'] = merged_data['safety_score_1'].fillna(mean_safety)
        merged_data['safety_score_2'] = merged_data['safety_score_2'].fillna(mean_safety)
        print(f"✓ Safety scores added (mean safety: {mean_safety:.3f})")
        
        # Add segmentation features if available
        if self.segmentation_data is not None:
            segmentation_dict = {}
            for _, row in self.segmentation_data.iterrows():
                img_name = row['filename_key'] + '.jpg'
                if pd.isna(row['filename_key']):
                    continue
                features = row.drop('filename_key').to_dict()
                segmentation_dict[img_name] = features
            
            # Get segmentation feature names
            seg_feature_names = [col for col in self.segmentation_data.columns if col != 'filename_key']
            
            # Create segmentation features efficiently using pd.concat
            seg_features_dict = {}
            for feature in seg_feature_names:
                seg_features_dict[f"{feature}_1"] = merged_data['IMG1'].map(lambda x: segmentation_dict.get(x, {}).get(feature, 0))
                seg_features_dict[f"{feature}_2"] = merged_data['IMG2'].map(lambda x: segmentation_dict.get(x, {}).get(feature, 0))
            
            # Add all segmentation features at once to avoid fragmentation
            seg_features_df = pd.DataFrame(seg_features_dict, index=merged_data.index)
            merged_data = pd.concat([merged_data, seg_features_df], axis=1)
            
            print(f"✓ Segmentation features added: {len(seg_feature_names)} features")
        
        self.merged_data = merged_data
        print(f"Final merged dataset: {len(merged_data)} observations, {len(merged_data.columns)} features")
        

        
    def _process_demographics(self):
        """Process demographic variables and create categories"""
        
        print("Processing demographic variables...")
        
        # Create all demographic category mappings at once to avoid fragmentation
        demo_categories = {}
        demo_categories['age_category'] = self.merged_data['age'].map(self.age_mapping)
        demo_categories['gender_category'] = self.merged_data['gender'].map(self.gender_mapping)
        
        # Map additional demographics if available
        if 'education' in self.merged_data.columns and self.education_mapping:
            demo_categories['education_category'] = self.merged_data['education'].map(self.education_mapping)
        if 'income' in self.merged_data.columns and self.income_mapping:
            demo_categories['income_category'] = self.merged_data['income'].map(self.income_mapping)
        if 'cyclingincident' in self.merged_data.columns and self.cycling_incident_mapping:
            demo_categories['cycling_incident_category'] = self.merged_data['cyclingincident'].map(self.cycling_incident_mapping)
        
        # Map new demographic variables
        if 'cycler' in self.merged_data.columns and self.cycler_mapping:
            demo_categories['cycler_category'] = self.merged_data['cycler'].map(self.cycler_mapping)
        if 'cyclinglike' in self.merged_data.columns and self.cyclinglike_mapping:
            demo_categories['cyclinglike_category'] = self.merged_data['cyclinglike'].map(self.cyclinglike_mapping)
        if 'cyclingunsafe' in self.merged_data.columns and self.cyclingunsafe_mapping:
            demo_categories['cyclingunsafe_category'] = self.merged_data['cyclingunsafe'].map(self.cyclingunsafe_mapping)
        if 'biketype' in self.merged_data.columns and self.biketype_mapping:
            demo_categories['biketype_category'] = self.merged_data['biketype'].map(self.biketype_mapping)
        if 'work' in self.merged_data.columns and self.work_mapping:
            demo_categories['work_category'] = self.merged_data['work'].map(self.work_mapping)
        if 'car' in self.merged_data.columns and self.car_mapping:
            demo_categories['car_category'] = self.merged_data['car'].map(self.car_mapping)

        # Add all demographic categories at once
        demo_df = pd.DataFrame(demo_categories, index=self.merged_data.index)
        self.merged_data = pd.concat([self.merged_data, demo_df], axis=1)
        
        # More lenient filtering - only filter out 'other' categories for age and gender
        before_filter = len(self.merged_data)
        
        # Basic filter for age and gender
        basic_filter = (
            (self.merged_data['age_category'] != 'other') & 
            (self.merged_data['gender_category'] != 'other') &
            (self.merged_data['age_category'].notna()) &
            (self.merged_data['gender_category'].notna())
        )
        
        # Apply basic filter
        self.merged_data = self.merged_data[basic_filter].copy()
        after_basic_filter = len(self.merged_data)
        
        print(f"After basic filtering (age/gender): {before_filter} -> {after_basic_filter} observations")
        
        # Print final distributions
        print("\nFinal demographic distributions:")
        print("Age categories:")
        print(self.merged_data['age_category'].value_counts())
        print("Gender categories:")
        print(self.merged_data['gender_category'].value_counts())
        
        if 'education_category' in self.merged_data.columns:
            print("Education categories:")
            print(self.merged_data['education_category'].value_counts())
        if 'income_category' in self.merged_data.columns:
            print("Income categories:")
            print(self.merged_data['income_category'].value_counts())
        if 'cycling_incident_category' in self.merged_data.columns:
            print("Cycling incident categories:")
            print(self.merged_data['cycling_incident_category'].value_counts())
        
        # Print distributions for new categories
        if 'cycler_category' in self.merged_data.columns:
            print("Cycler categories:")
            print(self.merged_data['cycler_category'].value_counts())
        if 'cyclinglike_category' in self.merged_data.columns:
            print("Cyclinglike categories:")
            print(self.merged_data['cyclinglike_category'].value_counts())
        if 'cyclingunsafe_category' in self.merged_data.columns:
            print("Cyclingunsafe categories:")
            print(self.merged_data['cyclingunsafe_category'].value_counts())
        if 'biketype_category' in self.merged_data.columns:
            print("Biketype categories:")
            print(self.merged_data['biketype_category'].value_counts())
        if 'work_category' in self.merged_data.columns:
            print("Work categories:")
            print(self.merged_data['work_category'].value_counts())
        if 'car_category' in self.merged_data.columns:
            print("Car categories:")
            print(self.merged_data['car_category'].value_counts())
    
    def create_demographic_dummy_variables(self, data_subset):
        """Create dummy variables for demographic categories and interactions"""
        
        print("Creating demographic dummy variables and interactions...")
        
        # Set reference categories based on most common categories in the data
        age_reference = '18-30'
        gender_reference = 'male'
        
        # Set reference categories for additional demographics if available
        education_reference = None
        if 'education_category' in data_subset.columns:
            education_reference = 'primary'
        
        income_reference = None
        if 'income_category' in data_subset.columns:
            income_reference = data_subset['income_category'].mode().iloc[0] if not data_subset['income_category'].mode().empty else 'medium'
        
        cycling_incident_reference = None
        if 'cycling_incident_category' in data_subset.columns:
            cycling_incident_reference = 'never'
        
        # Set reference categories for new demographics
        cycler_reference = 'daily'
        cyclinglike_reference = 'yes'
        cyclingunsafe_reference = 'never'
        biketype_reference = 'regular_bike'
        work_reference = 'full-time'
        car_reference = 'no_car'

        # Create age dummy variables (excluding reference)
        for age_cat in ['31-45', '46-60', '60+']:
            dummy_col = f'age_{age_cat}'
            data_subset[f'{dummy_col}_1'] = (data_subset['age_category'] == age_cat).astype(int)
            data_subset[f'{dummy_col}_2'] = data_subset[f'{dummy_col}_1']  # Same for both alternatives
            
            # Create safety × age interactions
            data_subset[f'safety_{dummy_col}_1'] = data_subset['safety_score_1'] * data_subset[f'{dummy_col}_1']
            data_subset[f'safety_{dummy_col}_2'] = data_subset['safety_score_2'] * data_subset[f'{dummy_col}_2']
        
        # Create gender dummy variables (excluding reference)
        data_subset['gender_female_1'] = (data_subset['gender_category'] == 'female').astype(int)
        data_subset['gender_female_2'] = data_subset['gender_female_1']  # Same for both alternatives
        
        # Create safety × gender interactions
        data_subset['safety_gender_female_1'] = data_subset['safety_score_1'] * data_subset['gender_female_1']
        data_subset['safety_gender_female_2'] = data_subset['safety_score_2'] * data_subset['gender_female_2']
        
        # Create education dummy variables if available
        education_categories = []
        if 'education_category' in data_subset.columns and education_reference is not None:
            available_education = data_subset['education_category'].unique()
            education_categories = [cat for cat in available_education if cat != education_reference and pd.notna(cat)]
            
            # Create dummy variables efficiently
            education_dummies = {}
            for edu_cat in education_categories:
                safe_cat_name = edu_cat.replace(' ', '_').replace('-', '_')
                dummy_col = f'education_{safe_cat_name}'
                education_dummies[f'{dummy_col}_1'] = (data_subset['education_category'] == edu_cat).astype(int)
                education_dummies[f'{dummy_col}_2'] = education_dummies[f'{dummy_col}_1']
                
                # Create safety × education interactions
                education_dummies[f'safety_{dummy_col}_1'] = data_subset['safety_score_1'] * education_dummies[f'{dummy_col}_1']
                education_dummies[f'safety_{dummy_col}_2'] = data_subset['safety_score_2'] * education_dummies[f'{dummy_col}_2']
            
            # Add education dummies to data
            if education_dummies:
                education_df = pd.DataFrame(education_dummies, index=data_subset.index)
                data_subset = pd.concat([data_subset, education_df], axis=1)
        
        # Create income dummy variables if available
        income_categories = []
        if 'income_category' in data_subset.columns and income_reference is not None:
            available_income = data_subset['income_category'].unique()
            income_categories = [cat for cat in available_income if cat != income_reference and pd.notna(cat)]
            
            # Create dummy variables efficiently
            income_dummies = {}
            for inc_cat in income_categories:
                safe_cat_name = inc_cat.replace(' ', '_').replace('-', '_')
                dummy_col = f'income_{safe_cat_name}'
                income_dummies[f'{dummy_col}_1'] = (data_subset['income_category'] == inc_cat).astype(int)
                income_dummies[f'{dummy_col}_2'] = income_dummies[f'{dummy_col}_1']
                
                # Create safety × income interactions
                income_dummies[f'safety_{dummy_col}_1'] = data_subset['safety_score_1'] * income_dummies[f'{dummy_col}_1']
                income_dummies[f'safety_{dummy_col}_2'] = data_subset['safety_score_2'] * income_dummies[f'{dummy_col}_2']
            
            # Add income dummies to data
            if income_dummies:
                income_df = pd.DataFrame(income_dummies, index=data_subset.index)
                data_subset = pd.concat([data_subset, income_df], axis=1)
        
        # Create cycling incident dummy variables if available
        cycling_incident_categories = []
        if 'cycling_incident_category' in data_subset.columns and cycling_incident_reference is not None:
            available_incident = data_subset['cycling_incident_category'].unique()
            cycling_incident_categories = [cat for cat in available_incident if cat != cycling_incident_reference and pd.notna(cat)]
            
            # Create dummy variables efficiently
            incident_dummies = {}
            for inc_cat in cycling_incident_categories:
                dummy_col = f'cycling_incident_{inc_cat}'
                incident_dummies[f'{dummy_col}_1'] = (data_subset['cycling_incident_category'] == inc_cat).astype(int)
                incident_dummies[f'{dummy_col}_2'] = incident_dummies[f'{dummy_col}_1']
                
                # Create safety × cycling incident interactions
                incident_dummies[f'safety_{dummy_col}_1'] = data_subset['safety_score_1'] * incident_dummies[f'{dummy_col}_1']
                incident_dummies[f'safety_{dummy_col}_2'] = data_subset['safety_score_2'] * incident_dummies[f'{dummy_col}_2']
            
            # Add cycling incident dummies to data
            if incident_dummies:
                incident_df = pd.DataFrame(incident_dummies, index=data_subset.index)
                data_subset = pd.concat([data_subset, incident_df], axis=1)
        
        # Create a helper function for creating dummies
        def create_dummies_for_feature(feature_name, reference_category, data_subset):
            category_col = f'{feature_name}_category'
            if category_col not in data_subset.columns or reference_category is None:
                return data_subset, []

            available_cats = data_subset[category_col].unique()
            categories = [cat for cat in available_cats if cat != reference_category and pd.notna(cat)]
            
            dummies = {}
            for cat in categories:
                safe_cat_name = str(cat).replace(' ', '_').replace('-', '_').replace('>', 'gt').replace('<', 'lt').replace('+', 'plus')
                dummy_col = f'{feature_name}_{safe_cat_name}'
                dummies[f'{dummy_col}_1'] = (data_subset[category_col] == cat).astype(int)
                dummies[f'{dummy_col}_2'] = dummies[f'{dummy_col}_1']
                
                # Create safety × feature interactions
                dummies[f'safety_{dummy_col}_1'] = data_subset['safety_score_1'] * dummies[f'{dummy_col}_1']
                dummies[f'safety_{dummy_col}_2'] = data_subset['safety_score_2'] * dummies[f'{dummy_col}_2']

            if dummies:
                dummies_df = pd.DataFrame(dummies, index=data_subset.index)
                data_subset = pd.concat([data_subset, dummies_df], axis=1)
                
            return data_subset, categories

        # Create dummies for new demographic features
        data_subset, cycler_categories = create_dummies_for_feature('cycler', cycler_reference, data_subset)
        data_subset, cyclinglike_categories = create_dummies_for_feature('cyclinglike', cyclinglike_reference, data_subset)
        data_subset, cyclingunsafe_categories = create_dummies_for_feature('cyclingunsafe', cyclingunsafe_reference, data_subset)
        data_subset, biketype_categories = create_dummies_for_feature('biketype', biketype_reference, data_subset)
        data_subset, work_categories = create_dummies_for_feature('work', work_reference, data_subset)
        data_subset, car_categories = create_dummies_for_feature('car', car_reference, data_subset)
        
        print(f"✓ Created dummy variables:")
        print(f"  Age reference: {age_reference}")
        print(f"  Gender reference: {gender_reference}")
        if education_categories:
            print(f"  Education reference: {education_reference}, categories: {education_categories}")
        if income_categories:
            print(f"  Income reference: {income_reference}, categories: {income_categories}")
        if cycling_incident_categories:
            print(f"  Cycling incident reference: {cycling_incident_reference}, categories: {cycling_incident_categories}")
        if cycler_categories:
            print(f"  Cycler reference: {cycler_reference}, categories: {cycler_categories}")
        if cyclinglike_categories:
            print(f"  Cyclinglike reference: {cyclinglike_reference}, categories: {cyclinglike_categories}")
        if cyclingunsafe_categories:
            print(f"  Cyclingunsafe reference: {cyclingunsafe_reference}, categories: {cyclingunsafe_categories}")
        if biketype_categories:
            print(f"  Biketype reference: {biketype_reference}, categories: {biketype_categories}")
        if work_categories:
            print(f"  Work reference: {work_reference}, categories: {work_categories}")
        if car_categories:
            print(f"  Car reference: {car_reference}, categories: {car_categories}")

        reference_info = {
            'age_reference': age_reference,
            'gender_reference': gender_reference,
            'education_reference': education_reference,
            'income_reference': income_reference,
            'cycling_incident_reference': cycling_incident_reference,
            'cycler_reference': cycler_reference,
            'cyclinglike_reference': cyclinglike_reference,
            'cyclingunsafe_reference': cyclingunsafe_reference,
            'biketype_reference': biketype_reference,
            'work_reference': work_reference,
            'car_reference': car_reference,
            'education_categories': education_categories,
            'income_categories': income_categories,
            'cycling_incident_categories': cycling_incident_categories,
            'cycler_categories': cycler_categories,
            'cyclinglike_categories': cyclinglike_categories,
            'cyclingunsafe_categories': cyclingunsafe_categories,
            'biketype_categories': biketype_categories,
            'work_categories': work_categories,
            'car_categories': car_categories
        }

        # Store dummy categories for model estimation
        self.demographic_categories = {
            'age': ['31-45', '46-60', '60+'],
            'gender': ['female', 'male'],
            'education': education_categories,
            'income': income_categories,
            'cycling_incident': cycling_incident_categories,
            'cycler': cycler_categories,
            'cyclinglike': cyclinglike_categories,
            'cyclingunsafe': cyclingunsafe_categories,
            'biketype': biketype_categories,
            'work': work_categories,
            'car': car_categories
        }
        
        return data_subset, reference_info
    
    def estimate_interaction_model(self, train_data, test_data=None):
        """Estimate the enhanced safety × demographics interaction model"""
        
        print("\nEstimating Enhanced Safety × Demographics Interaction Model")
        print("=" * 60)
        
        # Prepare the modeling data with demographic variables
        train_model_data, reference_info = self.create_demographic_dummy_variables(train_data.copy())
        
        # Start with required base columns (only interactions, no standalone demographics)
        required_cols = [
            'CHOICE', 'TL1', 'TT1', 'TL2', 'TT2', 'safety_score_1', 'safety_score_2',
            # Demographic interaction terms with safety (no standalone demographic variables)
            'safety_age_31-45_1', 'safety_age_31-45_2',
            'safety_age_46-60_1', 'safety_age_46-60_2', 
            'safety_age_60+_1', 'safety_age_60+_2',
            'safety_gender_female_1', 'safety_gender_female_2'
        ]
        
        # Add education interaction columns if available (no standalone education variables)
        education_cols = []
        for edu_cat in reference_info['education_categories']:
            safe_cat_name = edu_cat.replace(' ', '_').replace('-', '_')
            education_cols.extend([
                f'safety_education_{safe_cat_name}_1', f'safety_education_{safe_cat_name}_2'
            ])
        
        # Add income interaction columns if available (no standalone income variables)
        income_cols = []
        for inc_cat in reference_info['income_categories']:
            safe_cat_name = inc_cat.replace(' ', '_').replace('-', '_')
            income_cols.extend([
                f'safety_income_{safe_cat_name}_1', f'safety_income_{safe_cat_name}_2'
            ])
        
        # Add cycling incident interaction columns if available (no standalone cycling incident variables)
        cycling_incident_cols = []
        for inc_cat in reference_info['cycling_incident_categories']:
            cycling_incident_cols.extend([
                f'safety_cycling_incident_{inc_cat}_1', f'safety_cycling_incident_{inc_cat}_2'
            ])
        
        # Helper function to add new interaction columns
        def add_interaction_cols(feature_name, categories):
            cols = []
            for cat in categories:
                safe_cat_name = str(cat).replace(' ', '_').replace('-', '_').replace('>', 'gt').replace('<', 'lt').replace('+', 'plus')
                cols.extend([
                    f'safety_{feature_name}_{safe_cat_name}_1', 
                    f'safety_{feature_name}_{safe_cat_name}_2'
                ])
            return cols

        cycler_cols = add_interaction_cols('cycler', reference_info['cycler_categories'])
        cyclinglike_cols = add_interaction_cols('cyclinglike', reference_info['cyclinglike_categories'])
        cyclingunsafe_cols = add_interaction_cols('cyclingunsafe', reference_info['cyclingunsafe_categories'])
        biketype_cols = add_interaction_cols('biketype', reference_info['biketype_categories'])
        work_cols = add_interaction_cols('work', reference_info['work_categories'])
        car_cols = add_interaction_cols('car', reference_info['car_categories'])

        # Add segmentation features - extract available features from data
        seg_cols = []
        seg_features_used = []
        
        if self.segmentation_data is not None:
            print("Adding segmentation features...")
            
            # Get all segmentation feature names from the data
            seg_feature_names = [col for col in self.segmentation_data.columns if col != 'filename_key']
            
            # Map feature names using the feature mapping
            for feature in seg_feature_names:
                # Try different possible column name formats
                possible_names = [
                    feature,
                    feature.replace('_', ' '),
                    feature.replace(' ', '_'),
                    feature.title().replace(' ', '_'),
                    self.feature_name_mapping.get(feature, feature)
                ]
                
                feature_found = False
                for possible_name in possible_names:
                    col1 = f'{possible_name}_1'
                    col2 = f'{possible_name}_2'
                    if col1 in train_model_data.columns and col2 in train_model_data.columns:
                        seg_cols.extend([col1, col2])
                        seg_features_used.append(possible_name)
                        feature_found = True
                        print(f"✓ Added segmentation feature: '{feature}' -> '{possible_name}'")
                        break
                
                if not feature_found:
                    print(f"Warning: Could not find segmentation feature '{feature}' in data")
            
            print(f"Successfully included {len(seg_features_used)} segmentation features")
        
        # Combine all columns
        all_cols = (required_cols + education_cols + income_cols + cycling_incident_cols + 
                    cycler_cols + cyclinglike_cols + cyclingunsafe_cols + biketype_cols +
                    work_cols + car_cols + seg_cols)
        
        # Filter to only include columns that exist in the data
        available_cols = [col for col in all_cols if col in train_model_data.columns]
        train_model_data = train_model_data[available_cols].copy().dropna()
        
        print(f"Training data shape after feature selection: {train_model_data.shape}")
        print(f"Features included: {len(available_cols) - 1}")  # Exclude CHOICE
        
        # Create Biogeme database
        train_database = db.Database('safety_demographics_interaction', train_model_data)
        
        # Create variables
        TL1, TT1, TL2, TT2 = train_database.variables['TL1'], train_database.variables['TT1'], train_database.variables['TL2'], train_database.variables['TT2']
        safety1, safety2 = train_database.variables['safety_score_1'], train_database.variables['safety_score_2']
        CHOICE = train_database.variables['CHOICE']
        
        # Define base parameters
        B_TL = Beta('B_TL', 0, None, None, 0)
        B_TT = Beta('B_TT', 0, None, None, 0)
        B_SAFETY = Beta('B_SAFETY', 0, None, None, 0)
        
        # Start with base utility
        V1_components = [B_TL * TL1 / 3, B_TT * TT1 / 10, B_SAFETY * safety1]
        V2_components = [B_TL * TL2 / 3, B_TT * TT2 / 10, B_SAFETY * safety2]
        
        # Add age interactions with safety (no standalone demographic effects)
        age_categories = ['31-45', '46-60', '60+']
        for age_cat in age_categories:
            # Only interaction with safety
            interaction_param = Beta(f'B_SAFETY_AGE_{age_cat.replace("-", "_").replace("+", "plus")}', 0, None, None, 0)
            interaction_var1 = train_database.variables[f'safety_age_{age_cat}_1']
            interaction_var2 = train_database.variables[f'safety_age_{age_cat}_2']
            V1_components.append(interaction_param * interaction_var1)
            V2_components.append(interaction_param * interaction_var2)
        
        # Add gender × safety interaction (no standalone gender effect)
        gender_interaction_param = Beta('B_SAFETY_GENDER_FEMALE', 0, None, None, 0)
        gender_interaction_var1 = train_database.variables['safety_gender_female_1']
        gender_interaction_var2 = train_database.variables['safety_gender_female_2']
        V1_components.append(gender_interaction_param * gender_interaction_var1)
        V2_components.append(gender_interaction_param * gender_interaction_var2)
        
        # Add education interactions with safety (no standalone education effects)
        education_interaction_params = {}
        for edu_cat in reference_info['education_categories']:
            safe_cat_name = edu_cat.replace(' ', '_').replace('-', '_')
            
            # Only interaction with safety
            if f'safety_education_{safe_cat_name}_1' in train_database.variables:
                edu_interaction_param = Beta(f'B_SAFETY_EDUCATION_{safe_cat_name.upper()}', 0, None, None, 0)
                education_interaction_params[edu_cat] = edu_interaction_param
                
                edu_interaction_var1 = train_database.variables[f'safety_education_{safe_cat_name}_1']
                edu_interaction_var2 = train_database.variables[f'safety_education_{safe_cat_name}_2']
                V1_components.append(edu_interaction_param * edu_interaction_var1)
                V2_components.append(edu_interaction_param * edu_interaction_var2)
        
        # Add income interactions with safety (no standalone income effects)
        income_interaction_params = {}
        for inc_cat in reference_info['income_categories']:
            safe_cat_name = inc_cat.replace(' ', '_').replace('-', '_')
            
            # Only interaction with safety
            if f'safety_income_{safe_cat_name}_1' in train_database.variables:
                inc_interaction_param = Beta(f'B_SAFETY_INCOME_{safe_cat_name.upper()}', 0, None, None, 0)
                income_interaction_params[inc_cat] = inc_interaction_param
                
                inc_interaction_var1 = train_database.variables[f'safety_income_{safe_cat_name}_1']
                inc_interaction_var2 = train_database.variables[f'safety_income_{safe_cat_name}_2']
                V1_components.append(inc_interaction_param * inc_interaction_var1)
                V2_components.append(inc_interaction_param * inc_interaction_var2)
        
        # Add cycling incident interactions with safety (no standalone cycling incident effects)
        cycling_incident_interaction_params = {}
        for inc_cat in reference_info['cycling_incident_categories']:
            # Only interaction with safety
            if f'safety_cycling_incident_{inc_cat}_1' in train_database.variables:
                cycling_interaction_param = Beta(f'B_SAFETY_CYCLING_INCIDENT_{inc_cat.upper()}', 0, None, None, 0)
                cycling_incident_interaction_params[inc_cat] = cycling_interaction_param
                
                cycling_interaction_var1 = train_database.variables[f'safety_cycling_incident_{inc_cat}_1']
                cycling_interaction_var2 = train_database.variables[f'safety_cycling_incident_{inc_cat}_2']
                V1_components.append(cycling_interaction_param * cycling_interaction_var1)
                V2_components.append(cycling_interaction_param * cycling_interaction_var2)
        
        # Helper function to add interaction terms to the model
        def add_interaction_terms_to_model(feature_name, categories, database):
            interaction_params = {}
            for cat in categories:
                safe_cat_name = str(cat).replace(' ', '_').replace('-', '_').replace('>', 'gt').replace('<', 'lt').replace('+', 'plus')
                var_name_1 = f'safety_{feature_name}_{safe_cat_name}_1'
                if var_name_1 in database.variables:
                    param = Beta(f'B_SAFETY_{feature_name.upper()}_{safe_cat_name.upper()}', 0, None, None, 0)
                    interaction_params[cat] = param
                    
                    var1 = database.variables[var_name_1]
                    var2 = database.variables[f'safety_{feature_name}_{safe_cat_name}_2']
                    V1_components.append(param * var1)
                    V2_components.append(param * var2)
            return interaction_params

        # Add new interaction terms
        cycler_interaction_params = add_interaction_terms_to_model('cycler', reference_info['cycler_categories'], train_database)
        cyclinglike_interaction_params = add_interaction_terms_to_model('cyclinglike', reference_info['cyclinglike_categories'], train_database)
        cyclingunsafe_interaction_params = add_interaction_terms_to_model('cyclingunsafe', reference_info['cyclingunsafe_categories'], train_database)
        biketype_interaction_params = add_interaction_terms_to_model('biketype', reference_info['biketype_categories'], train_database)
        work_interaction_params = add_interaction_terms_to_model('work', reference_info['work_categories'], train_database)
        car_interaction_params = add_interaction_terms_to_model('car', reference_info['car_categories'], train_database)

        # Add segmentation features
        seg_params = {}
        for feature in seg_features_used:
            param_name = f"B_{feature.replace(' ', '_').replace('-', '_').upper()}"
            seg_params[feature] = Beta(param_name, 0, None, None, 0)
            
            seg_var1 = train_database.variables[f'{feature}_1']
            seg_var2 = train_database.variables[f'{feature}_2']
            
            V1_components.append(seg_params[feature] * seg_var1)
            V2_components.append(seg_params[feature] * seg_var2)
        
        # Define utility functions
        V1 = sum(V1_components)
        V2 = sum(V2_components)
        
        # Estimate the model
        train_results = self._estimate_mnl(V1, V2, CHOICE, train_database, 'safety_demographics_interaction')
        
        # Store model information
        self.model_info = {
            **reference_info,
            'age_categories': age_categories,
            'segmentation_features': seg_features_used,
            'features_included': {
                'base_effects': ['TL', 'TT', 'SAFETY'],
                'segmentation_effects': seg_features_used,
                'safety_interactions_only': {
                    'age_interactions': [f'safety_age_{cat}' for cat in age_categories],
                    'gender_interactions': ['safety_gender_female'],
                    'education_interactions': [f'safety_education_{cat}' for cat in reference_info['education_categories']],
                    'income_interactions': [f'safety_income_{cat}' for cat in reference_info['income_categories']],
                    'cycling_incident_interactions': [f'safety_cycling_incident_{cat}' for cat in reference_info['cycling_incident_categories']],
                    'cycler_interactions': [f'safety_cycler_{cat}' for cat in reference_info['cycler_categories']],
                    'cyclinglike_interactions': [f'safety_cyclinglike_{cat}' for cat in reference_info['cyclinglike_categories']],
                    'cyclingunsafe_interactions': [f'safety_cyclingunsafe_{cat}' for cat in reference_info['cyclingunsafe_categories']],
                    'biketype_interactions': [f'safety_biketype_{cat}' for cat in reference_info['biketype_categories']],
                    'work_interactions': [f'safety_work_{cat}' for cat in reference_info['work_categories']],
                    'car_interactions': [f'safety_car_{cat}' for cat in reference_info['car_categories']]
                },
                'note': 'No standalone demographic variables included - demographics only identified through interactions with choice attributes'
            }
        }
        
        return train_results
    
    def _estimate_mnl(self, V1, V2, Choice, database, name):
        """Estimate MNL model using Biogeme"""
        
        # Create utility dictionary
        V = {1: V1, 2: V2}
        av = {1: 1, 2: 1}
        
        # Define choice model
        prob = models.logit(V, av, Choice)
        LL = log(prob)
        
        # Create Biogeme object
        biogeme = bio.BIOGEME(database, LL)
        biogeme.modelName = name
        
        # Configure to save results
        biogeme.generate_pickle = True
        biogeme.generate_html = True
        biogeme.save_iterations = True
        
        # Change to output directory
        original_cwd = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Calculate null log-likelihood
            biogeme.calculate_null_loglikelihood(av)
            
            # Estimate model
            print("Starting model estimation...")
            results = biogeme.estimate()
            print("✓ Model estimation completed")
            
            # Save additional files
            try:
                results.write_latex()
                results.write_pickle()
                results.write_html()
            except AttributeError:
                results.writeLaTeX()
                results.writePickle()
                results.writeHTML()
                
        finally:
            os.chdir(original_cwd)
        
        return results
    
    def analyze_interaction_effects(self, results):
        """Analyze and interpret the safety × demographics interaction effects"""
        
        print("\nAnalyzing interaction effects...")
        
        # Get estimated parameters
        estimated_params = results.get_estimated_parameters()
        
        # Print available columns for debugging
        print(f"Available columns: {list(estimated_params.columns)}")
        
        # Determine which columns to use for p-values and t-stats
        p_value_col = None
        t_stat_col = None
        
        for col in estimated_params.columns:
            if 'p-value' in col.lower() or 'p.value' in col.lower():
                p_value_col = col
            if 't-test' in col.lower() or 't-stat' in col.lower() or 'tstat' in col.lower():
                t_stat_col = col
        
        print(f"Using p-value column: {p_value_col}")
        print(f"Using t-stat column: {t_stat_col}")
        
        # Extract interaction effects
        interaction_effects = {}
        main_safety_effect = estimated_params.loc['B_SAFETY', 'Value']
        
        # Helper function to get statistical values safely
        def get_stat_value(param_name, col_name):
            if col_name and param_name in estimated_params.index and col_name in estimated_params.columns:
                return estimated_params.loc[param_name, col_name]
            return None
        
        # Age interactions
        for age_cat in ['31-45', '46-60', '60+']:
            param_name = f"B_SAFETY_AGE_{age_cat.replace('-', '_').replace('+', 'plus')}"
            if param_name in estimated_params.index:
                interaction_coef = estimated_params.loc[param_name, 'Value']
                total_safety_effect = main_safety_effect + interaction_coef
                
                interaction_effects[f'age_{age_cat}'] = {
                    'interaction_coefficient': interaction_coef,
                    'main_safety_effect': main_safety_effect,
                    'total_safety_effect': total_safety_effect,
                    'p_value': get_stat_value(param_name, p_value_col),
                    't_stat': get_stat_value(param_name, t_stat_col)
                }
        
        # Gender interaction
        gender_param = 'B_SAFETY_GENDER_FEMALE'
        if gender_param in estimated_params.index:
            interaction_coef = estimated_params.loc[gender_param, 'Value']
            total_safety_effect = main_safety_effect + interaction_coef
            
            interaction_effects['gender_female'] = {
                'interaction_coefficient': interaction_coef,
                'main_safety_effect': main_safety_effect,
                'total_safety_effect': total_safety_effect,
                'p_value': get_stat_value(gender_param, p_value_col),
                't_stat': get_stat_value(gender_param, t_stat_col)
            }
        
        # Education interactions
        if 'education_categories' in self.model_info:
            for edu_cat in self.model_info['education_categories']:
                safe_cat_name = edu_cat.replace(' ', '_').replace('-', '_')
                param_name = f"B_SAFETY_EDUCATION_{safe_cat_name.upper()}"
                if param_name in estimated_params.index:
                    interaction_coef = estimated_params.loc[param_name, 'Value']
                    total_safety_effect = main_safety_effect + interaction_coef
                    
                    interaction_effects[f'education_{edu_cat}'] = {
                        'interaction_coefficient': interaction_coef,
                        'main_safety_effect': main_safety_effect,
                        'total_safety_effect': total_safety_effect,
                        'p_value': get_stat_value(param_name, p_value_col),
                        't_stat': get_stat_value(param_name, t_stat_col)
                    }
        
        # Income interactions
        if 'income_categories' in self.model_info:
            for inc_cat in self.model_info['income_categories']:
                safe_cat_name = inc_cat.replace(' ', '_').replace('-', '_')
                param_name = f"B_SAFETY_INCOME_{safe_cat_name.upper()}"
                if param_name in estimated_params.index:
                    interaction_coef = estimated_params.loc[param_name, 'Value']
                    total_safety_effect = main_safety_effect + interaction_coef
                    
                    interaction_effects[f'income_{inc_cat}'] = {
                        'interaction_coefficient': interaction_coef,
                        'main_safety_effect': main_safety_effect,
                        'total_safety_effect': total_safety_effect,
                        'p_value': get_stat_value(param_name, p_value_col),
                        't_stat': get_stat_value(param_name, t_stat_col)
                    }
        
        # Cycling incident interactions
        if 'cycling_incident_categories' in self.model_info:
            for inc_cat in self.model_info['cycling_incident_categories']:
                param_name = f"B_SAFETY_CYCLING_INCIDENT_{inc_cat.upper()}"
                if param_name in estimated_params.index:
                    interaction_coef = estimated_params.loc[param_name, 'Value']
                    total_safety_effect = main_safety_effect + interaction_coef
                    
                    interaction_effects[f'cycling_incident_{inc_cat}'] = {
                        'interaction_coefficient': interaction_coef,
                        'main_safety_effect': main_safety_effect,
                        'total_safety_effect': total_safety_effect,
                        'p_value': get_stat_value(param_name, p_value_col),
                        't_stat': get_stat_value(param_name, t_stat_col)
                    }
        
        # Helper function to analyze interactions for a feature
        def analyze_feature_interactions(feature_name, categories):
            for cat in categories:
                safe_cat_name = str(cat).replace(' ', '_').replace('-', '_').replace('>', 'gt').replace('<', 'lt').replace('+', 'plus')
                param_name = f"B_SAFETY_{feature_name.upper()}_{safe_cat_name.upper()}"
                if param_name in estimated_params.index:
                    interaction_coef = estimated_params.loc[param_name, 'Value']
                    total_safety_effect = main_safety_effect + interaction_coef
                    
                    interaction_effects[f'{feature_name}_{cat}'] = {
                        'interaction_coefficient': interaction_coef,
                        'main_safety_effect': main_safety_effect,
                        'total_safety_effect': total_safety_effect,
                        'p_value': get_stat_value(param_name, p_value_col),
                        't_stat': get_stat_value(param_name, t_stat_col)
                    }

        # Analyze new feature interactions
        analyze_feature_interactions('cycler', self.model_info.get('cycler_categories', []))
        analyze_feature_interactions('cyclinglike', self.model_info.get('cyclinglike_categories', []))
        analyze_feature_interactions('cyclingunsafe', self.model_info.get('cyclingunsafe_categories', []))
        analyze_feature_interactions('biketype', self.model_info.get('biketype_categories', []))
        analyze_feature_interactions('work', self.model_info.get('work_categories', []))
        analyze_feature_interactions('car', self.model_info.get('car_categories', []))

        # Reference categories
        interaction_effects['age_18-30'] = {
            'interaction_coefficient': 0,
            'main_safety_effect': main_safety_effect,
            'total_safety_effect': main_safety_effect,
            'p_value': get_stat_value('B_SAFETY', p_value_col),
            't_stat': get_stat_value('B_SAFETY', t_stat_col)
        }
        
        # Helper function to add reference category effects
        def add_reference_effect(feature_name, reference_value):
            if reference_value:
                interaction_effects[f'{feature_name}_{reference_value}'] = {
                    'interaction_coefficient': 0,
                    'main_safety_effect': main_safety_effect,
                    'total_safety_effect': main_safety_effect,
                    'p_value': get_stat_value('B_SAFETY', p_value_col),
                    't_stat': get_stat_value('B_SAFETY', t_stat_col)
                }
        
        add_reference_effect('cycler', self.model_info.get('cycler_reference'))
        add_reference_effect('cyclinglike', self.model_info.get('cyclinglike_reference'))
        add_reference_effect('cyclingunsafe', self.model_info.get('cyclingunsafe_reference'))
        add_reference_effect('biketype', self.model_info.get('biketype_reference'))
        add_reference_effect('work', self.model_info.get('work_reference'))
        add_reference_effect('car', self.model_info.get('car_reference'))

        # Final check for cycling incident reference
        if 'cycling_incident_reference' in self.model_info:
             interaction_effects[f'cycling_incident_{self.model_info["cycling_incident_reference"]}'] = {
                'interaction_coefficient': 0,
                'main_safety_effect': main_safety_effect,
                'total_safety_effect': main_safety_effect,
                'p_value': get_stat_value('B_SAFETY', p_value_col),
                't_stat': get_stat_value('B_SAFETY', t_stat_col)
            }
        
        return interaction_effects
    
    def save_results(self, train_results, interaction_effects):
        """Save all results and analysis"""
        
        print("\nSaving results...")
        
        # Calculate metrics - use getattr with fallback calculations
        train_metrics = {
            'log_likelihood': train_results.data.logLike,
            'n_parameters': len(train_results.data.betaValues),
            'n_observations': train_results.data.numberOfObservations,
            'AIC': getattr(train_results.data, 'akaike', 2 * len(train_results.data.betaValues) - 2 * train_results.data.logLike),
            'BIC': getattr(train_results.data, 'bayesianInformationCriterion', 
                          np.log(train_results.data.numberOfObservations) * len(train_results.data.betaValues) - 2 * train_results.data.logLike),
            'pseudo_r2': train_results.data.rhoSquare
        }
        
        # Save parameter estimates - handle different result types
        try:
            param_estimates = train_results.get_estimated_parameters()
            param_estimates.to_csv(self.output_dir / 'parameter_estimates.csv')
        except AttributeError:
            # Create parameter estimates from betaValues
            try:
                param_dict = {}
                if hasattr(train_results.data, 'betaValues'):
                    for param_name, value in train_results.data.betaValues.items():
                        param_dict[param_name] = {'Value': value}
                        if hasattr(train_results.data, 'stErr') and param_name in train_results.data.stErr:
                            param_dict[param_name]['Std err'] = train_results.data.stErr[param_name]
                
                param_estimates = pd.DataFrame(param_dict).T
                param_estimates.to_csv(self.output_dir / 'parameter_estimates.csv')
            except Exception as e:
                print(f"Warning: Could not save parameter estimates: {e}")
        
        # Save interaction effects analysis
        interaction_df = pd.DataFrame(interaction_effects).T
        interaction_df.to_csv(self.output_dir / 'interaction_effects_analysis.csv')
        
        # Save model summary
        model_summary = {
            'model_info': self.model_info,
            'train_metrics': train_metrics,
            'interaction_effects': interaction_effects,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_dir / 'model_summary.json', 'w') as f:
            json.dump(model_summary, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(interaction_effects, train_metrics)
        
        print(f"✓ Results saved to {self.output_dir}")
        return train_metrics
    
    def _create_summary_report(self, interaction_effects, train_metrics):
        """Create a human-readable summary report"""
        
        report_path = self.output_dir / 'demographics_interaction_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("SAFETY × DEMOGRAPHICS INTERACTION MODEL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model estimated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL PERFORMANCE:\n")
            f.write(f"  Log-likelihood: {train_metrics['log_likelihood']:.6f}\n")
            f.write(f"  Number of parameters: {train_metrics['n_parameters']}\n")
            f.write(f"  Number of observations: {train_metrics['n_observations']}\n")
            f.write(f"  AIC: {train_metrics['AIC']:.6f}\n")
            f.write(f"  BIC: {train_metrics['BIC']:.6f}\n")
            f.write(f"  Pseudo R²: {train_metrics['pseudo_r2']:.6f}\n\n")
            
            f.write("REFERENCE CATEGORIES:\n")
            f.write(f"  Age: {self.model_info['age_reference']}\n")
            f.write(f"  Gender: {self.model_info['gender_reference']}\n")
            if 'education_reference' in self.model_info:
                f.write(f"  Education: {self.model_info['education_reference']}\n")
            if 'income_reference' in self.model_info:
                f.write(f"  Income: {self.model_info['income_reference']}\n")
            if 'cycling_incident_reference' in self.model_info:
                f.write(f"  Cycling incident: {self.model_info['cycling_incident_reference']}\n")
            if 'cycler_reference' in self.model_info:
                f.write(f"  Cycler: {self.model_info['cycler_reference']}\n")
            if 'cyclinglike_reference' in self.model_info:
                f.write(f"  Cycling Like: {self.model_info['cyclinglike_reference']}\n")
            if 'cyclingunsafe_reference' in self.model_info:
                f.write(f"  Cycling Unsafe: {self.model_info['cyclingunsafe_reference']}\n")
            if 'biketype_reference' in self.model_info:
                f.write(f"  Biketype: {self.model_info['biketype_reference']}\n")
            if 'work_reference' in self.model_info:
                f.write(f"  Work: {self.model_info['work_reference']}\n")
            if 'car_reference' in self.model_info:
                f.write(f"  Car: {self.model_info['car_reference']}\n")
            f.write("\n")
            
            f.write("SAFETY EFFECTS BY DEMOGRAPHICS:\n")
            # Sort effects for consistent reporting
            sorted_effects = sorted(interaction_effects.items())

            for demo, effects in sorted_effects:
                f.write(f"  {demo}:\n")
                f.write(f"    Total safety effect: {effects['total_safety_effect']:.6f}\n")
                if effects['interaction_coefficient'] != 0:
                    f.write(f"    Interaction coefficient: {effects['interaction_coefficient']:.6f}\n")
                    if effects['p_value'] is not None:
                        significance = "***" if effects['p_value'] < 0.001 else "**" if effects['p_value'] < 0.01 else "*" if effects['p_value'] < 0.05 else ""
                        f.write(f"    P-value: {effects['p_value']:.6f} {significance}\n")
                        f.write(f"    T-statistic: {effects['t_stat']:.6f}\n")
                else:
                    f.write(f"    (Reference category)\n")
                f.write("\n")
            
            f.write("INTERPRETATION:\n")
            f.write("- Total safety effect = Main safety effect + Interaction coefficient\n")
            f.write("- Positive interaction coefficient means safety is MORE important for that group\n")
            f.write("- Negative interaction coefficient means safety is LESS important for that group\n")
            f.write("- Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")
        
        print(f"✓ Summary report saved to {report_path}")
    
    def run_analysis(self):
        """Run the complete safety × demographics interaction analysis"""
        
        print("Starting safety × demographics interaction analysis...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Split into train and test data
        if 'train' in self.merged_data.columns and 'test' in self.merged_data.columns:
            train_data = self.merged_data[self.merged_data['train'] == 1].copy()
            print(f"Using training data: {len(train_data)} observations")
        else:
            print("Warning: No train/test split found, using 80% for training")
            train_data = self.merged_data.sample(frac=0.8, random_state=42)
            print(f"Created training data: {len(train_data)} observations")
        
        # Estimate interaction model
        train_results = self.estimate_interaction_model(train_data)
        
        # Analyze interaction effects
        interaction_effects = self.analyze_interaction_effects(train_results)
        
        # Save results
        train_metrics = self.save_results(train_results, interaction_effects)
        
        print("\n✓ Safety × demographics interaction analysis completed!")
        print(f"Results saved to: {self.output_dir}")
        
        return train_results, interaction_effects, train_metrics


def main():
    """Main function to run safety × demographics interaction analysis"""
    
    # Initialize and run analysis
    interaction_model = SafetyDemographicsInteractionModelV2()
    train_results, interaction_effects, train_metrics = interaction_model.run_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ENHANCED INTERACTION EFFECTS")
    print("="*80)
    
    # Group effects by category
    age_effects = {k: v for k, v in interaction_effects.items() if k.startswith('age_')}
    gender_effects = {k: v for k, v in interaction_effects.items() if k.startswith('gender_')}
    education_effects = {k: v for k, v in interaction_effects.items() if k.startswith('education_')}
    income_effects = {k: v for k, v in interaction_effects.items() if k.startswith('income_')}
    cycling_effects = {k: v for k, v in interaction_effects.items() if k.startswith('cycling_incident_')}
    cycler_effects = {k: v for k, v in interaction_effects.items() if k.startswith('cycler_')}
    cyclinglike_effects = {k: v for k, v in interaction_effects.items() if k.startswith('cyclinglike_')}
    cyclingunsafe_effects = {k: v for k, v in interaction_effects.items() if k.startswith('cyclingunsafe_')}
    biketype_effects = {k: v for k, v in interaction_effects.items() if k.startswith('biketype_')}
    work_effects = {k: v for k, v in interaction_effects.items() if k.startswith('work_')}
    car_effects = {k: v for k, v in interaction_effects.items() if k.startswith('car_')}
    
    for category_name, effects_dict in [
        ("AGE EFFECTS", age_effects),
        ("GENDER EFFECTS", gender_effects),
        ("EDUCATION EFFECTS", education_effects),
        ("INCOME EFFECTS", income_effects),
        ("CYCLING INCIDENT EFFECTS", cycling_effects),
        ("CYCLER EFFECTS", cycler_effects),
        ("CYCLING LIKE EFFECTS", cyclinglike_effects),
        ("CYCLING UNSAFE EFFECTS", cyclingunsafe_effects),
        ("BIKETYPE EFFECTS", biketype_effects),
        ("WORK EFFECTS", work_effects),
        ("CAR EFFECTS", car_effects)
    ]:
        if effects_dict:
            print(f"\n{category_name}:")
            # Sort effects for consistent display
            sorted_effects = sorted(effects_dict.items())
            for category, effects in sorted_effects:
                print(f"  {category:25s}: Total safety effect = {effects['total_safety_effect']:8.4f}")
                if effects['interaction_coefficient'] != 0:
                    significance = ""
                    if effects['p_value'] is not None:
                        if effects['p_value'] < 0.001:
                            significance = " ***"
                        elif effects['p_value'] < 0.01:
                            significance = " **"
                        elif effects['p_value'] < 0.05:
                            significance = " *"
                    print(f"{'':27s}  Interaction coef = {effects['interaction_coefficient']:8.4f}{significance}")
    
    print(f"\nModel fit: LL = {train_metrics['log_likelihood']:.3f}, Pseudo R² = {train_metrics['pseudo_r2']:.4f}")
    print(f"Model performance: AIC = {train_metrics['AIC']:.1f}, BIC = {train_metrics['BIC']:.1f}")


if __name__ == "__main__":
    main() 