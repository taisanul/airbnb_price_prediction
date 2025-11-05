"""Training functions"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from src.utils import logger, save_pickle

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.results = []
        
    def prepare_features(self, df):
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        # Select feature types
        numerical_features = [
            'accommodates', 'bedrooms', 'beds', 'bathrooms_count',
            'minimum_nights', 'maximum_nights', 'availability_365',
            'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
            'host_listings_count', 'host_total_listings_count', 'host_days',
            'distance_to_center', 'amenities_count',
            'description_length', 'description_word_count', 'description_sentiment',
            'luxury_count', 'budget_count',
            'availability_rate', 'days_available',
            'total_reviews', 'recent_reviews_count', 'days_since_last_review',
            'image_quality_score',
            'neigh_price_mean', 'neigh_price_median', 'neigh_price_std'
        ]
        
        # Review scores
        review_score_features = [col for col in df.columns if 'review_scores' in col]
        numerical_features.extend(review_score_features)
        
        # Boolean features
        boolean_features = [
            'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
            'instant_bookable', 'entire_home', 'private_room'
        ]
        
        # Amenity features
        amenity_features = [col for col in df.columns if col.startswith('has_')]
        boolean_features.extend(amenity_features)
        
        # Categorical features
        categorical_features = [
            'property_type_group', 'neighbourhood_cleansed', 'location_cluster'
        ]
        
        # Filter existing features
        numerical_features = [f for f in numerical_features if f in df.columns]
        boolean_features = [f for f in boolean_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        # Create feature matrix
        features_to_use = numerical_features + boolean_features + categorical_features + ['price_clean', 'id']
        df_model = df[features_to_use].copy()
        
        # Handle missing values
        for col in numerical_features:
            df_model[col].fillna(df_model[col].median(), inplace=True)
            
        for col in boolean_features:
            df_model[col].fillna(0, inplace=True)
        
        # One-hot encode categorical features
        df_model = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
        
        logger.info(f"Final feature matrix shape: {df_model.shape}")
        
        return df_model, numerical_features
    
    def split_data(self, df_model, test_size=0.2):
        """Split data into train and test sets"""
        logger.info("Splitting data...")
        
        # Prepare features and target
        X = df_model.drop(['price_clean', 'id'], axis=1)
        y = df_model['price_clean']
        listing_ids = df_model['id']
        
        # Split
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, listing_ids, test_size=test_size, random_state=42
        )
        
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Test set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, ids_train, ids_test
    
    def scale_features(self, X_train, X_test, numerical_cols):
        """Scale numerical features"""
        logger.info("Scaling features...")
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Only scale numerical columns that exist
        numerical_cols_to_scale = [col for col in numerical_cols if col in X_train.columns]
        
        X_train_scaled[numerical_cols_to_scale] = self.scaler.fit_transform(X_train[numerical_cols_to_scale])
        X_test_scaled[numerical_cols_to_scale] = self.scaler.transform(X_test[numerical_cols_to_scale])
        
        # Save scaler
        save_pickle(self.scaler, 'data/processed/feature_scaler.pkl')
        
        return X_train_scaled, X_test_scaled
    
    def train_model(self, model, X_train, y_train, X_val=None, y_val=None):
        """Train a single model"""
        logger.info(f"Training {model.name}...")
        
        start_time = time.time()
        model.train(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time
        
        return train_time
    
    def evaluate_model(self, model, X_test, y_test, train_time):
        """Evaluate a model"""
        from src.evaluation import evaluate_predictions
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start_time
        
        metrics = evaluate_predictions(y_test, y_pred)
        
        result = {
            'Model': model.name,
            'RMSE': metrics['rmse'],
            #'MAE': metrics['mae'],
            'R2': metrics['r2'],
            'Training_Time': train_time,
            'Prediction_Time': pred_time
        }
        
        self.results.append(result)
        
        logger.info(f"{model.name} - RMSE: €{metrics['rmse']:.2f}, R²: {metrics['r2']:.4f}")
        
        return result, y_pred