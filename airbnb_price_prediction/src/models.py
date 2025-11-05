"""Model definitions and implementations"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from abc import ABC, abstractmethod
from src.utils import logger

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.feature_importance = None
        
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        if self.feature_importance is not None:
            return pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
        return None

class GlobalMeanModel(BaseModel):
    """Baseline: predict global mean"""
    
    def __init__(self):
        super().__init__("Global Mean")
        self.mean_price = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.mean_price = y_train.mean()
        
    def predict(self, X):
        return np.full(len(X), self.mean_price)

class NeighborhoodMeanModel(BaseModel):
    """Baseline: predict neighborhood mean"""
    
    def __init__(self, listings_df):
        super().__init__("Neighborhood Mean")
        self.neighborhood_means = {}
        self.global_mean = None
        self.listings_df = listings_df
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        # Get listing IDs from index
        train_ids = X_train.index
        
        # Get neighborhoods for training data
        train_data = self.listings_df[self.listings_df['id'].isin(train_ids)]
        
        # Calculate means
        for neighborhood in train_data['neighbourhood_cleansed'].unique():
            mask = train_data['neighbourhood_cleansed'] == neighborhood
            self.neighborhood_means[neighborhood] = train_data[mask]['price_clean'].mean()
        
        self.global_mean = y_train.mean()
        
    def predict(self, X):
        # Get listing IDs from index
        test_ids = X.index
        
        # Get neighborhoods for test data
        test_data = self.listings_df[self.listings_df['id'].isin(test_ids)]
        
        predictions = []
        for neighborhood in test_data['neighbourhood_cleansed']:
            if neighborhood in self.neighborhood_means:
                predictions.append(self.neighborhood_means[neighborhood])
            else:
                predictions.append(self.global_mean)
                
        return np.array(predictions)

class LinearRegressionModel(BaseModel):
    """Linear Regression"""
    
    def __init__(self):
        super().__init__("Linear Regression")
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        self.feature_importance = np.abs(self.model.coef_)

class RidgeModel(BaseModel):
    """Ridge Regression"""
    
    def __init__(self, alpha=0.01):
        super().__init__(f"Ridge (α={alpha})")
        self.alpha = alpha
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = Ridge(alpha=self.alpha, random_state=42)
        self.model.fit(X_train, y_train)
        self.feature_importance = np.abs(self.model.coef_)

class RandomForestModel(BaseModel):
    """Random Forest"""
    
    def __init__(self, **params):
        super().__init__("Random Forest")
        self.params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        self.params.update(params)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_

class XGBoostModel(BaseModel):
    """XGBoost"""
    
    def __init__(self, **params):
        super().__init__("XGBoost")
        self.params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1
        }
        self.params.update(params)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = xgb.XGBRegressor(**self.params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
            
        self.feature_importance = self.model.feature_importances_

class LightGBMModel(BaseModel):
    """LightGBM"""
    
    def __init__(self, **params):
        super().__init__("LightGBM")
        self.params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        self.params.update(params)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model = lgb.LGBMRegressor(**self.params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
            
        self.feature_importance = self.model.feature_importances_