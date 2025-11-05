"""Data loading and cleaning functions"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from src.utils import logger

class DataProcessor:
    def __init__(self, data_path='data/raw/'):
        self.data_path = data_path
        self.listings_detailed = None
        self.reviews = None
        self.calendar = None
        
    def load_data(self):
        """Load all datasets"""
        logger.info("Loading datasets...")
        
        # Load main datasets
        self.listings_detailed = pd.read_csv(f'{self.data_path}/listings_detailed.csv')
        self.reviews = pd.read_csv(f'{self.data_path}/reviews.csv')
        self.calendar = pd.read_csv(f'{self.data_path}/calendar.csv')
        
        logger.info(f"Listings shape: {self.listings_detailed.shape}")
        logger.info(f"Reviews shape: {self.reviews.shape}")
        logger.info(f"Calendar shape: {self.calendar.shape}")
        
        return self.listings_detailed, self.reviews, self.calendar
    
    def clean_price(self, price_str):
        """Convert price string to float"""
        if pd.isna(price_str):
            return np.nan
        price_cleaned = re.sub(r'[$,€]', '', str(price_str))
        try:
            return float(price_cleaned)
        except:
            return np.nan
    
    def clean_data(self, df):
        """Clean the listings data"""
        logger.info("Cleaning data...")
        
        # Clean price
        df['price_clean'] = df['price'].apply(self.clean_price)
        
        # Remove invalid prices
        df = df[df['price_clean'].notna()]
        df = df[df['price_clean'] > 0]
        
        # Remove outliers (prices between 10 and 1000)
        original_count = len(df)
        df = df[(df['price_clean'] >= 10) & (df['price_clean'] <= 1000)]
        logger.info(f"Removed {original_count - len(df)} price outliers")
        
        # Handle missing values
        # Bathrooms
        if 'bathrooms_text' in df.columns:
            df['bathrooms_text'].fillna('1 bath', inplace=True)
        
        # Bedrooms
        if 'bedrooms' in df.columns:
            df['bedrooms'].fillna(
                df.groupby('accommodates')['bedrooms'].transform('median'), 
                inplace=True
            )
        
        # Review scores
        review_cols = [col for col in df.columns if 'review_scores' in col]
        for col in review_cols:
            if col in df.columns:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Convert all boolean columns
        # Find all columns that might contain 't'/'f' values
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 3 and any(val in ['t', 'f'] for val in unique_vals):
                    logger.info(f"Converting boolean column: {col}")
                    df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)
        
        # Specific boolean conversions (in case they were missed)
        bool_columns = [
            'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
            'instant_bookable', 'has_availability', 'is_business_travel_ready'
        ]
        
        for col in bool_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)
        
        # Convert numeric columns that might be stored as strings
        numeric_columns = ['host_response_rate', 'host_acceptance_rate']
        for col in numeric_columns:
            if col in df.columns:
                # Remove percentage signs and convert
                df[col] = df[col].astype(str).str.rstrip('%').replace('N/A', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def extract_bathroom_count(self, bath_text):
        """Extract bathroom count from text"""
        if pd.isna(bath_text):
            return 1
        numbers = re.findall(r'\d+\.?\d*', str(bath_text))
        return float(numbers[0]) if numbers else 1