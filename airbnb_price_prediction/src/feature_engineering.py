"""Feature engineering functions"""

import pandas as pd
import numpy as np
from datetime import datetime
from textblob import TextBlob
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from src.utils import logger
import re

class FeatureEngineer:
    def __init__(self):
        self.neighborhood_stats = None
        self.location_clusters = None
        self.berlin_center = (52.5163, 13.3777)
        
    def create_basic_features(self, df):
        """Create basic features"""
        logger.info("Creating basic features...")
        
        # Property type grouping
        major_property_types = df['property_type'].value_counts().head(10).index.tolist()
        df['property_type_group'] = df['property_type'].apply(
            lambda x: x if x in major_property_types else 'Other'
        )
        
        # Room type encoding
        df['entire_home'] = (df['room_type'] == 'Entire home/apt').astype(int)
        df['private_room'] = (df['room_type'] == 'Private room').astype(int)
        
        # Extract bathroom count
        if 'bathrooms_text' in df.columns:
            df['bathrooms_count'] = df['bathrooms_text'].apply(self._extract_bathroom_count)
        elif 'bathrooms' in df.columns:
            df['bathrooms_count'] = df['bathrooms']
        else:
            df['bathrooms_count'] = 1
        
        # Host experience
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_days'] = (datetime.now() - df['host_since']).dt.days
            df['host_days'].fillna(365, inplace=True)  # Default to 1 year
        else:
            df['host_days'] = 365
        
        return df
    
    def create_text_features(self, df):
        """Create text-based features"""
        logger.info("Creating text features...")
        
        # Combine text fields
        text_fields = []
        if 'name' in df.columns:
            text_fields.append(df['name'].fillna(''))
        if 'description' in df.columns:
            text_fields.append(df['description'].fillna(''))
        if 'neighborhood_overview' in df.columns:
            text_fields.append(df['neighborhood_overview'].fillna(''))
        
        if text_fields:
            df['combined_text'] = text_fields[0]
            for field in text_fields[1:]:
                df['combined_text'] = df['combined_text'] + ' ' + field
        else:
            df['combined_text'] = ''
        
        # Basic text features
        df['description_length'] = df['combined_text'].str.len()
        df['description_word_count'] = df['combined_text'].str.split().str.len()
        
        # Sentiment analysis
        df['description_sentiment'] = df['combined_text'].apply(self._get_sentiment)
        
        # Keywords
        luxury_keywords = ['luxury', 'premium', 'high-end', 'exclusive', 'designer', 'penthouse', 'villa']
        budget_keywords = ['budget', 'cheap', 'affordable', 'economic', 'simple', 'basic']
        
        df['luxury_count'] = df['combined_text'].str.lower().apply(
            lambda x: sum(keyword in str(x) for keyword in luxury_keywords)
        )
        df['budget_count'] = df['combined_text'].str.lower().apply(
            lambda x: sum(keyword in str(x) for keyword in budget_keywords)
        )
        
        return df
    
    def create_location_features(self, df):
        """Create location-based features"""
        logger.info("Creating location features...")
        
        # Check if we have location data
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning("No location data available")
            return df
        
        # Neighborhood statistics
        if 'neighbourhood_cleansed' in df.columns:
            self.neighborhood_stats = df.groupby('neighbourhood_cleansed')['price_clean'].agg([
                'mean', 'median', 'std', 'count'
            ]).reset_index()
            self.neighborhood_stats.columns = ['neighbourhood_cleansed', 'neigh_price_mean', 
                                              'neigh_price_median', 'neigh_price_std', 'neigh_count']
            
            df = df.merge(self.neighborhood_stats, on='neighbourhood_cleansed', how='left')
        
        # Distance to center
        df['distance_to_center'] = df.apply(
            lambda row: self._calculate_distance_to_center(row['latitude'], row['longitude']), 
            axis=1
        )
        
        # Location clusters
        coords = df[['latitude', 'longitude']].values
        if len(coords) > 20:
            kmeans = KMeans(n_clusters=20, random_state=42)
            df['location_cluster'] = kmeans.fit_predict(coords)
            self.location_clusters = kmeans
        else:
            df['location_cluster'] = 0
        
        return df
    
    def create_amenities_features(self, df):
        """Create amenities-based features"""
        logger.info("Creating amenities features...")
        
        if 'amenities' not in df.columns:
            logger.warning("No amenities data available")
            df['amenities_count'] = 0
            return df
        
        # Parse amenities
        df['amenities_list'] = df['amenities'].apply(self._parse_amenities)
        df['amenities_count'] = df['amenities_list'].apply(len)
        
        # Important amenities
        important_amenities = [
            'Wifi', 'Kitchen', 'Washer', 'Dryer', 'Air conditioning', 'Heating',
            'TV', 'Pool', 'Hot tub', 'Gym', 'Elevator', 'Parking', 'Balcony'
        ]
        
        for amenity in important_amenities:
            col_name = f'has_{amenity.lower().replace(" ", "_")}'
            df[col_name] = df['amenities_list'].apply(
                lambda x: 1 if any(amenity.lower() in a.lower() for a in x) else 0
            )
        
        return df
    
    def create_calendar_features(self, df, calendar):
        """Create calendar-based features"""
        logger.info("Creating calendar features...")
        
        # Check if calendar data is empty
        if calendar.empty:
            logger.warning("Calendar data is empty, skipping calendar features")
            df['availability_rate'] = 0.5  # Default values
            df['days_available'] = 180
            return df
        
        # Process calendar data
        calendar['price_cal'] = calendar['price'].apply(self._clean_price)
        calendar['available'] = calendar['available'].map({'t': 1, 'f': 0})
        
        # Aggregate
        calendar_agg = calendar.groupby('listing_id').agg({
            'available': ['mean', 'sum'],
            'price_cal': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        calendar_agg.columns = ['listing_id', 'availability_rate', 'days_available', 
                               'cal_price_mean', 'cal_price_std', 'cal_price_min', 'cal_price_max']
        
        df = df.merge(calendar_agg, left_on='id', right_on='listing_id', how='left')
        
        # Fill missing values
        df['availability_rate'].fillna(0.5, inplace=True)
        df['days_available'].fillna(180, inplace=True)
        
        return df
    
    def create_review_features(self, df, reviews):
        """Create review-based features"""
        logger.info("Creating review features...")
        
        if reviews.empty:
            logger.warning("Reviews data is empty, using default values")
            df['total_reviews'] = df['number_of_reviews'] if 'number_of_reviews' in df.columns else 0
            df['recent_reviews_count'] = 0
            df['days_since_last_review'] = 365
            return df
        
        # Process reviews
        reviews['date'] = pd.to_datetime(reviews['date'])
        recent_date = reviews['date'].max() - pd.Timedelta(days=180)
        recent_reviews = reviews[reviews['date'] >= recent_date]
        
        # Count features
        review_counts = reviews.groupby('listing_id').agg({
            'id': 'count',
            'date': 'max'
        }).reset_index()
        review_counts.columns = ['listing_id', 'total_reviews', 'last_review_date']
        
        recent_review_counts = recent_reviews.groupby('listing_id').size().reset_index(name='recent_reviews_count')
        
        # Merge
        df = df.merge(review_counts, left_on='id', right_on='listing_id', how='left')
        df = df.merge(recent_review_counts, left_on='id', right_on='listing_id', how='left')
        
        # Fill missing
        df['total_reviews'].fillna(0, inplace=True)
        df['recent_reviews_count'].fillna(0, inplace=True)
        
        # Days since last review
        df['days_since_last_review'] = (
            datetime.now() - pd.to_datetime(df['last_review_date'])
        ).dt.days
        df['days_since_last_review'].fillna(365, inplace=True)
        
        return df
    
    def create_image_features(self, df):
        """Create simulated image features"""
        logger.info("Creating image features...")
        
        # Estimate photo quality based on other features
        df['estimated_photo_quality'] = (
            df['host_is_superhost'] * 0.3 +
            df['host_has_profile_pic'] * 0.1 +
            (df['review_scores_rating'].fillna(80) / 100) * 0.4 +
            (df['amenities_count'] / df['amenities_count'].max()) * 0.2
        )
        
        # Simulate image quality score
        np.random.seed(42)
        df['image_brightness_score'] = np.random.beta(
            a=2 + df['estimated_photo_quality'] * 3,
            b=2,
            size=len(df)
        )
        df['image_quality_score'] = df['estimated_photo_quality'] * 0.7 + df['image_brightness_score'] * 0.3
        
        return df
    
    def create_all_features(self, df, calendar, reviews):
        """Create all features"""
        df = self.create_basic_features(df)
        df = self.create_text_features(df)
        df = self.create_location_features(df)
        df = self.create_amenities_features(df)
        df = self.create_calendar_features(df, calendar)
        df = self.create_review_features(df, reviews)
        df = self.create_image_features(df)
        
        return df
    
    # Helper methods
    def _extract_bathroom_count(self, bath_text):
        if pd.isna(bath_text):
            return 1
        numbers = re.findall(r'\d+\.?\d*', str(bath_text))
        return float(numbers[0]) if numbers else 1
    
    def _get_sentiment(self, text):
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity
        except:
            return 0
    
    def _calculate_distance_to_center(self, lat, lon):
        try:
            return geodesic((lat, lon), self.berlin_center).kilometers
        except:
            return np.nan
    
    def _parse_amenities(self, amenities_str):
        if pd.isna(amenities_str):
            return []
        cleaned = amenities_str.strip('[]').replace('"', '').replace("'", "")
        return [a.strip() for a in cleaned.split(',')]
    
    def _clean_price(self, price_str):
        if pd.isna(price_str):
            return np.nan
        price_cleaned = re.sub(r'[$,€]', '', str(price_str))
        try:
            return float(price_cleaned)
        except:
            return np.nan