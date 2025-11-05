"""Utility functions for the project"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_pickle(obj, filepath):
    """Save object as pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved {filepath}")

def load_pickle(filepath):
    """Load pickle file"""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded {filepath}")
    return obj

def create_directories():
    """Create necessary directories"""
    dirs = ['data/processed', 'results', 'results/plots', 'results/feature_importance']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Created directories")

def plot_feature_importance(importance_df, title, top_n=20, save_path=None):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    importance_df.head(top_n).plot(x='feature', y='importance', kind='barh')
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()