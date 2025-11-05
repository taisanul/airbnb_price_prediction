"""Evaluation metrics and functions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from src.utils import logger

def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        #'mae': mae,
        'r2': r2
    }

def plot_model_comparison(results_df, save_path='results/plots/model_comparison.png'):
    """Plot model comparison"""
    logger.info("Creating model comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Sort by RMSE
    results_sorted = results_df.sort_values('RMSE')
    
    # RMSE comparison
    rmse_sorted = results_df.sort_values(by='RMSE')
    axes[0].bar(rmse_sorted['Model'], rmse_sorted['RMSE'], color=plt.cm.viridis(np.linspace(0, 1, len(rmse_sorted))))
    axes[0].set_title('Model Comparison - RMSE')
    axes[0].set_ylabel('RMSE (€)')
    axes[0].tick_params(axis='x', rotation=45)
    # R² comparison
    r2_sorted = results_df.sort_values(by='R2', ascending=False)
    axes[1].bar(r2_sorted['Model'], r2_sorted['R2'], color=plt.cm.viridis(np.linspace(0, 1, len(r2_sorted))))
    axes[1].set_title('Model Comparison - R²')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    # MAE comparison
    #ax3 = axes[1, 0]
    #results_sorted.plot(x='Model', y='MAE', kind='bar', ax=ax3, color='coral')
    #ax3.set_title('Model Comparison - MAE')
    #ax3.set_ylabel('MAE (€)')
    #ax3.set_xticklabels(results_sorted['Model'], rotation=45, ha='right')
    
    # Training time comparison
    results_df_melted = results_df.melt(id_vars='Model', value_vars=['Training_Time', 'Prediction_Time'], var_name='Time_Type', value_name='Time')
    # Need to handle grouping by Model and Time_Type for stacked or grouped bars if needed.
    # For simplicity, let's plot them side-by-side for each model.
    x = np.arange(len(results_df['Model']))
    width = 0.35
    training_times = results_df_melted[results_df_melted['Time_Type'] == 'Training_Time'].sort_values(by='Model')['Time']
    prediction_times = results_df_melted[results_df_melted['Time_Type'] == 'Prediction_Time'].sort_values(by='Model')['Time']

    rects1 = axes[2].bar(x - width/2, training_times, width, label='Training Time', color=plt.cm.viridis(0.3))
    rects2 = axes[2].bar(x + width/2, prediction_times, width, label='Prediction Time', color=plt.cm.viridis(0.7))

    axes[2].set_title('Model Training and Prediction Time')
    axes[2].set_ylabel('Time (s)')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_residual_analysis(y_true, y_pred, model_name, save_path=None):
    """Plot residual analysis"""
    logger.info(f"Creating residual plots for {model_name}...")
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Price (€)')
    ax1.set_ylabel('Residuals (€)')
    ax1.set_title(f'{model_name} - Residuals vs Predicted Values')
    
    # Residual distribution
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, edgecolor='black')
    ax2.set_xlabel('Residuals (€)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_name} - Distribution of Residuals')
    
    # Q-Q plot
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title(f'{model_name} - Q-Q Plot')
    
    # Actual vs Predicted
    ax4 = axes[1, 1]
    ax4.scatter(y_true, y_pred, alpha=0.5)
    ax4.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax4.set_xlabel('Actual Price (€)')
    ax4.set_ylabel('Predicted Price (€)')
    ax4.set_title(f'{model_name} - Actual vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_prediction_errors(y_true, y_pred, model_name):
    """Analyze prediction errors"""
    logger.info(f"Analyzing prediction errors for {model_name}...")
    
    errors = np.abs(y_true - y_pred)
    relative_errors = errors / y_true * 100
    
    print(f"\n=== Error Analysis for {model_name} ===")
    print(f"Mean Absolute Error: €{errors.mean():.2f}")
    print(f"Median Absolute Error: €{np.median(errors):.2f}")
    print(f"Mean Relative Error: {relative_errors.mean():.1f}%")
    print(f"Median Relative Error: {np.median(relative_errors):.1f}%")
    print(f"\nError Percentiles:")
    print(f"  25th percentile: €{np.percentile(errors, 25):.2f}")
    print(f"  50th percentile: €{np.percentile(errors, 50):.2f}")
    print(f"  75th percentile: €{np.percentile(errors, 75):.2f}")
    print(f"  90th percentile: €{np.percentile(errors, 90):.2f}")
    print(f"  95th percentile: €{np.percentile(errors, 95):.2f}")
    
    # Percentage of predictions within certain error bounds
    within_10 = (relative_errors <= 10).sum() / len(relative_errors) * 100
    within_20 = (relative_errors <= 20).sum() / len(relative_errors) * 100
    within_30 = (relative_errors <= 30).sum() / len(relative_errors) * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"  Within 10% of actual: {within_10:.1f}%")
    print(f"  Within 20% of actual: {within_20:.1f}%")
    print(f"  Within 30% of actual: {within_30:.1f}%")