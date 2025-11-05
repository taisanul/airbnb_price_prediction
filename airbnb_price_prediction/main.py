"""Main execution script"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from src.utils import create_directories, save_pickle, load_pickle, logger
from src.data_processing import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.models import (LinearRegressionModel,
                       RidgeModel, RandomForestModel, XGBoostModel, LightGBMModel)
from src.training import ModelTrainer
from src.evaluation import plot_model_comparison, plot_residual_analysis, analyze_prediction_errors

def main():
    # Create necessary directories
    create_directories()
    
    # Step 1: Load and clean data
    logger.info("="*50)
    logger.info("STEP 1: Loading and cleaning data")
    logger.info("="*50)
    
    processor = DataProcessor()
    listings_detailed, reviews, calendar = processor.load_data()
    listings_detailed = processor.clean_data(listings_detailed)
    
    # Step 2: Feature engineering
    logger.info("\n" + "="*50)
    logger.info("STEP 2: Feature engineering")
    logger.info("="*50)
    
    engineer = FeatureEngineer()
    listings_detailed = engineer.create_all_features(listings_detailed, calendar, reviews)
    
    # Step 3: Prepare data for modeling
    logger.info("\n" + "="*50)
    logger.info("STEP 3: Preparing data for modeling")
    logger.info("="*50)
    
    trainer = ModelTrainer()
    df_model, numerical_features = trainer.prepare_features(listings_detailed)
    
    # Save feature names
    save_pickle(df_model.columns.tolist(), 'data/processed/feature_names.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test, ids_train, ids_test = trainer.split_data(df_model)
    
    # Scale features
    X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test, numerical_features)
    
    # Save processed data
    save_pickle((X_train, X_test, y_train, y_test), 'data/processed/train_test_data.pkl')
    save_pickle((X_train_scaled, X_test_scaled), 'data/processed/scaled_data.pkl')
    
    # Step 4: Train models
    logger.info("\n" + "="*50)
    logger.info("STEP 4: Training models")
    logger.info("="*50)
    
    # Initialize models
    models = [
        LinearRegressionModel(),
        RidgeModel(alpha=0.1),
        RandomForestModel(),
        XGBoostModel(),
        LightGBMModel()
    ]
    
    # Train and evaluate each model
    predictions = {}
    
    for model in models:
        # Determine which data to use
        if isinstance(model, (LinearRegressionModel, RidgeModel)):
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # Set index for baseline models that need IDs
        X_train_use.index = ids_train.values
        X_test_use.index = ids_test.values
        
        # Train model
        train_time = trainer.train_model(model, X_train_use, y_train)
        
        # Evaluate model
        result, y_pred = trainer.evaluate_model(model, X_test_use, y_test, train_time)
        predictions[model.name] = y_pred
        
        # Save feature importance for tree-based models
        if hasattr(model, 'feature_importance') and model.feature_importance is not None:
            importance_df = model.get_feature_importance(X_train.columns)
            if importance_df is not None:
                importance_df.to_csv(f'results/feature_importance/{model.name.replace(" ", "_")}_importance.csv', index=False)
    
    # Step 5: Compare results
    logger.info("\n" + "="*50)
    logger.info("STEP 5: Comparing results")
    logger.info("="*50)
    
    # Create results dataframe
    results_df = pd.DataFrame(trainer.results)
    results_df.to_csv('results/model_performance.csv', index=False)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(results_df.sort_values('RMSE').to_string(index=False))
    
    # Plot comparisons
    plot_model_comparison(results_df)
    
    # Step 6: Analyze best model
    logger.info("\n" + "="*50)
    logger.info("STEP 6: Analyzing best model")
    logger.info("="*50)
    
    best_model_name = results_df.sort_values('RMSE').iloc[0]['Model']
    best_predictions = predictions[best_model_name]
    
    print(f"\nBest performing model: {best_model_name}")
    
    # Calculate improvement over baseline
    #baseline_rmse = results_df[results_df['Model'] == 'Global Mean']['RMSE'].values[0]
    #best_rmse = results_df.sort_values('RMSE').iloc[0]['RMSE']
    #improvement = (baseline_rmse - best_rmse) / baseline_rmse * 100
    #print(f"RMSE improvement over baseline: {improvement:.1f}%")
    
    # Residual analysis for best model
    plot_residual_analysis(y_test, best_predictions, best_model_name, 
                          f'results/plots/{best_model_name.replace(" ", "_")}_residuals.png')
    
    # Error analysis
    analyze_prediction_errors(y_test, best_predictions, best_model_name)
    
    logger.info("\n" + "="*50)
    logger.info("Project completed successfully!")
    logger.info("="*50)

if __name__ == "__main__":
    main()