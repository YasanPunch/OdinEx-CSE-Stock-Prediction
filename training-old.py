import pandas as pd
import torch
from algorithms import StockPredictor, get_default_config

def train_model(data_path, company, prediction_days, feature_columns=None, window_size=5):
    """
    Main training function called by the Streamlit app
    """
    # Load data
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get default configuration and update with parameters
    config = get_default_config()
    config['window_size'] = window_size
    
    # Initialize predictor
    predictor = StockPredictor(config)
    
    # Prepare data
    X, y = predictor.prepare_data(df, features=feature_columns)
    
    # Split data (similar to your original code)
    test_size = 30
    train_size = len(X) - test_size
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train models and get results
    model_results = predictor.train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Get best model
    best_model = max(model_results, key=lambda x: x['accuracy'])
    
    # Make future predictions
    last_sequence = X[-1]
    future_pred = predictor.predict_future(best_model['model'], last_sequence, prediction_days)
    
    return {
        'model': best_model['model'],
        'accuracy': best_model['accuracy'],
        'predictions': future_pred,
        'test_predictions': best_model['test_pred'],
        'test_actual': y_test,
        'scaler': predictor.scaler
    }