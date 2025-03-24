import numpy as np
from typing import Dict

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics"""
    
    def custom_accuracy(real, predict):
        """Custom accuracy: 1 - normalized RMSE (in percentage)"""
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    # Calculate metrics
    mse = np.mean(np.square(y_true - y_pred))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    accuracy = custom_accuracy(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy
    }

def calculate_prediction_confidence(model_predictions: np.ndarray) -> float:
    """Calculate prediction confidence based on model variance"""
    try:
        if isinstance(model_predictions, np.ndarray):
            if len(model_predictions.shape) > 1:
                variance = np.var(model_predictions, axis=0)
                confidence = 100 * (1 - np.mean(variance))
                return max(0, min(100, confidence))
            else:
                # For single predictions, use historical accuracy as base confidence
                return 85.0  # Base confidence level
    except Exception as e:
        print(f"Error calculating confidence: {str(e)}")
        return 0.0
    return 0.0