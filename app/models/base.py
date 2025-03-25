from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class BaseStockModel(ABC):
    """Abstract base class for all stock prediction models with enhanced functionality"""
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model description and parameters"""
        pass

    @abstractmethod
    def get_configurable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return list of parameters that can be configured via UI"""
        pass

    @abstractmethod
    def prepare_data(self, data: np.ndarray, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        pass

    @abstractmethod
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Train the model and return metrics"""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray, prediction_days: int) -> np.ndarray:
        """Make predictions for future days"""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return current model performance metrics"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass
        
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model on test data with comprehensive metrics
        
        Args:
            test_data: Tuple of (X_test, y_test)
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = test_data
        predictions = self.predict(X_test, 1)
        
        # Convert to numpy arrays if not already
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        
        # Ensure same shapes
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
            
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
            
        # Basic metrics
        mse = np.mean(np.square(y_test - predictions))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - predictions))
        
        # Custom accuracy (1 - normalized RMSE in percentage)
        real = np.array(y_test) + 1  # Add 1 to avoid division by zero issues
        predict = np.array(predictions) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        accuracy = percentage * 100
        
        # Direction accuracy for financial predictions
        direction_actual = np.diff(y_test)
        direction_pred = np.diff(predictions)
        direction_accuracy = np.mean(np.sign(direction_actual) == np.sign(direction_pred)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'direction_accuracy': direction_accuracy
        }
        
        try:
            # R-squared (coefficient of determination)
            ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
            ss_residual = np.sum((y_test - predictions) ** 2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            metrics['r_squared'] = r_squared
            
            # Mean Absolute Percentage Error
            mape = np.mean(np.abs((y_test - predictions) / np.maximum(1e-10, np.abs(y_test)))) * 100
            metrics['mape'] = mape
        except Exception:
            # Skip advanced metrics if they fail
            pass
            
        return metrics
    
    def validate_with_cross_validation(self, 
                                      data: Tuple[np.ndarray, np.ndarray], 
                                      n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation
        
        Args:
            data: Tuple of (X, y)
            n_splits: Number of cross-validation splits
            
        Returns:
            Dictionary with lists of metrics for each split
        """
        X, y = data
        fold_size = len(X) // (n_splits + 1)
        metrics_list = []
        
        for i in range(n_splits):
            train_end = len(X) - (n_splits - i) * fold_size
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:train_end+fold_size], y[train_end:train_end+fold_size]
            
            # Train model on this split
            self.train((X_train, y_train), (X_val, y_val))
            
            # Evaluate model on validation set
            fold_metrics = self.evaluate((X_val, y_val))
            metrics_list.append(fold_metrics)
            
        # Aggregate metrics across all folds
        aggregated_metrics = {}
        for metric in metrics_list[0].keys():
            values = [fold[metric] for fold in metrics_list]
            aggregated_metrics[metric] = {
                'values': values,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
        return aggregated_metrics