from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Tuple

class BaseStockModel(ABC):
    """Abstract base class for all stock prediction models"""
    
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
