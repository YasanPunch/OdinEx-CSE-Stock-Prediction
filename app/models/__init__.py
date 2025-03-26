from typing import Dict, Type, List
from .base import BaseStockModel
from .gru_bi import GRUPredictor
from .lstm_bi import LSTMPredictor

# Update the available_models dictionary
available_models: Dict[str, Dict] = {
    "Bidirectional GRU": {
        "class": GRUPredictor,
        "description": "Uses bidirectional GRU cells to capture stock price patterns",
        "complexity": "Medium",
        "recommended_for": ["Short-term predictions", "Trend analysis"],
        "min_data_points": 100
    },
    "Bidirectional LSTM": {
        "class": LSTMPredictor,
        "description": "Uses bidirectional LSTM cells to capture stock price patterns with improved memory retention",
        "complexity": "Medium-High",
        "recommended_for": ["Short-term predictions", "Trend analysis", "Complex pattern recognition"],
        "min_data_points": 120
    }
    # Future models will be added here
}

def get_model_class(model_name: str) -> Type[BaseStockModel]:
    """Get model class by name with error handling"""
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(available_models.keys())}")
    return available_models[model_name]["class"]

def get_model_instance(model_name: str) -> BaseStockModel:
    """Get initialized model instance by name"""
    model_class = get_model_class(model_name)
    return model_class()

def get_model_info(model_name: str = None) -> Dict:
    """Get info about available models or a specific model"""
    if model_name:
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(available_models.keys())}")
        return available_models[model_name]
    return available_models

def get_model_names() -> List[str]:
    """Get list of available model names"""
    return list(available_models.keys())