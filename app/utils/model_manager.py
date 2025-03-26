import os
import glob
import pickle
import torch
from pathlib import Path
import logging
import hashlib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelManager')

# Define the cache directory - adjust path if needed
CACHE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "cache" / "models"

def get_cached_models(company):
    """
    Find cached models for a specific company
    
    Args:
        company: Company symbol/name
        
    Returns:
        List of dictionaries containing model info
    """
    try:
        if not company:
            return []
            
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Find all pickle files in the cache directory
        model_files = list(CACHE_DIR.glob("*.pkl"))
        company_models = []
        
        for model_file in model_files:
            try:
                # Load each model file to check if it matches the company
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check if this model is for the requested company
                if model_data.get('company') == company:
                    # Extract relevant model information
                    params = model_data.get('parameters', {})
                    metrics = model_data.get('metrics', {})
                    
                    # Handle the training date - convert string to datetime if necessary
                    training_date = model_data.get('training_date', datetime.now())
                    if isinstance(training_date, str):
                        try:
                            training_date = datetime.strptime(training_date, '%Y-%m-%d %H:%M:%S')
                        except:
                            training_date = datetime.now()
                    
                    # Create a readable name for the model
                    model_type = params.get('model_type', type(model_data.get('model', '')).__name__)
                    model_name = f"{model_type} ({training_date.strftime('%Y-%m-%d %H:%M')})"
                    
                    # Add accuracy if available
                    if isinstance(metrics, dict) and 'accuracy' in metrics:
                        try:
                            acc = float(metrics['accuracy'])
                            model_name += f" - {acc:.2f}% acc"
                        except:
                            pass
                    
                    company_models.append({
                        'file_path': str(model_file),
                        'model_name': model_name,
                        'training_date': training_date,
                        'parameters': params,
                        'metrics': metrics,
                        'file_size': model_file.stat().st_size / (1024 * 1024)  # Size in MB
                    })
            except Exception as e:
                logger.warning(f"Error loading model from {model_file}: {str(e)}")
                continue
        
        # Sort models by training date (newest first)
        company_models.sort(key=lambda x: x['training_date'], reverse=True)
        return company_models
        
    except Exception as e:
        logger.error(f"Error getting cached models: {str(e)}")
        return []

def load_cached_model(file_path):
    """
    Load a model from cache with improved error handling for callback references
    
    Args:
        file_path: Path to the cached model file
        
    Returns:
        Loaded model data or None if error
    """
    try:
        # Define a custom unpickler to handle missing attributes
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Handle the 'progress_callback' attribute
                if name == 'progress_callback':
                    return None
                return super().find_class(module, name)
        
        with open(file_path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            model_data = unpickler.load()
            
            # If the model has a 'model' key that has a 'progress_callback' attribute,
            # set it to None to avoid reference errors
            if 'model' in model_data and hasattr(model_data['model'], 'progress_callback'):
                model_data['model'].progress_callback = None
                
            # If this is a dictionary with model_data, check that too
            if 'model_data' in model_data and 'model' in model_data['model_data']:
                if hasattr(model_data['model_data']['model'], 'progress_callback'):
                    model_data['model_data']['model'].progress_callback = None
                    
        return model_data
    except Exception as e:
        logging.error(f"Error loading cached model: {str(e)}")
        return None