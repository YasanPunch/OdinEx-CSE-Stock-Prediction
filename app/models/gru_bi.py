import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
import os
from pathlib import Path
from tqdm import tqdm

from .base import BaseStockModel

import torch.nn.utils as nn_utils

class GRUNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super(GRUNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add batch dimension if missing
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUPredictor(BaseStockModel):
    def __init__(self):
        # Set CUDA device if available, otherwise use CPU
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.models = []
        self.config = self.get_default_config()
        self.metrics = {}
        self.scaler = None
        self.progress_callback = None
        
        # Set seeds for reproducibility
        torch.manual_seed(1234)
        np.random.seed(1234)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'name': 'Bidirectional GRU Ensemble',
            'description': 'Multi-model ensemble using bidirectional GRU for time series prediction',
            'strengths': [
                'Effective for capturing long-term dependencies',
                'Ensemble approach reduces prediction variance',
                'Handles variable-length sequences well'
            ],
            'limitations': [
                'Requires substantial training data',
                'May overfit on small datasets'
            ]
        }

    def get_configurable_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'window_size': {
                'type': 'int',
                'min': 5,
                'max': 60,
                'default': 5,
                'description': 'Number of past days to consider'
            },
            'hidden_dim': {
                'type': 'int',
                'min': 32,
                'max': 256,
                'default': 64,
                'description': 'Hidden layer dimension'
            },
            'num_layers': {
                'type': 'int',
                'min': 1,
                'max': 4,
                'default': 2,
                'description': 'Number of GRU layers'
            },
            'dropout': {
                'type': 'float',
                'min': 0.0,
                'max': 0.5,
                'default': 0.2,
                'description': 'Dropout rate'
            },
            'learning_rate': {
                'type': 'float',
                'min': 0.0001,
                'max': 0.01,
                'default': 0.001,
                'description': 'Learning rate'
            }
        }

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            'window_size': 5,
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 150,
            'batch_size': 32,
            'num_models': 5,
            'patience': 60,
        }

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set a callback function to report training progress
        
        Args:
            callback: Function that takes progress (0-1) and status message
        """
        self.progress_callback = callback

    def report_progress(self, progress: float, message: str) -> None:
        """Report progress via callback if available"""
        if self.progress_callback:
            self.progress_callback(progress, message)

    def create_sequences(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Match notebook's sequence creation exactly"""
        X, Y = [], []
        if len(data) <= window:
            raise ValueError(f"Data length ({len(data)}) must be greater than window size ({window})")
            
        for i in range(len(data) - window):
            seq_x = data[i : i + window]
            seq_y = data[i + window]
            X.append(seq_x)
            Y.append(seq_y)
        return np.array(X), np.array(Y)

    def batch_generator(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        """Generate mini-batches from X and y"""
        indices = torch.randperm(len(X))
        X = X[indices]
        y = y[indices]
        for i in range(0, len(X), batch_size):
            yield X[i:i + batch_size], y[i:i + batch_size]

    def prepare_data(self, data: np.ndarray, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Data preparation with better error handling"""
        try:
            # Validate input data
            if data is None or data.size == 0:
                raise ValueError("Empty or invalid input data")
                
            X = torch.FloatTensor(data).to(self.device)
            
            # Ensure 3D tensor for GRU input [batch, sequence, features]
            if len(X.shape) == 1:
                X = X.unsqueeze(-1).unsqueeze(0)
            elif len(X.shape) == 2:
                X = X.unsqueeze(-1)
                
            return X, None
        except Exception as e:
            raise ValueError(f"Error preparing data: {str(e)}")

    def calculate_accuracy(self, real: np.ndarray, predict: np.ndarray) -> float:
        """Custom accuracy calculation with error handling"""
        try:
            # Validate inputs
            if len(real) != len(predict):
                raise ValueError(f"Length mismatch: real ({len(real)}) vs predict ({len(predict)})")
                
            if len(real) == 0:
                raise ValueError("Empty arrays provided")
            
            # Add small epsilon to avoid division by zero
            real = np.array(real) + 1
            predict = np.array(predict) + 1
            percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
            return percentage * 100
        except Exception as e:
            print(f"Error calculating accuracy: {str(e)}")
            return 0.0

    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Train multiple models for ensemble prediction without UI dependencies"""
        X_train, y_train = train_data
        X_val, y_val = validation_data
        
        # Validate input data
        if X_train is None or y_train is None or X_val is None or y_val is None:
            raise ValueError("Training data cannot be None")
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Empty training data provided")

        # Convert to PyTorch tensors if they aren't already
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)

        self.models = []
        model_results = []
        total_epochs = self.config['epochs'] * self.config['num_models']
        current_epoch = 0

        for m in range(self.config['num_models']):
            self.report_progress(
                m / self.config['num_models'], 
                f"Training model {m+1}/{self.config['num_models']}"
            )

            input_dim = X_train.shape[2] if len(X_train.shape) == 3 else 1
            model = GRUNetwork(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)

            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            # Initialize a step learning rate scheduler: reduces lr every 50 epochs by a factor of 0.5
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            criterion = nn.MSELoss()

            # Early stopping variables
            best_val_loss = float('inf')
            epochs_no_improve = 0
            
            # Training loop with progress reporting, gradient clipping, and learning rate scheduling
            for epoch in range(self.config['epochs']):
                current_epoch += 1
                model.train()
                train_losses = []

                # Process training batches
                for batch_x, batch_y in self.batch_generator(X_train, y_train, self.config['batch_size']):
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()

                    # Apply gradient clipping to stabilize training
                    nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    train_losses.append(loss.item())
                
                # Calculate average training loss
                avg_train_loss = np.mean(train_losses)
                
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val)
                    val_loss = criterion(val_output.squeeze(), y_val).item()
                
                # Step the scheduler at the end of each epoch
                scheduler.step()
                
                # Report progress for each epoch
                progress = (m * self.config['epochs'] + epoch + 1) / total_epochs
                self.report_progress(
                    progress, 
                    f"Model {m+1}/{self.config['num_models']} | Epoch {epoch+1}/{self.config['epochs']} | "
                    f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                
                # Early stopping check (if patience is set)
                if hasattr(self.config, 'patience') and self.config.get('patience', 0) > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= self.config['patience']:
                            self.report_progress(
                                progress,
                                f"Early stopping at epoch {epoch+1} for model {m+1}"
                            )
                            break
                        
            # Evaluate model on validation set
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).cpu().numpy()

            accuracy = self.calculate_accuracy(y_val.cpu().numpy(), val_pred)
            model_results.append({
                'model': model,
                'accuracy': accuracy,
                'predictions': val_pred,
                'val_loss': best_val_loss
            })
            self.models.append(model)

        self.report_progress(1.0, "Training completed")

        # Sort models by accuracy and keep the best ones
        model_results.sort(key=lambda x: x['accuracy'], reverse=True)
        self.models = [result['model'] for result in model_results]

        # Calculate ensemble metrics
        ensemble_predictions = self.predict(X_val.cpu().numpy(), 1)
        metrics = {
            'accuracy': self.calculate_accuracy(y_val.cpu().numpy(), ensemble_predictions),
            'individual_accuracies': [result['accuracy'] for result in model_results],
            'val_loss': model_results[0]['val_loss']  # Include validation loss in metrics
        }
        self.metrics = metrics
        return metrics


    def predict(self, input_data: np.ndarray, prediction_days: int) -> np.ndarray:
        """Make ensemble predictions with improved error handling"""
        try:
            if not self.models:
                raise ValueError("No trained models available")
                
            # Ensure input data has correct shape
            if input_data is None or input_data.size == 0:
                raise ValueError("Empty input data provided")
                
            # Ensure 3D input: [batch, sequence, features]
            input_shape = input_data.shape
            if len(input_shape) == 1:
                # Handle 1D data - single feature sequence
                input_data = input_data.reshape(1, -1, 1)
            elif len(input_shape) == 2:
                # Handle 2D data - either multiple sequences or seq+features
                if input_shape[0] < input_shape[1]:
                    # Likely a single sequence with multiple features
                    input_data = input_data.reshape(1, input_shape[0], input_shape[1])
                else:
                    # Likely multiple sequences with single feature
                    input_data = input_data.reshape(input_shape[0], input_shape[1], 1)
                    
            all_predictions = []
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data).to(self.device)
                
                for model in self.models:
                    model.eval()
                    if prediction_days == 1:
                        pred = model(input_tensor).cpu().numpy()
                    else:
                        # Sequential prediction for multiple days
                        pred = self.predict_sequence(model, input_tensor, prediction_days)
                        
                    all_predictions.append(pred)
            
            # Ensemble predictions by averaging across models
            ensemble_pred = np.mean(all_predictions, axis=0)
            return ensemble_pred
            
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def predict_sequence(self, model: nn.Module, initial_seq: torch.Tensor, days: int) -> np.ndarray:
        """Make sequential predictions for multiple days"""
        current_seq = initial_seq.clone()
        predictions = []
        
        for _ in range(days):
            # Get next prediction
            with torch.no_grad():
                output = model(current_seq)
                
            # Extract prediction
            if output.size(0) == 0:
                raise ValueError("Model produced empty output")
                
            pred = output[0, 0].cpu().numpy() if output.dim() > 1 else output[0].cpu().numpy()
            predictions.append(pred)
            
            # Update sequence for next prediction
            if days > 1 and _ < days - 1:
                # Roll the sequence and update the last element
                current_seq = current_seq.roll(-1, dims=1)
                current_seq[0, -1, 0] = torch.tensor(pred, device=self.device)
                
        return np.array(predictions)

    def get_metrics(self) -> Dict[str, float]:
        """Return current model metrics"""
        return self.metrics

    def save(self, path: str) -> None:
        """Save model to disk with better error handling"""
        try:
            if not self.models:
                raise ValueError("No models to save")
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            
            model_states = [model.state_dict() for model in self.models]
            save_dict = {
                'model_states': model_states,
                'config': self.config,
                'metrics': self.metrics
            }
            
            torch.save(save_dict, path)
            
        except Exception as e:
            raise IOError(f"Error saving model: {str(e)}")
            
    def load(self, path: str) -> None:
        """Load model from disk with better error handling"""
        try:
            if not Path(path).exists():
                raise FileNotFoundError(f"Model file not found: {path}")
                
            checkpoint = torch.load(path, map_location=self.device)
            
            if 'model_states' not in checkpoint:
                raise ValueError("Invalid model file format")
                
            self.models = []
            for state_dict in checkpoint['model_states']:
                model = GRUNetwork(
                    input_dim=1,  # Default dimension
                    hidden_dim=checkpoint.get('config', self.config)['hidden_dim'],
                    num_layers=checkpoint.get('config', self.config)['num_layers'],
                    dropout=checkpoint.get('config', self.config)['dropout']
                ).to(self.device)
                
                model.load_state_dict(state_dict)
                model.eval()  # Set to evaluation mode
                self.models.append(model)
                
            self.config = checkpoint.get('config', self.config)
            self.metrics = checkpoint.get('metrics', {})
            
        except Exception as e:
            raise IOError(f"Error loading model: {str(e)}")