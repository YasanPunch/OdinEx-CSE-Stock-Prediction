import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Tuple
import streamlit as st
from pathlib import Path
from tqdm import tqdm

from .base import BaseStockModel
from utils.metrics import calculate_metrics, calculate_prediction_confidence

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
            dropout=dropout,
            bidirectional=self.bidirectional
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUPredictor(BaseStockModel):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.config = self.get_default_config()
        self.metrics = {}
        self.scaler = None
        
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
            'num_models': 5
        }

    def create_sequences(self, data: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Match notebook's sequence creation exactly"""
        X, Y = [], []
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
        """Simplified data preparation matching notebook"""
        X = torch.FloatTensor(data).to(self.device)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        return X, None

    def calculate_accuracy(self, real: np.ndarray, predict: np.ndarray) -> float:
        """Custom accuracy calculation matching the notebook"""
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              validation_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Train multiple models for ensemble prediction"""
        X_train, y_train = train_data
        X_val, y_val = validation_data
        
        # Convert to PyTorch tensors if they aren't already
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).to(self.device)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_text = st.empty()
        
        self.models = []
        model_results = []
        
        for m in range(self.config['num_models']):
            input_dim = X_train.shape[2] if len(X_train.shape) == 3 else 1
            model = GRUNetwork(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            ).to(self.device)
            
            optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.MSELoss()
            
            # Training loop matching notebook implementation
            for epoch in tqdm(range(self.config['epochs'])):
                model.train()
                train_losses = []
                
                for batch_x, batch_y in self.batch_generator(X_train, y_train, self.config['batch_size']):
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                
                # Update progress
                progress = ((epoch + 1) + (m * self.config['epochs'])) / (self.config['epochs'] * self.config['num_models'])
                progress_bar.progress(progress)
                
                if (epoch + 1) % 20 == 0:
                    status_text.text(f"Model {m+1} | Epoch [{epoch+1}/{self.config['epochs']}]")
                    metrics_text.text(f"Training Loss: {np.mean(train_losses):.6f}")
            
            # Evaluate model
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val).cpu().numpy()
            
            accuracy = self.calculate_accuracy(y_val.cpu().numpy(), val_pred)
            model_results.append({
                'model': model,
                'accuracy': accuracy,
                'predictions': val_pred
            })
            
            self.models.append(model)
        
        # Sort models by accuracy and keep the best ones
        model_results.sort(key=lambda x: x['accuracy'], reverse=True)
        self.models = [result['model'] for result in model_results]
        
        # Calculate ensemble metrics
        ensemble_predictions = self.predict(X_val.cpu().numpy(), 1)
        metrics = {
            'accuracy': self.calculate_accuracy(y_val.cpu().numpy(), ensemble_predictions),
            'individual_accuracies': [result['accuracy'] for result in model_results]
        }
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        metrics_text.empty()
        
        return metrics

    def predict(self, input_data: np.ndarray, prediction_days: int) -> np.ndarray:
        """Make ensemble predictions matching notebook implementation"""
        if not self.models:
            raise ValueError("No trained models available")
            
        # Ensure input data has correct shape
        if len(input_data.shape) == 2:
            input_data = input_data.reshape(1, *input_data.shape)
            
        # Verify we have data to work with
        if input_data.size == 0:
            raise ValueError("Empty input data provided")
            
        all_predictions = []
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).to(self.device)
            
            for model in self.models:
                model.eval()
                if prediction_days == 1:
                    pred = model(input_tensor).cpu().numpy()
                else:
                    current_seq = input_tensor.clone()
                    predictions = []
                    
                    # Make first prediction
                    output = model(current_seq)
                    if output.size(0) == 0:
                        raise ValueError("Model produced empty output")
                    last_pred = output[0, -1] if output.dim() > 1 else output[0]
                    predictions.append(last_pred.cpu().numpy())
                    
                    # Continue with remaining predictions
                    for _ in range(prediction_days - 1):
                        # Update sequence with last prediction
                        if current_seq.size(0) > 0:  # Ensure sequence isn't empty
                            current_seq = torch.roll(current_seq, -1, dims=1)
                            current_seq[0, -1] = last_pred
                            
                            # Get next prediction
                            output = model(current_seq)
                            last_pred = output[0] if output.dim() == 1 else output[0, -1]
                            predictions.append(last_pred.cpu().numpy())
                        
                    pred = np.array(predictions)
                all_predictions.append(pred)
        
        ensemble_pred = np.mean(all_predictions, axis=0)
        return ensemble_pred

    def get_metrics(self) -> Dict[str, float]:
        """Return current model metrics"""
        return self.metrics

    def save(self, path: str) -> None:
        """Save model to disk"""
        if self.models:
            model_states = [model.state_dict() for model in self.models]
            torch.save({
                'model_states': model_states,
                'config': self.config,
                'metrics': self.metrics
            }, path)
            
    def load(self, path: str) -> None:
        """Load model from disk"""
        if Path(path).exists():
            checkpoint = torch.load(path)
            self.models = []
            for state_dict in checkpoint['model_states']:
                model = GRUNetwork(
                    input_dim=1,
                    hidden_dim=self.config['hidden_dim'],
                    num_layers=self.config['num_layers'],
                    dropout=self.config['dropout']
                ).to(self.device)
                model.load_state_dict(state_dict)
                self.models.append(model)
            self.config = checkpoint['config']
            self.metrics = checkpoint['metrics']

