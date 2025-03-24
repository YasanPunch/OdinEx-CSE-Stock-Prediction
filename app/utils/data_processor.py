import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self._scaler_fitted = False
        
    def get_available_companies(self) -> List[str]:
        """Get list of available company codes from processed data"""
        data_dir = Path("app/data/processed_data")
        files = list(data_dir.glob("*.N*_historical.csv"))  # Match your file pattern
        if not files:
            return []
        # Extract company code from filename (e.g., "JKH.N0000" from "JKH.N0000_historical.csv")
        return [f.stem.split('_')[0] for f in files]

    def load_stock_data(self, company: str) -> pd.DataFrame:
        """Load stock data from CSV file"""
        data_path = Path(f"app/data/processed_data/{company}_historical.csv")
        if not data_path.exists():
            raise FileNotFoundError(f"No data file found for company {company}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def prepare_features(self, 
                        df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """Prepare and scale features"""
        if feature_columns is None:
            feature_columns = ['Close']
            
        data = df[feature_columns].values.astype('float32')
        if not self._scaler_fitted:
            data = self.scaler.fit_transform(data)
            self._scaler_fitted = True
        else:
            data = self.scaler.transform(data)
            
        return data

    def create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        for i in range(len(data) - window_size):
            # Create sequence of shape (window_size, features)
            X.append(data[i:(i + window_size)])
            # Target is the next value after the sequence
            y.append(data[i + window_size, 0])  # Only predict the first feature (Close price)
        
        X = np.array(X)
        y = np.array(y)
        
        # Ensure correct shapes
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], X.shape[1], -1)  # (batch_size, window_size, features)
        y = y.reshape(-1)  # Flatten targets
        
        return X, y

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data"""
        if not self._scaler_fitted:
            raise ValueError("Scaler has not been fitted yet")
            
        # If data is 1D, reshape it
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        # If we're only predicting Close price, pad with zeros for other features
        if data.shape[1] == 1 and self.scaler.n_features_in_ > 1:
            pad_width = self.scaler.n_features_in_ - 1
            data = np.pad(data, ((0, 0), (0, pad_width)), 'constant')
            
        transformed = self.scaler.inverse_transform(data)
        return transformed[:, 0]  # Return only the Close price column

    def get_train_test_split(self, data: np.ndarray, test_size: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into training and test sets"""
        if len(data) <= test_size:
            raise ValueError(f"Insufficient data for splitting. Total size: {len(data)}, requested test size: {test_size}")
            
        train_size = len(data) - test_size
        return data[:train_size], data[train_size:]
