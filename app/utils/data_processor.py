import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import os
import logging
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataProcessor')

class DataProcessor:
    def __init__(self, data_dir: str = "app/data/processed_data", cache_size: int = 10):
        self.data_dir = Path(data_dir)
        self.scaler = MinMaxScaler()
        self._scaler_fitted = False
        self.cache_size = cache_size
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            os.makedirs(self.data_dir, exist_ok=True)
    
    @lru_cache(maxsize=10)
    def get_available_companies(self) -> List[str]:
        """Get list of available company codes from processed data with duplicate removal"""
        try:
            # Use a simpler pattern and ensure unique results with a set
            files = list(self.data_dir.glob("*_historical.csv"))
            
            if not files:
                logger.warning(f"No data files found in {self.data_dir}")
                return []
                
            # Use a set to ensure uniqueness
            companies = list(set([f.stem.split('_')[0] for f in files]))
            logger.info(f"Found {len(companies)} companies: {companies}")
            return companies
        except Exception as e:
            logger.error(f"Error getting available companies: {str(e)}")
            return []

    @lru_cache(maxsize=10)
    def load_stock_data(self, company: str) -> pd.DataFrame:
        """Load stock data from CSV file with enhanced error handling"""
        try:
            data_path = self.data_dir / f"{company}_historical.csv"
            if not data_path.exists():
                raise FileNotFoundError(f"No data file found for company {company}")
            
            logger.info(f"Loading data for {company} from {data_path}")
            
            try:
                df = pd.read_csv(data_path)
            except pd.errors.EmptyDataError:
                raise ValueError(f"Data file for {company} is empty")
            except pd.errors.ParserError:
                raise ValueError(f"Data file for {company} is corrupted or has invalid format")
            
            # Basic data validation and cleaning
            if 'Date' not in df.columns:
                raise ValueError(f"Date column missing in data for {company}")
            
            # Convert Date to datetime with error handling
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                raise ValueError(f"Failed to parse Date column: {str(e)}")
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Check for missing required columns
            required_columns = ['Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Verify data types and handle non-numeric data
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.warning(f"Converted non-numeric values in {col} to NaN")
                    except Exception as e:
                        raise ValueError(f"Column {col} contains invalid non-numeric data: {str(e)}")
            
            # Check for sufficient data
            if len(df) < 30:
                raise ValueError(f"Insufficient data for {company}: only {len(df)} records")
            
            # Handle missing values
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    logger.warning(f"Column {col} has {missing_count} missing values. Filling with appropriate methods.")
                    
                    if col == 'Date':
                        # Cannot have missing dates
                        raise ValueError("Dataset contains missing Date values")
                    elif col in ['Open', 'High', 'Low', 'Close']:
                        # For price columns, use forward fill then backward fill
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    else:
                        # For other numeric columns, use median
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            # For non-numeric, use most frequent value
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None)
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data for {company}: {str(e)}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with robust error handling"""
        try:
            # Make a copy to avoid modifying the original
            df = df.copy()
            
            # Validate that we have enough data for calculations
            if len(df) < 20:
                logger.warning("Insufficient data for some technical indicators")
                return df
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
            df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = delta.mask(delta < 0, 0).fillna(0)
            loss = (-delta).mask(delta > 0, 0).fillna(0)
            
            # Calculate average gain and loss with at least 1 period
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            
            # Avoid division by zero - use epsilon where avg_loss is zero
            rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Price rate of change
            df['ROC'] = df['Close'].pct_change(periods=5) * 100
            
            # MACD - Moving Average Convergence Divergence
            try:
                if len(df) >= 26:  # Minimum data points needed for MACD
                    df['EMA12'] = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
                    df['EMA26'] = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
                    df['MACD'] = df['EMA12'] - df['EMA26']
                    df['Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
                    df['MACD_Hist'] = df['MACD'] - df['Signal']
                else:
                    logger.warning("Insufficient data for MACD calculation")
            except Exception as e:
                logger.error(f"Error calculating MACD: {str(e)}")
            
            # Bollinger Bands
            try:
                if len(df) >= 20:  # Minimum data points needed for Bollinger Bands
                    df['BB_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
                    df['BB_std'] = df['Close'].rolling(window=20, min_periods=1).std()
                    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
                    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
                else:
                    logger.warning("Insufficient data for Bollinger Bands calculation")
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            
            # Fill NaN values that occur at the beginning due to calculations
            for col in df.columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                    fill_method = 'bfill' if col != 'Date' else None
                    if fill_method:
                        if fill_method == 'bfill':
                            df[col] = df[col].bfill()
                        elif fill_method == 'ffill':
                            df[col] = df[col].ffill()
                        else:
                            # Handle any other methods
                            df[col] = df[col].fillna(df[col].mean())
            
            return df
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            # Return original dataframe if calculation fails
            return df.copy()
    
    def prepare_features(self, 
                        df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None,
                        scale_data: bool = True) -> np.ndarray:
        """Prepare and scale features with configurable columns"""
        try:
            # Default to Close plus technical indicators if they exist
            if feature_columns is None:
                feature_columns = ['Close']
                # Add technical indicators if they exist in the dataframe
                for col in ['MA5', 'MA20', 'RSI', 'ROC']:
                    if col in df.columns:
                        feature_columns.append(col)
            
            # Check all requested columns exist
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}. Using only available columns.")
                feature_columns = [col for col in feature_columns if col in df.columns]
                
            if not feature_columns:
                raise ValueError("No valid feature columns available for processing")
                
            logger.info(f"Using features: {feature_columns}")
            data = df[feature_columns].values.astype('float32')
            
            if scale_data:
                if not self._scaler_fitted:
                    data = self.scaler.fit_transform(data)
                    self._scaler_fitted = True
                else:
                    data = self.scaler.transform(data)
            
            return data
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def create_sequences(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction with enhanced validation"""
        try:
            # Validate inputs
            if window_size <= 0:
                raise ValueError("Window size must be positive")
                
            if data is None or data.size == 0:
                raise ValueError("Empty data provided")
                
            if len(data) <= window_size:
                raise ValueError(f"Data length ({len(data)}) must be greater than window size ({window_size})")
                    
            X, y = [], []
            
            # Ensure data is 2D
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
                    
            for i in range(len(data) - window_size):
                # Create sequence of shape (window_size, features)
                X.append(data[i:(i + window_size)])
                # Target is the next value after the sequence
                y.append(data[i + window_size, 0])  # Only predict the first feature (Close price)
            
            if not X or not y:
                raise ValueError("Failed to create any valid sequences")
                
            X_array = np.array(X)
            y_array = np.array(y)
            
            # Ensure correct shapes
            if len(X_array.shape) == 3:
                X_array = X_array.reshape(X_array.shape[0], X_array.shape[1], -1)  # (batch_size, window_size, features)
            y_array = y_array.reshape(-1)  # Flatten targets
            
            logger.info(f"Created sequences with shape X: {X_array.shape}, y: {y_array.shape}")
            return X_array, y_array
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise ValueError(f"Failed to create sequences: {str(e)}")

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data with robust handling of different shapes"""
        try:
            if not self._scaler_fitted:
                raise ValueError("Scaler has not been fitted yet")
            
            # Handle different input shapes    
            original_shape = data.shape
            is_1d = len(original_shape) == 1
            
            # If data is 1D, reshape it
            if is_1d:
                data = data.reshape(-1, 1)
                
            # If we're only predicting Close price, pad with zeros for other features
            if data.shape[1] == 1 and self.scaler.n_features_in_ > 1:
                pad_width = self.scaler.n_features_in_ - 1
                data = np.pad(data, ((0, 0), (0, pad_width)), 'constant')
                
            transformed = self.scaler.inverse_transform(data)
            result = transformed[:, 0]  # Return only the Close price column
            
            # Restore original shape if it was 1D
            if is_1d and len(result.shape) > 1:
                result = result.reshape(original_shape)
                
            return result
            
        except Exception as e:
            logger.error(f"Error in inverse transform: {str(e)}")
            raise

    def get_train_test_split(self, 
                           data: np.ndarray, 
                           test_size: int = 30, 
                           validation_size: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training, validation and test sets"""
        try:
            total_size = len(data)
            
            if total_size <= test_size + validation_size:
                raise ValueError(f"Insufficient data for splitting. Total size: {total_size}, " +
                              f"requested test size: {test_size}, validation size: {validation_size}")
            
            train_size = total_size - test_size - validation_size
            
            if validation_size > 0:
                train_data = data[:train_size]
                val_data = data[train_size:train_size+validation_size]
                test_data = data[train_size+validation_size:]
                return train_data, val_data, test_data
            else:
                train_data = data[:train_size]
                test_data = data[train_size:]
                return train_data, test_data
                
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
            
    def get_dataset_statistics(self, company: str) -> Dict[str, Any]:
        """Get statistics about the dataset for a company"""
        try:
            df = self.load_stock_data(company)
            
            stats = {
                "total_records": len(df),
                "date_range": {
                    "start": df['Date'].min().strftime('%Y-%m-%d'),
                    "end": df['Date'].max().strftime('%Y-%m-%d')
                },
                "price_stats": {
                    "min": df['Close'].min(),
                    "max": df['Close'].max(),
                    "mean": df['Close'].mean(),
                    "median": df['Close'].median(),
                    "std": df['Close'].std()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {str(e)}")
            return {"error": str(e)}