import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
import logging
from scipy import stats
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Metrics')

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate various performance metrics with error handling"""
    try:
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")
            
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
            
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic metrics
        mse = np.mean(np.square(y_true - y_pred))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Custom accuracy (1 - normalized RMSE in percentage)
        real = np.array(y_true) + 1  # Add 1 to avoid division by zero
        predict = np.array(y_pred) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        accuracy = percentage * 100
        
        # Direction accuracy (% of correct trend predictions)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_match = np.sign(y_true_diff) == np.sign(y_pred_diff)
        direction_accuracy = np.mean(direction_match) * 100
        
        # R-squared (coefficient of determination)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, np.abs(y_true)))) * 100
        
        # Theil's U statistic (measure of forecasting accuracy)
        changes_true = np.diff(y_true)
        changes_pred = np.diff(y_pred)
        # Avoid division by zero
        u_numerator = np.sqrt(np.mean(np.square(changes_true - changes_pred)))
        u_denominator = np.sqrt(np.mean(np.square(changes_true))) + np.sqrt(np.mean(np.square(changes_pred)))
        theils_u = u_numerator / u_denominator if u_denominator != 0 else 1.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'direction_accuracy': direction_accuracy,
            'r_squared': r_squared,
            'mape': mape,
            'theils_u': theils_u
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        # Return a minimal set of metrics with error indicators
        return {
            'mse': float('nan'),
            'rmse': float('nan'),
            'mae': float('nan'),
            'accuracy': 0.0,
            'error': str(e)
        }

def calculate_prediction_confidence(model_predictions: Union[np.ndarray, List], confidence_level: float = 0.95) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate prediction confidence with statistical confidence intervals
    
    Args:
        model_predictions: Array of predictions from multiple models or ensemble runs
        confidence_level: Confidence level (0-1)
        
    Returns:
        Dictionary with confidence metrics and intervals
    """
    try:
        if isinstance(model_predictions, list):
            model_predictions = np.array(model_predictions)
            
        # If we have multiple model predictions (ensemble)
        if len(model_predictions.shape) > 1 and model_predictions.shape[0] > 1:
            # Calculate mean and standard deviation across models
            mean_pred = np.mean(model_predictions, axis=0)
            std_dev = np.std(model_predictions, axis=0)
            
            # Calculate confidence interval using t-distribution
            n = model_predictions.shape[0]  # Number of models
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin = t_value * std_dev / np.sqrt(n)
            
            lower_bound = mean_pred - margin
            upper_bound = mean_pred + margin
            
            # Calculate coefficient of variation as a confidence metric
            cv = std_dev / np.abs(np.maximum(1e-10, mean_pred)) * 100
            confidence = 100 - np.mean(cv)  # Higher variation = lower confidence
            
            return {
                'confidence': max(0, min(100, confidence)),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'mean': mean_pred,
                'std_dev': std_dev
            }
        else:
            # For single predictions, use a baseline confidence
            return {
                'confidence': 85.0,  # Base confidence level
                'lower_bound': model_predictions * 0.95 if hasattr(model_predictions, '__iter__') else model_predictions * 0.95,
                'upper_bound': model_predictions * 1.05 if hasattr(model_predictions, '__iter__') else model_predictions * 1.05,
                'mean': model_predictions,
                'std_dev': np.zeros_like(model_predictions) if hasattr(model_predictions, '__iter__') else 0
            }
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        return {
            'confidence': 0.0,
            'error': str(e)
        }

def get_trend_analysis(prices: np.ndarray) -> Dict[str, Union[str, float]]:
    """Analyze price trends and provide a summary"""
    try:
        if len(prices) < 2:
            return {'trend': 'Unknown', 'strength': 0.0}
            
        # Calculate overall trend
        start_price = prices[0]
        end_price = prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100
        
        # Calculate trend strength using linear regression
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        trend_strength = abs(r_value) * 100  # R-value as percentage
        
        # Determine trend direction
        if change_pct > 5:
            trend = "Strong Uptrend"
        elif change_pct > 1:
            trend = "Moderate Uptrend"
        elif change_pct >= -1:
            trend = "Sideways"
        elif change_pct >= -5:
            trend = "Moderate Downtrend"
        else:
            trend = "Strong Downtrend"
            
        return {
            'trend': trend,
            'change_percent': change_pct,
            'strength': trend_strength,
            'slope': slope
        }
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        return {'trend': 'Error', 'error': str(e)}

def create_performance_plot(dates, actual, predicted, title="Model Performance") -> go.Figure:
    """Create a performance comparison plot"""
    try:
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            name='Actual',
            line=dict(color='#FFD700', width=2),
            hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=dates,
            y=predicted,
            name='Predicted',
            line=dict(color='#00FF00', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            title=dict(
                text=title,
                font=dict(size=20, color='#FFD700'),
                x=0.5,
                xanchor='center'
            ),
            font=dict(color='#FFFFFF'),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='#FFD700'
            ),
            xaxis=dict(
                gridcolor='#333333',
                title='Date',
                title_font=dict(color='#FFD700'),
                tickfont=dict(color='#FFFFFF')
            ),
            yaxis=dict(
                gridcolor='#333333',
                title='Price (LKR)',
                title_font=dict(color='#FFD700'),
                tickfont=dict(color='#FFFFFF')
            ),
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance plot: {str(e)}")
        # Return an empty figure
        return go.Figure()