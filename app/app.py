import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
import logging
import time
import os
from scipy import stats
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional

# Set page config
st.set_page_config(
    page_title="OdinEx - CSE Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import models and utilities
from models import get_model_class, get_model_names, get_model_info
from utils.data_processor import DataProcessor
from utils.metrics import (
    calculate_metrics, 
    calculate_prediction_confidence, 
    get_trend_analysis,
    create_performance_plot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OdinEx')

# Add this at the beginning of your app to initialize the disclaimer state
if 'show_disclaimer' not in st.session_state:
    st.session_state.show_disclaimer = True

# Set up directories
CACHE_DIR = Path("cache/models")
DEMO_DIR = Path("demo_models")
LOG_DIR = Path("logs")

for directory in [CACHE_DIR, DEMO_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Initialize data processor
data_processor = DataProcessor()

# Initialize session state for model and related data if they don't exist
if 'model_type' not in st.session_state:
    model_names = get_model_names()
    st.session_state.model_type = model_names[0] if model_names else None
    st.session_state.model_instance = get_model_class(st.session_state.model_type)() if st.session_state.model_type else None
    st.session_state.current_params = {}  # Store current parameter values
    st.session_state.selected_company = None  # Store selected company
    st.session_state.training_history = []  # Store training history
    st.session_state.last_trained = None  # Store last training timestamp
    st.session_state.comparison_models = []  # Models for comparison
    st.session_state.active_tab = "Prediction"
    st.session_state.error_message = None
    st.session_state.data_info = {}
    st.session_state.tech_chart_type = "Line"
    st.session_state.tech_date_range = "6 Months"
    st.session_state.tech_indicator = "RSI"

def validate_data(df: pd.DataFrame) -> bool:
    """Enhanced data validation with comprehensive checks"""
    try:
        if df is None or df.empty:
            raise ValueError("Empty or invalid dataset")
            
        # Check required columns
        required_columns = ['Date', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate Date column
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                # Attempt to convert to datetime
                df['Date'] = pd.to_datetime(df['Date'])
            except:
                raise ValueError("Date column cannot be converted to datetime format")
        
        # Check for non-numeric data in Close column
        if not pd.api.types.is_numeric_dtype(df['Close']):
            raise ValueError("Close price column contains non-numeric data")
            
        # Check for negative or zero prices (invalid for stocks)
        if (df['Close'] <= 0).any():
            logger.warning(f"Dataset contains {(df['Close'] <= 0).sum()} non-positive price values")
        
        # Check for sufficient data
        if len(df) < 60:  # Minimum required for meaningful analysis
            raise ValueError(f"Insufficient data: only {len(df)} records available. At least 60 records required.")
            
        # Check for missing values
        missing_values = df[required_columns].isna().any().any()
        if missing_values:
            raise ValueError("Dataset contains missing values in required columns")
            
        # Check for time series completeness
        date_diff = df['Date'].diff().dropna()
        irregular_intervals = not (date_diff == date_diff.mode()[0]).all()
        if irregular_intervals:
            logger.warning("Dataset contains irregular time intervals (non-consecutive dates)")
            
        # Check for extreme outliers (more than 5 standard deviations)
        close_mean = df['Close'].mean()
        close_std = df['Close'].std()
        outlier_threshold = 5 * close_std
        outliers = df[abs(df['Close'] - close_mean) > outlier_threshold]
        
        if len(outliers) > 0:
            logger.warning(f"Dataset contains {len(outliers)} potential outliers in Close prices")
            
        return True
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        st.session_state.error_message = f"Data validation error: {str(e)}"
        return False

def safe_get_company_data(company: str) -> Optional[pd.DataFrame]:
    """Safely load company data with error handling"""
    try:
        if not company:
            st.session_state.error_message = "No company selected"
            return None
            
        df = data_processor.load_stock_data(company)
        
        if not validate_data(df):
            return None
            
        return df
    except Exception as e:
        logger.error(f"Error loading data for {company}: {str(e)}")
        st.session_state.error_message = f"Error loading data for {company}: {str(e)}"
        return None

# Function to align predictions with dates
def align_predictions_with_actual(predictions: np.ndarray, actual_dates: np.ndarray, actual_prices: np.ndarray) -> np.ndarray:
    """Ensure predictions align with actual data with better error handling"""
    try:
        if len(predictions) == 0:
            raise ValueError("Empty predictions array")
            
        if len(actual_dates) == 0:
            raise ValueError("Empty dates array")
            
        # Handle length mismatch
        if len(predictions) != len(actual_dates):
            logger.warning(f"Length mismatch: predictions ({len(predictions)}) vs dates ({len(actual_dates)})")
            # Truncate or pad predictions to match actual data length
            if len(predictions) > len(actual_dates):
                logger.info(f"Truncating predictions from {len(predictions)} to {len(actual_dates)}")
                predictions = predictions[:len(actual_dates)]
            else:
                logger.info(f"Padding predictions from {len(predictions)} to {len(actual_dates)}")
                pad_length = len(actual_dates) - len(predictions)
                # Pad with the last value or zeros
                pad_value = predictions[-1] if len(predictions) > 0 else 0
                predictions = np.append(predictions, np.full(pad_length, pad_value))
                
        return predictions
        
    except Exception as e:
        logger.error(f"Error aligning predictions: {str(e)}")
        # Return original predictions or empty array
        return predictions if 'predictions' in locals() else np.array([])

# Function to generate a unique key for model caching
def get_model_cache_key(model_type: str, company: str, parameters: Dict[str, Any]) -> str:
    """Generate a unique key for model caching with better serialization"""
    try:
        # Sort parameters for consistent hashing
        sorted_params = sorted(parameters.items())
        
        # Convert parameters to string, handling non-string types
        params_str = "_".join([f"{k}:{v}" for k, v in sorted_params])
        
        # Generate final key string
        key_str = f"{model_type}_{company}_{params_str}"
        
        # Hash the key string
        return hashlib.md5(key_str.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Error generating cache key: {str(e)}")
        # Fallback to a simple key based on time
        return f"fallback_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Function to train a model (without caching decorator)
def get_or_train_model(model_type: str, company: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get cached model or train a new one with better error handling and logging"""
    start_time = time.time()
    
    try:
        cache_key = get_model_cache_key(model_type, company, parameters)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    model_data = pickle.load(f)
                st.success("Loaded model from cache!")
                return model_data
            except Exception as e:
                st.warning(f"Cache load failed: {str(e)}. Training new model...")
        
        # Load and validate data
        df = data_processor.load_stock_data(company)
        validate_data(df)
        
        # Prepare data
        data = data_processor.prepare_features(df)
        X, y = data_processor.create_sequences(data, parameters['window_size'])
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Insufficient data for training after sequence creation")
        
        # Split data
        train_data, test_data = data_processor.get_train_test_split(X)
        train_labels, test_labels = data_processor.get_train_test_split(y)
        
        # Validate split data
        if len(test_data) == 0:
            raise ValueError("No test data available after splitting")
        
        # Check if model instance is passed from outside
        if 'model_instance' in st.session_state and st.session_state.model_instance:
            model_instance = st.session_state.model_instance
        else:
            # Create new model instance
            model_instance = get_model_class(model_type)()
        
        # Configure model with parameters
        for param, value in parameters.items():
            if hasattr(model_instance, 'config') and param in model_instance.config:
                model_instance.config[param] = value
                
        # Train model and get metrics
        metrics = model_instance.train((train_data, train_labels), (test_data, test_labels))
        
        try:
            # Make future predictions
            last_sequence = X[-1:] if len(X) > 0 else test_data[-1:]
            
            if last_sequence.size == 0:
                raise ValueError("No data available for prediction")
                
            # Make predictions for future days
            prediction_days = parameters.get('prediction_days', 7)
            predictions = model_instance.predict(last_sequence, prediction_days)
            
            # Inverse transform to get actual price values
            predictions = data_processor.inverse_transform(predictions)
            
            # Generate test predictions for historical comparison
            test_predictions = model_instance.predict(test_data, 1)
            test_predictions = data_processor.inverse_transform(test_predictions)
            
            # Calculate confidence intervals
            confidence_data = calculate_prediction_confidence(predictions)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            predictions = np.array([])
            test_predictions = np.array([])
            confidence_data = {'confidence': 0.0, 'error': str(e)}
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Compile model data
        model_data = {
            'model': model_instance,
            'metrics': metrics,
            'predictions': predictions,
            'test_predictions': test_predictions,
            'confidence_data': confidence_data,
            'training_date': datetime.now(),
            'parameters': parameters,
            'company': company,
            'training_time': training_time
        }
        
        # Save to cache
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(str(cache_file)), exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to cache: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving model to cache: {str(e)}")
            
        return model_data
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        st.error(f"Error during model training: {str(e)}")
        return None

# Function to render model comparison visualization
def render_model_comparison(comparison_models: List[Dict[str, Any]], company: str):
    """Render comparison of multiple models"""
    try:
        if not comparison_models:
            st.info("No models to compare. Train and add models to comparison first.")
            return
            
        # Get company data
        df = data_processor.load_stock_data(company)
        
        # Create tabs for different comparisons
        comp_tabs = st.tabs(["Prediction Comparison", "Performance Metrics", "Parameter Comparison"])
        
        with comp_tabs[0]:
            st.subheader("Prediction Comparison")
            
            # Create figure for prediction comparison
            fig = go.Figure()
            
            # Add actual historical prices (last 7 days)
            actual_dates = df['Date'].values[-7:]
            actual_prices = df['Close'].values[-7:]
            
            fig.add_trace(go.Scatter(
                x=actual_dates,
                y=actual_prices,
                name='Historical',
                line=dict(color='#FFD700', width=2),
                hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
            ))
            
            # Add future dates for predictions
            last_date = df['Date'].iloc[-1]
            
            # Find maximum prediction days across all models
            max_pred_days = max([
                len(model_data.get('predictions', [])) 
                for model_data in comparison_models
            ])
            
            future_dates = [last_date + timedelta(days=i+1) for i in range(max_pred_days)]
            
            # Add each model's predictions
            colors = ['#00FF00', '#00FFFF', '#FF00FF', '#FF4500', '#32CD32']
            
            for i, model_data in enumerate(comparison_models):
                model_type = model_data.get('parameters', {}).get('model_type', f"Model {i+1}")
                predictions = model_data.get('predictions', [])
                
                if len(predictions) > 0:
                    color_idx = i % len(colors)
                    model_future_dates = future_dates[:len(predictions)]
                    
                    fig.add_trace(go.Scatter(
                        x=model_future_dates,
                        y=predictions,
                        name=f"{model_type}",
                        line=dict(color=colors[color_idx], width=2),
                        mode='lines+markers',
                        hovertemplate='Date: %{x}<br>Predicted: LKR %{y:.2f}<extra></extra>'
                    ))
            
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#1a1a1a',
                paper_bgcolor='#1a1a1a',
                title=dict(
                    text=f'Model Prediction Comparison - {company}',
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
            
            st.plotly_chart(fig, use_container_width=True)
            
        with comp_tabs[1]:
            st.subheader("Performance Metrics Comparison")
            
            # Create metrics dataframe
            metrics_data = []
            
            for i, model_data in enumerate(comparison_models):
                model_type = model_data.get('parameters', {}).get('model_type', f"Model {i+1}")
                metrics = model_data.get('metrics', {})
                
                metrics_row = {
                    'Model': model_type,
                    'Accuracy (%)': metrics.get('accuracy', 0),
                    'Direction Accuracy (%)': metrics.get('direction_accuracy', 0),
                    'RMSE': metrics.get('rmse', 0),
                    'MAE': metrics.get('mae', 0),
                    'R²': metrics.get('r_squared', 0),
                    'Training Time (s)': model_data.get('training_time', 0)
                }
                
                metrics_data.append(metrics_row)
                
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Create metrics chart
                st.subheader("Accuracy Comparison")
                
                metrics_fig = go.Figure()
                models = metrics_df['Model'].tolist()
                
                metrics_fig.add_trace(go.Bar(
                    x=models,
                    y=metrics_df['Accuracy (%)'].tolist(),
                    name='Accuracy (%)',
                    marker_color='#FFD700'
                ))
                
                metrics_fig.add_trace(go.Bar(
                    x=models,
                    y=metrics_df['Direction Accuracy (%)'].tolist(),
                    name='Direction Accuracy (%)',
                    marker_color='#00FF00'
                ))
                
                metrics_fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    font=dict(color='#FFFFFF'),
                    legend=dict(
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='#FFD700'
                    ),
                    xaxis=dict(
                        gridcolor='#333333',
                        title='Model',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF')
                    ),
                    yaxis=dict(
                        gridcolor='#333333',
                        title='Accuracy (%)',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF')
                    ),
                    barmode='group'
                )
                
                st.plotly_chart(metrics_fig, use_container_width=True)
            else:
                st.info("No metrics data available for comparison")
                
        with comp_tabs[2]:
            st.subheader("Parameter Comparison")
            
            # Create parameters dataframe
            params_data = []
            
            for i, model_data in enumerate(comparison_models):
                model_type = model_data.get('parameters', {}).get('model_type', f"Model {i+1}")
                params = model_data.get('parameters', {})
                
                params_row = {
                    'Model': model_type,
                    'Window Size': params.get('window_size', 'N/A'),
                    'Hidden Dim': params.get('hidden_dim', 'N/A'),
                    'Num Layers': params.get('num_layers', 'N/A'),
                    'Dropout': params.get('dropout', 'N/A'),
                    'Learning Rate': params.get('learning_rate', 'N/A')
                }
                
                params_data.append(params_row)
                
            if params_data:
                params_df = pd.DataFrame(params_data)
                st.dataframe(params_df, use_container_width=True)
            else:
                st.info("No parameter data available for comparison")
                
    except Exception as e:
        logger.error(f"Error rendering model comparison: {str(e)}")
        st.error(f"Error comparing models: {str(e)}")

# Custom CSS with enhanced styling and dark mode optimizations
st.markdown("""
    <style>
        /* Main page background */
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        /* Headers */
        .main-header {
            color: #FFD700;
            font-family: 'Arial Black', sans-serif;
            font-size: 2.5em;
            margin-bottom: 0.5em;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        /* Sub-headers */
        .sub-header {
            color: #FFD700;
            font-family: 'Arial', sans-serif;
            font-size: 1.5em;
            margin-bottom: 0.5em;
            opacity: 0.9;
        }
        
        /* Card containers */
        .info-card {
            background-color: #232323;
            padding: 1em;
            border-radius: 10px;
            border: 1px solid #3a3a3a;
            margin-bottom: 1em;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            background-color: #232323;
            padding: 1.2em;
            border-radius: 10px;
            border: 1px solid #FFD700;
            margin-bottom: 1em;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            box-shadow: 0 0 15px rgba(255,215,0,0.3);
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #232323;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #1a1a1a;
            color: #FFD700;
            border: 2px solid #FFD700;
            border-radius: 6px;
            padding: 0.5em 1em;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #FFD700;
            color: #1a1a1a;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Metrics */
        .stMetric {
            background-color: #232323;
            padding: 1em;
            border-radius: 8px;
            border: 1px solid #FFD700;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Inputs */
        .stSlider>div {
            color: #FFD700;
        }
        
        .stSelectbox>div>div {
            background-color: #2d2d2d;
            color: white;
            border-radius: 6px;
            border: 1px solid #3a3a3a;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2d2d2d;
            border-radius: 8px;
            padding: 0.5em;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #FFD700;
            border-radius: 6px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3d3d3d;
            font-weight: bold;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            color: #FFD700 !important;
            font-weight: 600;
        }
        
        /* Tooltips */
        .stTooltipIcon {
            color: #FFD700 !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">OdinEx 📈</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #FFD700; margin-bottom: 2em;">CSE Stock Prediction Platform</p>', unsafe_allow_html=True)

# Handle error messages
if st.session_state.error_message:
    st.error(st.session_state.error_message)
    if st.button("Clear Error"):
        st.session_state.error_message = None
    st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown('<p style="color: #FFD700; font-size: 1.2em;">Model Configuration</p>', unsafe_allow_html=True)
    
    # Model Selection
    available_models = get_model_names()
    model_type = st.selectbox(
        "Select Model",
        options=available_models,
        index=available_models.index(st.session_state.model_type) if st.session_state.model_type in available_models else 0,
        help="Choose the prediction model to use"
    )
    
    # Update model type in session state
    if model_type != st.session_state.model_type:
        st.session_state.model_type = model_type
        st.session_state.model_instance = get_model_class(model_type)()
    
    # Show model information
    with st.expander("Model Information", expanded=False):
        model_info = get_model_info(model_type)
        if model_info:
            st.markdown(f"**Description:** {model_info.get('description', 'No description')}")
            st.markdown(f"**Complexity:** {model_info.get('complexity', 'Unknown')}")
            
            st.markdown("**Recommended for:**")
            for use_case in model_info.get('recommended_for', ['General predictions']):
                st.markdown(f"- {use_case}")
    
    # Company Selection with validation
    companies = data_processor.get_available_companies()
    if not companies:
        st.error("No company data files found! Please check data/processed_data directory.")
    else:
        company = st.selectbox(
            "Select Company",
            companies,
            index=companies.index(st.session_state.selected_company) if st.session_state.selected_company in companies else 0,
            help="Select the company to analyze"
        )
        
        # Update company in session state
        if company != st.session_state.selected_company:
            st.session_state.selected_company = company
            
            # Get dataset info for the company
            try:
                st.session_state.data_info = data_processor.get_dataset_statistics(company)
            except Exception as e:
                logger.error(f"Error getting dataset info: {str(e)}")
                st.session_state.error_message = f"Error getting dataset info: {str(e)}"
        
        # Show dataset information
        with st.expander("Dataset Information", expanded=False):
            data_info = st.session_state.data_info
            if data_info:
                st.markdown(f"**Total Records:** {data_info.get('total_records', 'Unknown')}")
                st.markdown(f"**Date Range:** {data_info.get('date_range', {}).get('start', 'Unknown')} to {data_info.get('date_range', {}).get('end', 'Unknown')}")
                
                price_stats = data_info.get('price_stats', {})
                st.markdown(f"**Min Price:** LKR {price_stats.get('min', 'Unknown'):.2f}")
                st.markdown(f"**Max Price:** LKR {price_stats.get('max', 'Unknown'):.2f}")
                st.markdown(f"**Mean Price:** LKR {price_stats.get('mean', 'Unknown'):.2f}")
                st.markdown(f"**Std Deviation:** LKR {price_stats.get('std', 'Unknown'):.2f}")
            else:
                st.info("No dataset information available")
        
        # Verify data file is readable
        try:
            df = data_processor.load_stock_data(company)
            validate_data(df)
        except Exception as e:
            st.error(f"Error loading company data: {str(e)}")
            company = None

    # Dynamic parameter inputs
    st.markdown("### Model Parameters")
    current_params = {}

    # Basic Parameters
    st.markdown("#### Basic Parameters")
    col1, col2 = st.columns([3, 1])
    with col1:
        current_params['window_size'] = st.slider(
            "Number of Past Days to Consider",
            min_value=5,
            max_value=365,
            value=st.session_state.current_params.get('window_size', 30),
            step=1,
            help="How many days of historical data to use for prediction. More data can help identify longer-term patterns."
        )
    with col2:
        current_params['window_size'] = st.number_input(
            "Window Size",
            min_value=5,
            max_value=365,
            value=current_params['window_size'],
            step=1,
            label_visibility="collapsed"
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        current_params['prediction_days'] = st.slider(
            "Prediction Window (Days)",
            min_value=1,
            max_value=30,
            value=st.session_state.current_params.get('prediction_days', 7),
            step=1,
            help="How many days into the future to predict. Longer predictions tend to be less accurate."
        )
    with col2:
        current_params['prediction_days'] = st.number_input(
            "Prediction Days",
            min_value=1,
            max_value=30,
            value=current_params['prediction_days'],
            step=1,
            label_visibility="collapsed"
        )

    # Advanced Parameters Toggle
    show_advanced = st.checkbox("Show Advanced Parameters", 
                            value=st.session_state.get('show_advanced', False))
    st.session_state['show_advanced'] = show_advanced

    if show_advanced:
        st.markdown("#### Advanced Parameters")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            current_params['hidden_dim'] = st.slider(
                "Hidden Layer Size",
                min_value=32,
                max_value=256,
                value=st.session_state.current_params.get('hidden_dim', 64),
                step=8,  # Step by 8 for better granularity
                help="Technical: Number of neurons in hidden layers.\n\n" +
                    "Simple: Controls how much information the model can process at once. " +
                    "Larger values might improve accuracy but make training slower."
            )
        with col2:
            current_params['hidden_dim'] = st.number_input(
                "Hidden Dim",
                min_value=32,
                max_value=256,
                value=current_params['hidden_dim'],
                step=1,
                label_visibility="collapsed"
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            current_params['num_layers'] = st.slider(
                "Number of Model Layers",
                min_value=1,
                max_value=4,
                value=st.session_state.current_params.get('num_layers', 2),
                step=1,
                help="Technical: Number of stacked layers.\n\n" +
                    "Simple: Controls how complex patterns the model can learn. " +
                    "More layers can capture more complex patterns but might overfit."
            )
        with col2:
            current_params['num_layers'] = st.number_input(
                "Num Layers",
                min_value=1,
                max_value=4,
                value=current_params['num_layers'],
                step=1,
                label_visibility="collapsed"
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            current_params['dropout'] = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=st.session_state.current_params.get('dropout', 0.2),
                step=0.01,  # Small step for fine control
                format="%.2f",
                help="Technical: Probability of neurons being disabled during training.\n\n" +
                    "Simple: Controls how aggressively the model prevents overfitting. " +
                    "Higher values make the model more conservative but might reduce accuracy."
            )
        with col2:
            current_params['dropout'] = st.number_input(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=current_params['dropout'],
                step=0.01,
                format="%.2f",
                label_visibility="collapsed"
            )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            current_params['learning_rate'] = st.slider(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=st.session_state.current_params.get('learning_rate', 0.001),
                step=0.0001,  # Much smaller step for precise control
                format="%.5f",
                help="Technical: Step size during optimization.\n\n" +
                    "Simple: Controls how quickly the model learns. " +
                    "Too high might make training unstable, too low might make it too slow."
            )
        with col2:
            current_params['learning_rate'] = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=current_params['learning_rate'],
                step=0.0001,
                format="%.5f",
                label_visibility="collapsed"
            )
    else:
        # Set default values for advanced parameters when hidden
        model_instance = st.session_state.model_instance
        if hasattr(model_instance, 'get_configurable_parameters'):
            configurable_params = model_instance.get_configurable_parameters()
            for param_name, param_info in configurable_params.items():
                if param_name not in current_params:
                    current_params[param_name] = param_info.get('default', 0)
    
    # Update parameters in session state
    st.session_state.current_params = current_params
    
    # Cache Management
    with st.expander("Cache Management", expanded=False):
        st.markdown("### Cached Models")
        cached_models = list(CACHE_DIR.glob("*.pkl"))
        st.write(f"Currently cached: {len(cached_models)} models")
        
        if st.button("Clear All Cached Models"):
            for file in CACHE_DIR.glob("*.pkl"):
                file.unlink()
            st.cache_resource.clear()
            st.success("Cache cleared!")
    
    # Create a placeholder for training progress
    if 'train_progress_placeholder' not in st.session_state:
        st.session_state.train_progress_placeholder = st.empty()
    
    # Create a placeholder for current epoch display
    if 'epoch_placeholder' not in st.session_state:
        st.session_state.epoch_placeholder = st.empty()
    
    # Training Button
    if st.button("Train Model"):
        if not companies:
            st.error("Please add company data files before training!")
        elif not company:
            st.error("Please select a valid company!")
        else:
            # Simplified training progress with expandable section
            st.markdown("### Training Progress")
            
            # Initialize progress bar
            progress_bar = st.session_state.train_progress_placeholder.progress(0.0)
            epoch_status = st.session_state.epoch_placeholder.empty()
            
            # Define progress callback function
            def progress_callback(progress, message, current_model=None, total_models=None, metrics=None):
                # Update progress bar
                progress_bar.progress(progress)
                
                # Update status
                model_info = f"Model {current_model}/{total_models} | " if current_model and total_models else ""
                status_msg = f"{model_info}{message}"
                epoch_status.info(status_msg)
            
            with st.expander("Show Training Details", expanded=True):
                # Get or create model instance and set progress callback
                model_instance = get_model_class(model_type)()
                if hasattr(model_instance, 'set_progress_callback'):
                    model_instance.set_progress_callback(progress_callback)
                
                # Store the model instance in session state for get_or_train_model to use
                st.session_state.model_instance = model_instance
                
                # Start training
                model_data = get_or_train_model(model_type, company, current_params)
                if model_data:
                    st.session_state['current_model'] = model_data
                    st.session_state.last_trained = datetime.now()
                    
                    # Add to training history
                    history_entry = {
                        'timestamp': datetime.now(),
                        'model_type': model_type,
                        'company': company,
                        'metrics': model_data.get('metrics', {})
                    }
                    st.session_state.training_history.append(history_entry)
                    
                    # Show success message and update progress to 100%
                    progress_bar.progress(1.0)
                    epoch_status.success("Training completed successfully!")
                    
                    # Wait briefly to show completion before rerunning
                    time.sleep(1)
                    st.rerun()

    # Training history expander
    with st.expander("Training History", expanded=False):
        if st.session_state.training_history:
            # Show history in reverse chronological order
            for i, entry in enumerate(reversed(st.session_state.training_history)):
                st.markdown(f"**Training #{len(st.session_state.training_history) - i}**")
                st.markdown(f"- Time: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"- Model: {entry['model_type']}")
                st.markdown(f"- Company: {entry['company']}")
                st.markdown(f"- Accuracy: {entry.get('metrics', {}).get('accuracy', 0):.2f}%")
                st.markdown("---")
        else:
            st.info("No training history available")

# Main Content Tabs
tabs = st.tabs(["Prediction", "Model Comparison", "Technical Analysis", "Data Explorer"])

# Prediction Tab
with tabs[0]:
    st.session_state.active_tab = "Prediction"
    
    if 'current_model' in st.session_state:
        try:
            model_data = st.session_state['current_model']
            
            # Check if model_data contains all required fields
            required_fields = ['metrics', 'predictions', 'test_predictions', 'parameters']
            if not all(field in model_data for field in required_fields):
                raise ValueError("Model data is incomplete")
                
            # Get company from session state since it's not in model_data
            company = st.session_state.selected_company
            if not company:
                raise ValueError("No company selected")
                
            # Add company to model_data if not present
            if 'company' not in model_data:
                model_data['company'] = company
            
            # Continue with visualization code
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Historical Performance")
                try:
                    # Get and validate historical data
                    df = data_processor.load_stock_data(company)
                    validate_data(df)
                    
                    # Get the last 30 days of data
                    actual_dates = df['Date'].values[-30:]
                    actual_prices = df['Close'].values[-30:]
                    
                    # Align predictions with actual data
                    predicted_prices = align_predictions_with_actual(
                        model_data['test_predictions'],
                        actual_dates,
                        actual_prices
                    )

                    # Create historical performance plot
                    fig1 = go.Figure()
                    
                    # Add actual values
                    fig1.add_trace(go.Scatter(
                        x=actual_dates,
                        y=actual_prices,
                        name='Actual',
                        line=dict(color='#FFD700', width=2),
                        hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add predicted values
                    fig1.add_trace(go.Scatter(
                        x=actual_dates,
                        y=predicted_prices,
                        name='Predicted',
                        line=dict(color='#00FF00', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Update layout
                    fig1.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'Historical Price Comparison - {company}',
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
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Add historical performance metrics
                    if len(actual_prices) > 0 and len(predicted_prices) > 0:
                        st.markdown("### Performance Metrics")
                        metrics = calculate_metrics(actual_prices, predicted_prices)
                        
                        metrics_cols = st.columns(4)
                        with metrics_cols[0]:
                            st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
                        with metrics_cols[1]:
                            st.metric("Direction Accuracy", f"{metrics.get('direction_accuracy', 0):.2f}%")
                        with metrics_cols[2]:
                            st.metric("RMSE", f"LKR {metrics['rmse']:.2f}")
                        with metrics_cols[3]:
                            st.metric("R²", f"{metrics.get('r_squared', 0):.3f}")
                    
                except Exception as e:
                    logger.error(f"Error displaying historical performance: {str(e)}")
                    st.error(f"Error displaying historical performance: {str(e)}")
            
            with col2:
                st.markdown("### Future Predictions")
                
                try:
                    # Get actual data if not already available
                    if 'actual_dates' not in locals() or 'actual_prices' not in locals():
                        df = data_processor.load_stock_data(company)
                        actual_dates = df['Date'].values[-30:]
                        actual_prices = df['Close'].values[-30:]
                
                    # Get future predictions data
                    last_date = df['Date'].iloc[-1]
                    prediction_days = model_data['parameters'].get('prediction_days', 7)
                    future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                    future_predictions = model_data['predictions']
                    
                    if len(future_predictions) != len(future_dates):
                        future_predictions = future_predictions[:len(future_dates)]
                    
                    # Create predictions plot
                    fig2 = go.Figure()
                    
                    # Add historical context (last 7 days)
                    fig2.add_trace(go.Scatter(
                        x=actual_dates[-7:],
                        y=actual_prices[-7:],
                        name='Historical',
                        line=dict(color='#FFD700', width=2),
                        hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add predictions
                    fig2.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_predictions,
                        name='Prediction',
                        line=dict(color='#00FF00', width=2),
                        mode='lines+markers',
                        hovertemplate='Date: %{x}<br>Predicted: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add confidence interval if available
                    if 'confidence_data' in model_data:
                        confidence_data = model_data['confidence_data']
                        
                        if 'lower_bound' in confidence_data and 'upper_bound' in confidence_data:
                            lower_bound = confidence_data['lower_bound']
                            upper_bound = confidence_data['upper_bound']
                            
                            if len(lower_bound) == len(future_dates) and len(upper_bound) == len(future_dates):
                                fig2.add_trace(go.Scatter(
                                    x=future_dates + future_dates[::-1],
                                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(0,255,0,0.1)',
                                    line=dict(color='rgba(255,255,255,0)'),
                                    name='Confidence Interval',
                                    hoverinfo='skip'
                                ))
                    
                    # Update layout
                    fig2.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'Price Predictions - {company}',
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
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Add prediction details with error handling
                    try:
                        if len(future_predictions) > 0 and len(actual_prices) > 0:
                            price_change = ((future_predictions[-1] - actual_prices[-1]) / actual_prices[-1]) * 100
                            
                            # Get trend analysis
                            trend_data = get_trend_analysis(future_predictions)
                            
                            st.markdown(f"""
                                <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #FFD700;'>
                                    <h4 style='color: #FFD700;'>Prediction Details</h4>
                                    <p><b>Last Known Price:</b> LKR {actual_prices[-1]:.2f}</p>
                                    <p><b>Predicted End Price:</b> LKR {future_predictions[-1]:.2f}</p>
                                    <p><b>Predicted Change:</b> {price_change:.2f}%</p>
                                    <p><b>Predicted Trend:</b> {trend_data['trend']} ({trend_data['strength']:.1f}% confidence)</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Add confidence information
                            if 'confidence_data' in model_data:
                                confidence = model_data['confidence_data'].get('confidence', 0)
                                st.progress(confidence / 100)
                                st.caption(f"Prediction Confidence: {confidence:.1f}%")
                    
                    except Exception as e:
                        logger.error(f"Error calculating prediction details: {str(e)}")
                        st.error(f"Error calculating prediction details: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"Error displaying future predictions: {str(e)}")
                    st.error(f"Error displaying future predictions: {str(e)}")
            
            # Model Information Section
            st.markdown("### Model Information")
            
            try:
                info_cols = st.columns(2)
                
                with info_cols[0]:
                    st.markdown("""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>Training Parameters</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    params_table = pd.DataFrame([{
                        'Parameter': key,
                        'Value': value
                    } for key, value in model_data['parameters'].items()])
                    
                    st.dataframe(params_table, use_container_width=True, hide_index=True)
                    
                    # Training timestamp
                    training_date = model_data.get('training_date', datetime.now())
                    st.caption(f"Model trained on: {training_date.strftime('%Y-%m-%d %H:%M:%S')}")
                
                with info_cols[1]:
                    st.markdown("""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>Model Controls</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    control_cols = st.columns(2)
                    
                    with control_cols[0]:
                        if st.button("Add to Comparison"):
                            if model_data not in st.session_state.comparison_models:
                                st.session_state.comparison_models.append(model_data)
                                st.success("Model added to comparison!")
                            else:
                                st.warning("This model is already in comparison!")
                    
                    with control_cols[1]:
                        # Save as demo model
                        demo_file = DEMO_DIR / f"demo_{company}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                        if st.button("Save as Demo"):
                            try:
                                with open(demo_file, 'wb') as f:
                                    pickle.dump(model_data, f)
                                st.success(f"Saved demo model: {demo_file.name}")
                            except Exception as e:
                                logger.error(f"Error saving demo model: {str(e)}")
                                st.error(f"Error saving demo model: {str(e)}")
                                
            except Exception as e:
                logger.error(f"Error displaying model information: {str(e)}")
                st.error(f"Error displaying model information: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error displaying results: {str(e)}")
            st.error(f"Error displaying results: {str(e)}")
    else:
        st.info("No trained model available. Please train a model first.")

# Model Comparison Tab
with tabs[1]:
    st.session_state.active_tab = "Model Comparison"
    
    st.markdown("## Model Comparison")
    
    # Show comparison controls
    control_cols = st.columns(3)
    
    with control_cols[0]:
        st.markdown(f"**Models in comparison:** {len(st.session_state.comparison_models)}")
        
    with control_cols[1]:
        if st.button("Clear Comparison"):
            st.session_state.comparison_models = []
            st.success("Comparison cleared!")
            
    with control_cols[2]:
        if len(st.session_state.comparison_models) > 0:
            if st.button("Remove Last Model"):
                st.session_state.comparison_models.pop()
                st.success("Last model removed!")
    
    # Render comparison if there are models to compare
    if len(st.session_state.comparison_models) > 0:
        company = st.session_state.selected_company
        if company:
            render_model_comparison(st.session_state.comparison_models, company)
        else:
            st.warning("Please select a company to compare models")
    else:
        st.info("Add models to comparison from the Prediction tab")

# Technical Analysis Tab
with tabs[2]:
    st.session_state.active_tab = "Technical Analysis"
    
    st.markdown("## Technical Analysis")
    
    # Get company from session state directly
    company = st.session_state.selected_company
    
    if not company:
        st.warning("Please select a company first")
    else:
        try:
            # Load data
            df = data_processor.load_stock_data(company)
            
            # Create sub-tabs
            tech_tabs = st.tabs(["Price Chart", "Moving Averages", "Indicators", "Volume Analysis"])
            
            with tech_tabs[0]:
                st.markdown("### Price Chart")
                
                # Use session state for selections
                date_cols = st.columns([1, 1, 1])
                with date_cols[0]:
                    chart_type = st.selectbox(
                        "Chart Type",
                        ["Line", "Candlestick", "OHLC"],
                        index=["Line", "Candlestick", "OHLC"].index(st.session_state.tech_chart_type)
                    )
                
                with date_cols[1]:
                    date_range = st.selectbox(
                        "Date Range",
                        ["1 Month", "3 Months", "6 Months", "1 Year", "All"],
                        index=2
                    )
                    
                with date_cols[2]:
                    log_scale = st.checkbox("Log Scale", value=False)
                    
                    # Update state
                    st.session_state.tech_chart_type = chart_type
                
                # Filter data based on date range
                if date_range == "1 Month":
                    cutoff_date = df['Date'].max() - pd.Timedelta(days=30)
                elif date_range == "3 Months":
                    cutoff_date = df['Date'].max() - pd.Timedelta(days=90)
                elif date_range == "6 Months":
                    cutoff_date = df['Date'].max() - pd.Timedelta(days=180)
                elif date_range == "1 Year":
                    cutoff_date = df['Date'].max() - pd.Timedelta(days=365)
                else:  # All
                    cutoff_date = df['Date'].min()
                
                filtered_df = df[df['Date'] >= cutoff_date]
                
                # Create appropriate chart
                fig = go.Figure()
                
                if chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['Close'],
                        name='Close Price',
                        line=dict(color='#FFD700', width=2),
                        hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                    ))
                elif chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=filtered_df['Date'],
                        open=filtered_df['Open'] if 'Open' in filtered_df else filtered_df['Close'],
                        high=filtered_df['High'] if 'High' in filtered_df else filtered_df['Close'],
                        low=filtered_df['Low'] if 'Low' in filtered_df else filtered_df['Close'],
                        close=filtered_df['Close'],
                        name='OHLC',
                        increasing=dict(line=dict(color='#00FF00')),
                        decreasing=dict(line=dict(color='#FF0000'))
                    ))
                elif chart_type == "OHLC":
                    fig.add_trace(go.Ohlc(
                        x=filtered_df['Date'],
                        open=filtered_df['Open'] if 'Open' in filtered_df else filtered_df['Close'],
                        high=filtered_df['High'] if 'High' in filtered_df else filtered_df['Close'],
                        low=filtered_df['Low'] if 'Low' in filtered_df else filtered_df['Close'],
                        close=filtered_df['Close'],
                        name='OHLC',
                        increasing=dict(line=dict(color='#00FF00')),
                        decreasing=dict(line=dict(color='#FF0000'))
                    ))
                
                # Update layout
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    title=dict(
                        text=f'{company} - {date_range} Price Chart',
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
                        tickfont=dict(color='#FFFFFF'),
                        type='log' if log_scale else 'linear'
                    ),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with tech_tabs[1]:
                st.markdown("### Moving Averages")
                
                # Moving average controls
                ma_cols = st.columns(4)
                
                with ma_cols[0]:
                    show_ma5 = st.checkbox("5-Day MA", value=True)
                
                with ma_cols[1]:
                    show_ma20 = st.checkbox("20-Day MA", value=True)
                
                with ma_cols[2]:
                    show_ma50 = st.checkbox("50-Day MA", value=False)
                
                with ma_cols[3]:
                    show_ma100 = st.checkbox("100-Day MA", value=False)
                
                # Calculate moving averages
                filtered_df = filtered_df.copy()  # Create explicit copy
                filtered_df.loc[:, 'MA5'] = filtered_df['Close'].rolling(window=5).mean()
                filtered_df.loc[:, 'MA20'] = filtered_df['Close'].rolling(window=20).mean()
                filtered_df.loc[:, 'MA50'] = filtered_df['Close'].rolling(window=50).mean()
                filtered_df.loc[:, 'MA100'] = filtered_df['Close'].rolling(window=100).mean()
                
                # Create moving average chart
                fig = go.Figure()
                
                # Add price
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['Close'],
                    name='Close Price',
                    line=dict(color='#FFD700', width=2),
                    hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                ))
                
                # Add moving averages
                if show_ma5:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA5'],
                        name='5-Day MA',
                        line=dict(color='#00FFFF', width=1.5),
                        hovertemplate='Date: %{x}<br>MA5: LKR %{y:.2f}<extra></extra>'
                    ))
                
                if show_ma20:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA20'],
                        name='20-Day MA',
                        line=dict(color='#FF00FF', width=1.5),
                        hovertemplate='Date: %{x}<br>MA20: LKR %{y:.2f}<extra></extra>'
                    ))
                
                if show_ma50:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA50'],
                        name='50-Day MA',
                        line=dict(color='#00FF00', width=1.5),
                        hovertemplate='Date: %{x}<br>MA50: LKR %{y:.2f}<extra></extra>'
                    ))
                
                if show_ma100:
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA100'],
                        name='100-Day MA',
                        line=dict(color='#FF4500', width=1.5),
                        hovertemplate='Date: %{x}<br>MA100: LKR %{y:.2f}<extra></extra>'
                    ))
                
                # Update layout
                fig.update_layout(
                    template='plotly_dark',
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    title=dict(
                        text=f'{company} - Moving Averages',
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Moving average crossover analysis
                if show_ma5 and show_ma20:
                    st.markdown("### Moving Average Crossover Analysis")
                    
                    # Detect crossovers
                    filtered_df = filtered_df.copy()  # Create explicit copy
                    filtered_df.loc[:, 'MA5_gt_MA20'] = filtered_df['MA5'] > filtered_df['MA20']
                    filtered_df.loc[:, 'Crossover'] = filtered_df['MA5_gt_MA20'].ne(filtered_df['MA5_gt_MA20'].shift())

                    # Get last crossover
                    crossovers = filtered_df[filtered_df['Crossover']]
                    
                    if len(crossovers) > 0:
                        last_crossover = crossovers.iloc[-1]
                        crossover_date = last_crossover['Date']
                        is_golden = last_crossover['MA5_gt_MA20']
                        
                        crossover_type = "Golden Crossover (Bullish)" if is_golden else "Death Crossover (Bearish)"
                        
                        st.markdown(f"""
                            <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                                <h4 style='color: #FFD700;'>Last Crossover Signal</h4>
                                <p><b>Type:</b> {crossover_type}</p>
                                <p><b>Date:</b> {crossover_date.strftime('%Y-%m-%d')}</p>
                                <p><b>Signal:</b> {
                                    "Potential uptrend, consider buying" if is_golden else 
                                    "Potential downtrend, consider selling"
                                }</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No crossovers detected in the selected time period")
                
            with tech_tabs[2]:
                st.markdown("### Technical Indicators")
                
                # Indicator selection
                indicator = st.selectbox(
                    "Select Indicator",
                    ["RSI", "ROC", "MACD", "Bollinger Bands"],
                    index=0
                )
                
                # RSI
                if indicator == "RSI":
                    st.markdown("### Relative Strength Index (RSI)")
                    
                    # Calculate RSI if not already available
                    if 'RSI' not in filtered_df.columns:
                        delta = filtered_df['Close'].diff()
                        gain = delta.mask(delta < 0, 0).fillna(0)
                        loss = (-delta).mask(delta > 0, 0).fillna(0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
                        filtered_df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # Create RSI chart
                    fig = go.Figure()
                    
                    # Add RSI line
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['RSI'],
                        name='RSI',
                        line=dict(color='#00FFFF', width=2),
                        hovertemplate='Date: %{x}<br>RSI: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add overbought and oversold lines
                    fig.add_shape(
                        type="line",
                        x0=filtered_df['Date'].min(),
                        y0=70,
                        x1=filtered_df['Date'].max(),
                        y1=70,
                        line=dict(color="red", width=1, dash="dash"),
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=filtered_df['Date'].min(),
                        y0=30,
                        x1=filtered_df['Date'].max(),
                        y1=30,
                        line=dict(color="green", width=1, dash="dash"),
                    )
                    
                    # Add annotations
                    fig.add_annotation(
                        x=filtered_df['Date'].min(),
                        y=70,
                        text="Overbought (70)",
                        showarrow=False,
                        yshift=10,
                        font=dict(color="red")
                    )
                    
                    fig.add_annotation(
                        x=filtered_df['Date'].min(),
                        y=30,
                        text="Oversold (30)",
                        showarrow=False,
                        yshift=-10,
                        font=dict(color="green")
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - Relative Strength Index (RSI)',
                            font=dict(size=20, color='#FFD700'),
                            x=0.5,
                            xanchor='center'
                        ),
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            gridcolor='#333333',
                            title='Date',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF')
                        ),
                        yaxis=dict(
                            gridcolor='#333333',
                            title='RSI',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF'),
                            range=[0, 100]
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # RSI analysis
                    last_rsi = filtered_df['RSI'].iloc[-1]
                    rsi_message = ""
                    
                    if last_rsi > 70:
                        rsi_message = "Overbought - The stock may be overvalued and could soon correct downward"
                    elif last_rsi < 30:
                        rsi_message = "Oversold - The stock may be undervalued and could soon correct upward"
                    elif last_rsi > 50:
                        rsi_message = "Bullish - RSI above 50 indicates positive momentum"
                    else:
                        rsi_message = "Bearish - RSI below 50 indicates negative momentum"
                    
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>RSI Analysis</h4>
                            <p><b>Current RSI:</b> {last_rsi:.2f}</p>
                            <p><b>Signal:</b> {rsi_message}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # ROC
                elif indicator == "ROC":
                    st.markdown("### Rate of Change (ROC)")
                    
                    # Calculate ROC if not already available
                    if 'ROC' not in filtered_df.columns:
                        filtered_df['ROC'] = filtered_df['Close'].pct_change(periods=5) * 100
                    
                    # Create ROC chart
                    fig = go.Figure()
                    
                    # Add ROC line
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['ROC'],
                        name='ROC (5-day)',
                        line=dict(color='#00FF00', width=2),
                        hovertemplate='Date: %{x}<br>ROC: %{y:.2f}%<extra></extra>'
                    ))
                    
                    # Add zero line
                    fig.add_shape(
                        type="line",
                        x0=filtered_df['Date'].min(),
                        y0=0,
                        x1=filtered_df['Date'].max(),
                        y1=0,
                        line=dict(color="white", width=1, dash="dash"),
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - Rate of Change (ROC)',
                            font=dict(size=20, color='#FFD700'),
                            x=0.5,
                            xanchor='center'
                        ),
                        font=dict(color='#FFFFFF'),
                        xaxis=dict(
                            gridcolor='#333333',
                            title='Date',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF')
                        ),
                        yaxis=dict(
                            gridcolor='#333333',
                            title='ROC (%)',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF')
                        ),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROC analysis
                    last_roc = filtered_df['ROC'].iloc[-1]
                    roc_message = ""
                    
                    if last_roc > 5:
                        roc_message = "Strong upward momentum - Potentially bullish signal"
                    elif last_roc > 0:
                        roc_message = "Positive momentum - Weak bullish signal"
                    elif last_roc > -5:
                        roc_message = "Negative momentum - Weak bearish signal"
                    else:
                        roc_message = "Strong downward momentum - Potentially bearish signal"
                    
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>ROC Analysis</h4>
                            <p><b>Current ROC:</b> {last_roc:.2f}%</p>
                            <p><b>Signal:</b> {roc_message}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # MACD
                elif indicator == "MACD":
                    st.markdown("### Moving Average Convergence Divergence (MACD)")
                    
                    # Calculate MACD if not already available
                    if 'MACD' not in filtered_df.columns:
                        filtered_df['EMA12'] = filtered_df['Close'].ewm(span=12, adjust=False).mean()
                        filtered_df['EMA26'] = filtered_df['Close'].ewm(span=26, adjust=False).mean()
                        filtered_df['MACD'] = filtered_df['EMA12'] - filtered_df['EMA26']
                        filtered_df['Signal'] = filtered_df['MACD'].ewm(span=9, adjust=False).mean()
                        filtered_df['Histogram'] = filtered_df['MACD'] - filtered_df['Signal']
                    
                    # Create MACD chart
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f"{company} - Price", "MACD, Signal & Histogram"),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Add price line to top subplot
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['Close'],
                            name='Close Price',
                            line=dict(color='#FFD700', width=2),
                            hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Add MACD line
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['MACD'],
                            name='MACD',
                            line=dict(color='#00FFFF', width=2),
                            hovertemplate='Date: %{x}<br>MACD: %{y:.4f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Add Signal line
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['Signal'],
                            name='Signal',
                            line=dict(color='#FF00FF', width=2),
                            hovertemplate='Date: %{x}<br>Signal: %{y:.4f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Add Histogram
                    colors = ['green' if val >= 0 else 'red' for val in filtered_df['Histogram']]
                    
                    fig.add_trace(
                        go.Bar(
                            x=filtered_df['Date'],
                            y=filtered_df['Histogram'],
                            name='Histogram',
                            marker_color=colors,
                            hovertemplate='Date: %{x}<br>Histogram: %{y:.4f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - MACD Analysis',
                            font=dict(size=20, color='#FFD700'),
                            x=0.5,
                            xanchor='center'
                        ),
                        font=dict(color='#FFFFFF'),
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='#FFD700'
                        ),
                        hovermode='x unified',
                        height=700
                    )
                    
                    # Update axes
                    fig.update_xaxes(
                        gridcolor='#333333',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=2, col=1
                    )
                    
                    fig.update_yaxes(
                        gridcolor='#333333',
                        title='Price (LKR)',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=1, col=1
                    )
                    
                    fig.update_yaxes(
                        gridcolor='#333333',
                        title='MACD',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=2, col=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # MACD analysis
                    last_macd = filtered_df['MACD'].iloc[-1]
                    last_signal = filtered_df['Signal'].iloc[-1]
                    last_histogram = filtered_df['Histogram'].iloc[-1]
                    
                    # Detect crossovers for recent signals
                    filtered_df['MACD_gt_Signal'] = filtered_df['MACD'] > filtered_df['Signal']
                    filtered_df['MACD_Crossover'] = filtered_df['MACD_gt_Signal'].ne(filtered_df['MACD_gt_Signal'].shift())
                    
                    recent_crossovers = filtered_df[filtered_df['MACD_Crossover']].tail(3)
                    
                    macd_message = ""
                    if last_macd > last_signal:
                        macd_message = "Bullish - MACD above Signal line suggests upward momentum"
                    else:
                        macd_message = "Bearish - MACD below Signal line suggests downward momentum"
                        
                    histogram_message = ""
                    if last_histogram > 0 and last_histogram > filtered_df['Histogram'].iloc[-2]:
                        histogram_message = "Increasing positive histogram - Strong bullish momentum"
                    elif last_histogram > 0:
                        histogram_message = "Positive histogram - Bullish momentum"
                    elif last_histogram < 0 and last_histogram < filtered_df['Histogram'].iloc[-2]:
                        histogram_message = "Decreasing negative histogram - Strong bearish momentum"
                    else:
                        histogram_message = "Negative histogram - Bearish momentum"
                    
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>MACD Analysis</h4>
                            <p><b>Current MACD:</b> {last_macd:.4f}</p>
                            <p><b>Current Signal:</b> {last_signal:.4f}</p>
                            <p><b>Current Histogram:</b> {last_histogram:.4f}</p>
                            <p><b>MACD Signal:</b> {macd_message}</p>
                            <p><b>Histogram Signal:</b> {histogram_message}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Recent crossovers
                    if len(recent_crossovers) > 0:
                        st.markdown("### Recent MACD Crossover Signals")
                        
                        for i, crossover in recent_crossovers.iterrows():
                            is_bullish = crossover['MACD_gt_Signal']
                            signal_type = "Bullish (Buy)" if is_bullish else "Bearish (Sell)"
                            signal_date = crossover['Date']
                            
                            st.markdown(f"""
                                <div style='background-color: #232323; padding: 0.5em; border-radius: 5px; border: 1px solid #3a3a3a; margin-bottom: 0.5em;'>
                                    <p><b>Date:</b> {signal_date.strftime('%Y-%m-%d')}</p>
                                    <p><b>Signal:</b> {signal_type}</p>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Bollinger Bands
                elif indicator == "Bollinger Bands":
                    st.markdown("### Bollinger Bands")
                    
                    # Bollinger Bands parameters
                    bb_period = st.slider("Period", min_value=5, max_value=50, value=20)
                    bb_std_dev = st.slider("Standard Deviations", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
                    
                    # Calculate Bollinger Bands
                    filtered_df['MA'] = filtered_df['Close'].rolling(window=bb_period).mean()
                    filtered_df['BB_upper'] = filtered_df['MA'] + bb_std_dev * filtered_df['Close'].rolling(window=bb_period).std()
                    filtered_df['BB_lower'] = filtered_df['MA'] - bb_std_dev * filtered_df['Close'].rolling(window=bb_period).std()
                    
                    # Create Bollinger Bands chart
                    fig = go.Figure()
                    
                    # Add price line
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['Close'],
                        name='Close Price',
                        line=dict(color='#FFD700', width=2),
                        hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add middle band (MA)
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['MA'],
                        name=f'Middle Band (MA{bb_period})',
                        line=dict(color='#FFFFFF', width=1),
                        hovertemplate='Date: %{x}<br>MA: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add upper band
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['BB_upper'],
                        name='Upper Band',
                        line=dict(color='#00FF00', width=1, dash='dash'),
                        hovertemplate='Date: %{x}<br>Upper Band: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add lower band
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'],
                        y=filtered_df['BB_lower'],
                        name='Lower Band',
                        line=dict(color='#FF0000', width=1, dash='dash'),
                        hovertemplate='Date: %{x}<br>Lower Band: LKR %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add filled area between bands
                    fig.add_trace(go.Scatter(
                        x=filtered_df['Date'].tolist() + filtered_df['Date'].tolist()[::-1],
                        y=filtered_df['BB_upper'].tolist() + filtered_df['BB_lower'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,255,255,0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Band Range',
                        hoverinfo='skip'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - Bollinger Bands ({bb_period}, {bb_std_dev})',
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
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Bollinger Bands analysis
                    last_close = filtered_df['Close'].iloc[-1]
                    last_upper = filtered_df['BB_upper'].iloc[-1]
                    last_lower = filtered_df['BB_lower'].iloc[-1]
                    last_ma = filtered_df['MA'].iloc[-1]
                    
                    # Calculate band width and %B
                    filtered_df['BB_width'] = (filtered_df['BB_upper'] - filtered_df['BB_lower']) / filtered_df['MA']
                    filtered_df['BB_percent'] = (filtered_df['Close'] - filtered_df['BB_lower']) / (filtered_df['BB_upper'] - filtered_df['BB_lower'])
                    
                    last_width = filtered_df['BB_width'].iloc[-1]
                    last_percent_b = filtered_df['BB_percent'].iloc[-1]
                    
                    # Determine signal
                    bb_signal = ""
                    if last_close > last_upper:
                        bb_signal = "Overbought - Price above upper band suggests potential reversal or continued strong uptrend"
                    elif last_close < last_lower:
                        bb_signal = "Oversold - Price below lower band suggests potential reversal or continued strong downtrend"
                    elif last_close > last_ma:
                        bb_signal = "Bullish - Price above middle band (MA) suggests positive momentum"
                    else:
                        bb_signal = "Bearish - Price below middle band (MA) suggests negative momentum"
                    
                    # Volatility analysis
                    width_change = last_width - filtered_df['BB_width'].iloc[-5]  # Change over 5 periods
                    volatility_signal = ""
                    
                    if last_width > 0.1:
                        if width_change > 0:
                            volatility_signal = "Increasing volatility - Potential for significant price move"
                        else:
                            volatility_signal = "High volatility - Use caution when trading"
                    else:
                        if width_change < 0:
                            volatility_signal = "Decreasing volatility - Potential consolidation phase"
                        else:
                            volatility_signal = "Low volatility - Potential for breakout"
                    
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>Bollinger Bands Analysis</h4>
                            <p><b>Current Price:</b> LKR {last_close:.2f}</p>
                            <p><b>Upper Band:</b> LKR {last_upper:.2f}</p>
                            <p><b>Middle Band (MA):</b> LKR {last_ma:.2f}</p>
                            <p><b>Lower Band:</b> LKR {last_lower:.2f}</p>
                            <p><b>%B:</b> {last_percent_b:.2f} (0-1 scale, >1 overbought, <0 oversold)</p>
                            <p><b>Band Width:</b> {last_width:.4f} (measure of volatility)</p>
                            <p><b>Price Signal:</b> {bb_signal}</p>
                            <p><b>Volatility Signal:</b> {volatility_signal}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with tech_tabs[3]:
                st.markdown("### Volume Analysis")
                
                # Check if Volume data is available
                if 'Volume' not in filtered_df.columns:
                    st.warning("Volume data is not available for this company")
                else:
                    # Volume chart with price
                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f"{company} - Price", "Volume"),
                        row_heights=[0.7, 0.3]
                    )
                    
                    # Add price line to top subplot
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['Close'],
                            name='Close Price',
                            line=dict(color='#FFD700', width=2),
                            hovertemplate='Date: %{x}<br>Price: LKR %{y:.2f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
                    
                    # Add volume bars
                    colors = ['green' if filtered_df['Close'].iloc[i] > filtered_df['Close'].iloc[i-1] 
                             else 'red' for i in range(1, len(filtered_df))]
                    colors.insert(0, 'green')  # Default color for first bar
                    
                    fig.add_trace(
                        go.Bar(
                            x=filtered_df['Date'],
                            y=filtered_df['Volume'],
                            name='Volume',
                            marker_color=colors,
                            hovertemplate='Date: %{x}<br>Volume: %{y:,.0f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Calculate and add volume MA
                    filtered_df = filtered_df.copy()  # Create explicit copy
                    filtered_df['Volume_MA20'] = filtered_df['Volume'].rolling(window=20).mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=filtered_df['Date'],
                            y=filtered_df['Volume_MA20'],
                            name='Volume MA (20)',
                            line=dict(color='#FFFFFF', width=1),
                            hovertemplate='Date: %{x}<br>Volume MA: %{y:,.0f}<extra></extra>'
                        ),
                        row=2, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - Volume Analysis',
                            font=dict(size=20, color='#FFD700'),
                            x=0.5,
                            xanchor='center'
                        ),
                        font=dict(color='#FFFFFF'),
                        legend=dict(
                            bgcolor='rgba(0,0,0,0)',
                            bordercolor='#FFD700'
                        ),
                        hovermode='x unified',
                        height=700
                    )
                    
                    # Update axes
                    fig.update_xaxes(
                        gridcolor='#333333',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=2, col=1
                    )
                    
                    fig.update_yaxes(
                        gridcolor='#333333',
                        title='Price (LKR)',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=1, col=1
                    )
                    
                    fig.update_yaxes(
                        gridcolor='#333333',
                        title='Volume',
                        title_font=dict(color='#FFD700'),
                        tickfont=dict(color='#FFFFFF'),
                        row=2, col=1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume analysis
                    last_volume = filtered_df['Volume'].iloc[-1]
                    last_volume_ma = filtered_df['Volume_MA20'].iloc[-1]
                    
                    # Calculate recent volume trend
                    volume_change = ((last_volume / last_volume_ma) - 1) * 100
                    
                    # Determine signals
                    volume_signal = ""
                    if last_volume > 1.5 * last_volume_ma:
                        volume_signal = "Significant volume spike - Potential breakout or trend reversal"
                    elif last_volume > last_volume_ma:
                        volume_signal = "Above average volume - Confirms current price movement"
                    elif last_volume < 0.5 * last_volume_ma:
                        volume_signal = "Very low volume - Lack of interest, potential consolidation"
                    else:
                        volume_signal = "Below average volume - Weakening of current trend"
                    
                    # Price-volume trend analysis
                    price_change = ((filtered_df['Close'].iloc[-1] / filtered_df['Close'].iloc[-5]) - 1) * 100
                    
                    trend_signal = ""
                    if price_change > 0 and volume_change > 0:
                        trend_signal = "Bullish confirmation - Price up with increasing volume"
                    elif price_change > 0 and volume_change < 0:
                        trend_signal = "Weak bullish - Price up with decreasing volume"
                    elif price_change < 0 and volume_change > 0:
                        trend_signal = "Strong bearish - Price down with increasing volume"
                    else:
                        trend_signal = "Weak bearish - Price down with decreasing volume"
                    
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                            <h4 style='color: #FFD700;'>Volume Analysis</h4>
                            <p><b>Latest Volume:</b> {last_volume:,.0f}</p>
                            <p><b>20-Day Average Volume:</b> {last_volume_ma:,.0f}</p>
                            <p><b>Volume vs Average:</b> {volume_change:+.2f}%</p>
                            <p><b>Volume Signal:</b> {volume_signal}</p>
                            <p><b>Price-Volume Trend:</b> {trend_signal}</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            st.error(f"Error generating technical analysis: {str(e)}")

# Data Explorer Tab
with tabs[3]:
    st.session_state.active_tab = "Data Explorer"
    
    st.markdown("## Data Explorer")
    
    # Check if company is selected
    company = st.session_state.selected_company
    if not company:
        st.warning("Please select a company first")
    else:
        try:
            # Load data
            df = data_processor.load_stock_data(company)
            
            # Date range filter
            date_cols = st.columns([1, 1])
            with date_cols[0]:
                start_date = st.date_input(
                    "Start Date",
                    value=df['Date'].min().date(),
                    min_value=df['Date'].min().date(),
                    max_value=df['Date'].max().date()
                )
                
            with date_cols[1]:
                end_date = st.date_input(
                    "End Date",
                    value=df['Date'].max().date(),
                    min_value=df['Date'].min().date(),
                    max_value=df['Date'].max().date()
                )
            
            # Filter data by date range
            filtered_df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
            
            # Data visualization tabs
            data_tabs = st.tabs(["Table View", "Summary Statistics", "Correlation Analysis", "Distribution Analysis"])
            
            with data_tabs[0]:
                st.markdown("### Data Table")
                
                # Show dataframe with date sorting options
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                download_cols = st.columns([1, 3])
                with download_cols[0]:
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{company}_data_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                    
                with download_cols[1]:
                    st.caption(f"Showing {len(filtered_df)} records from {start_date} to {end_date}")
            
            with data_tabs[1]:
                st.markdown("### Summary Statistics")
                
                # Calculate basic statistics
                numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
                stats_df = filtered_df[numeric_cols].describe().T
                
                # Add additional statistics
                stats_df['median'] = filtered_df[numeric_cols].median()
                stats_df['skew'] = filtered_df[numeric_cols].skew()
                stats_df['kurtosis'] = filtered_df[numeric_cols].kurtosis()
                
                # Display statistics
                st.dataframe(stats_df, use_container_width=True)
                
                # Price change analysis
                st.markdown("### Price Change Analysis")
                
                if 'Close' in filtered_df.columns:
                    # Calculate returns
                    filtered_df.loc[:, 'daily_return'] = filtered_df['Close'].pct_change() * 100
                    
                    # Summary metrics
                    first_price = filtered_df['Close'].iloc[0]
                    last_price = filtered_df['Close'].iloc[-1]
                    total_change = ((last_price / first_price) - 1) * 100
                    
                    # Calculate annualized return if data spans multiple years
                    days_span = (filtered_df['Date'].max() - filtered_df['Date'].min()).days
                    years_span = days_span / 365.25
                    
                    if years_span > 0:
                        annualized_return = ((last_price / first_price) ** (1 / years_span) - 1) * 100
                    else:
                        annualized_return = 0
                        
                    # Volatility (standard deviation of returns)
                    volatility = filtered_df['daily_return'].std()
                    
                    # Maximum drawdown
                    cumulative_returns = (1 + filtered_df['daily_return'] / 100).cumprod()
                    rolling_max = cumulative_returns.cummax()
                    drawdowns = (cumulative_returns / rolling_max) - 1
                    max_drawdown = drawdowns.min() * 100
                    
                    # Display metrics
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.metric(
                            "Total Return", 
                            f"{total_change:.2f}%",
                            delta=f"{total_change:.2f}%",
                            delta_color="normal" if total_change >= 0 else "inverse"
                        )
                        
                    with metrics_cols[1]:
                        st.metric(
                            "Annualized Return", 
                            f"{annualized_return:.2f}%" if years_span > 0 else "N/A"
                        )
                        
                    with metrics_cols[2]:
                        st.metric(
                            "Daily Volatility", 
                            f"{volatility:.2f}%"
                        )
                        
                    with metrics_cols[3]:
                        st.metric(
                            "Maximum Drawdown", 
                            f"{max_drawdown:.2f}%",
                            delta=None
                        )
                        
                    # Return distribution chart
                    st.markdown("### Return Distribution")
                    
                    # Create histogram of returns
                    fig = go.Figure()
                    
                    fig.add_trace(go.Histogram(
                        x=filtered_df['daily_return'].dropna(),
                        nbinsx=30,
                        marker_color='#00FFFF',
                        opacity=0.7,
                        name='Daily Returns',
                        hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
                    ))
                    
                    # Add normal distribution line for comparison
                    x_range = np.linspace(
                        filtered_df['daily_return'].min(),
                        filtered_df['daily_return'].max(),
                        100
                    )
                    
                    # Calculate normal distribution values
                    mean = filtered_df['daily_return'].mean()
                    std = filtered_df['daily_return'].std()
                    
                    # Scale y to match histogram height
                    hist, bin_edges = np.histogram(
                        filtered_df['daily_return'].dropna(),
                        bins=30,
                        density=True
                    )
                    
                    y_norm = stats.norm.pdf(x_range, mean, std) * len(filtered_df) * (bin_edges[1] - bin_edges[0])
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_norm,
                        mode='lines',
                        name='Normal Distribution',
                        line=dict(color='#FFD700', width=2),
                        hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add mean line
                    fig.add_shape(
                        type="line",
                        x0=mean,
                        y0=0,
                        x1=mean,
                        y1=max(hist) * len(filtered_df) * (bin_edges[1] - bin_edges[0]),
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    fig.add_annotation(
                        x=mean,
                        y=max(hist) * len(filtered_df) * (bin_edges[1] - bin_edges[0]) * 0.95,
                        text=f"Mean: {mean:.2f}%",
                        showarrow=False,
                        font=dict(color="red", size=12)
                    )
                    
                    # Update layout
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#1a1a1a',
                        paper_bgcolor='#1a1a1a',
                        title=dict(
                            text=f'{company} - Daily Return Distribution',
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
                            title='Daily Return (%)',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF')
                        ),
                        yaxis=dict(
                            gridcolor='#333333',
                            title='Frequency',
                            title_font=dict(color='#FFD700'),
                            tickfont=dict(color='#FFFFFF')
                        ),
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
            with data_tabs[2]:
                st.markdown("### Correlation Analysis")
                
                # Select features for correlation
                numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                
                # Only proceed if there are enough numeric columns
                if len(numeric_cols) > 1:
                    selected_features = st.multiselect(
                        "Select Features",
                        options=numeric_cols,
                        default=numeric_cols[:min(5, len(numeric_cols))]
                    )
                    
                    if selected_features and len(selected_features) > 1:
                        # Calculate correlation matrix
                        corr_matrix = filtered_df[selected_features].corr()
                        
                        # Create heatmap
                        fig = go.Figure()
                        
                        fig.add_trace(go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.index,
                            colorscale='RdBu_r',
                            zmid=0,
                            text=corr_matrix.round(2).values,
                            texttemplate='%{text:.2f}',
                            textfont=dict(color='white'),
                            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.4f}<extra></extra>'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            title=dict(
                                text=f'{company} - Feature Correlation Heatmap',
                                font=dict(size=20, color='#FFD700'),
                                x=0.5,
                                xanchor='center'
                            ),
                            font=dict(color='#FFFFFF'),
                            xaxis=dict(
                                title='',
                                tickfont=dict(color='#FFFFFF')
                            ),
                            yaxis=dict(
                                title='',
                                tickfont=dict(color='#FFFFFF')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation insights
                        st.markdown("### Correlation Insights")
                        
                        # Find highest correlations (excluding self-correlations)
                        corr_pairs = []
                        
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_value = corr_matrix.iloc[i, j]
                                corr_pairs.append({
                                    'feature1': corr_matrix.columns[i],
                                    'feature2': corr_matrix.columns[j],
                                    'correlation': corr_value,
                                    'abs_correlation': abs(corr_value)
                                })
                        
                        if corr_pairs:
                            # Sort by absolute correlation (highest first)
                            corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
                            
                            # Display top correlations
                            top_corr_df = pd.DataFrame(corr_pairs[:5])
                            
                            # Format correlation values
                            top_corr_df['correlation'] = top_corr_df['correlation'].map(lambda x: f"{x:.4f}")
                            top_corr_df['relationship'] = top_corr_df['correlation'].astype(float).map(
                                lambda x: "Strong Positive" if x > 0.7 else
                                "Moderate Positive" if x > 0.3 else
                                "Weak Positive" if x > 0 else
                                "Weak Negative" if x > -0.3 else
                                "Moderate Negative" if x > -0.7 else
                                "Strong Negative"
                            )
                            
                            # Display as colored dataframe
                            st.dataframe(
                                top_corr_df[['feature1', 'feature2', 'correlation', 'relationship']],
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("Not enough features to calculate correlations")
                    else:
                        st.info("Please select at least two features for correlation analysis")
                else:
                    st.info("Not enough numeric features available for correlation analysis")
            
            with data_tabs[3]:
                st.markdown("### Distribution Analysis")
                
                # Select feature for distribution
                feature_cols = st.columns([1, 1])
                
                with feature_cols[0]:
                    selected_feature = st.selectbox(
                        "Select Feature",
                        options=filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
                        index=filtered_df.columns.get_loc('Close') if 'Close' in filtered_df.columns else 0
                    )
                
                with feature_cols[1]:
                    plot_type = st.selectbox(
                        "Plot Type",
                        options=["Histogram", "Box Plot", "Violin Plot", "QQ Plot"],
                        index=0
                    )
                
                if selected_feature:
                    feature_data = filtered_df[selected_feature].dropna()
                    
                    if plot_type == "Histogram":
                        # Create histogram with distribution
                        fig = go.Figure()
                        
                        fig.add_trace(go.Histogram(
                            x=feature_data,
                            nbinsx=30,
                            marker_color='#00FFFF',
                            opacity=0.7,
                            name=selected_feature,
                            hovertemplate=f'{selected_feature}: %{{x}}<br>Frequency: %{{y}}<extra></extra>'
                        ))
                        
                        # Add KDE (kernel density estimation)
                        try:
                            kde_x = np.linspace(feature_data.min(), feature_data.max(), 100)
                            kde = stats.gaussian_kde(feature_data)
                            kde_y = kde(kde_x) * len(feature_data) * (feature_data.max() - feature_data.min()) / 30
                            
                            fig.add_trace(go.Scatter(
                                x=kde_x,
                                y=kde_y,
                                mode='lines',
                                name='Density',
                                line=dict(color='#FFD700', width=2),
                                hovertemplate=f'{selected_feature}: %{{x}}<br>Density: %{{y:.4f}}<extra></extra>'
                            ))
                        except Exception as e:
                            logger.warning(f"Could not generate KDE: {str(e)}")
                        
                        # Update layout
                        fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            title=dict(
                                text=f'{company} - {selected_feature} Distribution',
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
                                title=selected_feature,
                                title_font=dict(color='#FFD700'),
                                tickfont=dict(color='#FFFFFF')
                            ),
                            yaxis=dict(
                                gridcolor='#333333',
                                title='Frequency',
                                title_font=dict(color='#FFD700'),
                                tickfont=dict(color='#FFFFFF')
                            ),
                            bargap=0.1
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif plot_type == "Box Plot":
                        # Create box plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=feature_data,
                            name=selected_feature,
                            boxpoints='all',
                            jitter=0.3,
                            pointpos=-1.8,
                            marker=dict(
                                color='#00FFFF',
                                size=4,
                                opacity=0.5
                            ),
                            line=dict(color='#FFD700'),
                            fillcolor='rgba(0,255,255,0.1)',
                            hovertemplate=f'{selected_feature}: %{{y}}<extra></extra>'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            title=dict(
                                text=f'{company} - {selected_feature} Box Plot',
                                font=dict(size=20, color='#FFD700'),
                                x=0.5,
                                xanchor='center'
                            ),
                            font=dict(color='#FFFFFF'),
                            yaxis=dict(
                                gridcolor='#333333',
                                title=selected_feature,
                                title_font=dict(color='#FFD700'),
                                tickfont=dict(color='#FFFFFF')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif plot_type == "Violin Plot":
                        # Create violin plot
                        fig = go.Figure()
                        
                        fig.add_trace(go.Violin(
                            y=feature_data,
                            name=selected_feature,
                            box_visible=True,
                            meanline_visible=True,
                            fillcolor='rgba(0,255,255,0.3)',
                            line_color='#FFD700',
                            marker=dict(color='#00FFFF', opacity=0.5),
                            hovertemplate=f'{selected_feature}: %{{y}}<extra></extra>'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            template='plotly_dark',
                            plot_bgcolor='#1a1a1a',
                            paper_bgcolor='#1a1a1a',
                            title=dict(
                                text=f'{company} - {selected_feature} Violin Plot',
                                font=dict(size=20, color='#FFD700'),
                                x=0.5,
                                xanchor='center'
                            ),
                            font=dict(color='#FFFFFF'),
                            yaxis=dict(
                                gridcolor='#333333',
                                title=selected_feature,
                                title_font=dict(color='#FFD700'),
                                tickfont=dict(color='#FFFFFF')
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    elif plot_type == "QQ Plot":
                        # Create QQ Plot (Quantile-Quantile)
                        fig = go.Figure()
                        
                        try:
                            # Calculate quantiles
                            feature_sorted = np.sort(feature_data)
                            n = len(feature_sorted)
                            theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
                            
                            # Standardize actual data
                            feature_standardized = (feature_sorted - feature_sorted.mean()) / feature_sorted.std()
                            
                            # Create QQ Plot
                            fig.add_trace(go.Scatter(
                                x=theoretical_quantiles,
                                y=feature_standardized,
                                mode='markers',
                                name='Data',
                                marker=dict(color='#00FFFF', size=6),
                                hovertemplate='Theoretical: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>'
                            ))
                            
                            # Add reference line
                            min_val = min(theoretical_quantiles.min(), feature_standardized.min())
                            max_val = max(theoretical_quantiles.max(), feature_standardized.max())
                            
                            fig.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                name='Normal Reference',
                                line=dict(color='#FFD700', width=2, dash='dash'),
                                hoverinfo='skip'
                            ))
                            
                            # Update layout
                            fig.update_layout(
                                template='plotly_dark',
                                plot_bgcolor='#1a1a1a',
                                paper_bgcolor='#1a1a1a',
                                title=dict(
                                    text=f'{company} - {selected_feature} QQ Plot',
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
                                    title='Theoretical Quantiles',
                                    title_font=dict(color='#FFD700'),
                                    tickfont=dict(color='#FFFFFF')
                                ),
                                yaxis=dict(
                                    gridcolor='#333333',
                                    title='Sample Quantiles',
                                    title_font=dict(color='#FFD700'),
                                    tickfont=dict(color='#FFFFFF')
                                )
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Calculate normality test
                            shapiro_test = stats.shapiro(feature_data)
                            
                            st.markdown("### Normality Test (Shapiro-Wilk)")
                            
                            st.markdown(f"""
                                <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #3a3a3a;'>
                                    <p><b>Shapiro-Wilk Test Statistic:</b> {shapiro_test[0]:.4f}</p>
                                    <p><b>p-value:</b> {shapiro_test[1]:.6f}</p>
                                    <p><b>Interpretation:</b> {
                                        "Data is likely normally distributed" if shapiro_test[1] > 0.05 else 
                                        "Data is not normally distributed"
                                    }</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Could not generate QQ Plot: {str(e)}")
                else:
                    st.info("Please select a feature for distribution analysis")
                    
        except Exception as e:
            logger.error(f"Error in data explorer: {str(e)}")
            st.error(f"Error exploring data: {str(e)}")

# Risk Disclaimer with close button
if st.session_state.show_disclaimer:
    disclaimer_container = st.container()
    with disclaimer_container:
        cols = st.columns([20, 1])
        
        with cols[0]:
            st.markdown("""
                <div style='background-color: #232323; padding: 1em; border-radius: 10px; border: 1px solid #FFD700; margin-top: 2em;'>
                    <p style='color: #FFD700; margin-bottom: 0.5em; font-weight: bold;'>⚠️ Risk Disclaimer</p>
                    <p style='font-size: 0.9em; color: #ffffff;'>
                        Past performance does not guarantee future results. Market predictions are subject to various external factors including economic conditions, 
                        political events, and market sentiment. Always conduct thorough research before making investment decisions.
                        This tool is for educational and research purposes only.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            if st.button("✕", key="close_disclaimer"):
                st.session_state.show_disclaimer = False
                st.rerun()

# Footer
st.markdown(f"""
    <div style='display: flex; justify-content: space-between; color: #666666; font-size: 0.8em; margin-top: 2em; padding-top: 1em; border-top: 1px solid #333333;'>
        <div>OdinEx - CSE Stock Prediction Platform</div>
        <div>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
""", unsafe_allow_html=True)
                            