#Code in app.py file
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
import numpy as np

# Import models and utilities
from models import available_models
from utils.data_processor import DataProcessor
from utils.metrics import calculate_metrics, calculate_prediction_confidence

# Dictionary of available models
AVAILABLE_MODELS = available_models

# Set up directories
CACHE_DIR = Path("cache/models")
DEMO_DIR = Path("demo_models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# Initialize data processor
data_processor = DataProcessor()

# Initialize session state for model and related data if they don't exist
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Bidirectional GRU"
    st.session_state.model_instance = AVAILABLE_MODELS[st.session_state.model_type]()
    st.session_state.current_params = {}  # Store current parameter values
    st.session_state.selected_company = None  # Store selected company
    st.session_state.training_history = []  # Store training history
    st.session_state.last_trained = None  # Store last training timestamp
    
def validate_data(df):
    """Validate input data"""
    if df is None or df.empty:
        raise ValueError("Empty or invalid dataset")
    required_columns = ['Date', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def align_predictions_with_actual(predictions, actual_dates, actual_prices):
    """Ensure predictions align with actual data"""
    if len(predictions) != len(actual_dates):
        # Truncate or pad predictions to match actual data length
        predictions = predictions[:len(actual_dates)]
    return predictions

def get_model_cache_key(model_type, company, parameters):
    """Generate a unique key for model caching"""
    params_str = f"{model_type}_{company}_{str(sorted(parameters.items()))}"
    return hashlib.md5(params_str.encode()).hexdigest()

@st.cache_resource(max_entries=10)
def get_or_train_model(model_type, company, parameters):
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
        
        # Train model
        model_instance = st.session_state.model_instance
        metrics = model_instance.train((train_data, train_labels), (test_data, test_labels))
        
        try:
            # Make future predictions
            last_sequence = test_data[-1:]  # Take last sequence as batch
            if last_sequence.size == 0:
                raise ValueError("No data available for prediction")
                
            predictions = model_instance.predict(last_sequence, parameters['prediction_days'])
            predictions = data_processor.inverse_transform(predictions)
            
            # Generate test predictions for historical comparison
            test_predictions = model_instance.predict(test_data, 1)
            test_predictions = data_processor.inverse_transform(test_predictions)
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            raise
        
        model_data = {
            'model': model_instance,
            'metrics': metrics,
            'predictions': predictions,
            'test_predictions': test_predictions,
            'training_date': datetime.now(),
            'parameters': parameters,
            'company': company
        }
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data
        
    except Exception as e:
        st.error(f"Error during model training: {str(e)}")
        return None
    
# Set page config
st.set_page_config(
    page_title="OdinEx - CSE Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
            margin-bottom: 1em;
            text-align: center;
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
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #FFD700;
            color: #1a1a1a;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #232323;
            padding: 1em;
            border-radius: 5px;
            border: 1px solid #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">OdinEx</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #FFD700;">CSE Stock Prediction Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<p style="color: #FFD700;">Model Configuration</p>', unsafe_allow_html=True)
    
    # Model Selection
    model_type = st.selectbox(
        "Select Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.model_type)
    )
    
    # Company Selection with validation
    companies = data_processor.get_available_companies()
    if not companies:
        st.error("No company data files found! Please check data/processed_data directory.")
    else:
        company = st.selectbox(
            "Select Company",
            companies,
            help="Select the company to analyze"
        )
        st.session_state.selected_company = company
        
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
    current_params['window_size'] = st.slider(
        "Number of Past Days to Consider 🛈",
        min_value=5,
        max_value=365,
        value=30,
        help="How many days of historical data to use for prediction. More data can help identify longer-term patterns."
    )
    
    current_params['prediction_days'] = st.slider(
        "Prediction Window (Days) 🛈",
        min_value=1,
        max_value=30,
        value=7,
        help="How many days into the future to predict. Longer predictions tend to be less accurate."
    )
    
    # Advanced Parameters Toggle
    show_advanced = st.checkbox("Show Advanced Parameters")
    
    if show_advanced:
        st.markdown("#### Advanced Parameters")
        
        current_params['hidden_dim'] = st.slider(
            "Hidden Layer Size 🛈",
            min_value=32,
            max_value=256,
            value=64,
            help="Technical: Number of neurons in hidden layers.\n\n" +
                 "Simple: Controls how much information the model can process at once. " +
                 "Larger values might improve accuracy but make training slower."
        )
        
        current_params['num_layers'] = st.slider(
            "Number of GRU Layers 🛈",
            min_value=1,
            max_value=4,
            value=2,
            help="Technical: Number of stacked GRU layers.\n\n" +
                 "Simple: Controls how complex patterns the model can learn. " +
                 "More layers can capture more complex patterns but might overfit."
        )
        
        current_params['dropout'] = st.slider(
            "Dropout Rate 🛈",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.1,
            help="Technical: Probability of neurons being disabled during training.\n\n" +
                 "Simple: Controls how aggressively the model prevents overfitting. " +
                 "Higher values make the model more conservative but might reduce accuracy."
        )
        
        current_params['learning_rate'] = st.slider(
            "Learning Rate 🛈",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            format="%.4f",
            help="Technical: Step size during optimization.\n\n" +
                 "Simple: Controls how quickly the model learns. " +
                 "Too high might make training unstable, too low might make it too slow."
        )
    else:
        # Set default values for advanced parameters when hidden
        configurable_params = st.session_state.model_instance.get_configurable_parameters()
        for param_name, param_info in configurable_params.items():
            if param_name not in current_params:
                current_params[param_name] = param_info['default']
    
    st.session_state.current_params = current_params
    
    # Cache Management
    st.markdown("### Cached Models")
    cached_models = list(CACHE_DIR.glob("*.pkl"))
    st.write(f"Currently cached: {len(cached_models)} models")
    
    if st.button("Clear All Cached Models"):
        for file in CACHE_DIR.glob("*.pkl"):
            file.unlink()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    # Training Button
    if st.button("Train Model"):
        if not companies:
            st.error("Please add company data files before training!")
        else:
            st.markdown("### Training Progress")
            with st.expander("Show Training Details", expanded=True):
                model_data = get_or_train_model(model_type, company, current_params)
                if model_data:
                    st.session_state['current_model'] = model_data
                    st.session_state.last_trained = datetime.now()

# Main Content
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
            
        # Add company to model_data
        model_data['company'] = company
        
        # Continue with original visualization code
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
                        text=f'Historical Price Comparison - {company}',  # Add explicit title
                        font=dict(size=20, color='#FFD700'),
                        x=0.5,  # Center the title
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
                
            except Exception as e:
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
                prediction_days = model_data['parameters'].get('prediction_days', 7)  # Default to 7 if not specified
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
                if 'confidence_intervals' in model_data:
                    lower_bound = model_data['confidence_intervals']['lower']
                    upper_bound = model_data['confidence_intervals']['upper']
                    
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
                        text=f'Price Predictions - {company}',  # Add explicit title
                        font=dict(size=20, color='#FFD700'),
                        x=0.5,  # Center the title
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
                    price_change = ((future_predictions[-1] - actual_prices[-1]) / actual_prices[-1]) * 100
                    st.markdown(f"""
                        <div style='background-color: #232323; padding: 1em; border-radius: 5px; border: 1px solid #FFD700;'>
                            <h4 style='color: #FFD700;'>Prediction Details</h4>
                            <p>Last Known Price: LKR {actual_prices[-1]:.2f}</p>
                            <p>Predicted End Price: LKR {future_predictions[-1]:.2f}</p>
                            <p>Predicted Change: {price_change:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error calculating prediction details: {str(e)}")
                    
            except Exception as e:
                st.error(f"Error displaying future predictions: {str(e)}")
        
        # Model Performance Metrics
        try:
            st.markdown("### Model Performance Metrics")
            metrics = model_data.get('metrics', {})
            
            if metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    accuracy = metrics.get('accuracy', 0)
                    st.metric(
                        label="Model Accuracy",
                        value=f"{accuracy:.2f}%",
                        delta=f"{accuracy - 90:.2f}%" if accuracy > 90 else None
                    )
                
                with col2:
                    rmse = metrics.get('rmse', 0)
                    st.metric(
                        label="Prediction RMSE",
                        value=f"LKR {rmse:.2f}",
                        delta=None
                    )
                
                with col3:
                    predictions = model_data.get('predictions', np.array([]))
                    confidence = calculate_prediction_confidence(predictions)
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.1f}%",
                        delta=None
                    )
            else:
                st.warning("No metrics available")
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
            
        try:
            # Demo Model Management
            if company:  # Check if company exists
                demo_file = DEMO_DIR / f"demo_{company}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
                if st.button("Save Current Model as Presentation Demo"):
                    try:
                        with open(demo_file, 'wb') as f:
                            pickle.dump(model_data, f)
                        st.success(f"Saved demo model: {demo_file.name}")
                    except Exception as e:
                        st.error(f"Error saving demo model: {str(e)}")
        except Exception as e:
            st.error(f"Error in demo model section: {str(e)}")
                
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")

# Risk Disclaimer
st.markdown("""
    <div style='background-color: #232323; padding: 1em; border-radius: 5px; border: 1px solid #FFD700; margin-top: 2em;'>
        <p style='color: #FFD700; margin-bottom: 0.5em;'>⚠️ Risk Disclaimer</p>
        <p style='font-size: 0.9em; color: #ffffff;'>
            Past performance does not guarantee future results. Market predictions are subject to various external factors including economic conditions, 
            political events, and market sentiment. Always conduct thorough research before making investment decisions.
        </p>
    </div>
""", unsafe_allow_html=True)

# Data Last Updated
st.markdown(f"""
    <div style='text-align: right; color: #666666; font-size: 0.8em; margin-top: 2em;'>
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
""", unsafe_allow_html=True)
