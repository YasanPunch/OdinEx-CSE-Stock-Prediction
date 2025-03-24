import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import pickle
from pathlib import Path
from models.gru_bi import GRUBiModel

# Set up cache directory
CACHE_DIR = Path("cache/models")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache management functions
def get_model_cache_key(company, lookback_period, prediction_days, other_params=None):
    """Generate a unique key for model caching"""
    params_str = f"{company}_{lookback_period}_{prediction_days}"
    if other_params:
        params_str += f"_{str(other_params)}"
    return hashlib.md5(params_str.encode()).hexdigest()

@st.cache_resource(max_entries=10)
def get_or_train_model(company, lookback_period, prediction_days):
    cache_key = get_model_cache_key(company, lookback_period, prediction_days)
    cache_file = CACHE_DIR / f"{cache_key}.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)
            st.success("Loaded model from cache!")
            return model_data
        except Exception as e:
            st.warning("Cache load failed, training new model...")
    
    with st.spinner('Training new model...'):
        # Your training code here - placeholder for now
        model_data = {
            'model': None,  # Replace with actual trained model
            'accuracy': 0.95,  # Replace with actual accuracy
            'training_date': datetime.now(),
            'parameters': {
                'company': company,
                'lookback_period': lookback_period,
                'prediction_days': prediction_days
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data

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
    
    # Company Selection
    company = st.selectbox(
        "Select Company",
        ["Company A", "Company B", "Company C"]  # Replace with actual company list
    )
    
    # Model Parameters
    st.markdown("### Model Parameters")
    lookback_period = st.slider("Lookback Period (days)", 5, 60, 30)
    prediction_days = st.slider("Prediction Window (days)", 1, 30, 7)
    
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
        model_data = get_or_train_model(company, lookback_period, prediction_days)
        st.session_state['current_model'] = model_data

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Historical Performance")
    # Placeholder for historical performance chart
    st.markdown("```Chart will be displayed here```")
    
with col2:
    st.markdown("### Predictions")
    # Placeholder for predictions chart
    st.markdown("```Predictions will be displayed here```")

# Demo Model Management
if 'current_model' in st.session_state:
    if st.button("Save Current Model as Presentation Demo"):
        demo_file = Path("demo_models") / f"demo_{company}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        demo_file.parent.mkdir(exist_ok=True)
        with open(demo_file, 'wb') as f:
            pickle.dump(st.session_state['current_model'], f)
        st.success(f"Saved demo model: {demo_file.name}")

# Model Metrics
if 'current_model' in st.session_state:
    st.markdown("### Model Metrics")
    metric1, metric2, metric3 = st.columns(3)
    
    with metric1:
        st.metric(label="Training Accuracy", value="95%")
        
    with metric2:
        st.metric(label="Validation RMSE", value="0.023")
        
    with metric3:
        st.metric(label="Prediction Confidence", value="87%")

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