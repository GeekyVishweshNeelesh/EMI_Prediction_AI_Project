"""
EMIPredict AI - Intelligent Financial Risk Assessment Platform
Main Application Entry Point (app.py)

This file serves as the main hub that handles:
- Page configuration
- Sidebar navigation
- Page routing to different features
- Global model loading and caching
- Application theming
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings


warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 20px;
    }

    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: bold;
    }

    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    # Logo and Title
    st.markdown("# 💰 EMIPredict AI")
    st.markdown("### Intelligent Financial Risk Assessment Platform")
    st.markdown("---")

    # Navigation Menu
    st.markdown("## 📍 NAVIGATION")
    selected_page = st.radio(
        "Choose a page:",
        ["🏠 Home",
         "🎯 EMI Classification",
         "💸 EMI Amount Prediction",
         "📊 Model Comparison",
         "⚙️ Admin Panel"],
        key="page_selector"
    )

    st.markdown("---")

    # Project Information
    st.markdown("## 📊 PROJECT INFORMATION")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", "400K")
        st.metric("Features", "22")
    with col2:
        st.metric("ML Models", "8")
        st.metric("Predictions", "Dual Task")

    st.markdown("---")

    # Model Information
    st.markdown("## 🤖 BEST MODELS")
    st.info("""
    **Classification:**
    - Model: XGBoost (Bayesian)
    - Accuracy: 95.90%
    - ROC-AUC: 0.9962

    **Regression:**
    - Model: XGBoost (Bayesian)
    - RMSE: ₹973.08
    - R² Score: 0.9840
    """)

    st.markdown("---")

    # Quick Links
    st.markdown("## 🔗 QUICK LINKS")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[📚 Docs](#)", unsafe_allow_html=True)
    with col2:
        st.markdown("[🐙 GitHub](#)", unsafe_allow_html=True)
    with col3:
        st.markdown("[📧 Contact](#)", unsafe_allow_html=True)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; color: #888;'>
        <p><small>EMIPredict AI v1.0</small></p>
        <p><small>Powered by XGBoost & Streamlit</small></p>
        <p><small>© 2024 Financial Risk Assessment</small></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (CACHED FOR PERFORMANCE)
# ============================================================================

@st.cache_resource
def load_all_models():
    """
    Load all 8 ML models and their scalers
    Models are cached to avoid reloading on every page change
    """
    from utils.model_loader import load_models
    return load_models()

# Load models at startup
try:
    models = load_all_models()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False
    models = None

# ============================================================================
# PAGE ROUTING LOGIC
# ============================================================================

if selected_page == "🏠 Home":
    from pages import home
    home.main()

elif selected_page == "🎯 EMI Classification":
    from pages import classification
    if models_loaded:
        classification.main(models)
    else:
        st.error("Models not loaded. Please refresh the page.")

elif selected_page == "💸 EMI Amount Prediction":
    from pages import regression
    if models_loaded:
        regression.main(models)
    else:
        st.error("Models not loaded. Please refresh the page.")

elif selected_page == "📊 Model Comparison":
    from pages import model_comparison
    model_comparison.main()

elif selected_page == "⚙️ Admin Panel":
    from pages import admin
    admin.main(models)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; margin-top: 50px;'>
    <p><small>EMIPredict AI - Intelligent Financial Risk Assessment Platform</small></p>
    <p><small>Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</small></p>
    <p><small>For questions or support, please contact the development team.</small></p>
</div>
""", unsafe_allow_html=True)
