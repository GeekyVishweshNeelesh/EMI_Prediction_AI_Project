"""
app.py - Main Streamlit Application for EMI Prediction AI

This is the main entry point for the EMI Prediction application.
It handles page navigation and loads the appropriate page based on user selection.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils.model_loader import load_all_models

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="EMI Predictor AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# EMI Predictor AI\nFinancial Risk Assessment Platform"
    }
)

# ============================================================================
# CUSTOM CSS (OPTIONAL)
# ============================================================================

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS (CACHED)
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models once and cache them"""
    try:
        return load_all_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {'classification': {}, 'regression': {}}

# ============================================================================
# SIDEBAR NAVIGATION (AT THE TOP)
# ============================================================================

with st.sidebar:
    st.title("💰 EMI Predictor AI")
    st.markdown("---")

    # Navigation Menu at the top
    st.header("📍 Navigation")

    page = st.radio(
        "Choose a page:",
        [
            "🏠 Home",
            "🎯 Classification",
            "💰 Regression",
            "📊 Model Comparison",
            "⚙️ Admin Panel"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Project Information
    st.subheader("📊 Project Info")
    st.info("""
    **Dataset:** 400K+ Records
    **Models:** 8 ML Models
    **Features:** 22 Input Features
    **Tasks:** Classification & Regression
    """)

    st.markdown("---")

    # Model Status
    st.subheader("🤖 Model Status")
    with st.spinner("Loading models..."):
        try:
            models = load_models()

            # Count available models
            num_classification = len(models.get('classification', {}))
            num_regression = len(models.get('regression', {}))

            if num_classification > 0:
                st.success(f"✅ {num_classification} Classification Models")
            else:
                st.error("❌ No Classification Models")

            if num_regression > 0:
                st.success(f"✅ {num_regression} Regression Models")
            else:
                st.error("❌ No Regression Models")

        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            models = {'classification': {}, 'regression': {}}
            num_classification = 0
            num_regression = 0

    st.markdown("---")

    # Quick Stats
    st.subheader("📈 Quick Stats")
    st.metric("Total Models", num_classification + num_regression)
    st.metric("Data Points", "400K+")
    st.metric("Accuracy", ">90%")

    st.markdown("---")

    # Footer
    st.caption("© 2024 EMI Predictor AI")
    st.caption("Built with Streamlit & MLflow")

# ============================================================================
# PAGE ROUTING
# ============================================================================

try:
    # Route to the selected page
    if page == "🏠 Home":
        # Import and run home page
        from pages import home
        home.main()

    elif page == "🎯 Classification":
        # Import and run classification page
        from pages import classification
        classification.main(models)

    elif page == "💰 Regression":
        # Import and run regression page
        from pages import regression
        regression.main(models)

    elif page == "📊 Model Comparison":
        # Import and run model comparison page
        from pages import model_comparison
        model_comparison.main(models)

    elif page == "⚙️ Admin Panel":
        # Import and run admin panel page
        from pages import admin
        admin.main()

except Exception as e:
    st.error(f"❌ Error loading page: {str(e)}")
    st.exception(e)

# ============================================================================
# FOOTER (MAIN CONTENT AREA)
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>EMI Predictor AI | Machine Learning for Financial Risk Assessment</p>
        <p>Powered by XGBoost, Random Forest, and Advanced ML Algorithms</p>
    </div>
    """,
    unsafe_allow_html=True
)
