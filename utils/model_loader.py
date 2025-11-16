"""
utils/model_loader.py - Load saved models and scalers (WITH DEBUG INFO)
"""

import pickle
import joblib
import streamlit as st
from pathlib import Path
import os

# Base path for saved models
MODELS_PATH = "saved_models/"

# Check if folder exists
if not os.path.exists(MODELS_PATH):
    st.error(f"❌ Folder '{MODELS_PATH}' does not exist!")
else:
    st.success(f"✅ Folder '{MODELS_PATH}' exists")
    # List all files in the folder
    files = os.listdir(MODELS_PATH)
    st.write(f"📁 Found {len(files)} files in saved_models/")
    if len(files) > 0:
        st.write("Files:", files)

# Model file mappings
MODEL_FILES = {
    'classification': {
        'Logistic Regression': {
            'model': '01_logistic_regression_base.pkl',
            'scaler': '01_logistic_regression_base_scaler.pkl'
        },
        'XGBoost Classifier': {
            'model': '03_xgboost_classifier_bayesian.pkl',
            'scaler': '03_xgboost_classifier_bayesian_scaler.pkl'
        },
        'Decision Tree': {
            'model': '07_decision_tree_classifier_base.pkl',
            'scaler': '07_decision_tree_classifier_base_scaler.pkl'
        },
        'Gradient Boosting': {
            'model': '08_gradient_boosting_classifier_optimized.pkl',
            'scaler': '08_gradient_boosting_classifier_optimized_scaler.pkl'
        }
    },
    'regression': {
        'Ridge Regression': {
            'model': '04_ridge_regression_base.pkl',
            'scaler': '04_ridge_regression_base_scaler.pkl'
        },
        'XGBoost Regressor': {
            'model': '06_xgboost_regressor_bayesian.pkl',
            'scaler': '06_xgboost_regressor_bayesian_scaler.pkl'
        }
    }
}

@st.cache_resource
def load_model(filename):
    """Load a pickle model file"""
    try:
        filepath = Path(MODELS_PATH) / filename
        st.write(f"🔍 Trying to load: {filepath}")

        if not filepath.exists():
            st.error(f"❌ File not found: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        st.success(f"✅ Loaded: {filename}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {str(e)}")
        return None

@st.cache_resource
def load_scaler(filename):
    """Load a pickle scaler file"""
    try:
        filepath = Path(MODELS_PATH) / filename
        st.write(f"🔍 Trying to load: {filepath}")

        if not filepath.exists():
            st.error(f"❌ File not found: {filepath}")
            return None

        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)

        st.success(f"✅ Loaded: {filename}")
        return scaler
    except Exception as e:
        st.error(f"❌ Error loading {filename}: {str(e)}")
        return None

def load_all_models():
    """
    Load all available models and scalers

    Returns:
    --------
    dict: Dictionary containing classification and regression models
    """
    st.write("📦 Starting to load all models...")

    models_dict = {
        'classification': {},
        'regression': {}
    }

    # Load classification models
    st.write("📊 Loading classification models...")
    for model_name, files in MODEL_FILES['classification'].items():
        st.write(f"  ➡️ Loading {model_name}...")
        model = load_model(files['model'])
        scaler = load_scaler(files['scaler'])

        if model is not None and scaler is not None:
            models_dict['classification'][model_name] = {
                'model': model,
                'scaler': scaler
            }
            st.success(f"  ✅ {model_name} loaded successfully")
        else:
            st.warning(f"  ⚠️ {model_name} failed to load")

    # Load regression models
    st.write("💰 Loading regression models...")
    for model_name, files in MODEL_FILES['regression'].items():
        st.write(f"  ➡️ Loading {model_name}...")
        model = load_model(files['model'])
        scaler = load_scaler(files['scaler'])

        if model is not None and scaler is not None:
            models_dict['regression'][model_name] = {
                'model': model,
                'scaler': scaler
            }
            st.success(f"  ✅ {model_name} loaded successfully")
        else:
            st.warning(f"  ⚠️ {model_name} failed to load")

    st.write(f"📊 Classification models loaded: {len(models_dict['classification'])}")
    st.write(f"💰 Regression models loaded: {len(models_dict['regression'])}")

    return models_dict

def get_available_models():
    """Get list of available model names"""
    models = load_all_models()
    return {
        'classification': list(models['classification'].keys()),
        'regression': list(models['regression'].keys())
    }
