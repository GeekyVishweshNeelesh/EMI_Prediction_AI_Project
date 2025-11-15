"""
utils/model_loader.py - Load saved models and scalers
"""

import pickle
import joblib
import streamlit as st
from pathlib import Path

# Base path for saved models
MODELS_PATH = "saved_models/"

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
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename}: {str(e)}")
        return None

@st.cache_resource
def load_scaler(filename):
    """Load a pickle scaler file"""
    try:
        filepath = Path(MODELS_PATH) / filename
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename}: {str(e)}")
        return None

def load_all_models():
    """
    Load all available models and scalers

    Returns:
    --------
    dict: Dictionary containing classification and regression models
    """
    models_dict = {
        'classification': {},
        'regression': {}
    }

    # Load classification models
    for model_name, files in MODEL_FILES['classification'].items():
        model = load_model(files['model'])
        scaler = load_scaler(files['scaler'])

        if model is not None and scaler is not None:
            models_dict['classification'][model_name] = {
                'model': model,
                'scaler': scaler
            }

    # Load regression models
    for model_name, files in MODEL_FILES['regression'].items():
        model = load_model(files['model'])
        scaler = load_scaler(files['scaler'])

        if model is not None and scaler is not None:
            models_dict['regression'][model_name] = {
                'model': model,
                'scaler': scaler
            }

    return models_dict

def get_available_models():
    """Get list of available model names"""
    models = load_all_models()
    return {
        'classification': list(models['classification'].keys()),
        'regression': list(models['regression'].keys())
    }
