"""
utils/model_loader.py - Load saved models and scalers

This module handles loading of all trained ML models and their scalers
from the saved_models directory.
"""

import pickle
import streamlit as st
from pathlib import Path
import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_models_path():
    """
    Find the correct path to saved_models directory
    Works on both local machine and Streamlit Cloud
    """
    # Try multiple possible paths
    possible_paths = [
        Path("saved_models"),                                      # Relative to project root
        Path(__file__).parent.parent / "saved_models",            # Relative to this file
        Path("/mount/src/emi_prediction_ai_project/saved_models/") # Streamlit Cloud absolute
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # If nothing found, return default
    return Path("saved_models")

# Get the correct models path
MODELS_PATH = get_models_path()

# ============================================================================
# MODEL FILE MAPPINGS
# ============================================================================

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

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_pickle_file(filepath):
    """
    Load a pickle file (model or scaler)

    Parameters:
    -----------
    filepath : Path
        Path to the pickle file

    Returns:
    --------
    object : Loaded pickle object or None if failed
    """
    try:
        if not filepath.exists():
            return None

        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        return obj

    except Exception as e:
        st.warning(f"⚠️ Could not load {filepath.name}: {str(e)}")
        return None

def load_all_models():
    """
    Load all available classification and regression models with their scalers

    Returns:
    --------
    dict : Dictionary with structure:
        {
            'classification': {
                'Model Name': {'model': model_obj, 'scaler': scaler_obj},
                ...
            },
            'regression': {
                'Model Name': {'model': model_obj, 'scaler': scaler_obj},
                ...
            }
        }
    """

    models_dict = {
        'classification': {},
        'regression': {}
    }

    # Check if models directory exists
    if not MODELS_PATH.exists():
        st.error(f"❌ Models directory not found: {MODELS_PATH}")
        st.info("💡 Please ensure the 'saved_models/' folder exists in your project")
        return models_dict

    # Load classification models
    for model_name, files in MODEL_FILES['classification'].items():
        model_path = MODELS_PATH / files['model']
        scaler_path = MODELS_PATH / files['scaler']

        model = load_pickle_file(model_path)
        scaler = load_pickle_file(scaler_path)

        if model is not None and scaler is not None:
            models_dict['classification'][model_name] = {
                'model': model,
                'scaler': scaler
            }

    # Load regression models
    for model_name, files in MODEL_FILES['regression'].items():
        model_path = MODELS_PATH / files['model']
        scaler_path = MODELS_PATH / files['scaler']

        model = load_pickle_file(model_path)
        scaler = load_pickle_file(scaler_path)

        if model is not None and scaler is not None:
            models_dict['regression'][model_name] = {
                'model': model,
                'scaler': scaler
            }

    return models_dict

def get_available_models():
    """
    Get list of available model names

    Returns:
    --------
    dict : Dictionary with lists of available classification and regression models
    """
    models = load_all_models()
    return {
        'classification': list(models['classification'].keys()),
        'regression': list(models['regression'].keys())
    }

def verify_model_files():
    """
    Verify which model files exist in the saved_models directory

    Returns:
    --------
    dict : Status of each model file
    """
    status = {
        'models_path': str(MODELS_PATH),
        'path_exists': MODELS_PATH.exists(),
        'classification': {},
        'regression': {}
    }

    if not MODELS_PATH.exists():
        return status

    # Check classification models
    for model_name, files in MODEL_FILES['classification'].items():
        model_path = MODELS_PATH / files['model']
        scaler_path = MODELS_PATH / files['scaler']

        status['classification'][model_name] = {
            'model_exists': model_path.exists(),
            'scaler_exists': scaler_path.exists(),
            'model_file': files['model'],
            'scaler_file': files['scaler']
        }

    # Check regression models
    for model_name, files in MODEL_FILES['regression'].items():
        model_path = MODELS_PATH / files['model']
        scaler_path = MODELS_PATH / files['scaler']

        status['regression'][model_name] = {
            'model_exists': model_path.exists(),
            'scaler_exists': scaler_path.exists(),
            'model_file': files['model'],
            'scaler_file': files['scaler']
        }

    return status

# ============================================================================
# DEBUG HELPER FUNCTION
# ============================================================================

def show_debug_info():
    """Show debug information about models path and available files"""
    st.write("### 🔍 Model Loader Debug Info")
    st.write(f"**Models Path:** `{MODELS_PATH}`")
    st.write(f"**Path Exists:** {'✅ Yes' if MODELS_PATH.exists() else '❌ No'}")

    if MODELS_PATH.exists():
        # List all .pkl files
        pkl_files = list(MODELS_PATH.glob("*.pkl"))
        st.write(f"**Total .pkl files found:** {len(pkl_files)}")

        if pkl_files:
            st.write("**Files in saved_models/:**")
            for file in sorted(pkl_files):
                st.write(f"  - `{file.name}`")

        # Show verification status
        st.write("\n**Model Files Status:**")
        status = verify_model_files()

        st.write("**Classification Models:**")
        for model_name, info in status['classification'].items():
            model_status = "✅" if info['model_exists'] else "❌"
            scaler_status = "✅" if info['scaler_exists'] else "❌"
            st.write(f"  - {model_name}: Model {model_status} | Scaler {scaler_status}")

        st.write("\n**Regression Models:**")
        for model_name, info in status['regression'].items():
            model_status = "✅" if info['model_exists'] else "❌"
            scaler_status = "✅" if info['scaler_exists'] else "❌"
            st.write(f"  - {model_name}: Model {model_status} | Scaler {scaler_status}")
    else:
        st.error("Models directory does not exist!")
        st.write("**Current working directory:**", os.getcwd())
