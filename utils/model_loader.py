"""
utils/model_loader.py - Load saved models and scalers
Aligned with EMI Predictor AI Project Requirements
"""

import pickle
import joblib
import streamlit as st
from pathlib import Path
import os

# ============================================================================
# PATH CONFIGURATION - As per project structure
# ============================================================================

# Project structure from PDF:
# emi_prediction_app/
# ├── saved_models/    ← Models are saved here
# ├── utils/
# ├── pages/
# └── app.py

def get_models_path():
    """
    Get the correct path to saved_models directory
    Works for both local development and Streamlit Cloud deployment
    """
    # Try multiple possible paths
    possible_paths = [
        Path("saved_models"),                                       # Standard location
        Path(__file__).parent.parent / "saved_models",             # Relative to utils/
        Path("/mount/src/emi_prediction_ai_project/saved_models")  # Streamlit Cloud
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            pkl_files = list(path.glob("*.pkl"))
            if pkl_files:  # Directory exists and has .pkl files
                return path

    # Default fallback
    return Path("saved_models")

# Get models path
MODELS_PATH = get_models_path()

# ============================================================================
# MODEL FILE MAPPINGS - As per PDF requirements
# ============================================================================

# Classification Models (Minimum 3 required)
# Regression Models (Minimum 3 required)

MODEL_FILES = {
    'classification': {
        'Logistic Regression': {
            'model': '01_logistic_regression_base.pkl',
            'scaler': '01_logistic_regression_base_scaler.pkl'
        },
        'Random Forest Classifier': {
            'model': '02_random_forest_classifier_base.pkl',
            'scaler': '02_random_forest_classifier_base_scaler.pkl'
        },
        'XGBoost Classifier': {
            'model': '03_xgboost_classifier_bayesian.pkl',
            'scaler': '03_xgboost_classifier_bayesian_scaler.pkl'
        },
        'Decision Tree Classifier': {
            'model': '07_decision_tree_classifier_base.pkl',
            'scaler': '07_decision_tree_classifier_base_scaler.pkl'
        },
        'Gradient Boosting Classifier': {
            'model': '08_gradient_boosting_classifier_optimized.pkl',
            'scaler': '08_gradient_boosting_classifier_optimized_scaler.pkl'
        }
    },
    'regression': {
        'Linear Regression': {
            'model': '04_linear_regression_base.pkl',
            'scaler': '04_linear_regression_base_scaler.pkl'
        },
        'Random Forest Regressor': {
            'model': '05_random_forest_regressor_base.pkl',
            'scaler': '05_random_forest_regressor_base_scaler.pkl'
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
    Supports both pickle and joblib formats
    """
    try:
        if not filepath.exists():
            return None

        # Try pickle first
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    except Exception:
        try:
            # Try joblib as fallback
            return joblib.load(filepath)
        except Exception as e:
            st.warning(f"⚠️ Could not load {filepath.name}: {str(e)}")
            return None

def load_all_models():
    """
    Load all classification and regression models with their scalers

    Returns:
    --------
    dict : Dictionary containing all loaded models
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
        st.info("💡 Expected location: saved_models/ folder in project root")
        st.info("📋 Please ensure model .pkl files are uploaded to GitHub")
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

def verify_model_files():
    """
    Verify which model files exist in saved_models directory

    Returns:
    --------
    dict : Status of each expected model file
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

def get_available_models():
    """Get lists of available model names"""
    models = load_all_models()
    return {
        'classification': list(models['classification'].keys()),
        'regression': list(models['regression'].keys())
    }

# ============================================================================
# DEBUG HELPER
# ============================================================================

def show_debug_info():
    """Display debug information about model loading"""
    st.write("### 🔍 Model Loader Debug Information")
    st.write("---")

    st.write(f"**Models Path:** `{MODELS_PATH}`")
    st.write(f"**Path Exists:** {'✅ Yes' if MODELS_PATH.exists() else '❌ No'}")
    st.write(f"**Absolute Path:** `{MODELS_PATH.resolve()}`")

    st.write("---")

    if MODELS_PATH.exists():
        pkl_files = list(MODELS_PATH.glob("*.pkl"))
        st.write(f"**Total .pkl files found:** {len(pkl_files)}")

        if pkl_files:
            st.write("**Files in saved_models/:**")
            for f in sorted(pkl_files):
                st.write(f"  - `{f.name}`")

        st.write("---")

        # Verification
        st.write("**📋 Expected Model Files Status:**")
        status = verify_model_files()

        st.write("\n**Classification Models:**")
        for model_name, info in status['classification'].items():
            model_status = "✅" if info['model_exists'] else "❌"
            scaler_status = "✅" if info['scaler_exists'] else "❌"
            st.write(f"  - {model_name}")
            st.write(f"    - Model: {model_status} `{info['model_file']}`")
            st.write(f"    - Scaler: {scaler_status} `{info['scaler_file']}`")

        st.write("\n**Regression Models:**")
        for model_name, info in status['regression'].items():
            model_status = "✅" if info['model_exists'] else "❌"
            scaler_status = "✅" if info['scaler_exists'] else "❌"
            st.write(f"  - {model_name}")
            st.write(f"    - Model: {model_status} `{info['model_file']}`")
            st.write(f"    - Scaler: {scaler_status} `{info['scaler_file']}`")
    else:
        st.error(f"❌ Directory not found: {MODELS_PATH}")
        st.write("**Current working directory:**", os.getcwd())
