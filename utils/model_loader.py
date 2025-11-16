"""
utils/model_loader.py - Load saved models and scalers

This module handles loading of all trained ML models and their scalers.
Supports custom path configuration and automatic model discovery.
"""

import pickle
import joblib
import streamlit as st
from pathlib import Path
import os
from typing import Dict, Optional, Tuple, Any

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION TO CHANGE PATHS
# ============================================================================

# Option 1: Set a custom absolute path (uncomment and edit)
#CUSTOM_MODELS_PATH = "/mount/src/emi_prediction_ai_project/saved_models"

# Option 2: Set relative path from project root (uncomment and edit)
#CUSTOM_MODELS_PATH = "saved_models"

# Option 3: Use default auto-detection (recommended)
#CUSTOM_MODELS_PATH = "saved_models"


# Check if running on Streamlit Cloud or locally
if os.path.exists("/mount/src/emi_prediction_ai_project/saved_models"):
    # Streamlit Cloud path
    CUSTOM_MODELS_PATH = "/mount/src/emi_prediction_ai_project/saved_models"
else:
    # Local development path
    CUSTOM_MODELS_PATH = "saved_models"



# ============================================================================
# PATH DETECTION AND CONFIGURATION
# ============================================================================

def find_models_directory(custom_path: Optional[str] = None) -> Path:
    """
    Find the models directory using multiple strategies

    Parameters:
    -----------
    custom_path : str, optional
        Custom path to models directory

    Returns:
    --------
    Path : Path to models directory
    """

    # Strategy 1: Use custom path if provided
    if custom_path:
        custom = Path(custom_path)
        if custom.exists() and custom.is_dir():
            return custom
        else:
            st.warning(f"⚠️ Custom path '{custom_path}' not found, trying alternatives...")

    # Strategy 2: Try common locations
    possible_paths = [
        # Current directory
        Path("saved_models"),

        # Relative to this file's location
        Path(__file__).parent.parent / "saved_models",

        # Streamlit Cloud deployment path
        Path("/mount/src/emi_prediction_ai_project/saved_models"),

        # Alternative names
        Path("models"),
        Path("trained_models"),
        Path(__file__).parent.parent / "models",

        # One level up
        Path("../saved_models"),
    ]

    # Try each path
    for path in possible_paths:
        try:
            resolved_path = path.resolve()
            if resolved_path.exists() and resolved_path.is_dir():
                # Check if it contains .pkl files
                pkl_files = list(resolved_path.glob("*.pkl"))
                if pkl_files:
                    return resolved_path
        except Exception:
            continue

    # Strategy 3: Search for saved_models folder in current directory tree
    current_dir = Path.cwd()
    for root, dirs, files in os.walk(current_dir):
        if "saved_models" in dirs:
            potential_path = Path(root) / "saved_models"
            pkl_files = list(potential_path.glob("*.pkl"))
            if pkl_files:
                return potential_path

    # Default fallback
    return Path("saved_models")


# Get the models directory
MODELS_PATH = find_models_directory(CUSTOM_MODELS_PATH)

# ============================================================================
# AUTOMATIC MODEL DISCOVERY
# ============================================================================

def discover_model_files() -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Automatically discover model and scaler files in the models directory

    Returns:
    --------
    dict : Dictionary mapping model names to their file paths
    """

    if not MODELS_PATH.exists():
        return {'classification': {}, 'regression': {}}

    # Get all pickle files
    all_pkl_files = sorted(MODELS_PATH.glob("*.pkl"))

    # Separate models and scalers
    model_files = [f for f in all_pkl_files if 'scaler' not in f.name.lower()]
    scaler_files = [f for f in all_pkl_files if 'scaler' in f.name.lower()]

    discovered = {
        'classification': {},
        'regression': {}
    }

    # Match models with their scalers
    for model_file in model_files:
        model_name = model_file.stem

        # Try to find corresponding scaler
        scaler_file = None
        for scaler in scaler_files:
            # Check if scaler name matches model name
            if model_name in scaler.stem or scaler.stem in model_name:
                scaler_file = scaler
                break

        if scaler_file:
            # Determine if classification or regression based on filename
            is_classification = any(keyword in model_name.lower() for keyword in [
                'classification', 'classifier', 'logistic', 'svm', 'svc'
            ])

            is_regression = any(keyword in model_name.lower() for keyword in [
                'regression', 'regressor', 'ridge', 'lasso', 'svr'
            ])

            # Create friendly name
            friendly_name = create_friendly_name(model_name)

            model_info = {
                'model': model_file.name,
                'scaler': scaler_file.name
            }

            if is_classification:
                discovered['classification'][friendly_name] = model_info
            elif is_regression:
                discovered['regression'][friendly_name] = model_info
            else:
                # Default to classification if unclear
                discovered['classification'][friendly_name] = model_info

    return discovered


def create_friendly_name(filename: str) -> str:
    """
    Convert filename to friendly display name

    Parameters:
    -----------
    filename : str
        Model filename

    Returns:
    --------
    str : Friendly name
    """
    # Remove common prefixes (numbers, underscores)
    name = filename.replace('_', ' ').strip()

    # Remove leading numbers
    while name and name[0].isdigit():
        name = name[1:].strip()

    # Capitalize words
    words = name.split()
    friendly_words = []

    for word in words:
        if word.lower() in ['xgboost', 'mlp', 'svm', 'knn', 'rf']:
            friendly_words.append(word.upper())
        else:
            friendly_words.append(word.capitalize())

    return ' '.join(friendly_words)


# ============================================================================
# MANUAL MODEL CONFIGURATION (Fallback)
# ============================================================================

MANUAL_MODEL_FILES = {
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
def load_pickle_file(filepath: Path) -> Optional[Any]:
    """
    Load a pickle file with multiple library support

    Parameters:
    -----------
    filepath : Path
        Path to the pickle file

    Returns:
    --------
    object : Loaded object or None if failed
    """
    if not filepath.exists():
        return None

    try:
        # Try pickle first
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            # Try joblib
            return joblib.load(filepath)
        except Exception as e2:
            st.warning(f"⚠️ Could not load {filepath.name}: {str(e1)}")
            return None


def load_model_pair(model_filename: str, scaler_filename: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load a model and its corresponding scaler

    Parameters:
    -----------
    model_filename : str
        Name of the model file
    scaler_filename : str
        Name of the scaler file

    Returns:
    --------
    tuple : (model, scaler) or (None, None) if failed
    """
    model_path = MODELS_PATH / model_filename
    scaler_path = MODELS_PATH / scaler_filename

    model = load_pickle_file(model_path)
    scaler = load_pickle_file(scaler_path)

    return model, scaler


def load_all_models(use_auto_discovery: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Load all available models

    Parameters:
    -----------
    use_auto_discovery : bool
        If True, automatically discover models. If False, use manual configuration.

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

    # Check if directory exists
    if not MODELS_PATH.exists():
        st.error(f"❌ Models directory not found: {MODELS_PATH}")
        st.info("💡 Please ensure model files exist in the correct location")
        return models_dict

    # Get model file mappings
    if use_auto_discovery:
        model_files = discover_model_files()
        if not model_files['classification'] and not model_files['regression']:
            # Fallback to manual configuration
            model_files = MANUAL_MODEL_FILES
    else:
        model_files = MANUAL_MODEL_FILES

    # Load classification models
    for model_name, files in model_files['classification'].items():
        model, scaler = load_model_pair(files['model'], files['scaler'])

        if model is not None and scaler is not None:
            models_dict['classification'][model_name] = {
                'model': model,
                'scaler': scaler,
                'model_file': files['model'],
                'scaler_file': files['scaler']
            }

    # Load regression models
    for model_name, files in model_files['regression'].items():
        model, scaler = load_model_pair(files['model'], files['scaler'])

        if model is not None and scaler is not None:
            models_dict['regression'][model_name] = {
                'model': model,
                'scaler': scaler,
                'model_file': files['model'],
                'scaler_file': files['scaler']
            }

    return models_dict


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_models_info() -> Dict[str, Any]:
    """
    Get information about the models directory and available files

    Returns:
    --------
    dict : Information about models
    """
    info = {
        'models_path': str(MODELS_PATH),
        'path_exists': MODELS_PATH.exists(),
        'total_pkl_files': 0,
        'model_files': [],
        'scaler_files': []
    }

    if MODELS_PATH.exists():
        pkl_files = list(MODELS_PATH.glob("*.pkl"))
        info['total_pkl_files'] = len(pkl_files)
        info['model_files'] = [f.name for f in pkl_files if 'scaler' not in f.name.lower()]
        info['scaler_files'] = [f.name for f in pkl_files if 'scaler' in f.name.lower()]

    return info


def verify_model_files() -> Dict[str, Any]:
    """
    Verify which model files exist

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

    # Check manual configuration
    for model_type in ['classification', 'regression']:
        for model_name, files in MANUAL_MODEL_FILES[model_type].items():
            model_path = MODELS_PATH / files['model']
            scaler_path = MODELS_PATH / files['scaler']

            status[model_type][model_name] = {
                'model_exists': model_path.exists(),
                'scaler_exists': scaler_path.exists(),
                'model_file': files['model'],
                'scaler_file': files['scaler']
            }

    return status


def show_debug_info():
    """Display detailed debug information about model loading"""
    st.write("### 🔍 Model Loader Debug Information")
    st.write("---")

    # Path info
    st.write("**📁 Path Configuration:**")
    st.code(f"Models Path: {MODELS_PATH}")
    st.write(f"Path Exists: {'✅ Yes' if MODELS_PATH.exists() else '❌ No'}")
    st.write(f"Absolute Path: `{MODELS_PATH.resolve()}`")

    if CUSTOM_MODELS_PATH:
        st.write(f"Custom Path Configured: `{CUSTOM_MODELS_PATH}`")

    st.write("---")

    # Files info
    info = get_models_info()
    st.write("**📊 Files Summary:**")
    st.write(f"- Total .pkl files: {info['total_pkl_files']}")
    st.write(f"- Model files: {len(info['model_files'])}")
    st.write(f"- Scaler files: {len(info['scaler_files'])}")

    if info['model_files']:
        st.write("\n**Model Files:**")
        for f in sorted(info['model_files']):
            st.write(f"  - `{f}`")

    if info['scaler_files']:
        st.write("\n**Scaler Files:**")
        for f in sorted(info['scaler_files']):
            st.write(f"  - `{f}`")

    st.write("---")

    # Discovery info
    st.write("**🔍 Auto-Discovery Results:**")
    discovered = discover_model_files()

    st.write(f"**Classification Models Found:** {len(discovered['classification'])}")
    for name in discovered['classification'].keys():
        st.write(f"  - {name}")

    st.write(f"\n**Regression Models Found:** {len(discovered['regression'])}")
    for name in discovered['regression'].keys():
        st.write(f"  - {name}")

    st.write("---")

    # Verification
    st.write("**✅ Manual Configuration Verification:**")
    status = verify_model_files()

    for model_type in ['classification', 'regression']:
        st.write(f"\n**{model_type.capitalize()}:**")
        for model_name, info in status[model_type].items():
            model_status = "✅" if info['model_exists'] else "❌"
            scaler_status = "✅" if info['scaler_exists'] else "❌"
            st.write(f"  - {model_name}: Model {model_status} | Scaler {scaler_status}")


# ============================================================================
# CONFIGURATION HELPER
# ============================================================================

def set_custom_path(path: str):
    """
    Set a custom path for models directory

    Parameters:
    -----------
    path : str
        Custom path to models directory
    """
    global MODELS_PATH, CUSTOM_MODELS_PATH
    CUSTOM_MODELS_PATH = path
    MODELS_PATH = find_models_directory(CUSTOM_MODELS_PATH)
    st.cache_resource.clear()  # Clear cache to reload models
    st.success(f"✅ Models path updated to: {MODELS_PATH}")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'load_all_models',
    'get_models_info',
    'verify_model_files',
    'show_debug_info',
    'set_custom_path',
    'MODELS_PATH'
]
