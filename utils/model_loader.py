"""
utils/model_loader.py - Load and manage all ML models and scalers

This module handles:
- Loading all 8 ML models
- Loading corresponding scalers
- Caching for performance
- Error handling
"""

import pickle
import joblib
import streamlit as st
from config import MODEL_PATHS, SCALER_PATHS

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model(model_path):
    """
    Load a single model from pickle file

    Parameters:
    -----------
    model_path : str
        Path to the model file

    Returns:
    --------
    model : object
        Loaded model object
    """
    try:
        model = pickle.load(open(model_path, "rb"))
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def load_scaler(scaler_path):
    """
    Load a single scaler from joblib file

    Parameters:
    -----------
    scaler_path : str
        Path to the scaler file

    Returns:
    --------
    scaler : object
        Loaded scaler object
    """
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from {scaler_path}: {str(e)}")
        return None

# ============================================================================
# LOAD ALL MODELS AT ONCE
# ============================================================================

def load_models():
    """
    Load all 8 ML models and their scalers

    Returns:
    --------
    dict : Dictionary containing all models organized by type
    """

    models_dict = {
        'classification': {},
        'regression': {}
    }

    # ========================================================================
    # CLASSIFICATION MODELS
    # ========================================================================

    # Model 1: Logistic Regression
    models_dict['classification']['Logistic Regression (Base)'] = {
        'model': load_model(MODEL_PATHS['logistic_regression_base']),
        'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
    }

    models_dict['classification']['Logistic Regression (Tuned)'] = {
        'model': load_model(MODEL_PATHS['logistic_regression_tuned']),
        'scaler': load_scaler(SCALER_PATHS['logistic_regression_tuned_scaler'])
    }

    # Model 2: Random Forest Classifier
    models_dict['classification']['Random Forest Classifier (Base)'] = {
        'model': load_model(MODEL_PATHS['random_forest_classifier_base']),
        'scaler': load_scaler(SCALER_PATHS['random_forest_classifier_base_scaler'])
    }

    models_dict['classification']['Random Forest Classifier (Tuned)'] = {
        'model': load_model(MODEL_PATHS['random_forest_classifier_tuned']),
        'scaler': load_scaler(SCALER_PATHS['random_forest_classifier_tuned_scaler'])
    }

    # Model 3: XGBoost Classifier
    models_dict['classification']['XGBoost Classifier (Base)'] = {
        'model': load_model(MODEL_PATHS['xgboost_classifier_base']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
    }

    models_dict['classification']['🏆 XGBoost Classifier (Bayesian)'] = {
        'model': load_model(MODEL_PATHS['xgboost_classifier_bayesian']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_bayesian_scaler'])
    }

    # Model 7: Decision Tree Classifier
    models_dict['classification']['Decision Tree Classifier (Base)'] = {
        'model': load_model(MODEL_PATHS['decision_tree_classifier']),
        'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
    }

    models_dict['classification']['Decision Tree Classifier (Optimized)'] = {
        'model': load_model(MODEL_PATHS['decision_tree_classifier_optimized']),
        'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
    }

    # Model 8: Gradient Boosting Classifier
    models_dict['classification']['Gradient Boosting Classifier (Base)'] = {
        'model': load_model(MODEL_PATHS['gradient_boosting_classifier']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
    }

    models_dict['classification']['Gradient Boosting Classifier (Optimized)'] = {
        'model': load_model(MODEL_PATHS['gradient_boosting_classifier_optimized']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
    }

    # ========================================================================
    # REGRESSION MODELS
    # ========================================================================

    # Model 4: Linear Regression
    models_dict['regression']['Linear Regression (Base)'] = {
        'model': load_model(MODEL_PATHS['linear_regression_base']),
        'scaler': load_scaler(SCALER_PATHS['linear_regression_base_scaler'])
    }

    models_dict['regression']['Ridge Regression (Tuned)'] = {
        'model': load_model(MODEL_PATHS['linear_regression_tuned']),
        'scaler': load_scaler(SCALER_PATHS['linear_regression_tuned_scaler'])
    }

    # Model 5: Random Forest Regressor
    models_dict['regression']['Random Forest Regressor (Base)'] = {
        'model': load_model(MODEL_PATHS['random_forest_regressor_base']),
        'scaler': load_scaler(SCALER_PATHS['random_forest_regressor_base_scaler'])
    }

    models_dict['regression']['Random Forest Regressor (Tuned)'] = {
        'model': load_model(MODEL_PATHS['random_forest_regressor_tuned']),
        'scaler': load_scaler(SCALER_PATHS['random_forest_regressor_tuned_scaler'])
    }

    # Model 6: XGBoost Regressor
    models_dict['regression']['XGBoost Regressor (Base)'] = {
        'model': load_model(MODEL_PATHS['xgboost_regressor_base']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_regressor_base_scaler'])
    }

    models_dict['regression']['🏆 XGBoost Regressor (Bayesian)'] = {
        'model': load_model(MODEL_PATHS['xgboost_regressor_bayesian']),
        'scaler': load_scaler(SCALER_PATHS['xgboost_regressor_bayesian_scaler'])
    }

    return models_dict

# ============================================================================
# GET SPECIFIC MODELS
# ============================================================================

def get_best_classification_model(models_dict):
    """
    Get the best classification model (XGBoost Bayesian)

    Parameters:
    -----------
    models_dict : dict
        Dictionary containing all models

    Returns:
    --------
    tuple : (model, scaler)
    """
    best_model = models_dict['classification']['🏆 XGBoost Classifier (Bayesian)']
    return best_model['model'], best_model['scaler']

def get_best_regression_model(models_dict):
    """
    Get the best regression model (XGBoost Bayesian)

    Parameters:
    -----------
    models_dict : dict
        Dictionary containing all models

    Returns:
    --------
    tuple : (model, scaler)
    """
    best_model = models_dict['regression']['🏆 XGBoost Regressor (Bayesian)']
    return best_model['model'], best_model['scaler']

def get_classification_model_names(models_dict):
    """Get all classification model names"""
    return list(models_dict['classification'].keys())

def get_regression_model_names(models_dict):
    """Get all regression model names"""
    return list(models_dict['regression'].keys())

# ============================================================================
# VERIFY MODELS
# ============================================================================

def verify_models(models_dict):
    """
    Verify that all models loaded successfully

    Parameters:
    -----------
    models_dict : dict
        Dictionary containing all models

    Returns:
    --------
    bool : True if all models loaded, False otherwise
    """

    all_loaded = True

    # Check classification models
    for model_name, model_data in models_dict['classification'].items():
        if model_data['model'] is None or model_data['scaler'] is None:
            print(f"⚠️ Issue loading: {model_name}")
            all_loaded = False

    # Check regression models
    for model_name, model_data in models_dict['regression'].items():
        if model_data['model'] is None or model_data['scaler'] is None:
            print(f"⚠️ Issue loading: {model_name}")
            all_loaded = False

    if all_loaded:
        print("✓ All models loaded successfully!")

    return all_loaded
