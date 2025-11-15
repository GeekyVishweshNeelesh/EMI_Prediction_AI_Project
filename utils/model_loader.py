"""
utils/model_loader.py - Load and manage all ML models and scalers

This module handles:
- Loading all ML models (including Random Forest)
- Loading corresponding scalers
- Caching for performance
- Error handling with better diagnostics
"""

import pickle
import joblib
import streamlit as st
from config import MODEL_PATHS, SCALER_PATHS

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_model(model_path):
    """Load a single model from pickle file"""
    try:
        model = pickle.load(open(model_path, "rb"))
        return model
    except Exception as e:
        st.warning(f"⚠️ Error loading model from {model_path}: {str(e)}")
        print(f"❌ Model loading failed: {model_path}")
        print(f"   Error: {str(e)}")
        return None

def load_scaler(scaler_path):
    """Load a single scaler from joblib file"""
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.warning(f"⚠️ Error loading scaler from {scaler_path}: {str(e)}")
        print(f"❌ Scaler loading failed: {scaler_path}")
        print(f"   Error: {str(e)}")
        return None

# ============================================================================
# LOAD ALL MODELS
# ============================================================================

def load_models():
    """
    Load ALL ML models and their scalers

    Returns:
    --------
    dict : Dictionary containing all models organized by type
    """

    models_dict = {
        'classification': {},
        'regression': {}
    }

    print("\n" + "="*80)
    print("LOADING ALL ML MODELS")
    print("="*80 + "\n")

    # ========================================================================
    # CLASSIFICATION MODELS
    # ========================================================================

    print("📊 LOADING CLASSIFICATION MODELS...\n")

    # Model 1: Logistic Regression
    print("1. Loading Logistic Regression...")
    try:
        models_dict['classification']['Logistic Regression (Base)'] = {
            'model': load_model(MODEL_PATHS['logistic_regression_base']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
        }
        print("   ✅ Logistic Regression (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['classification']['Logistic Regression (Tuned)'] = {
            'model': load_model(MODEL_PATHS['logistic_regression_tuned']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_tuned_scaler'])
        }
        print("   ✅ Logistic Regression (Tuned) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 2: Random Forest Classifier
    print("2. Loading Random Forest Classifier...")
    try:
        models_dict['classification']['Random Forest Classifier (Base)'] = {
            'model': load_model(MODEL_PATHS['random_forest_classifier_base']),
            'scaler': load_scaler(SCALER_PATHS['random_forest_classifier_base_scaler'])
        }
        print("   ✅ Random Forest Classifier (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['classification']['Random Forest Classifier (Tuned)'] = {
            'model': load_model(MODEL_PATHS['random_forest_classifier_tuned']),
            'scaler': load_scaler(SCALER_PATHS['random_forest_classifier_tuned_scaler'])
        }
        print("   ✅ Random Forest Classifier (Tuned) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 3: XGBoost Classifier
    print("3. Loading XGBoost Classifier...")
    try:
        models_dict['classification']['XGBoost Classifier (Base)'] = {
            'model': load_model(MODEL_PATHS['xgboost_classifier_base']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
        }
        print("   ✅ XGBoost Classifier (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['classification']['🏆 XGBoost Classifier (Bayesian)'] = {
            'model': load_model(MODEL_PATHS['xgboost_classifier_bayesian']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_bayesian_scaler'])
        }
        print("   ✅ XGBoost Classifier (Bayesian) - BEST - loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 7: Decision Tree Classifier
    print("4. Loading Decision Tree Classifier...")
    try:
        models_dict['classification']['Decision Tree Classifier (Base)'] = {
            'model': load_model(MODEL_PATHS['decision_tree_classifier']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
        }
        print("   ✅ Decision Tree Classifier (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['classification']['Decision Tree Classifier (Optimized)'] = {
            'model': load_model(MODEL_PATHS['decision_tree_classifier_optimized']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_base_scaler'])
        }
        print("   ✅ Decision Tree Classifier (Optimized) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 8: Gradient Boosting Classifier
    print("5. Loading Gradient Boosting Classifier...")
    try:
        models_dict['classification']['Gradient Boosting Classifier (Base)'] = {
            'model': load_model(MODEL_PATHS['gradient_boosting_classifier']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
        }
        print("   ✅ Gradient Boosting Classifier (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['classification']['Gradient Boosting Classifier (Optimized)'] = {
            'model': load_model(MODEL_PATHS['gradient_boosting_classifier_optimized']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_base_scaler'])
        }
        print("   ✅ Gradient Boosting Classifier (Optimized) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # ========================================================================
    # REGRESSION MODELS
    # ========================================================================

    print("💰 LOADING REGRESSION MODELS...\n")

    # Model 4: Linear Regression
    print("1. Loading Linear Regression...")
    try:
        models_dict['regression']['Linear Regression (Base)'] = {
            'model': load_model(MODEL_PATHS['linear_regression_base']),
            'scaler': load_scaler(SCALER_PATHS['linear_regression_base_scaler'])
        }
        print("   ✅ Linear Regression (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['regression']['Ridge Regression (Tuned)'] = {
            'model': load_model(MODEL_PATHS['linear_regression_tuned']),
            'scaler': load_scaler(SCALER_PATHS['linear_regression_tuned_scaler'])
        }
        print("   ✅ Ridge Regression (Tuned) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 5: Random Forest Regressor
    print("2. Loading Random Forest Regressor...")
    try:
        models_dict['regression']['Random Forest Regressor (Base)'] = {
            'model': load_model(MODEL_PATHS['random_forest_regressor_base']),
            'scaler': load_scaler(SCALER_PATHS['random_forest_regressor_base_scaler'])
        }
        print("   ✅ Random Forest Regressor (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['regression']['Random Forest Regressor (Tuned)'] = {
            'model': load_model(MODEL_PATHS['random_forest_regressor_tuned']),
            'scaler': load_scaler(SCALER_PATHS['random_forest_regressor_tuned_scaler'])
        }
        print("   ✅ Random Forest Regressor (Tuned) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    # Model 6: XGBoost Regressor
    print("3. Loading XGBoost Regressor...")
    try:
        models_dict['regression']['XGBoost Regressor (Base)'] = {
            'model': load_model(MODEL_PATHS['xgboost_regressor_base']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_regressor_base_scaler'])
        }
        print("   ✅ XGBoost Regressor (Base) loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    try:
        models_dict['regression']['🏆 XGBoost Regressor (Bayesian)'] = {
            'model': load_model(MODEL_PATHS['xgboost_regressor_bayesian']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_regressor_bayesian_scaler'])
        }
        print("   ✅ XGBoost Regressor (Bayesian) - BEST - loaded\n")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}\n")

    return models_dict

# ============================================================================
# GET SPECIFIC MODELS
# ============================================================================

def get_best_classification_model(models_dict):
    """Get the best classification model (XGBoost Bayesian)"""
    try:
        best_model = models_dict['classification']['🏆 XGBoost Classifier (Bayesian)']
        return best_model['model'], best_model['scaler']
    except KeyError:
        st.error("❌ Best classification model not available")
        return None, None

def get_best_regression_model(models_dict):
    """Get the best regression model (XGBoost Bayesian)"""
    try:
        best_model = models_dict['regression']['🏆 XGBoost Regressor (Bayesian)']
        return best_model['model'], best_model['scaler']
    except KeyError:
        st.error("❌ Best regression model not available")
        return None, None

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
    """Verify that all models loaded successfully"""

    all_loaded = True
    classification_count = 0
    regression_count = 0

    print("\n" + "="*80)
    print("MODEL LOADING VERIFICATION")
    print("="*80)

    # Check classification models
    print("\n✅ CLASSIFICATION MODELS:")
    for model_name, model_data in models_dict['classification'].items():
        if model_data['model'] is not None and model_data['scaler'] is not None:
            print(f"   ✅ {model_name}")
            classification_count += 1
        else:
            print(f"   ❌ {model_name}")
            all_loaded = False

    # Check regression models
    print("\n✅ REGRESSION MODELS:")
    for model_name, model_data in models_dict['regression'].items():
        if model_data['model'] is not None and model_data['scaler'] is not None:
            print(f"   ✅ {model_name}")
            regression_count += 1
        else:
            print(f"   ❌ {model_name}")
            all_loaded = False

    print("\n" + "="*80)

    if all_loaded:
        print(f"✅ ALL MODELS LOADED SUCCESSFULLY!")
        print(f"   Classification Models: {classification_count}")
        print(f"   Regression Models: {regression_count}")
        print(f"   Total Models: {classification_count + regression_count}")
    else:
        print("⚠️ Some models failed to load. Check errors above.")

    print("="*80 + "\n")

    return all_loaded
