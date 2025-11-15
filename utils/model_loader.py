"""
utils/model_loader.py - Load only WORKING models

DELETED FILES (corrupted by Git LFS):
- 02_random_forest_classifier_base.pkl
- 02_random_forest_classifier_tuned.pkl
- 05_random_forest_regressor_base.pkl
- 05_random_forest_regressor_tuned.pkl
"""

import pickle
import joblib
import streamlit as st
from config import MODEL_PATHS, SCALER_PATHS

def load_model(model_path):
    try:
        model = pickle.load(open(model_path, "rb"))
        return model
    except Exception as e:
        print(f"⚠️ Error: {model_path} - {str(e)}")
        return None

def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"⚠️ Error: {scaler_path} - {str(e)}")
        return None

def load_models():
    """Load ONLY working models (skip corrupted Random Forest files)"""

    models_dict = {
        'classification': {},
        'regression': {}
    }

    # ✅ CLASSIFICATION MODELS

    # Model 1: Logistic Regression
    try:
        models_dict['classification']['Logistic Regression (Tuned)'] = {
            'model': load_model(MODEL_PATHS['logistic_regression_tuned']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_tuned_scaler'])
        }
        print("✅ Logistic Regression loaded")
    except:
        pass

    # ⚠️ Model 2: Random Forest Classifier - DELETED - SKIPPED
    print("⏭️ Skipped: Random Forest Classifier (files deleted)")

    # Model 3: XGBoost Classifier (BEST)
    try:
        models_dict['classification']['🏆 XGBoost Classifier (Bayesian)'] = {
            'model': load_model(MODEL_PATHS['xgboost_classifier_bayesian']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_bayesian_scaler'])
        }
        print("✅ XGBoost Classifier (BEST) loaded")
    except:
        pass

    # Model 7: Decision Tree
    try:
        models_dict['classification']['Decision Tree Classifier (Optimized)'] = {
            'model': load_model(MODEL_PATHS['decision_tree_classifier_optimized']),
            'scaler': load_scaler(SCALER_PATHS['logistic_regression_tuned_scaler'])
        }
        print("✅ Decision Tree Classifier loaded")
    except:
        pass

    # Model 8: Gradient Boosting
    try:
        models_dict['classification']['Gradient Boosting Classifier (Optimized)'] = {
            'model': load_model(MODEL_PATHS['gradient_boosting_classifier_optimized']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_classifier_bayesian_scaler'])
        }
        print("✅ Gradient Boosting Classifier loaded")
    except:
        pass

    # ✅ REGRESSION MODELS

    # Model 4: Ridge Regression
    try:
        models_dict['regression']['Ridge Regression (Tuned)'] = {
            'model': load_model(MODEL_PATHS['linear_regression_tuned']),
            'scaler': load_scaler(SCALER_PATHS['linear_regression_tuned_scaler'])
        }
        print("✅ Ridge Regression loaded")
    except:
        pass

    # ⚠️ Model 5: Random Forest Regressor - DELETED - SKIPPED
    print("⏭️ Skipped: Random Forest Regressor (files deleted)")

    # Model 6: XGBoost Regressor (BEST)
    try:
        models_dict['regression']['🏆 XGBoost Regressor (Bayesian)'] = {
            'model': load_model(MODEL_PATHS['xgboost_regressor_bayesian']),
            'scaler': load_scaler(SCALER_PATHS['xgboost_regressor_bayesian_scaler'])
        }
        print("✅ XGBoost Regressor (BEST) loaded")
    except:
        pass

    return models_dict

def get_best_classification_model(models_dict):
    try:
        best_model = models_dict['classification']['🏆 XGBoost Classifier (Bayesian)']
        return best_model['model'], best_model['scaler']
    except KeyError:
        st.error("❌ Best classification model not available")
        return None, None

def get_best_regression_model(models_dict):
    try:
        best_model = models_dict['regression']['🏆 XGBoost Regressor (Bayesian)']
        return best_model['model'], best_model['scaler']
    except KeyError:
        st.error("❌ Best regression model not available")
        return None, None

def get_classification_model_names(models_dict):
    return list(models_dict['classification'].keys())

def get_regression_model_names(models_dict):
    return list(models_dict['regression'].keys())

def verify_models(models_dict):
    print("\n✅ LOADED MODELS:")
    for name in get_classification_model_names(models_dict):
        print(f"  ✅ {name}")
    for name in get_regression_model_names(models_dict):
        print(f"  ✅ {name}")
    return True
