"""
config.py - Configuration Settings for EMIPredict AI Application

This file contains all configuration settings and constants used across the application
"""

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_CONFIG = {
    "app_name": "EMIPredict AI",
    "app_version": "1.0.0",
    "description": "Intelligent Financial Risk Assessment Platform",
    "author": "FinTech Development Team",
    "year": 2024
}

# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    # Classification Models
    "logistic_regression_base": "/saved_models/01_logistic_regression_base.pkl",
    "logistic_regression_tuned": "/saved_models/01_logistic_regression_tuned.pkl",
    "random_forest_classifier_base": "/saved_models/02_random_forest_classifier_base.pkl",
    "random_forest_classifier_tuned": "/saved_models/02_random_forest_classifier_tuned.pkl",
    "xgboost_classifier_base": "/saved_models/03_xgboost_classifier_base.pkl",
    "xgboost_classifier_bayesian": "/saved_models/03_xgboost_classifier_bayesian.pkl",
    "decision_tree_classifier": "/saved_models/07_decision_tree_classifier_base.pkl",
    "decision_tree_classifier_optimized": "/saved_models/07_decision_tree_classifier_optimized.pkl",
    "gradient_boosting_classifier": "//saved_models/08_gradient_boosting_classifier_base.pkl",
    "gradient_boosting_classifier_optimized": "/saved_models/08_gradient_boosting_classifier_optimized.pkl",

    # Regression Models
    "linear_regression_base": "/saved_models/04_linear_regression_base.pkl",
    "linear_regression_tuned": "/saved_models/04_ridge_regression_tuned.pkl",
    "random_forest_regressor_base": "/saved_models/05_random_forest_regressor_base.pkl",
    "random_forest_regressor_tuned": "/saved_models/05_random_forest_regressor_tuned.pkl",
    "xgboost_regressor_base": "/saved_models/06_xgboost_regressor_base.pkl",
    "xgboost_regressor_bayesian": "/saved_models/06_xgboost_regressor_bayesian.pkl",
}

SCALER_PATHS = {
    # Classification Scalers
    "logistic_regression_base_scaler": "/saved_models/01_logistic_regression_base_scaler.pkl",
    "logistic_regression_tuned_scaler": "/saved_models/01_logistic_regression_tuned_scaler.pkl",
    "random_forest_classifier_base_scaler": "/saved_models/02_random_forest_classifier_base_scaler.pkl",
    "random_forest_classifier_tuned_scaler": "/saved_models/02_random_forest_classifier_tuned_scaler.pkl",
    "xgboost_classifier_base_scaler": "/saved_models/03_xgboost_classifier_base_scaler.pkl",
    "xgboost_classifier_bayesian_scaler": "/saved_models/03_xgboost_classifier_bayesian_scaler.pkl",

    # Regression Scalers
    "linear_regression_base_scaler": "/saved_models/04_linear_regression_base_scaler.pkl",
    "linear_regression_tuned_scaler": "/saved_models/04_ridge_regression_tuned_scaler.pkl",
    "random_forest_regressor_base_scaler": "/saved_models/05_random_forest_regressor_base_scaler.pkl",
    "random_forest_regressor_tuned_scaler": "/saved_models/05_random_forest_regressor_tuned_scaler.pkl",
    "xgboost_regressor_base_scaler": "/saved_models/06_xgboost_regressor_base_scaler.pkl",
    "xgboost_regressor_bayesian_scaler": "/saved_models/06_xgboost_regressor_bayesian_scaler.pkl",
}

# ============================================================================
# FEATURE NAMES AND TYPES
# ============================================================================

FEATURE_NAMES = [
    'age', 'gender', 'marital_status', 'education',
    'monthly_salary', 'employment_type', 'years_of_employment', 'company_type',
    'house_type', 'monthly_rent', 'family_size', 'dependents',
    'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
    'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
    'credit_score', 'bank_balance', 'emergency_fund'
]

NUMERICAL_FEATURES = [
    'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
    'family_size', 'dependents', 'school_fees', 'college_fees',
    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
    'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund'
]

CATEGORICAL_FEATURES = [
    'gender', 'marital_status', 'education', 'employment_type',
    'company_type', 'house_type', 'existing_loans'
]

# ============================================================================
# FEATURE RANGES AND DEFAULTS
# ============================================================================

FEATURE_RANGES = {
    'age': (25, 60),
    'monthly_salary': (15000, 200000),
    'years_of_employment': (0, 30),
    'monthly_rent': (0, 50000),
    'family_size': (1, 8),
    'dependents': (0, 5),
    'school_fees': (0, 50000),
    'college_fees': (0, 100000),
    'travel_expenses': (0, 20000),
    'groceries_utilities': (5000, 50000),
    'other_monthly_expenses': (0, 30000),
    'current_emi_amount': (0, 100000),
    'credit_score': (300, 850),
    'bank_balance': (0, 1000000),
    'emergency_fund': (0, 500000),
}

# ============================================================================
# CATEGORICAL OPTIONS
# ============================================================================

CATEGORICAL_OPTIONS = {
    'gender': ['Male', 'Female'],
    'marital_status': ['Single', 'Married'],
    'education': ['High School', 'Graduate', 'Post Graduate', 'Professional'],
    'employment_type': ['Private', 'Government', 'Self-employed'],
    'company_type': ['Startup', 'Small', 'Medium', 'Large', 'Corporate'],
    'house_type': ['Rented', 'Own', 'Family'],
    'existing_loans': ['Yes', 'No']
}

# ============================================================================
# EMI SCENARIOS
# ============================================================================

EMI_SCENARIOS = {
    'E-commerce Shopping': {'min': 10000, 'max': 200000, 'tenure_min': 3, 'tenure_max': 24},
    'Home Appliances': {'min': 20000, 'max': 300000, 'tenure_min': 6, 'tenure_max': 36},
    'Vehicle': {'min': 80000, 'max': 1500000, 'tenure_min': 12, 'tenure_max': 84},
    'Personal Loan': {'min': 50000, 'max': 1000000, 'tenure_min': 12, 'tenure_max': 60},
    'Education': {'min': 50000, 'max': 500000, 'tenure_min': 6, 'tenure_max': 48}
}

# ============================================================================
# MODEL PERFORMANCE METRICS
# ============================================================================

MODEL_PERFORMANCE = {
    "Classification": {
        "Model 1: Logistic Regression (Base)": {
            "Accuracy": 0.8881,
            "Precision": 0.8546,
            "Recall": 0.8881,
            "F1-Score": 0.8683,
            "ROC-AUC": 0.9518
        },
        "Model 2: Random Forest (Base)": {
            "Accuracy": 0.9098,
            "Precision": 0.9082,
            "Recall": 0.9098,
            "F1-Score": 0.9082,
            "ROC-AUC": 0.9790
        },
        "Model 3: XGBoost (Base)": {
            "Accuracy": 0.9286,
            "Precision": 0.8982,
            "Recall": 0.9286,
            "F1-Score": 0.9080,
            "ROC-AUC": 0.9863
        },
        "Model 7: Decision Tree (Base)": {
            "Accuracy": 0.8708,
            "Precision": 0.8439,
            "Recall": 0.8708,
            "F1-Score": 0.8549,
            "ROC-AUC": 0.9201
        },
        "Model 8: Gradient Boosting (Base)": {
            "Accuracy": 0.9142,
            "Precision": 0.8740,
            "Recall": 0.9142,
            "F1-Score": 0.8929,
            "ROC-AUC": 0.9786
        },
        "🏆 Model 3: XGBoost (Bayesian)": {
            "Accuracy": 0.9590,
            "Precision": 0.9527,
            "Recall": 0.9590,
            "F1-Score": 0.9517,
            "ROC-AUC": 0.9962
        }
    },
    "Regression": {
        "Model 4: Linear Regression (Base)": {
            "RMSE": 4288.96,
            "MAE": 3095.10,
            "R²": 0.6884,
            "MAPE": 193.91
        },
        "Model 5: Random Forest (Base)": {
            "RMSE": 1903.47,
            "MAE": 1186.86,
            "R²": 0.9386,
            "MAPE": 46.19
        },
        "Model 6: XGBoost (Base)": {
            "RMSE": 1560.28,
            "MAE": 1049.22,
            "R²": 0.9588,
            "MAPE": 54.33
        },
        "🏆 Model 6: XGBoost (Bayesian)": {
            "RMSE": 973.08,
            "MAE": 542.19,
            "R²": 0.9840,
            "MAPE": 28.60
        }
    }
}

# ============================================================================
# ELIGIBILITY CLASSES
# ============================================================================

ELIGIBILITY_CLASSES = {
    0: {'name': 'Not_Eligible', 'description': 'High risk, loan not recommended', 'color': 'red'},
    1: {'name': 'High_Risk', 'description': 'Marginal case, requires higher interest rates', 'color': 'orange'},
    2: {'name': 'Eligible', 'description': 'Low risk, comfortable EMI affordability', 'color': 'green'}
}

# ============================================================================
# COLOR SCHEME
# ============================================================================

COLORS = {
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',
    'primary': '#007bff',
    'secondary': '#6c757d',
}

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

DISPLAY_CONFIG = {
    'font_size': 16,
    'chart_height': 500,
    'chart_width': 800,
    'decimals': 4,
    'currency_format': '₹',
}

# ============================================================================
# MESSAGES
# ============================================================================

MESSAGES = {
    'welcome': 'Welcome to EMIPredict AI - Intelligent Financial Risk Assessment Platform',
    'select_page': 'Select a page from the sidebar to get started',
    'loading_models': 'Loading ML models...',
    'models_loaded': 'Models loaded successfully!',
    'prediction_success': 'Prediction completed successfully!',
    'invalid_input': 'Please check your input values',
    'error': 'An error occurred. Please try again.',
}
