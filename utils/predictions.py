"""
utils/predictions.py - Make predictions using loaded models
FIXED: Handles 25 features (22 base + 3 derived)
"""

import numpy as np
import streamlit as st

def create_25_features(input_data):
    """
    Convert 22 base features to 25 features by adding derived features

    Base 22 features:
    0. age
    1. gender
    2. marital_status
    3. education
    4. monthly_salary
    5. employment_type
    6. years_of_employment
    7. company_type
    8. house_type
    9. monthly_rent
    10. family_size
    11. dependents
    12. school_fees
    13. college_fees
    14. travel_expenses
    15. groceries_utilities
    16. other_monthly_expenses
    17. existing_loans
    18. current_emi_amount
    19. credit_score
    20. bank_balance
    21. emergency_fund

    Derived 3 features (added):
    22. total_monthly_expenses
    23. disposable_income
    24. debt_to_income_ratio

    Parameters:
    -----------
    input_data : array-like
        22 base features

    Returns:
    --------
    np.array : 25 features
    """

    # Extract base features
    age = input_data[0]
    gender = input_data[1]
    marital_status = input_data[2]
    education = input_data[3]
    monthly_salary = input_data[4]
    employment_type = input_data[5]
    years_of_employment = input_data[6]
    company_type = input_data[7]
    house_type = input_data[8]
    monthly_rent = input_data[9]
    family_size = input_data[10]
    dependents = input_data[11]
    school_fees = input_data[12]
    college_fees = input_data[13]
    travel_expenses = input_data[14]
    groceries_utilities = input_data[15]
    other_monthly_expenses = input_data[16]
    existing_loans = input_data[17]
    current_emi_amount = input_data[18]
    credit_score = input_data[19]
    bank_balance = input_data[20]
    emergency_fund = input_data[21]

    # Calculate derived features
    # Feature 23: Total monthly expenses
    total_monthly_expenses = (
        monthly_rent +
        school_fees +
        college_fees +
        travel_expenses +
        groceries_utilities +
        other_monthly_expenses +
        current_emi_amount
    )

    # Feature 24: Disposable income (salary - expenses)
    disposable_income = monthly_salary - total_monthly_expenses

    # Feature 25: Debt to income ratio
    debt_to_income_ratio = (current_emi_amount / monthly_salary) if monthly_salary > 0 else 0

    # Combine all 25 features
    features_25 = np.array([
        # Original 22 features (indices 0-21)
        age,
        gender,
        marital_status,
        education,
        monthly_salary,
        employment_type,
        years_of_employment,
        company_type,
        house_type,
        monthly_rent,
        family_size,
        dependents,
        school_fees,
        college_fees,
        travel_expenses,
        groceries_utilities,
        other_monthly_expenses,
        existing_loans,
        current_emi_amount,
        credit_score,
        bank_balance,
        emergency_fund,
        # Derived 3 features (indices 22-24)
        total_monthly_expenses,   # Feature 22
        disposable_income,         # Feature 23
        debt_to_income_ratio       # Feature 24
    ])

    return features_25


def predict_emi_eligibility(model, scaler, input_data):
    """
    Predict EMI eligibility using classification model

    Parameters:
    -----------
    model : object
        Trained classification model
    scaler : object
        Scaler for input features (expects 25 features)
    input_data : array-like
        Input features (22 base features)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """
    try:
        # Convert 22 features to 25 features
        features_25 = create_25_features(input_data)

        # Reshape for sklearn
        input_array = features_25.reshape(1, -1)

        # Debug: Check feature count
        if input_array.shape[1] != 25:
            return {
                'success': False,
                'error': f'Feature engineering failed. Expected 25 features, got {input_array.shape[1]}'
            }

        # Scale the input
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Get probabilities if model supports it
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_input)[0]
            probability = probabilities[prediction]
        else:
            probabilities = [0.5, 0.5]
            probability = 0.5

        return {
            'success': True,
            'prediction': int(prediction),
            'probability': float(probability),
            'probabilities': [float(p) for p in probabilities]
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        return {
            'success': False,
            'error': str(e),
            'details': error_details
        }


def predict_max_emi(model, scaler, input_data):
    """
    Predict maximum EMI amount using regression model

    Parameters:
    -----------
    model : object
        Trained regression model
    scaler : object
        Scaler for input features (expects 25 features)
    input_data : array-like
        Input features (22 base features)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """
    try:
        # Convert 22 features to 25 features
        features_25 = create_25_features(input_data)

        # Reshape for sklearn
        input_array = features_25.reshape(1, -1)

        # Debug: Check feature count
        if input_array.shape[1] != 25:
            return {
                'success': False,
                'error': f'Feature engineering failed. Expected 25 features, got {input_array.shape[1]}'
            }

        # Scale the input
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Ensure non-negative EMI amount
        prediction = max(0, prediction)

        return {
            'success': True,
            'prediction': float(prediction)
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()

        return {
            'success': False,
            'error': str(e),
            'details': error_details
        }
