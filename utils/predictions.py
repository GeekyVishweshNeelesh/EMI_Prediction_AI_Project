"""
utils/predictions.py - Make predictions using loaded models

This module handles:
- Preparing input data for models
- Making classification predictions
- Making regression predictions
- Formatting prediction results
"""

import numpy as np
import pandas as pd
import streamlit as st
from config import ELIGIBILITY_CLASSES

# ============================================================================
# CLASSIFICATION PREDICTIONS
# ============================================================================

def prepare_classification_input(input_data, scaler):
    """
    Prepare input data for classification model

    Parameters:
    -----------
    input_data : np.array or pd.DataFrame
        Input features for prediction
    scaler : object
        Scaler object to scale features

    Returns:
    --------
    np.array : Scaled input data
    """
    try:
        # Scale the input
        scaled_input = scaler.transform([input_data])
        return scaled_input
    except Exception as e:
        st.error(f"Error preparing classification input: {str(e)}")
        return None

def predict_emi_eligibility(model, scaler, input_data):
    """
    Predict EMI eligibility using classification model

    Parameters:
    -----------
    model : object
        Trained classification model
    scaler : object
        Scaler for input features
    input_data : np.array
        Input features (22 dimensions)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """

    try:
        # Prepare input
        scaled_input = prepare_classification_input(input_data, scaler)

        if scaled_input is None:
            return {
                'success': False,
                'error': 'Error preparing input'
            }

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        # Get prediction details
        eligibility_class = ELIGIBILITY_CLASSES[prediction]

        # Calculate confidence
        confidence = max(probabilities) * 100

        # Prepare result
        result = {
            'success': True,
            'prediction': prediction,
            'class_name': eligibility_class['name'],
            'description': eligibility_class['description'],
            'color': eligibility_class['color'],
            'confidence': confidence,
            'probabilities': {
                'Not_Eligible': probabilities[0] * 100,
                'High_Risk': probabilities[1] * 100,
                'Eligible': probabilities[2] * 100
            }
        }

        return result

    except Exception as e:
        st.error(f"Error making classification prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# REGRESSION PREDICTIONS
# ============================================================================

def prepare_regression_input(input_data, scaler):
    """
    Prepare input data for regression model

    Parameters:
    -----------
    input_data : np.array or pd.DataFrame
        Input features for prediction
    scaler : object
        Scaler object to scale features

    Returns:
    --------
    np.array : Scaled input data
    """
    try:
        # Scale the input
        scaled_input = scaler.transform([input_data])
        return scaled_input
    except Exception as e:
        st.error(f"Error preparing regression input: {str(e)}")
        return None

def predict_max_emi(model, scaler, input_data):
    """
    Predict maximum EMI amount using regression model

    Parameters:
    -----------
    model : object
        Trained regression model
    scaler : object
        Scaler for input features
    input_data : np.array
        Input features (22 dimensions)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """

    try:
        # Prepare input
        scaled_input = prepare_regression_input(input_data, scaler)

        if scaled_input is None:
            return {
                'success': False,
                'error': 'Error preparing input'
            }

        # Make prediction
        max_emi = model.predict(scaled_input)[0]

        # Ensure non-negative prediction
        max_emi = max(0, max_emi)

        # Prepare result
        result = {
            'success': True,
            'max_emi': max_emi,
            'formatted_emi': f"₹{max_emi:,.2f}",
            'prediction_range': {
                'low': max_emi * 0.8,
                'high': max_emi * 1.2
            }
        }

        return result

    except Exception as e:
        st.error(f"Error making regression prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# ============================================================================
# BATCH PREDICTIONS
# ============================================================================

def batch_predict_eligibility(model, scaler, data_df):
    """
    Make predictions for multiple customers

    Parameters:
    -----------
    model : object
        Trained classification model
    scaler : object
        Scaler object
    data_df : pd.DataFrame
        DataFrame with multiple customers

    Returns:
    --------
    pd.DataFrame : DataFrame with predictions added
    """

    try:
        # Prepare data
        X_scaled = scaler.transform(data_df)

        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        # Add to dataframe
        data_df['prediction'] = predictions
        data_df['class'] = data_df['prediction'].map(
            lambda x: ELIGIBILITY_CLASSES[x]['name']
        )
        data_df['confidence'] = np.max(probabilities, axis=1) * 100

        return data_df

    except Exception as e:
        st.error(f"Error making batch predictions: {str(e)}")
        return None

def batch_predict_emi(model, scaler, data_df):
    """
    Make EMI predictions for multiple customers

    Parameters:
    -----------
    model : object
        Trained regression model
    scaler : object
        Scaler object
    data_df : pd.DataFrame
        DataFrame with multiple customers

    Returns:
    --------
    pd.DataFrame : DataFrame with predictions added
    """

    try:
        # Prepare data
        X_scaled = scaler.transform(data_df)

        # Make predictions
        predictions = model.predict(X_scaled)

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        # Add to dataframe
        data_df['predicted_max_emi'] = predictions
        data_df['formatted_emi'] = data_df['predicted_max_emi'].apply(
            lambda x: f"₹{x:,.2f}"
        )

        return data_df

    except Exception as e:
        st.error(f"Error making batch EMI predictions: {str(e)}")
        return None

# ============================================================================
# EMI BREAKDOWN
# ============================================================================

def calculate_emi_breakdown(max_emi, tenure_months, interest_rate=12):
    """
    Calculate EMI breakdown for different tenures

    Parameters:
    -----------
    max_emi : float
        Maximum EMI amount
    tenure_months : int
        Tenure in months
    interest_rate : float
        Annual interest rate (%)

    Returns:
    --------
    dict : EMI breakdown details
    """

    try:
        # Convert annual rate to monthly
        monthly_rate = interest_rate / 100 / 12

        # Calculate principal using EMI formula
        # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        # P = EMI * ((1+r)^n - 1) / (r * (1+r)^n)

        if monthly_rate > 0:
            denominator = (1 + monthly_rate) ** tenure_months
            principal = max_emi * (denominator - 1) / (monthly_rate * denominator)
        else:
            principal = max_emi * tenure_months

        # Calculate total amount
        total_amount = max_emi * tenure_months

        # Calculate interest
        total_interest = total_amount - principal

        breakdown = {
            'max_emi': max_emi,
            'tenure_months': tenure_months,
            'principal': principal,
            'total_amount': total_amount,
            'total_interest': total_interest,
            'interest_rate': interest_rate,
            'monthly_rate': monthly_rate * 100
        }

        return breakdown

    except Exception as e:
        st.error(f"Error calculating EMI breakdown: {str(e)}")
        return None

# ============================================================================
# VALIDATION
# ============================================================================

def validate_prediction(prediction_result):
    """
    Validate prediction result

    Parameters:
    -----------
    prediction_result : dict
        Prediction result from model

    Returns:
    --------
    bool : True if valid, False otherwise
    """

    if prediction_result.get('success', False):
        return True
    else:
        return False
