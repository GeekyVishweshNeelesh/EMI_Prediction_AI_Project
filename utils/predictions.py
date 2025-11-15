"""
utils/predictions.py - Make predictions using loaded models
"""

import numpy as np
import streamlit as st

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
        Input features (22 features)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """
    try:
        # Reshape input
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Get probabilities
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
        return {
            'success': False,
            'error': str(e)
        }

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
        Input features (22 features)

    Returns:
    --------
    dict : Dictionary containing prediction results
    """
    try:
        # Reshape input
        input_array = np.array(input_data).reshape(1, -1)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        return {
            'success': True,
            'prediction': float(max(0, prediction))  # Ensure non-negative
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
