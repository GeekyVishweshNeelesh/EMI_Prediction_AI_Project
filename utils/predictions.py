"""
utils/predictions.py - Make predictions using loaded models
UPDATED: Handles 25-feature scalers with 22 input features
"""

import numpy as np
import streamlit as st

def prepare_input_for_scaler(input_data):
    """
    Prepare input data by adding dummy features if needed

    Parameters:
    -----------
    input_data : np.array
        Input array with 22 features

    Returns:
    --------
    np.array : Array with correct number of features
    """
    # Check if we have 22 features
    if len(input_data) == 22:
        # Add 3 dummy features (zeros) to make it 25
        dummy_features = np.zeros(3)
        extended_input = np.concatenate([input_data, dummy_features])
        return extended_input
    else:
        return input_data

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
        # Convert to array and reshape
        input_array = np.array(input_data).reshape(1, -1)

        # Check if scaler expects more features
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
            current_features = input_array.shape[1]

            if current_features < expected_features:
                # Add dummy features
                num_dummy = expected_features - current_features
                dummy = np.zeros((1, num_dummy))
                input_array = np.concatenate([input_array, dummy], axis=1)
                st.info(f"ℹ️ Added {num_dummy} dummy features to match scaler expectations")

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Make prediction (use only first 22 features for model)
        if scaled_input.shape[1] > 22:
            scaled_input = scaled_input[:, :22]  # Take only first 22 features

        prediction = model.predict(scaled_input)[0]

        # Get probabilities if available
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
        # Convert to array and reshape
        input_array = np.array(input_data).reshape(1, -1)

        # Check if scaler expects more features
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
            current_features = input_array.shape[1]

            if current_features < expected_features:
                # Add dummy features
                num_dummy = expected_features - current_features
                dummy = np.zeros((1, num_dummy))
                input_array = np.concatenate([input_array, dummy], axis=1)
                st.info(f"ℹ️ Added {num_dummy} dummy features to match scaler expectations")

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Make prediction (use only first 22 features for model)
        if scaled_input.shape[1] > 22:
            scaled_input = scaled_input[:, :22]  # Take only first 22 features

        prediction = model.predict(scaled_input)[0]

        # Ensure non-negative prediction
        prediction = max(0, prediction)

        return {
            'success': True,
            'prediction': float(prediction)
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
