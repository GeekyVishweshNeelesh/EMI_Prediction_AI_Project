"""
pages/classification.py - EMI Eligibility Classification Page

This page handles:
- Input form for 22 customer features
- Real-time classification prediction
- Eligibility status display
- Confidence scores
- Probability breakdown
"""

import streamlit as st
import numpy as np
import pandas as pd
from config import (
    FEATURE_RANGES, CATEGORICAL_OPTIONS, FEATURE_NAMES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES
)
from utils.model_loader import get_best_classification_model
from utils.predictions import predict_emi_eligibility
from utils.visualizations import plot_eligibility_prediction, display_eligibility_status

def main(models):
    """Main function for classification page"""

    st.markdown("# 🎯 EMI Eligibility Classification")
    st.markdown("Predict customer EMI eligibility and risk level")

    st.markdown("---")

    # Get best model
    best_model, best_scaler = get_best_classification_model(models)

    if best_model is None or best_scaler is None:
        st.error("Classification model not loaded. Please refresh the page.")
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Upload"])

    with tab1:
        st.markdown("## Enter Customer Details")
        st.markdown("Fill in all fields to get eligibility prediction")

        # Create columns for better layout
        col1, col2 = st.columns(2)

        # Dictionary to store inputs
        customer_data = {}

        # ====================================================================
        # DEMOGRAPHIC INFORMATION
        # ====================================================================

        with st.expander("👤 Personal Demographics", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider("Age (Years)",
                               min_value=int(FEATURE_RANGES['age'][0]),
                               max_value=int(FEATURE_RANGES['age'][1]),
                               value=35)
                customer_data['age'] = age

                gender = st.radio("Gender",
                                 options=CATEGORICAL_OPTIONS['gender'])
                customer_data['gender'] = 1 if gender == 'Male' else 0

            with col2:
                marital_status = st.radio("Marital Status",
                                         options=CATEGORICAL_OPTIONS['marital_status'])
                customer_data['marital_status'] = 1 if marital_status == 'Married' else 0

                education = st.selectbox("Education",
                                        options=CATEGORICAL_OPTIONS['education'])
                education_map = {'High School': 0, 'Graduate': 1, 'Post Graduate': 2, 'Professional': 3}
                customer_data['education'] = education_map[education]

        # ====================================================================
        # EMPLOYMENT & INCOME
        # ====================================================================

        with st.expander("💼 Employment & Income", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                monthly_salary = st.number_input("Monthly Salary (₹)",
                                                 min_value=int(FEATURE_RANGES['monthly_salary'][0]),
                                                 max_value=int(FEATURE_RANGES['monthly_salary'][1]),
                                                 value=100000)
                customer_data['monthly_salary'] = monthly_salary

                employment_type = st.selectbox("Employment Type",
                                              options=CATEGORICAL_OPTIONS['employment_type'])
                emp_map = {'Private': 0, 'Government': 1, 'Self-employed': 2}
                customer_data['employment_type'] = emp_map[employment_type]

            with col2:
                years_of_employment = st.number_input("Years of Employment",
                                                       min_value=0,
                                                       max_value=30,
                                                       value=5)
                customer_data['years_of_employment'] = years_of_employment

                company_type = st.selectbox("Company Type",
                                           options=CATEGORICAL_OPTIONS['company_type'])
                company_map = {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Corporate': 4}
                customer_data['company_type'] = company_map[company_type]

        # ====================================================================
        # HOUSING & FAMILY
        # ====================================================================

        with st.expander("🏠 Housing & Family", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                house_type = st.selectbox("House Type",
                                         options=CATEGORICAL_OPTIONS['house_type'],
                                         key="house_type")
                house_map = {'Rented': 0, 'Own': 1, 'Family': 2}
                customer_data['house_type'] = house_map[house_type]

                monthly_rent = st.number_input("Monthly Rent (₹)",
                                              min_value=0,
                                              max_value=50000,
                                              value=10000)
                customer_data['monthly_rent'] = monthly_rent

            with col2:
                family_size = st.number_input("Family Size",
                                             min_value=1,
                                             max_value=8,
                                             value=4)
                customer_data['family_size'] = family_size

                dependents = st.number_input("Number of Dependents",
                                            min_value=0,
                                            max_value=5,
                                            value=1)
                customer_data['dependents'] = dependents

        # ====================================================================
        # FINANCIAL OBLIGATIONS
        # ====================================================================

        with st.expander("💸 Monthly Financial Obligations", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                school_fees = st.number_input("School Fees (₹)",
                                             min_value=0,
                                             max_value=50000,
                                             value=5000)
                customer_data['school_fees'] = school_fees

                college_fees = st.number_input("College Fees (₹)",
                                              min_value=0,
                                              max_value=100000,
                                              value=10000)
                customer_data['college_fees'] = college_fees

                travel_expenses = st.number_input("Travel Expenses (₹)",
                                                 min_value=0,
                                                 max_value=20000,
                                                 value=3000)
                customer_data['travel_expenses'] = travel_expenses

            with col2:
                groceries_utilities = st.number_input("Groceries & Utilities (₹)",
                                                      min_value=5000,
                                                      max_value=50000,
                                                      value=15000)
                customer_data['groceries_utilities'] = groceries_utilities

                other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)",
                                                        min_value=0,
                                                        max_value=30000,
                                                        value=5000)
                customer_data['other_monthly_expenses'] = other_monthly_expenses

        # ====================================================================
        # FINANCIAL STATUS & CREDIT
        # ====================================================================

        with st.expander("💰 Financial Status & Credit", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                existing_loans = st.radio("Existing Loans?",
                                         options=CATEGORICAL_OPTIONS['existing_loans'])
                customer_data['existing_loans'] = 1 if existing_loans == 'Yes' else 0

                current_emi_amount = st.number_input("Current EMI Amount (₹)",
                                                     min_value=0,
                                                     max_value=100000,
                                                     value=10000)
                customer_data['current_emi_amount'] = current_emi_amount

            with col2:
                credit_score = st.number_input("Credit Score",
                                              min_value=300,
                                              max_value=850,
                                              value=700)
                customer_data['credit_score'] = credit_score

                bank_balance = st.number_input("Bank Balance (₹)",
                                              min_value=0,
                                              max_value=1000000,
                                              value=200000)
                customer_data['bank_balance'] = bank_balance

            emergency_fund = st.number_input("Emergency Fund (₹)",
                                            min_value=0,
                                            max_value=500000,
                                            value=100000)
            customer_data['emergency_fund'] = emergency_fund

        st.markdown("---")

        # ====================================================================
        # MAKE PREDICTION
        # ====================================================================

        if st.button("🚀 Get EMI Eligibility Prediction", use_container_width=True, type="primary"):
            # Convert to array in correct order
            feature_order = [
                'age', 'gender', 'marital_status', 'education',
                'monthly_salary', 'employment_type', 'years_of_employment', 'company_type',
                'house_type', 'monthly_rent', 'family_size', 'dependents',
                'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
                'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
                'credit_score', 'bank_balance', 'emergency_fund'
            ]

            input_array = np.array([customer_data[feat] for feat in feature_order])

            # Make prediction
            result = predict_emi_eligibility(best_model, best_scaler, input_array)

            if result.get('success', False):
                st.success("✅ Prediction completed successfully!")

                # Display prediction
                col1, col2 = st.columns([1, 1])

                with col1:
                    display_eligibility_status(result['class_name'])

                with col2:
                    st.markdown("### 📊 Prediction Details")
                    st.metric("Confidence Score", f"{result['confidence']:.2f}%")
                    st.markdown(f"**Description:** {result['description']}")

                # Display visualizations
                st.markdown("---")
                plot_eligibility_prediction(result)

                # Display detailed probabilities
                st.markdown("---")
                st.markdown("### 📈 Detailed Probabilities")
                prob_data = pd.DataFrame({
                    'Class': ['Not Eligible', 'High Risk', 'Eligible'],
                    'Probability (%)': [
                        result['probabilities']['Not_Eligible'],
                        result['probabilities']['High_Risk'],
                        result['probabilities']['Eligible']
                    ]
                })
                st.dataframe(prob_data, use_container_width=True, hide_index=True)

            else:
                st.error(f"Error making prediction: {result.get('error', 'Unknown error')}")

    with tab2:
        st.markdown("## 📊 Batch Predictions")
        st.markdown("Upload a CSV file with customer data for batch predictions")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records")
                st.dataframe(df.head())

                if st.button("🚀 Run Batch Predictions"):
                    st.info("Batch prediction feature coming soon!")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
