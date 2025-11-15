"""
pages/regression.py - Maximum EMI Amount Prediction Page

This page handles:
- Input form for 22 customer features
- Real-time maximum EMI prediction
- EMI breakdown visualization
- Affordability analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
from config import (
    FEATURE_RANGES, CATEGORICAL_OPTIONS, EMI_SCENARIOS
)
from utils.model_loader import get_best_regression_model
from utils.predictions import predict_max_emi, calculate_emi_breakdown
from utils.visualizations import plot_max_emi_prediction, plot_emi_breakdown_chart

def main(models):
    """Main function for regression page"""

    st.markdown("# 💸 Maximum EMI Amount Prediction")
    st.markdown("Calculate the maximum safe EMI amount for customers")

    st.markdown("---")

    # Get best model
    best_model, best_scaler = get_best_regression_model(models)

    if best_model is None or best_scaler is None:
        st.error("Regression model not loaded. Please refresh the page.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📊 Batch Upload"])

    with tab1:
        st.markdown("## Enter Customer Details")

        # Dictionary to store inputs
        customer_data = {}

        # ====================================================================
        # DEMOGRAPHIC INFORMATION
        # ====================================================================

        with st.expander("👤 Personal Demographics", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                age = st.slider("Age (Years)",
                               min_value=25, max_value=60, value=35)
                customer_data['age'] = age

                gender = st.radio("Gender", options=['Male', 'Female'])
                customer_data['gender'] = 1 if gender == 'Male' else 0

            with col2:
                marital_status = st.radio("Marital Status",
                                         options=['Single', 'Married'])
                customer_data['marital_status'] = 1 if marital_status == 'Married' else 0

                education = st.selectbox("Education",
                                        options=['High School', 'Graduate', 'Post Graduate', 'Professional'])
                education_map = {'High School': 0, 'Graduate': 1, 'Post Graduate': 2, 'Professional': 3}
                customer_data['education'] = education_map[education]

        # ====================================================================
        # EMPLOYMENT & INCOME
        # ====================================================================

        with st.expander("💼 Employment & Income", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                monthly_salary = st.number_input("Monthly Salary (₹)",
                                                 min_value=15000, max_value=200000, value=100000)
                customer_data['monthly_salary'] = monthly_salary

                employment_type = st.selectbox("Employment Type",
                                              options=['Private', 'Government', 'Self-employed'])
                emp_map = {'Private': 0, 'Government': 1, 'Self-employed': 2}
                customer_data['employment_type'] = emp_map[employment_type]

            with col2:
                years_of_employment = st.number_input("Years of Employment",
                                                       min_value=0, max_value=30, value=5)
                customer_data['years_of_employment'] = years_of_employment

                company_type = st.selectbox("Company Type",
                                           options=['Startup', 'Small', 'Medium', 'Large', 'Corporate'])
                company_map = {'Startup': 0, 'Small': 1, 'Medium': 2, 'Large': 3, 'Corporate': 4}
                customer_data['company_type'] = company_map[company_type]

        # ====================================================================
        # HOUSING & FAMILY
        # ====================================================================

        with st.expander("🏠 Housing & Family", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                house_type = st.selectbox("House Type",
                                         options=['Rented', 'Own', 'Family'],
                                         key="house_type_reg")
                house_map = {'Rented': 0, 'Own': 1, 'Family': 2}
                customer_data['house_type'] = house_map[house_type]

                monthly_rent = st.number_input("Monthly Rent (₹)",
                                              min_value=0, max_value=50000, value=10000)
                customer_data['monthly_rent'] = monthly_rent

            with col2:
                family_size = st.number_input("Family Size",
                                             min_value=1, max_value=8, value=4)
                customer_data['family_size'] = family_size

                dependents = st.number_input("Number of Dependents",
                                            min_value=0, max_value=5, value=1)
                customer_data['dependents'] = dependents

        # ====================================================================
        # FINANCIAL OBLIGATIONS
        # ====================================================================

        with st.expander("💸 Monthly Financial Obligations", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                school_fees = st.number_input("School Fees (₹)",
                                             min_value=0, max_value=50000, value=5000)
                customer_data['school_fees'] = school_fees

                college_fees = st.number_input("College Fees (₹)",
                                              min_value=0, max_value=100000, value=10000)
                customer_data['college_fees'] = college_fees

                travel_expenses = st.number_input("Travel Expenses (₹)",
                                                 min_value=0, max_value=20000, value=3000)
                customer_data['travel_expenses'] = travel_expenses

            with col2:
                groceries_utilities = st.number_input("Groceries & Utilities (₹)",
                                                      min_value=5000, max_value=50000, value=15000)
                customer_data['groceries_utilities'] = groceries_utilities

                other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)",
                                                        min_value=0, max_value=30000, value=5000)
                customer_data['other_monthly_expenses'] = other_monthly_expenses

        # ====================================================================
        # FINANCIAL STATUS & CREDIT
        # ====================================================================

        with st.expander("💰 Financial Status & Credit", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                existing_loans = st.radio("Existing Loans?",
                                         options=['Yes', 'No'])
                customer_data['existing_loans'] = 1 if existing_loans == 'Yes' else 0

                current_emi_amount = st.number_input("Current EMI Amount (₹)",
                                                     min_value=0, max_value=100000, value=10000)
                customer_data['current_emi_amount'] = current_emi_amount

            with col2:
                credit_score = st.number_input("Credit Score",
                                              min_value=300, max_value=850, value=700)
                customer_data['credit_score'] = credit_score

                bank_balance = st.number_input("Bank Balance (₹)",
                                              min_value=0, max_value=1000000, value=200000)
                customer_data['bank_balance'] = bank_balance

            emergency_fund = st.number_input("Emergency Fund (₹)",
                                            min_value=0, max_value=500000, value=100000)
            customer_data['emergency_fund'] = emergency_fund

        st.markdown("---")

        # ====================================================================
        # EMI SCENARIO SELECTION
        # ====================================================================

        st.markdown("## 🎁 EMI Scenario")

        scenario = st.selectbox("Select EMI Scenario",
                               options=list(EMI_SCENARIOS.keys()))

        scenario_info = EMI_SCENARIOS[scenario]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Amount Range", f"₹{scenario_info['min']:,} - ₹{scenario_info['max']:,}")

        with col2:
            st.metric("Tenure Range", f"{scenario_info['tenure_min']} - {scenario_info['tenure_max']} months")

        with col3:
            st.metric("Scenario", scenario)

        st.markdown("---")

        # ====================================================================
        # MAKE PREDICTION
        # ====================================================================

        if st.button("🚀 Calculate Maximum Safe EMI", use_container_width=True, type="primary"):
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

        # VALIDATE: Must be exactly 22 features
        if not validate_features(input_array):
            st.error("❌ Cannot make prediction - feature count mismatch")
        else:
            # Make prediction
            result = predict_max_emi(best_model, best_scaler, input_array)

            if result.get('success', False):
                st.success("✅ Prediction completed successfully!")

                # Display main prediction
                max_emi = result['max_emi']

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 💰 Maximum Safe EMI")
                    st.markdown(f"""
                    <div style='background-color: #007bff; color: white; padding: 30px; border-radius: 10px; text-align: center;'>
                        <h1 style='margin: 0;'>{result['formatted_emi']}</h1>
                        <p style='margin: 10px 0 0 0;'>Monthly Payment Capacity</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown("### ✅ Affordability Check")

                    monthly_income = customer_data['monthly_salary']
                    emi_to_income = (max_emi / monthly_income) * 100

                    if emi_to_income < 40:
                        status = "✅ Affordable"
                        color = "green"
                    elif emi_to_income < 50:
                        status = "⚠️ Manageable"
                        color = "orange"
                    else:
                        status = "❌ High Burden"
                        color = "red"

                    st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 20px; border-radius: 10px;'>
                        <h3 style='margin: 0;'>{status}</h3>
                        <p style='margin: 10px 0 0 0;'>EMI/Income Ratio: {emi_to_income:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Display visualizations
                st.markdown("---")
                plot_max_emi_prediction(result)

                # EMI Breakdown
                st.markdown("---")
                st.markdown("### 📋 EMI Breakdown Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    selected_tenure = st.selectbox("Select Tenure for Breakdown",
                                                  options=range(scenario_info['tenure_min'],
                                                              scenario_info['tenure_max'] + 1, 6),
                                                  value=scenario_info['tenure_min'])

                with col2:
                    interest_rate = st.slider("Interest Rate (%)",
                                             min_value=5, max_value=20, value=12)

                # Calculate breakdown
                breakdown = calculate_emi_breakdown(max_emi, selected_tenure, interest_rate)

                if breakdown:
                    st.markdown(f"""
                    #### 💹 Loan Breakdown for {selected_tenure} Months

                    - **Principal Amount:** ₹{breakdown['principal']:,.2f}
                    - **Monthly EMI:** ₹{max_emi:,.2f}
                    - **Total Amount Paid:** ₹{breakdown['total_amount']:,.2f}
                    - **Total Interest:** ₹{breakdown['total_interest']:,.2f}
                    - **Interest Rate:** {breakdown['interest_rate']:.2f}% p.a.
                    """)

                # Tenure comparison
                st.markdown("---")
                st.markdown("### 📊 EMI Across Different Tenures")

                tenures = list(range(scenario_info['tenure_min'],
                                    scenario_info['tenure_max'] + 1, 6))

                plot_emi_breakdown_chart(max_emi, tenures)

                # Summary
                st.markdown("---")
                st.markdown("### 📌 Summary")

                summary_data = {
                    'Metric': ['Monthly Salary', 'Maximum EMI', 'EMI to Income Ratio', 'Current EMI', 'Remaining Capacity'],
                    'Value': [
                        f"₹{customer_data['monthly_salary']:,.0f}",
                        f"₹{max_emi:,.2f}",
                        f"{emi_to_income:.1f}%",
                        f"₹{customer_data['current_emi_amount']:,.0f}",
                        f"₹{max_emi - customer_data['current_emi_amount']:,.2f}"
                    ]
                }

                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            else:
                st.error(f"Error making prediction: {result.get('error', 'Unknown error')}")

    with tab2:
        st.markdown("## 📊 Batch Predictions")
        st.markdown("Upload a CSV file with customer data for batch predictions")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv", key="regression_batch")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} records")
                st.dataframe(df.head())

                if st.button("🚀 Run Batch Predictions", key="batch_reg_button"):
                    st.info("Batch prediction feature coming soon!")

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
