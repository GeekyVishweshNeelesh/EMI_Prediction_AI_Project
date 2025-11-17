"""
pages/classification.py - EMI Eligibility Classification Page
UPDATED: Uses 25 features (22 base + 3 derived)
"""

import streamlit as st
import numpy as np
import pandas as pd
from utils.predictions import predict_emi_eligibility

def main(models=None):
    st.title("🎯 EMI Eligibility Classification")
    st.markdown("**Predict whether a customer is eligible for an EMI loan**")

    # ========================================================================
    # LOAD MODELS
    # ========================================================================

    from utils.model_loader import load_all_models

    with st.spinner("Loading classification models..."):
        models = load_all_models()

    classification_models = models.get('classification', {})

    if not classification_models:
        st.error("❌ No classification models available!")
        st.warning("💡 Please ensure model files exist in 'saved_models/' folder")

        with st.expander("🔍 Click here for detailed diagnostic information"):
            from utils.model_loader import show_debug_info
            show_debug_info()

        st.info("""
        **Expected files in saved_models/ folder:**
        - 01_logistic_regression_base.pkl
        - 01_logistic_regression_base_scaler.pkl
        - 03_xgboost_classifier_bayesian.pkl
        - 03_xgboost_classifier_bayesian_scaler.pkl
        - (and other model files...)
        """)
        return

    # ========================================================================
    # MODEL SELECTION
    # ========================================================================

    st.sidebar.header("⚙️ Model Selection")
    st.sidebar.success(f"✅ {len(classification_models)} models loaded")

    selected_model_name = st.sidebar.selectbox(
        "Choose Classification Model:",
        list(classification_models.keys()),
        help="Select a machine learning model for prediction"
    )

    model_data = classification_models[selected_model_name]
    model = model_data['model']
    scaler = model_data['scaler']

    st.sidebar.info(f"**Active Model:** {selected_model_name}")

    # ========================================================================
    # INPUT FORM - Personal Details
    # ========================================================================

    st.header("📝 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])

    with col2:
        st.subheader("💼 Employment Details")
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, max_value=1000000, value=50000, step=5000)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        company_type = st.selectbox("Company Type", ["Private", "Government", "MNC"])

    with col3:
        st.subheader("🏠 Housing Details")
        house_type = st.selectbox("House Type", ["Rented", "Owned", "Parental"])
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, max_value=100000, value=10000, step=1000)
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=4)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)

    # ========================================================================
    # INPUT FORM - Financial Information
    # ========================================================================

    st.header("💰 Financial Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("📊 Monthly Expenses")
        school_fees = st.number_input("School Fees (₹)", min_value=0, max_value=100000, value=5000, step=500)
        college_fees = st.number_input("College Fees (₹)", min_value=0, max_value=200000, value=0, step=1000)
        travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, max_value=50000, value=3000, step=500)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, max_value=100000, value=8000, step=500)

    with col5:
        st.subheader("💳 Loan Details")
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, max_value=100000, value=5000, step=500)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, max_value=200000, value=0, step=1000)

    with col6:
        st.subheader("💎 Financial Health")
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0, max_value=10000000, value=100000, step=10000)
        emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, max_value=5000000, value=50000, step=5000)

    # ========================================================================
    # DATA PREPARATION - 22 Base Features
    # ========================================================================

    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    marital_encoded = {"Single": 0, "Married": 1, "Divorced": 2}[marital_status]
    education_encoded = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}[education]
    employment_encoded = {"Salaried": 0, "Self-Employed": 1, "Business": 2}[employment_type]
    company_encoded = {"Private": 0, "Government": 1, "MNC": 2}[company_type]
    house_encoded = {"Rented": 0, "Owned": 1, "Parental": 2}[house_type]

    # Create 22 base features in correct order
    feature_order = [
        'age', 'gender', 'marital_status', 'education',
        'monthly_salary', 'employment_type', 'years_of_employment', 'company_type',
        'house_type', 'monthly_rent', 'family_size', 'dependents',
        'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
        'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
        'credit_score', 'bank_balance', 'emergency_fund'
    ]

    customer_data = {
        'age': age,
        'gender': gender_encoded,
        'marital_status': marital_encoded,
        'education': education_encoded,
        'monthly_salary': monthly_salary,
        'employment_type': employment_encoded,
        'years_of_employment': years_of_employment,
        'company_type': company_encoded,
        'house_type': house_encoded,
        'monthly_rent': monthly_rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'emergency_fund': emergency_fund
    }

    # Convert to numpy array (22 base features)
    # The predictions.py will convert this to 25 features by adding:
    # - Feature 23: total_monthly_expenses
    # - Feature 24: disposable_income
    # - Feature 25: debt_to_income_ratio
    input_array = np.array([customer_data[feat] for feat in feature_order])

    # ========================================================================
    # PREDICTION
    # ========================================================================

    st.markdown("---")

    # Show input summary
    with st.expander("📋 View Input Summary (22 Base Features)"):
        summary_df = pd.DataFrame({
            'Feature': feature_order,
            'Value': input_array
        })
        st.dataframe(summary_df, use_container_width=True)

        st.info("""
        **Note:** The model uses 25 features total:
        - 22 base features (shown above)
        - 3 derived features (calculated automatically):
          * total_monthly_expenses
          * disposable_income
          * debt_to_income_ratio
        """)

    # Predict button
    if st.button("🔮 Predict EMI Eligibility", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            # predict_emi_eligibility will convert 22 features to 25 features
            result = predict_emi_eligibility(model, scaler, input_array)

        if result.get('success', False):
            st.success("✅ Prediction completed successfully!")

            # Display results
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                eligibility = result['prediction']
                if eligibility == 1:
                    st.success("### ✅ ELIGIBLE for EMI")
                else:
                    st.error("### ❌ NOT ELIGIBLE for EMI")

            with col_res2:
                confidence = result['probability'] * 100
                st.metric("Prediction Confidence", f"{confidence:.2f}%", help="Model's confidence in this prediction")

            # Show detailed probabilities
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")

            # Safely extract probabilities
            probabilities = result.get('probabilities', [0.5, 0.5])

            if isinstance(probabilities, (list, tuple, np.ndarray)):
                if len(probabilities) == 2:
                    prob_not_eligible = float(probabilities[0])
                    prob_eligible = float(probabilities[1])
                elif len(probabilities) == 1:
                    prob_eligible = float(probabilities[0])
                    prob_not_eligible = 1.0 - prob_eligible
                else:
                    prob_not_eligible = 0.5
                    prob_eligible = 0.5
            else:
                prob_not_eligible = 0.5
                prob_eligible = 0.5

            # Create probability DataFrame
            prob_df = pd.DataFrame({
                'Eligibility': ['Not Eligible', 'Eligible'],
                'Probability': [prob_not_eligible, prob_eligible]
            })

            # Display chart and table
            st.bar_chart(prob_df.set_index('Eligibility'))
            st.dataframe(prob_df, use_container_width=True)

            # Financial insights
            st.markdown("---")
            st.subheader("💡 Financial Insights")

            total_monthly_expenses = (
                monthly_rent +
                school_fees +
                college_fees +
                travel_expenses +
                groceries_utilities +
                other_monthly_expenses +
                current_emi_amount
            )

            disposable_income = monthly_salary - total_monthly_expenses
            debt_to_income_ratio = (current_emi_amount / monthly_salary * 100) if monthly_salary > 0 else 0

            col_insight1, col_insight2, col_insight3, col_insight4 = st.columns(4)

            with col_insight1:
                st.metric("Total Monthly Expenses", f"₹{total_monthly_expenses:,.0f}")

            with col_insight2:
                st.metric("Disposable Income", f"₹{disposable_income:,.0f}")

            with col_insight3:
                expense_ratio = (total_monthly_expenses / monthly_salary * 100) if monthly_salary > 0 else 0
                st.metric("Expense Ratio", f"{expense_ratio:.1f}%")

            with col_insight4:
                st.metric("Debt-to-Income Ratio", f"{debt_to_income_ratio:.1f}%")

            # Show derived features
            with st.expander("🔍 View Derived Features (3 additional)"):
                derived_df = pd.DataFrame({
                    'Derived Feature': [
                        'total_monthly_expenses',
                        'disposable_income',
                        'debt_to_income_ratio'
                    ],
                    'Value': [
                        total_monthly_expenses,
                        disposable_income,
                        debt_to_income_ratio
                    ],
                    'Description': [
                        'Sum of all monthly expenses including rent and current EMI',
                        'Monthly salary minus total expenses',
                        'Current EMI as percentage of monthly salary'
                    ]
                })
                st.dataframe(derived_df, use_container_width=True)

        else:
            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

            if 'details' in result:
                with st.expander("🔍 Error Details"):
                    st.code(result['details'])

            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
