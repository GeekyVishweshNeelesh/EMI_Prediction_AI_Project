"""
pages/regression.py - Maximum EMI Amount Prediction Page
"""

import streamlit as st
import numpy as np
import pandas as pd
from utils.predictions import predict_max_emi

def main(models=None):  # ← Added models parameter with default None
    st.title("💰 Maximum EMI Amount Prediction")
    st.markdown("**Predict the maximum EMI amount a customer can afford**")

    # Load models if not provided
    if models is None:
        from utils.model_loader import load_all_models
        models = load_all_models()

    regression_models = models.get('regression', {})

    if not regression_models:
        st.error("❌ No regression models available!")
        return

    # Model selection
    st.sidebar.header("⚙️ Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Choose Regression Model:",
        list(regression_models.keys())
    )

    # Get selected model and scaler
    model_data = regression_models[selected_model_name]
    model = model_data['model']
    scaler = model_data['scaler']

    # Input form (same as classification)
    st.header("📝 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])

    with col2:
        st.subheader("Employment Details")
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=50000)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business"])
        years_of_employment = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        company_type = st.selectbox("Company Type", ["Private", "Government", "MNC"])

    with col3:
        st.subheader("Housing Details")
        house_type = st.selectbox("House Type", ["Rented", "Owned", "Parental"])
        monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=10000)
        family_size = st.number_input("Family Size", min_value=1, max_value=20, value=4)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2)

    st.header("💰 Financial Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.subheader("Expenses")
        school_fees = st.number_input("School Fees (₹)", min_value=0, value=5000)
        college_fees = st.number_input("College Fees (₹)", min_value=0, value=0)
        travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=3000)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, value=8000)

    with col5:
        st.subheader("Other Details")
        other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", min_value=0, value=5000)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
        current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, value=0)

    with col6:
        st.subheader("Financial Health")
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=700)
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=100000)
        emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=50000)

    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    marital_encoded = {"Single": 0, "Married": 1, "Divorced": 2}[marital_status]
    education_encoded = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}[education]
    employment_encoded = {"Salaried": 0, "Self-Employed": 1, "Business": 2}[employment_type]
    company_encoded = {"Private": 0, "Government": 1, "MNC": 2}[company_type]
    house_encoded = {"Rented": 0, "Owned": 1, "Parental": 2}[house_type]

    # Create feature array
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

    # Convert to array
    input_array = np.array([customer_data[feat] for feat in feature_order])

    # Predict button
    if st.button("🔮 Predict Maximum EMI", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = predict_max_emi(model, scaler, input_array)

            if result.get('success', False):
                st.success("✅ Prediction completed successfully!")

                # Display result
                max_emi = result['prediction']
                st.metric("💰 Maximum EMI Amount", f"₹{max_emi:,.2f}")

                # Show affordability breakdown
                st.subheader("📊 Affordability Analysis")

                col_a1, col_a2, col_a3 = st.columns(3)

                with col_a1:
                    emi_to_income = (max_emi / monthly_salary) * 100 if monthly_salary > 0 else 0
                    st.metric("EMI to Income Ratio", f"{emi_to_income:.1f}%")

                with col_a2:
                    disposable = monthly_salary - max_emi
                    st.metric("Disposable Income", f"₹{disposable:,.2f}")

                with col_a3:
                    total_expenses = (school_fees + college_fees + travel_expenses +
                                    groceries_utilities + other_monthly_expenses)
                    remaining = disposable - total_expenses
                    st.metric("Remaining After Expenses", f"₹{remaining:,.2f}")

            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
