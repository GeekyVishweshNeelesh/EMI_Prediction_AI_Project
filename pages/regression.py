"""
pages/regression.py - Maximum EMI Amount Prediction Page
"""

import streamlit as st
import numpy as np
import pandas as pd
from utils.predictions import predict_max_emi

def main(models=None):
    st.title("💰 Maximum EMI Amount Prediction")
    st.markdown("**Predict the maximum EMI amount a customer can afford**")

    # ========================================================================
    # LOAD MODELS - Always load fresh to avoid issues
    # ========================================================================

    from utils.model_loader import load_all_models

    with st.spinner("Loading regression models..."):
        models = load_all_models()

    regression_models = models.get('regression', {})

    # Check if models are available
    if not regression_models:
        st.error("❌ No regression models available!")
        st.warning("💡 Please ensure model files exist in 'saved_models/' folder")

        # Show debug information
        with st.expander("🔍 Click here for detailed diagnostic information"):
            from utils.model_loader import show_debug_info
            show_debug_info()

        st.info("""
        **Expected files in saved_models/ folder:**
        - 04_linear_regression_base.pkl
        - 04_linear_regression_base_scaler.pkl
        - 06_xgboost_regressor_bayesian.pkl
        - 06_xgboost_regressor_bayesian_scaler.pkl
        - (and other model files...)
        """)
        return

    # ========================================================================
    # MODEL SELECTION
    # ========================================================================

    st.sidebar.header("⚙️ Model Selection")
    st.sidebar.success(f"✅ {len(regression_models)} models loaded")

    selected_model_name = st.sidebar.selectbox(
        "Choose Regression Model:",
        list(regression_models.keys()),
        help="Select a machine learning model for prediction"
    )

    # Get selected model and scaler
    model_data = regression_models[selected_model_name]
    model = model_data['model']
    scaler = model_data['scaler']

    st.sidebar.info(f"**Active Model:** {selected_model_name}")

    # ========================================================================
    # INPUT FORM - Customer Information
    # ========================================================================

    st.header("📝 Customer Information")

    col1, col2, col3 = st.columns(3)

    # Column 1: Personal Details
    with col1:
        st.subheader("👤 Personal Details")
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            help="Customer's age in years"
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )
        education = st.selectbox(
            "Education",
            ["High School", "Bachelor", "Master", "PhD"]
        )

    # Column 2: Employment Details
    with col2:
        st.subheader("💼 Employment Details")
        monthly_salary = st.number_input(
            "Monthly Salary (₹)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=5000,
            help="Monthly salary in Indian Rupees"
        )
        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-Employed", "Business"]
        )
        years_of_employment = st.number_input(
            "Years of Employment",
            min_value=0,
            max_value=50,
            value=5,
            help="Total years of work experience"
        )
        company_type = st.selectbox(
            "Company Type",
            ["Private", "Government", "MNC"]
        )

    # Column 3: Housing Details
    with col3:
        st.subheader("🏠 Housing Details")
        house_type = st.selectbox(
            "House Type",
            ["Rented", "Owned", "Parental"]
        )
        monthly_rent = st.number_input(
            "Monthly Rent (₹)",
            min_value=0,
            max_value=100000,
            value=10000,
            step=1000,
            help="Monthly rent if applicable"
        )
        family_size = st.number_input(
            "Family Size",
            min_value=1,
            max_value=20,
            value=4,
            help="Total number of family members"
        )
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=2,
            help="Number of dependent family members"
        )

    # ========================================================================
    # FINANCIAL INFORMATION
    # ========================================================================

    st.header("💰 Financial Information")

    col4, col5, col6 = st.columns(3)

    # Column 4: Monthly Expenses
    with col4:
        st.subheader("📊 Monthly Expenses")
        school_fees = st.number_input(
            "School Fees (₹)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500
        )
        college_fees = st.number_input(
            "College Fees (₹)",
            min_value=0,
            max_value=200000,
            value=0,
            step=1000
        )
        travel_expenses = st.number_input(
            "Travel Expenses (₹)",
            min_value=0,
            max_value=50000,
            value=3000,
            step=500
        )
        groceries_utilities = st.number_input(
            "Groceries & Utilities (₹)",
            min_value=0,
            max_value=100000,
            value=8000,
            step=500
        )

    # Column 5: Other Financial Details
    with col5:
        st.subheader("💳 Loan Details")
        other_monthly_expenses = st.number_input(
            "Other Monthly Expenses (₹)",
            min_value=0,
            max_value=100000,
            value=5000,
            step=500
        )
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of existing loans"
        )
        current_emi_amount = st.number_input(
            "Current EMI Amount (₹)",
            min_value=0,
            max_value=200000,
            value=0,
            step=1000,
            help="Total current EMI payments"
        )

    # Column 6: Financial Health
    with col6:
        st.subheader("💎 Financial Health")
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700,
            help="CIBIL/Credit score"
        )
        bank_balance = st.number_input(
            "Bank Balance (₹)",
            min_value=0,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Current bank account balance"
        )
        emergency_fund = st.number_input(
            "Emergency Fund (₹)",
            min_value=0,
            max_value=5000000,
            value=50000,
            step=5000,
            help="Emergency savings"
        )

    # ========================================================================
    # DATA PREPARATION
    # ========================================================================

    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0
    marital_encoded = {"Single": 0, "Married": 1, "Divorced": 2}[marital_status]
    education_encoded = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}[education]
    employment_encoded = {"Salaried": 0, "Self-Employed": 1, "Business": 2}[employment_type]
    company_encoded = {"Private": 0, "Government": 1, "MNC": 2}[company_type]
    house_encoded = {"Rented": 0, "Owned": 1, "Parental": 2}[house_type]

    # Create feature array (22 features in correct order)
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

    # Convert to array in correct order
    input_array = np.array([customer_data[feat] for feat in feature_order])

    # ========================================================================
    # PREDICTION
    # ========================================================================

    st.markdown("---")

    # Show input summary
    with st.expander("📋 View Input Summary"):
        summary_df = pd.DataFrame({
            'Feature': feature_order,
            'Value': input_array
        })
        st.dataframe(summary_df, use_container_width=True)

    # Predict button
    if st.button("🔮 Predict Maximum EMI", type="primary", use_container_width=True):
        with st.spinner("Making prediction..."):
            result = predict_max_emi(model, scaler, input_array)

            if result.get('success', False):
                st.success("✅ Prediction completed successfully!")

                # Get prediction
                max_emi = result['prediction']

                # Display main result
                st.markdown("---")
                st.subheader("💰 Prediction Result")

                # Big metric display
                col_main = st.columns([1, 2, 1])
                with col_main[1]:
                    st.metric(
                        label="Maximum Affordable EMI",
                        value=f"₹{max_emi:,.0f}",
                        help="Maximum EMI amount customer can afford per month"
                    )

                # Affordability Analysis
                st.markdown("---")
                st.subheader("📊 Affordability Analysis")

                col_a1, col_a2, col_a3 = st.columns(3)

                with col_a1:
                    emi_to_income = (max_emi / monthly_salary * 100) if monthly_salary > 0 else 0
                    st.metric(
                        "EMI to Income Ratio",
                        f"{emi_to_income:.1f}%",
                        help="Percentage of income going to EMI"
                    )

                with col_a2:
                    disposable = monthly_salary - max_emi
                    st.metric(
                        "Disposable Income After EMI",
                        f"₹{disposable:,.0f}",
                        help="Income remaining after EMI payment"
                    )

                with col_a3:
                    total_expenses = (
                        school_fees + college_fees + travel_expenses +
                        groceries_utilities + other_monthly_expenses + current_emi_amount
                    )
                    remaining = disposable - total_expenses

                    st.metric(
                        "Surplus After All Expenses",
                        f"₹{remaining:,.0f}",
                        delta=f"{'Surplus' if remaining > 0 else 'Deficit'}",
                        help="Amount left after EMI and all expenses"
                    )

                # Financial Breakdown
                st.markdown("---")
                st.subheader("💡 Financial Breakdown")

                breakdown_data = {
                    'Category': [
                        'Monthly Salary',
                        'Predicted Max EMI',
                        'Current EMI',
                        'School Fees',
                        'College Fees',
                        'Travel Expenses',
                        'Groceries & Utilities',
                        'Other Expenses',
                        'Total Expenses',
                        'Net Remaining'
                    ],
                    'Amount (₹)': [
                        monthly_salary,
                        max_emi,
                        current_emi_amount,
                        school_fees,
                        college_fees,
                        travel_expenses,
                        groceries_utilities,
                        other_monthly_expenses,
                        total_expenses + max_emi,
                        monthly_salary - (total_expenses + max_emi)
                    ]
                }

                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, use_container_width=True)

                # Recommendation
                st.markdown("---")
                st.subheader("✅ Recommendation")

                if emi_to_income <= 40:
                    st.success(f"""
                    **EMI is well within affordable range!**
                    - Your EMI to income ratio is {emi_to_income:.1f}%, which is healthy (ideally <40%)
                    - You have sufficient disposable income of ₹{disposable:,.0f}
                    - Consider this as a safe EMI amount
                    """)
                elif emi_to_income <= 50:
                    st.warning(f"""
                    **EMI is at the upper limit of affordability**
                    - Your EMI to income ratio is {emi_to_income:.1f}% (moderate risk)
                    - Disposable income: ₹{disposable:,.0f}
                    - Consider reducing other expenses or opting for a slightly lower EMI
                    """)
                else:
                    st.error(f"""
                    **EMI might be too high for your income**
                    - Your EMI to income ratio is {emi_to_income:.1f}% (high risk)
                    - This leaves limited disposable income
                    - Recommended: Reduce loan amount or extend tenure to lower EMI
                    """)

            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
