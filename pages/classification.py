"""
pages/classification.py - EMI Eligibility Classification Page
"""

import streamlit as st
import numpy as np
import pandas as pd
from utils.predictions import predict_emi_eligibility

def main(models=None):
    st.title("🎯 EMI Eligibility Classification")
    st.markdown("**Predict whether a customer is eligible for an EMI loan**")

    # ========================================================================
    # LOAD MODELS - Always load fresh to avoid issues
    # ========================================================================

    from utils.model_loader import load_all_models

    with st.spinner("Loading classification models..."):
        models = load_all_models()

    classification_models = models.get('classification', {})

    # Check if models are available
    if not classification_models:
        st.error("❌ No classification models available!")
        st.warning("💡 Please ensure model files exist in 'saved_models/' folder")

        # Show debug information
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

    # Get selected model and scaler
    model_data = classification_models[selected_model_name]
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
        if st.button("🔮 Predict EMI Eligibility", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
        result = predict_emi_eligibility(model, scaler, input_array)

        if result.get('success', False):
            st.success("✅ Prediction completed successfully!")

            # Display results in columns
            col_res1, col_res2 = st.columns(2)

            with col_res1:
                eligibility = result['prediction']
                if eligibility == 1:
                    st.success("### ✅ ELIGIBLE for EMI")
                else:
                    st.error("### ❌ NOT ELIGIBLE for EMI")

            with col_res2:
                confidence = result['probability'] * 100
                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%",
                    help="Model's confidence in this prediction"
                )

            # Show detailed probabilities
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")

            # Safely get probabilities
            probabilities = result.get('probabilities', [0.5, 0.5])

            # Ensure we have exactly 2 probabilities
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

            # Create DataFrame with guaranteed 2 values
            prob_df = pd.DataFrame({
                'Eligibility': ['Not Eligible', 'Eligible'],
                'Probability': [prob_not_eligible, prob_eligible]
            })

            # Display as bar chart
            st.bar_chart(prob_df.set_index('Eligibility'))

            # Display as table
            st.dataframe(prob_df, use_container_width=True)

            # Additional insights
            st.markdown("---")
            st.subheader("💡 Financial Insights")

            total_monthly_expenses = (
                school_fees + college_fees + travel_expenses +
                groceries_utilities + other_monthly_expenses + current_emi_amount
            )

            disposable_income = monthly_salary - total_monthly_expenses

            col_insight1, col_insight2, col_insight3 = st.columns(3)

            with col_insight1:
                st.metric("Total Monthly Expenses", f"₹{total_monthly_expenses:,.0f}")

            with col_insight2:
                st.metric("Disposable Income", f"₹{disposable_income:,.0f}")

            with col_insight3:
                expense_ratio = (total_monthly_expenses / monthly_salary * 100) if monthly_salary > 0 else 0
                st.metric("Expense Ratio", f"{expense_ratio:.1f}%")

        else:
            st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

            # Show detailed error if available
            if 'details' in result:
                with st.expander("🔍 Error Details"):
                    st.code(result['details'])

            st.info("Please check your input values and try again.")

               # Ensure probabilities has exactly 2 values
                probabilities = result.get('probabilities', [0.5, 0.5])
                if len(probabilities) != 2:
                    probabilities = [probabilities[0], 1 - probabilities[0]] if len(probabilities) == 1 else [0.5, 0.5]

                prob_df = pd.DataFrame({
                    'Eligibility': ['Not Eligible', 'Eligible'],
                    'Probability': probabilities
                })

                # Display as bar chart
                st.bar_chart(prob_df.set_index('Eligibility'))

                # Display as table
                st.dataframe(prob_df, use_container_width=True)

                # Additional insights
                st.markdown("---")
                st.subheader("💡 Financial Insights")

                total_monthly_expenses = (
                    school_fees + college_fees + travel_expenses +
                    groceries_utilities + other_monthly_expenses + current_emi_amount
                )

                disposable_income = monthly_salary - total_monthly_expenses

                col_insight1, col_insight2, col_insight3 = st.columns(3)

                with col_insight1:
                    st.metric("Total Monthly Expenses", f"₹{total_monthly_expenses:,.0f}")

                with col_insight2:
                    st.metric("Disposable Income", f"₹{disposable_income:,.0f}")

                with col_insight3:
                    expense_ratio = (total_monthly_expenses / monthly_salary * 100) if monthly_salary > 0 else 0
                    st.metric("Expense Ratio", f"{expense_ratio:.1f}%")

            else:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
