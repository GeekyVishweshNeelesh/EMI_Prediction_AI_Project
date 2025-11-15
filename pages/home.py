"""
pages/home.py - Home Dashboard Page

This page displays:
- Project overview
- Key statistics
- Quick links
- Project information
"""

import streamlit as st
import pandas as pd
from config import APP_CONFIG, MODEL_PERFORMANCE

def main():
    """Main function for home page"""

    # Page title
    st.markdown("# 🏠 Welcome to EMIPredict AI")

    # Introduction
    st.markdown("""
    ## 💰 Intelligent Financial Risk Assessment Platform

    EMIPredict AI is a comprehensive machine learning platform designed to help financial institutions,
    banks, and FinTech companies make data-driven decisions for EMI (Equated Monthly Installment)
    predictions and risk assessment.
    """)

    st.markdown("---")

    # Key Statistics
    st.markdown("## 📊 Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📈 Total Records", "400,000", "Customers Analyzed")

    with col2:
        st.metric("🔧 Features", "22", "Input Variables")

    with col3:
        st.metric("🤖 ML Models", "8", "Trained Models")

    with col4:
        st.metric("🎯 Predictions", "Dual Task", "Classification + Regression")

    st.markdown("---")

    # Best Models Summary
    st.markdown("## 🏆 Best Performing Models")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Classification")
        st.markdown("""
        **Model:** XGBoost Classifier (Bayesian Optimized)

        - **Accuracy:** 95.90% ✅
        - **Precision:** 95.27%
        - **Recall:** 95.90%
        - **F1-Score:** 95.17%
        - **ROC-AUC:** 0.9962 🎯

        **Task:** Predict EMI Eligibility (Eligible, High Risk, Not Eligible)
        """)

    with col2:
        st.markdown("### 💸 Regression")
        st.markdown("""
        **Model:** XGBoost Regressor (Bayesian Optimized)

        - **RMSE:** ₹973.08 ✅
        - **MAE:** ₹542.19
        - **R² Score:** 0.9840 🎯
        - **MAPE:** 28.60%

        **Task:** Predict Maximum Safe EMI Amount
        """)

    st.markdown("---")

    # Features & Capabilities
    st.markdown("## ✨ Key Features")

    features_data = {
        "Feature": [
            "🎯 Real-time Predictions",
            "📊 Multiple Models",
            "🔍 Model Comparison",
            "📈 Data Visualization",
            "⚙️ Admin Panel",
            "💾 Batch Processing"
        ],
        "Description": [
            "Get instant EMI eligibility and maximum amount predictions",
            "Compare 8 different ML models side by side",
            "Analyze performance metrics across all models",
            "Interactive charts and visualizations",
            "Manage customer data and run batch predictions",
            "Process multiple customers at once"
        ]
    }

    features_df = pd.DataFrame(features_data)
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # EMI Scenarios
    st.markdown("## 🎁 EMI Scenarios Supported")

    scenarios_data = {
        "Scenario": [
            "E-commerce Shopping",
            "Home Appliances",
            "Vehicle",
            "Personal Loan",
            "Education"
        ],
        "Amount Range": [
            "₹10K - ₹200K",
            "₹20K - ₹300K",
            "₹80K - ₹1500K",
            "₹50K - ₹1000K",
            "₹50K - ₹500K"
        ],
        "Tenure Range": [
            "3-24 months",
            "6-36 months",
            "12-84 months",
            "12-60 months",
            "6-48 months"
        ],
        "Records": [
            "80,000",
            "80,000",
            "80,000",
            "80,000",
            "80,000"
        ]
    }

    scenarios_df = pd.DataFrame(scenarios_data)
    st.dataframe(scenarios_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Use Cases
    st.markdown("## 💼 Business Use Cases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏦 Financial Institutions")
        st.markdown("""
        - Automate loan approval processes
        - Reduce manual underwriting time by 80%
        - Implement risk-based pricing strategies
        - Real-time eligibility assessment
        """)

        st.markdown("### 💳 FinTech Companies")
        st.markdown("""
        - Instant EMI eligibility checks
        - Digital lending platform integration
        - Pre-qualification services
        - Automated risk scoring
        """)

    with col2:
        st.markdown("### 🏛️ Banks & Credit Agencies")
        st.markdown("""
        - Data-driven loan recommendations
        - Portfolio risk management
        - Default prediction
        - Regulatory compliance
        """)

        st.markdown("### 👔 Loan Officers")
        st.markdown("""
        - AI-powered recommendations
        - Financial profile analysis
        - Historical performance tracking
        - Model accuracy monitoring
        """)

    st.markdown("---")

    # How to Use
    st.markdown("## 🚀 How to Use")

    st.markdown("""
    ### Step 1: Classification (EMI Eligibility)
    Navigate to **🎯 EMI Classification** page to:
    - Enter customer financial details
    - Get eligibility prediction (Eligible/High Risk/Not Eligible)
    - View confidence scores and probability breakdown

    ### Step 2: Regression (Maximum EMI)
    Navigate to **💸 EMI Amount Prediction** page to:
    - Enter customer information
    - Get maximum safe EMI amount
    - View EMI breakdown for different tenures

    ### Step 3: Model Comparison
    Navigate to **📊 Model Comparison** page to:
    - Compare all 8 models
    - Analyze performance metrics
    - Understand model strengths and weaknesses

    ### Step 4: Admin Panel
    Navigate to **⚙️ Admin Panel** page to:
    - Upload customer data
    - Run batch predictions
    - Download results
    - Manage datasets
    """)

    st.markdown("---")

    # Technology Stack
    st.markdown("## 🔧 Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Machine Learning")
        st.markdown("""
        - XGBoost
        - Random Forest
        - Logistic Regression
        - Gradient Boosting
        - Decision Trees
        """)

    with col2:
        st.markdown("### Optimization")
        st.markdown("""
        - Bayesian Optimization
        - GridSearchCV
        - RandomizedSearchCV
        - Optuna TPE Sampler
        - Hyperparameter Tuning
        """)

    with col3:
        st.markdown("### Frameworks")
        st.markdown("""
        - Streamlit (Frontend)
        - Scikit-learn (ML)
        - Pandas (Data)
        - NumPy (Computation)
        - Matplotlib (Visualization)
        """)

    st.markdown("---")

    # Getting Started
    st.markdown("## ⚡ Getting Started")

    st.info("""
    **👉 Quick Start Guide:**

    1. Select a page from the navigation menu on the left
    2. For **Classification**: Enter customer details to predict EMI eligibility
    3. For **Regression**: Get the maximum safe EMI amount
    4. For **Model Comparison**: See how all 8 models perform
    5. For **Admin Panel**: Manage data and run batch predictions

    **Need Help?**
    - Hover over fields for detailed descriptions
    - Check the sidebar for model performance info
    - All predictions include confidence scores
    """)

    st.markdown("---")

    # Footer with stats
    st.markdown("## 📈 Performance Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Classification Metrics")
        st.markdown(f"""
        - Best Accuracy: **95.90%**
        - Best ROC-AUC: **0.9962**
        - Best F1-Score: **0.9517**
        - Average Accuracy: **91.5%**
        """)

    with col2:
        st.markdown("### ✅ Regression Metrics")
        st.markdown(f"""
        - Best RMSE: **₹973.08**
        - Best R² Score: **0.9840**
        - Best MAE: **₹542.19**
        - Baseline RMSE: **₹4,288.96**
        """)

    st.markdown("---")

    # Final CTA
    st.success("""
    ### 🎯 Ready to Get Started?

    Select a page from the navigation menu to begin making predictions!

    Start with **🎯 EMI Classification** to predict customer eligibility,
    or **💸 EMI Amount Prediction** to calculate maximum safe EMI amounts.
    """)
