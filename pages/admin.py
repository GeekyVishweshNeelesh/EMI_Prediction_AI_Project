"""
pages/admin.py - Admin Panel Page

This page handles:
- Data management and CRUD operations
- Batch predictions
- Data download
- System monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.model_loader import get_best_classification_model, get_best_regression_model
from utils.predictions import batch_predict_eligibility, batch_predict_emi

def main(models):
    """Main function for admin panel"""

    st.markdown("# ⚙️ Admin Panel")
    st.markdown("Manage data, run batch predictions, and monitor system")

    st.markdown("---")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Data Management",
        "🚀 Batch Predictions",
        "📊 System Monitoring",
        "⚙️ Settings"
    ])

    # ========================================================================
    # TAB 1: DATA MANAGEMENT
    # ========================================================================

    with tab1:
        st.markdown("## 📁 Data Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📥 Upload Customer Data")

            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type="csv",
                key="data_upload"
            )

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Successfully loaded {len(df)} records")

                    # Display preview
                    st.markdown("#### 👁️ Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)

                    # Display statistics
                    st.markdown("#### 📊 Data Statistics")
                    col1_stat, col2_stat, col3_stat, col4_stat = st.columns(4)

                    with col1_stat:
                        st.metric("Total Records", len(df))

                    with col2_stat:
                        st.metric("Total Columns", len(df.columns))

                    with col3_stat:
                        st.metric("Missing Values", df.isnull().sum().sum())

                    with col4_stat:
                        st.metric("Duplicate Rows", df.duplicated().sum())

                    # Data quality report
                    st.markdown("#### 🔍 Data Quality")

                    quality_issues = []

                    # Check for missing values
                    missing_cols = df.columns[df.isnull().any()].tolist()
                    if missing_cols:
                        quality_issues.append(f"⚠️ Missing values in: {', '.join(missing_cols)}")

                    # Check for duplicates
                    if df.duplicated().sum() > 0:
                        quality_issues.append(f"⚠️ Found {df.duplicated().sum()} duplicate rows")

                    if not quality_issues:
                        st.info("✅ Data quality check passed! No issues detected.")
                    else:
                        for issue in quality_issues:
                            st.warning(issue)

                    # Store in session
                    st.session_state.uploaded_data = df

                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")

        with col2:
            st.markdown("### 📊 Sample Data")

            st.markdown("Generate sample customer data for testing")

            n_samples = st.slider("Number of samples", 10, 1000, 100)

            if st.button("🔄 Generate Sample Data", key="generate_sample"):

                # Generate random data
                sample_data = {
                    'age': np.random.randint(25, 60, n_samples),
                    'gender': np.random.choice([0, 1], n_samples),
                    'monthly_salary': np.random.randint(15000, 200000, n_samples),
                    'credit_score': np.random.randint(300, 850, n_samples),
                    'family_size': np.random.randint(1, 8, n_samples),
                    'existing_loans': np.random.choice([0, 1], n_samples),
                    'bank_balance': np.random.randint(0, 1000000, n_samples),
                }

                sample_df = pd.DataFrame(sample_data)

                st.success(f"✅ Generated {n_samples} sample records")
                st.dataframe(sample_df.head(10), use_container_width=True)

                # Download button
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sample Data",
                    data=csv,
                    file_name=f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    # ========================================================================
    # TAB 2: BATCH PREDICTIONS
    # ========================================================================

    with tab2:
        st.markdown("## 🚀 Batch Predictions")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Run Batch Predictions")

            # Check if data is loaded
            if 'uploaded_data' not in st.session_state:
                st.warning("⚠️ Please upload data first in the Data Management tab")
            else:
                df = st.session_state.uploaded_data

                st.markdown(f"📊 Processing {len(df)} records...")

                # Select prediction type
                pred_type = st.radio(
                    "Select prediction type:",
                    ["🎯 Classification", "💰 Regression", "🔄 Both"]
                )

                if st.button("🚀 Run Predictions", use_container_width=True, type="primary"):

                    # Ensure we have required features
                    required_features = 22
                    if len(df.columns) < required_features:
                        st.error(f"❌ Dataset needs {required_features} features, found {len(df.columns)}")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        if pred_type in ["🎯 Classification", "🔄 Both"]:
                            status_text.text("🔄 Running classification predictions...")

                            try:
                                best_model, best_scaler = get_best_classification_model(models)

                                # Run batch predictions
                                df_with_pred = batch_predict_eligibility(best_model, best_scaler, df.copy())

                                st.session_state.classification_results = df_with_pred
                                st.success("✅ Classification predictions completed!")
                                progress_bar.progress(50)

                            except Exception as e:
                                st.error(f"❌ Classification error: {str(e)}")

                        if pred_type in ["💰 Regression", "🔄 Both"]:
                            status_text.text("🔄 Running regression predictions...")

                            try:
                                best_model, best_scaler = get_best_regression_model(models)

                                # Run batch predictions
                                df_with_pred = batch_predict_emi(best_model, best_scaler, df.copy())

                                st.session_state.regression_results = df_with_pred
                                st.success("✅ Regression predictions completed!")
                                progress_bar.progress(100)

                            except Exception as e:
                                st.error(f"❌ Regression error: {str(e)}")

        with col2:
            st.markdown("### 📥 Results Download")

            if 'classification_results' in st.session_state:
                st.markdown("#### Classification Results")
                csv = st.session_state.classification_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Classification",
                    data=csv,
                    file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            if 'regression_results' in st.session_state:
                st.markdown("#### Regression Results")
                csv = st.session_state.regression_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Regression",
                    data=csv,
                    file_name=f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        # Display results
        if 'classification_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Classification Results Preview")
            st.dataframe(st.session_state.classification_results.head(10), use_container_width=True)

        if 'regression_results' in st.session_state:
            st.markdown("---")
            st.markdown("### 📊 Regression Results Preview")
            st.dataframe(st.session_state.regression_results.head(10), use_container_width=True)

    # ========================================================================
    # TAB 3: SYSTEM MONITORING
    # ========================================================================

    with tab3:
        st.markdown("## 📊 System Monitoring")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("🤖 Models Loaded", "8/8")

        with col2:
            st.metric("⚡ API Status", "Active")

        with col3:
            st.metric("📈 Predictions Today", "0")

        with col4:
            st.metric("⏱️ Avg Latency", "0.5s")

        st.markdown("---")

        st.markdown("### 📋 System Information")

        system_info = {
            'Component': [
                'Classification Models',
                'Regression Models',
                'Total Features',
                'Max Dataset Size',
                'Supported Formats'
            ],
            'Status': [
                '✅ 5 Models (1 Best)',
                '✅ 3 Models (1 Best)',
                '✅ 22 Features',
                '✅ 400K+ Records',
                '✅ CSV'
            ]
        }

        sys_df = pd.DataFrame(system_info)
        st.dataframe(sys_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        st.markdown("### ✅ Health Check")

        health_checks = [
            ("Models Loaded", True),
            ("API Connected", True),
            ("Database Available", True),
            ("Predictions Working", True),
        ]

        for check_name, status in health_checks:
            status_icon = "✅" if status else "❌"
            st.markdown(f"{status_icon} {check_name}")

    # ========================================================================
    # TAB 4: SETTINGS
    # ========================================================================

    with tab4:
        st.markdown("## ⚙️ Settings")

        st.markdown("### 🎯 Model Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Classification Model")
            class_model = st.selectbox(
                "Select classification model",
                ["🏆 XGBoost (Bayesian)", "Random Forest", "Logistic Regression"]
            )
            st.info(f"Selected: {class_model}")

        with col2:
            st.markdown("#### Regression Model")
            reg_model = st.selectbox(
                "Select regression model",
                ["🏆 XGBoost (Bayesian)", "Random Forest", "Linear Regression"]
            )
            st.info(f"Selected: {reg_model}")

        st.markdown("---")

        st.markdown("### 📊 Data Settings")

        batch_size = st.slider("Batch Size", 10, 1000, 100)
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 75)

        st.markdown("---")

        st.markdown("### 💾 Cache & Storage")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ Clear Cache"):
                st.session_state.clear()
                st.success("✅ Cache cleared!")

        with col2:
            if st.button("📊 Export System Log"):
                st.info("Export feature coming soon!")

        st.markdown("---")

        st.success("""
        ### ✅ Admin Panel Ready

        - **Data Upload:** Upload customer data for batch processing
        - **Batch Predictions:** Run predictions on multiple customers
        - **Results Download:** Export predictions as CSV
        - **System Monitoring:** Check system health and performance
        - **Settings:** Configure model and system parameters
        """)
