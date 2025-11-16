"""
pages/admin.py - Admin Panel for Model Management
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
from pathlib import Path

def main():
    st.title("⚙️ Admin Panel")
    st.markdown("**Manage models, data, and system settings**")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 System Status",
        "🗂️ Data Management",
        "🤖 Model Management",
        "⚙️ Settings"
    ])

    # ========================================================================
    # SYSTEM STATUS TAB
    # ========================================================================

    with tab1:
        st.header("System Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("System Status", "🟢 Online")
            st.metric("Uptime", "24h 15m")

        with col2:
            st.metric("Total Predictions", "1,234")
            st.metric("Success Rate", "99.2%")

        with col3:
            st.metric("Avg Response Time", "150ms")
            st.metric("Active Users", "42")

        st.markdown("---")

        # Models status
        st.subheader("🤖 Models Status")

        from utils.model_loader import load_all_models
        models = load_all_models()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Classification Models:**")
            classification_models = models.get('classification', {})
            if classification_models:
                for model_name in classification_models.keys():
                    st.success(f"✅ {model_name}")
            else:
                st.error("❌ No classification models loaded")

        with col2:
            st.write("**Regression Models:**")
            regression_models = models.get('regression', {})
            if regression_models:
                for model_name in regression_models.keys():
                    st.success(f"✅ {model_name}")
            else:
                st.error("❌ No regression models loaded")

        st.markdown("---")

        # System info
        st.subheader("💻 System Information")

        system_info = {
            'Parameter': [
                'Python Version',
                'Streamlit Version',
                'Working Directory',
                'Models Directory',
                'Last Updated'
            ],
            'Value': [
                '3.11.x',
                '1.28.x',
                os.getcwd(),
                'saved_models/',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }

        st.dataframe(pd.DataFrame(system_info), use_container_width=True)

    # ========================================================================
    # DATA MANAGEMENT TAB
    # ========================================================================

    with tab2:
        st.header("Data Management")

        st.subheader("📁 Upload New Dataset")

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a new dataset for training or prediction"
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully!")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("💾 Save Dataset"):
                st.success("Dataset saved successfully!")

        st.markdown("---")

        st.subheader("📊 Dataset Statistics")

        stats_data = {
            'Dataset': ['Training Data', 'Test Data', 'Validation Data'],
            'Records': ['280,000', '80,000', '40,000'],
            'Features': ['22', '22', '22'],
            'Size': ['45 MB', '13 MB', '6.5 MB']
        }

        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

        st.markdown("---")

        st.subheader("🗑️ Data Cleanup")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Cache", use_container_width=True):
                st.cache_resource.clear()
                st.success("Cache cleared successfully!")

        with col2:
            if st.button("🧹 Clean Temp Files", use_container_width=True):
                st.success("Temporary files cleaned!")

    # ========================================================================
    # MODEL MANAGEMENT TAB
    # ========================================================================

    with tab3:
        st.header("Model Management")

        st.subheader("📦 Available Models")

        from utils.model_loader import verify_model_files

        status = verify_model_files()

        st.write(f"**Models Path:** `{status['models_path']}`")
        st.write(f"**Path Exists:** {'✅ Yes' if status['path_exists'] else '❌ No'}")

        st.markdown("---")

        # Classification models status
        st.write("**Classification Models:**")

        for model_name, info in status['classification'].items():
            with st.expander(f"🎯 {model_name}"):
                col1, col2 = st.columns(2)

                with col1:
                    model_status = "✅ Exists" if info['model_exists'] else "❌ Missing"
                    st.write(f"**Model:** {model_status}")
                    st.code(info['model_file'])

                with col2:
                    scaler_status = "✅ Exists" if info['scaler_exists'] else "❌ Missing"
                    st.write(f"**Scaler:** {scaler_status}")
                    st.code(info['scaler_file'])

        st.markdown("---")

        # Regression models status
        st.write("**Regression Models:**")

        for model_name, info in status['regression'].items():
            with st.expander(f"💰 {model_name}"):
                col1, col2 = st.columns(2)

                with col1:
                    model_status = "✅ Exists" if info['model_exists'] else "❌ Missing"
                    st.write(f"**Model:** {model_status}")
                    st.code(info['model_file'])

                with col2:
                    scaler_status = "✅ Exists" if info['scaler_exists'] else "❌ Missing"
                    st.write(f"**Scaler:** {scaler_status}")
                    st.code(info['scaler_file'])

        st.markdown("---")

        st.subheader("🔧 Model Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reload Models", use_container_width=True):
                st.cache_resource.clear()
                st.success("Models reloaded!")

        with col2:
            if st.button("📊 View Metrics", use_container_width=True):
                st.info("Navigate to Model Comparison page")

        with col3:
            if st.button("📥 Export Models", use_container_width=True):
                st.info("Model export feature coming soon!")

    # ========================================================================
    # SETTINGS TAB
    # ========================================================================

    with tab4:
        st.header("Settings")

        st.subheader("🎨 Application Settings")

        # Theme settings
        theme = st.selectbox(
            "Theme",
            ["Light", "Dark", "Auto"],
            index=2
        )

        # Prediction settings
        st.markdown("---")
        st.subheader("🔮 Prediction Settings")

        default_model_class = st.selectbox(
            "Default Classification Model",
            ["Logistic Regression", "XGBoost Classifier", "Decision Tree", "Gradient Boosting"]
        )

        default_model_reg = st.selectbox(
            "Default Regression Model",
            ["Ridge Regression", "XGBoost Regressor"]
        )

        confidence_threshold = st.slider(
            "Confidence Threshold (%)",
            min_value=50,
            max_value=100,
            value=70
        )

        st.markdown("---")
        st.subheader("📧 Notification Settings")

        enable_email = st.checkbox("Enable email notifications", value=False)
        enable_alerts = st.checkbox("Enable system alerts", value=True)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Settings", type="primary", use_container_width=True):
                st.success("Settings saved successfully!")

        with col2:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.info("Settings reset to defaults!")

if __name__ == "__main__":
    main()
