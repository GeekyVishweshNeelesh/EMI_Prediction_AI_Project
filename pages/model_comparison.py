"""
pages/model_comparison.py - Model Performance Comparison Page

This page displays:
- Comparison of all 8 models
- Performance metrics
- Model rankings
- Best model recommendations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import MODEL_PERFORMANCE
from utils.visualizations import (
    plot_classification_model_comparison,
    plot_regression_model_comparison
)

def main():
    """Main function for model comparison page"""

    st.markdown("# 📊 Model Performance Comparison")
    st.markdown("Compare all 8 ML models and their performance metrics")

    st.markdown("---")

    # Create tabs for classification and regression
    tab1, tab2 = st.tabs(["🎯 Classification Models", "💰 Regression Models"])

    with tab1:
        st.markdown("## 🎯 EMI Eligibility Classification")
        st.markdown("Compare all classification models for EMI eligibility prediction")

        # Get classification data
        classification_data = MODEL_PERFORMANCE['Classification']

        # Create comparison dataframe
        class_df = pd.DataFrame(classification_data).T
        class_df = class_df.round(4)

        # Display table
        st.markdown("### 📋 Classification Metrics Table")
        st.dataframe(class_df, use_container_width=True)

        # Display visualizations
        st.markdown("---")
        st.markdown("### 📊 Detailed Performance Charts")
        plot_classification_model_comparison()

        # Model Rankings
        st.markdown("---")
        st.markdown("### 🏆 Model Rankings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### By Accuracy")
            accuracy_sorted = class_df['Accuracy'].sort_values(ascending=False)
            for i, (model, acc) in enumerate(accuracy_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: {acc:.4f}")

        with col2:
            st.markdown("#### By ROC-AUC")
            auc_sorted = class_df['ROC-AUC'].sort_values(ascending=False)
            for i, (model, auc) in enumerate(auc_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: {auc:.4f}")

        with col3:
            st.markdown("#### By F1-Score")
            f1_sorted = class_df['F1-Score'].sort_values(ascending=False)
            for i, (model, f1) in enumerate(f1_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: {f1:.4f}")

        # Best Model Info
        st.markdown("---")
        st.markdown("### 🏆 Recommended Model")

        best_accuracy_model = class_df['Accuracy'].idxmax()
        best_accuracy_value = class_df['Accuracy'].max()

        st.success(f"""
        **Recommended:** {best_accuracy_model}

        ✅ **Accuracy:** {best_accuracy_value:.4f} (95.90%)
        ✅ **ROC-AUC:** {class_df.loc[best_accuracy_model, 'ROC-AUC']:.4f}
        ✅ **F1-Score:** {class_df.loc[best_accuracy_model, 'F1-Score']:.4f}
        ✅ **Precision:** {class_df.loc[best_accuracy_model, 'Precision']:.4f}
        ✅ **Recall:** {class_df.loc[best_accuracy_model, 'Recall']:.4f}

        **Optimization Method:** Bayesian Optimization with Optuna TPE Sampler

        **Why This Model?**
        - Highest accuracy (95.90%) among all classifiers
        - Excellent ROC-AUC score (0.9962) for discrimination
        - Balanced precision and recall
        - Handles non-linear relationships well
        - Production-ready performance
        """)

        # Model Details
        st.markdown("---")
        st.markdown("### 📋 Detailed Model Information")

        for model_name, metrics in classification_data.items():
            with st.expander(f"📌 {model_name}", expanded=model_name.startswith('🏆')):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    st.metric("Precision", f"{metrics['Precision']:.4f}")

                with col2:
                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                    st.metric("F1-Score", f"{metrics['F1-Score']:.4f}")

                with col3:
                    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")

    with tab2:
        st.markdown("## 💰 Maximum EMI Amount Regression")
        st.markdown("Compare all regression models for maximum EMI prediction")

        # Get regression data
        regression_data = MODEL_PERFORMANCE['Regression']

        # Create comparison dataframe
        reg_df = pd.DataFrame(regression_data).T
        reg_df = reg_df.round(4)

        # Display table
        st.markdown("### 📋 Regression Metrics Table")
        st.dataframe(reg_df, use_container_width=True)

        # Display visualizations
        st.markdown("---")
        st.markdown("### 📊 Detailed Performance Charts")
        plot_regression_model_comparison()

        # Model Rankings
        st.markdown("---")
        st.markdown("### 🏆 Model Rankings")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### By R² Score (↑)")
            r2_sorted = reg_df['R²'].sort_values(ascending=False)
            for i, (model, r2) in enumerate(r2_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: {r2:.4f}")

        with col2:
            st.markdown("#### By RMSE (↓)")
            rmse_sorted = reg_df['RMSE'].sort_values(ascending=True)
            for i, (model, rmse) in enumerate(rmse_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: ₹{rmse:,.2f}")

        with col3:
            st.markdown("#### By MAE (↓)")
            mae_sorted = reg_df['MAE'].sort_values(ascending=True)
            for i, (model, mae) in enumerate(mae_sorted.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{model}**: ₹{mae:,.2f}")

        # Best Model Info
        st.markdown("---")
        st.markdown("### 🏆 Recommended Model")

        best_r2_model = reg_df['R²'].idxmax()
        best_r2_value = reg_df['R²'].max()
        best_rmse_value = reg_df.loc[best_r2_model, 'RMSE']

        st.success(f"""
        **Recommended:** {best_r2_model}

        ✅ **R² Score:** {best_r2_value:.4f} (98.40% variance explained)
        ✅ **RMSE:** ₹{best_rmse_value:,.2f} (error within ₹1000)
        ✅ **MAE:** ₹{reg_df.loc[best_r2_model, 'MAE']:,.2f} (average error)
        ✅ **MAPE:** {reg_df.loc[best_r2_model, 'MAPE']:.2f}% (percentage error)

        **Optimization Method:** Bayesian Optimization with Optuna TPE Sampler

        **Improvement Over Baseline:**
        - 76.3% improvement in RMSE vs Linear Regression
        - 37.6% improvement in RMSE vs Base XGBoost
        - 98.4% variance explained (vs 68.8% for Linear)

        **Why This Model?**
        - Best R² score (0.9840) among all regressors
        - Lowest RMSE (₹973.08) for accurate predictions
        - Handles non-linear patterns in EMI relationships
        - Excellent generalization capability
        - Production-ready performance
        """)

        # Model Details
        st.markdown("---")
        st.markdown("### 📋 Detailed Model Information")

        for model_name, metrics in regression_data.items():
            with st.expander(f"📌 {model_name}", expanded=model_name.startswith('🏆')):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("R² Score", f"{metrics['R²']:.4f}")
                    st.metric("RMSE", f"₹{metrics['RMSE']:,.2f}")

                with col2:
                    st.metric("MAE", f"₹{metrics['MAE']:,.2f}")
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

    # Summary and Insights
    st.markdown("---")
    st.markdown("## 💡 Key Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Classification Insights")
        st.markdown("""
        - **XGBoost Bayesian** significantly outperforms baseline models
        - Bayesian optimization improved performance by 3.04% over base XGBoost
        - All models achieve >87% accuracy - suitable for deployment
        - ROC-AUC >0.92 indicates excellent class discrimination
        - Ensemble methods (Random Forest, Gradient Boosting) perform better than single models
        """)

    with col2:
        st.markdown("### 💰 Regression Insights")
        st.markdown("""
        - **XGBoost Bayesian** reduces RMSE by 76.3% vs Linear Regression
        - Bayesian optimization improved R² score by 2.52 percentage points
        - Gradient boosting methods dominate regression tasks
        - Linear models inadequate for complex EMI relationships
        - Model performance validates XGBoost as production choice
        """)
