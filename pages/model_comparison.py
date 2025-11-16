"""
pages/model_comparison.py - Model Performance Comparison Page
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def main(models=None):
    st.title("📊 Model Performance Comparison")
    st.markdown("**Compare the performance of all trained ML models**")

    # Load models if not provided
    if models is None:
        from utils.model_loader import load_all_models
        models = load_all_models()

    if not models:
        st.error("❌ No models available for comparison!")
        return

    # Create tabs for classification and regression
    tab1, tab2 = st.tabs(["🎯 Classification Models", "💰 Regression Models"])

    # ========================================================================
    # CLASSIFICATION MODELS TAB
    # ========================================================================

    with tab1:
        st.header("Classification Models Performance")

        classification_models = models.get('classification', {})

        if not classification_models:
            st.warning("⚠️ No classification models available")
        else:
            st.success(f"✅ {len(classification_models)} classification models loaded")

            # Display available models
            st.subheader("📋 Available Models")
            for idx, model_name in enumerate(classification_models.keys(), 1):
                st.write(f"{idx}. **{model_name}**")

            # Sample performance metrics (you can replace with actual metrics)
            st.subheader("📊 Performance Metrics")

            # Create sample data - replace with your actual metrics
            metrics_data = {
                'Model': list(classification_models.keys()),
                'Accuracy': [0.92, 0.95, 0.89, 0.94],  # Sample values
                'Precision': [0.91, 0.94, 0.88, 0.93],
                'Recall': [0.90, 0.93, 0.87, 0.92],
                'F1-Score': [0.90, 0.93, 0.87, 0.92]
            }

            # Adjust length to match available models
            num_models = len(classification_models)
            for key in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                metrics_data[key] = metrics_data[key][:num_models]

            df_class = pd.DataFrame(metrics_data)

            # Display table
            st.dataframe(df_class, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for accuracy
                fig1 = px.bar(
                    df_class,
                    x='Model',
                    y='Accuracy',
                    title='Model Accuracy Comparison',
                    color='Accuracy',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # Radar chart for all metrics
                fig2 = go.Figure()

                for idx, row in df_class.iterrows():
                    fig2.add_trace(go.Scatterpolar(
                        r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        fill='toself',
                        name=row['Model']
                    ))

                fig2.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title='All Metrics Comparison'
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ========================================================================
    # REGRESSION MODELS TAB
    # ========================================================================

    with tab2:
        st.header("Regression Models Performance")

        regression_models = models.get('regression', {})

        if not regression_models:
            st.warning("⚠️ No regression models available")
        else:
            st.success(f"✅ {len(regression_models)} regression models loaded")

            # Display available models
            st.subheader("📋 Available Models")
            for idx, model_name in enumerate(regression_models.keys(), 1):
                st.write(f"{idx}. **{model_name}**")

            # Sample performance metrics
            st.subheader("📊 Performance Metrics")

            # Create sample data - replace with actual metrics
            metrics_data = {
                'Model': list(regression_models.keys()),
                'R² Score': [0.85, 0.92],  # Sample values
                'RMSE': [2500, 1800],
                'MAE': [1800, 1200],
                'MAPE': [8.5, 6.2]
            }

            df_reg = pd.DataFrame(metrics_data)

            # Display table
            st.dataframe(df_reg, use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Bar chart for R² Score
                fig3 = px.bar(
                    df_reg,
                    x='Model',
                    y='R² Score',
                    title='R² Score Comparison',
                    color='R² Score',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                # Bar chart for RMSE
                fig4 = px.bar(
                    df_reg,
                    x='Model',
                    y='RMSE',
                    title='RMSE Comparison (Lower is Better)',
                    color='RMSE',
                    color_continuous_scale='Reds_r'
                )
                st.plotly_chart(fig4, use_container_width=True)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    st.markdown("---")
    st.subheader("📈 Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Models",
            len(classification_models) + len(regression_models)
        )

    with col2:
        st.metric(
            "Classification Models",
            len(classification_models)
        )

    with col3:
        st.metric(
            "Regression Models",
            len(regression_models)
        )

    st.info("💡 **Note:** The metrics shown are sample values. Replace them with actual model performance metrics from your training results.")

if __name__ == "__main__":
    main()
