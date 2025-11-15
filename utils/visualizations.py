"""
utils/visualizations.py - Create visualizations and charts

This module handles:
- Creating prediction visualizations
- Model comparison charts
- Data distribution plots
- Performance metrics visualization
"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import MODEL_PERFORMANCE, COLORS

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# ============================================================================
# ELIGIBILITY PREDICTION VISUALIZATION
# ============================================================================

def plot_eligibility_prediction(prediction_result):
    """
    Visualize EMI eligibility prediction with probabilities

    Parameters:
    -----------
    prediction_result : dict
        Result from predict_emi_eligibility function
    """

    if not prediction_result.get('success', False):
        st.error("Invalid prediction result")
        return

    col1, col2 = st.columns(2)

    # Main prediction
    with col1:
        st.markdown(f"### 🎯 Prediction Result")

        class_name = prediction_result['class_name']
        color = prediction_result['color']
        confidence = prediction_result['confidence']
        description = prediction_result['description']

        st.markdown(f"""
        <div style='background-color: {color}; color: white; padding: 20px; border-radius: 10px; text-align: center;'>
            <h2 style='margin: 0;'>{class_name}</h2>
            <p style='margin: 10px 0 0 0;'>{description}</p>
            <h3 style='margin: 10px 0 0 0;'>Confidence: {confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    # Probability chart
    with col2:
        st.markdown(f"### 📊 Class Probabilities")

        probs = prediction_result['probabilities']
        prob_data = pd.DataFrame({
            'Class': list(probs.keys()),
            'Probability': list(probs.values())
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        colors_list = ['#dc3545', '#ffc107', '#28a745']
        bars = ax.bar(prob_data['Class'], prob_data['Probability'], color=colors_list, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('EMI Eligibility Probabilities', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

# ============================================================================
# MAX EMI PREDICTION VISUALIZATION
# ============================================================================

def plot_max_emi_prediction(prediction_result, tenure_months=60):
    """
    Visualize maximum EMI prediction

    Parameters:
    -----------
    prediction_result : dict
        Result from predict_max_emi function
    tenure_months : int
        Tenure for EMI breakdown
    """

    if not prediction_result.get('success', False):
        st.error("Invalid prediction result")
        return

    max_emi = prediction_result['max_emi']
    formatted_emi = prediction_result['formatted_emi']

    col1, col2 = st.columns(2)

    # Main EMI value
    with col1:
        st.markdown(f"### 💰 Maximum Safe EMI")
        st.markdown(f"""
        <div style='background-color: #007bff; color: white; padding: 30px; border-radius: 10px; text-align: center;'>
            <h1 style='margin: 0;'>{formatted_emi}</h1>
            <p style='margin: 10px 0 0 0;'>Monthly Payment Capacity</p>
        </div>
        """, unsafe_allow_html=True)

    # Prediction range
    with col2:
        st.markdown(f"### 📈 Prediction Range")

        pred_range = prediction_result['prediction_range']
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create range visualization
        range_low = pred_range['low']
        range_high = pred_range['high']

        categories = ['Low\n(80%)', 'Predicted', 'High\n(120%)']
        values = [range_low, max_emi, range_high]
        colors_range = ['#FFA500', '#28a745', '#FF6347']

        bars = ax.bar(categories, values, color=colors_range, alpha=0.8, edgecolor='black', width=0.6)

        ax.set_ylabel('EMI Amount (₹)', fontsize=12, fontweight='bold')
        ax.set_title('Maximum EMI Range', fontsize=13, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'₹{val:,.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

def plot_emi_breakdown_chart(max_emi, tenures):
    """
    Plot EMI amounts for different tenures

    Parameters:
    -----------
    max_emi : float
        Maximum EMI amount
    tenures : list
        List of tenure values
    """

    st.markdown("### 📊 EMI Breakdown by Tenure")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create data
    tenure_data = pd.DataFrame({
        'Tenure (Months)': tenures,
        'Monthly EMI': [max_emi * (t / 60) for t in tenures]  # Scale EMI based on tenure
    })

    # Plot
    ax.plot(tenure_data['Tenure (Months)'], tenure_data['Monthly EMI'],
           marker='o', linewidth=2, markersize=8, color='#007bff')

    ax.fill_between(tenure_data['Tenure (Months)'],
                    tenure_data['Monthly EMI'] * 0.8,
                    tenure_data['Monthly EMI'] * 1.2,
                    alpha=0.2, color='#007bff')

    ax.set_xlabel('Tenure (Months)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly EMI (₹)', fontsize=12, fontweight='bold')
    ax.set_title('EMI vs Tenure Analysis', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/1000:.0f}K'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================================

def plot_classification_model_comparison():
    """
    Plot comparison of all classification models
    """

    st.markdown("### 📊 Classification Models Comparison")

    # Get data from config
    classification_models = MODEL_PERFORMANCE['Classification']

    # Prepare data
    model_names = list(classification_models.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # Create subplots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Classification Models Performance Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    colors_models = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [classification_models[model].get(metric, 0) for model in model_names]

        bars = ax.bar(range(len(model_names)), values, color=colors_models, alpha=0.8, edgecolor='black')

        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.replace(' (', '\n(') for name in model_names], fontsize=9)
        ax.set_ylim([0.8, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.tight_layout()
    st.pyplot(fig)

def plot_regression_model_comparison():
    """
    Plot comparison of all regression models
    """

    st.markdown("### 📊 Regression Models Comparison")

    # Get data from config
    regression_models = MODEL_PERFORMANCE['Regression']

    model_names = list(regression_models.keys())
    metrics = ['RMSE', 'MAE', 'R²', 'MAPE']

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Regression Models Performance Comparison', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    colors_models = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [regression_models[model].get(metric, 0) for model in model_names]

        bars = ax.bar(range(len(model_names)), values, color=colors_models, alpha=0.8, edgecolor='black')

        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.replace(' (', '\n(') for name in model_names], fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            if metric in ['RMSE', 'MAE']:
                label = f'₹{val:,.0f}'
            else:
                label = f'{val:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# HEATMAP VISUALIZATION
# ============================================================================

def plot_feature_correlation_heatmap(data_df, features):
    """
    Plot correlation heatmap for features

    Parameters:
    -----------
    data_df : pd.DataFrame
        Dataset
    features : list
        List of feature names to include
    """

    st.markdown("### 📈 Feature Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate correlation
    corr_matrix = data_df[features].corr()

    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               cbar_kws={'label': 'Correlation'}, ax=ax, vmin=-1, vmax=1)

    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# ============================================================================
# METRICS CARDS
# ============================================================================

def display_metrics_cards(metrics_dict):
    """
    Display metrics in nice cards

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names and values
    """

    cols = st.columns(len(metrics_dict))

    for col, (metric_name, metric_value) in zip(cols, metrics_dict.items()):
        with col:
            st.metric(metric_name, f"{metric_value:.4f}")

# ============================================================================
# STATUS INDICATORS
# ============================================================================

def display_eligibility_status(eligibility_class):
    """
    Display eligibility status with color coding

    Parameters:
    -----------
    eligibility_class : str
        Eligibility class name
    """

    status_colors = {
        'Eligible': '#28a745',
        'High_Risk': '#ffc107',
        'Not_Eligible': '#dc3545'
    }

    status_icons = {
        'Eligible': '✅',
        'High_Risk': '⚠️',
        'Not_Eligible': '❌'
    }

    color = status_colors.get(eligibility_class, '#6c757d')
    icon = status_icons.get(eligibility_class, '❓')

    st.markdown(f"""
    <div style='background-color: {color}; color: white; padding: 15px; border-radius: 10px; text-align: center;'>
        <h3 style='margin: 0;'>{icon} {eligibility_class}</h3>
    </div>
    """, unsafe_allow_html=True)
