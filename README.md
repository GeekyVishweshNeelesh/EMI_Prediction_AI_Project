# 💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform

## 📋 Project Overview

EMIPredict AI is a comprehensive machine learning platform designed to help financial institutions, banks, and FinTech companies make data-driven decisions for EMI (Equated Monthly Installment) predictions and risk assessment.

**Live Demo:** [Deploy on Streamlit Cloud]

---

## 🎯 Key Features

✅ **Real-time Predictions** - Instant EMI eligibility and maximum amount predictions  
✅ **8 ML Models** - Classification and regression with multiple algorithms  
✅ **Bayesian Optimization** - Best-in-class models using Optuna TPE sampler  
✅ **Batch Processing** - Process multiple customers at once  
✅ **Interactive Dashboard** - Beautiful visualizations and analytics  
✅ **Admin Panel** - Data management and system monitoring  

---

## 📊 Performance Metrics

### Classification (EMI Eligibility)
- **Best Model:** XGBoost (Bayesian Optimized)
- **Accuracy:** 95.90%
- **ROC-AUC:** 0.9962
- **F1-Score:** 0.9517

### Regression (Maximum EMI Amount)
- **Best Model:** XGBoost (Bayesian Optimized)
- **RMSE:** ₹973.08
- **R² Score:** 0.9840
- **MAE:** ₹542.19

---

## 📁 Project Structure

```
emi_prediction_app/
│
├── app.py                              # Main entry point
├── config.py                           # Configuration settings
├── requirements.txt                    # Dependencies
├── README.md                          # Documentation
│
├── saved_models/                       # All 32 model files
│   ├── 01_logistic_regression_*.pkl
│   ├── 02_random_forest_classifier_*.pkl
│   ├── 03_xgboost_classifier_*.pkl
│   ├── 04_linear_regression_*.pkl
│   ├── 05_random_forest_regressor_*.pkl
│   ├── 06_xgboost_regressor_*.pkl
│   ├── 07_decision_tree_*.pkl
│   └── 08_gradient_boosting_*.pkl
│
├── pages/                              # Multi-page apps
│   ├── home.py                        # Dashboard
│   ├── classification.py              # EMI Eligibility
│   ├── regression.py                  # Maximum EMI
│   ├── model_comparison.py            # Performance Metrics
│   └── admin.py                       # Data Management
│
└── utils/                              # Utilities
    ├── __init__.py
    ├── model_loader.py                # Load models
    ├── predictions.py                 # Make predictions
    └── visualizations.py              # Charts & plots
```

---

## 🚀 Installation & Setup

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd emi_prediction_app
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Place Model Files
Ensure all 32 model files are in the `saved_models/` folder:
- 01_logistic_regression_base.pkl
- 01_logistic_regression_base_scaler.pkl
- 01_logistic_regression_tuned.pkl
- 01_logistic_regression_tuned_scaler.pkl
- [... and so on for all 8 models]

### Step 5: Run Locally
```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)

### Deployment Steps

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy EMI Prediction App"
git push origin main
```

2. **Connect to Streamlit Cloud**
   - Go to share.streamlit.io
   - Click "New app"
   - Select GitHub repository
   - Choose `main` branch
   - Set main file to `app.py`
   - Click "Deploy"

3. **Share Public URL**
   - Your app is now live at: `https://share.streamlit.io/your-username/repo-name`

---

## 📖 How to Use

### 1️⃣ Classification (EMI Eligibility)
**Navigate:** 🎯 EMI Classification

- Enter all 22 customer financial details
- Click "Get EMI Eligibility Prediction"
- View result: Eligible / High Risk / Not Eligible
- See confidence score and probability breakdown

### 2️⃣ Regression (Maximum EMI)
**Navigate:** 💸 EMI Amount Prediction

- Fill customer information form
- Select EMI scenario
- Click "Calculate Maximum Safe EMI"
- View maximum amount and affordability analysis
- See EMI breakdown for different tenures

### 3️⃣ Model Comparison
**Navigate:** 📊 Model Comparison

- Compare all 8 models side-by-side
- View performance metrics
- See model rankings by accuracy, RMSE, etc.
- Understand why XGBoost is best

### 4️⃣ Admin Panel
**Navigate:** ⚙️ Admin Panel

- Upload customer data (CSV)
- Run batch predictions
- Download results
- Monitor system health

---

## 📊 Dataset Features (22 Variables)

### Personal Demographics
- age, gender, marital_status, education

### Employment & Income
- monthly_salary, employment_type, years_of_employment, company_type

### Housing & Family
- house_type, monthly_rent, family_size, dependents

### Financial Obligations
- school_fees, college_fees, travel_expenses, groceries_utilities, other_monthly_expenses

### Financial Status & Credit
- existing_loans, current_emi_amount, credit_score, bank_balance, emergency_fund

---

## 🔧 Technology Stack

**Frontend:**
- Streamlit (Interactive web application)

**Machine Learning:**
- XGBoost (Gradient boosting)
- Random Forest (Ensemble learning)
- Scikit-learn (ML library)
- Logistic Regression (Baseline)

**Optimization:**
- Bayesian Optimization (Optuna)
- GridSearchCV & RandomizedSearchCV

**Data Processing:**
- Pandas (Data manipulation)
- NumPy (Numerical computing)

**Visualization:**
- Matplotlib & Seaborn (Static plots)
- Plotly (Interactive charts)

---

## 🤖 Models Included

### Classification Models (EMI Eligibility)
1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier
4. Decision Tree Classifier
5. Gradient Boosting Classifier

### Regression Models (Maximum EMI)
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

**Each model includes:**
- Base version (default hyperparameters)
- Tuned version (GridSearchCV/RandomizedSearchCV)
- Or Bayesian Optimized version

---

## 📈 Performance Comparison

| Model | Task | Accuracy/R² | Best For |
|-------|------|-------------|----------|
| **XGBoost (Bayesian)** | Classification | 95.90% | ⭐ Recommended |
| XGBoost (Bayesian) | Regression | 0.9840 R² | ⭐ Recommended |
| Random Forest | Classification | 91.27% | Good |
| Random Forest | Regression | 0.9386 R² | Good |
| Logistic Regression | Classification | 88.81% | Baseline |
| Linear Regression | Regression | 0.6884 R² | Baseline |

---

## 🔐 File Format

All models are saved as `.pkl` (Pickle) files:
- **Models:** `*.pkl` files (using `pickle` module)
- **Scalers:** `*_scaler.pkl` files (using `joblib`)

Load in your code:
```python
import pickle
import joblib

# Load model
model = pickle.load(open("saved_models/model_name.pkl", "rb"))

# Load scaler
scaler = joblib.load("saved_models/scaler_name.pkl")
```

---

## 🚨 Troubleshooting

### Models Not Loading
- Check all 32 files are in `saved_models/` folder
- Verify file names match exactly
- Ensure enough memory (models need ~500MB RAM)

### Predictions Not Working
- Verify input data has 22 features
- Check feature values are within valid ranges
- Ensure scalers loaded correctly

### Streamlit Cache Issues
- Clear browser cache
- Run: `streamlit cache clear`
- Restart app: `streamlit run app.py --logger.level=debug`

---

## 📝 Configuration

Edit `config.py` to customize:

```python
# Model paths
MODEL_PATHS = {...}
SCALER_PATHS = {...}

# Feature ranges
FEATURE_RANGES = {...}

# Categorical options
CATEGORICAL_OPTIONS = {...}

# Model performance metrics
MODEL_PERFORMANCE = {...}
```

---

## 💡 Usage Examples

### Single Prediction (Classification)
```python
from utils.model_loader import get_best_classification_model
from utils.predictions import predict_emi_eligibility
import numpy as np

# Load model
model, scaler = get_best_classification_model(models)

# Prepare input (22 features)
customer_features = np.array([...22 values...])

# Predict
result = predict_emi_eligibility(model, scaler, customer_features)
print(result['class_name'])  # Output: Eligible/High_Risk/Not_Eligible
print(result['confidence'])   # Output: 95.27
```

### Single Prediction (Regression)
```python
from utils.predictions import predict_max_emi

# Predict maximum EMI
result = predict_max_emi(model, scaler, customer_features)
print(result['formatted_emi'])  # Output: ₹25,000.00
```

---

## 🔄 Batch Processing

```python
from utils.predictions import batch_predict_eligibility
import pandas as pd

# Load customer data
df = pd.read_csv("customers.csv")

# Run batch predictions
results = batch_predict_eligibility(model, scaler, df)

# Save results
results.to_csv("predictions.csv")
```

---

## 📞 Support & Contact

- **Issues:** Open GitHub issue
- **Email:** support@emipredict.com
- **Documentation:** Check README.md

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contributors

- Development Team
- Data Science Team
- QA Team

---

## 🎯 Roadmap

- [ ] Mobile app integration
- [ ] REST API endpoints
- [ ] Advanced SHAP explainability
- [ ] Real-time model updates
- [ ] Multi-language support
- [ ] Enhanced security features

---

## ✅ Checklist Before Deployment

- [ ] All 32 model files in `saved_models/`
- [ ] `requirements.txt` with correct versions
- [ ] `config.py` properly configured
- [ ] All page files in `pages/` folder
- [ ] Utils files in `utils/` folder
- [ ] `.gitignore` configured
- [ ] Tested locally (streamlit run app.py)
- [ ] Pushed to GitHub
- [ ] Deployed on Streamlit Cloud
- [ ] Public URL accessible

---

**Last Updated:** 2024  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
