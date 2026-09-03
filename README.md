# 🏦 Credit Card Customer Default Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/)

> An end-to-end Machine Learning solution for credit risk assessment, trained on 30,000 customer records to predict credit card payment defaults with high recall, integrated with an interactive real-time Streamlit web dashboard.

---

### 👨‍💻 Project Creators
This project was designed, engineered, and developed by:
* **Harsh Mishra**
* **Sarthak Tajane**

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Banking Context](#-problem-statement--banking-context)
3. [System Architecture](#-system-architecture)
4. [Dataset Description](#-dataset-description)
5. [Data Cleaning & Preprocessing Pipeline](#-data-cleaning--preprocessing-pipeline)
6. [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
7. [Machine Learning Modeling & Benchmark](#-machine-learning-modeling--benchmark)
8. [Feature Importance & Risk Drivers](#-feature-importance--risk-drivers)
9. [Streamlit Web Application](#-streamlit-web-application)
10. [Security, Data Privacy & Breach Prevention](#-security-data-privacy--breach-prevention)
11. [Repository Structure](#-repository-structure)
12. [Installation & Local Setup](#-installation--local-setup)
13. [Deployment Guide (Streamlit Cloud & GitHub)](#-deployment-guide-streamlit-cloud--github)
14. [Future Scope & Roadmap](#-future-scope--roadmap)
15. [License & Acknowledgments](#-license--acknowledgments)

---

## 📌 Project Overview

In the commercial banking and retail lending sector, extending credit involves continuous risk evaluation. When credit card holders default on their monthly dues, lending institutions face liquidity crunches, provisions for bad debts, and severe balance-sheet impairment.

This repository provides a production-grade machine learning pipeline that:
- Ingests and cleans multi-dimensional financial and demographic customer records.
- Handles class imbalance and resolves non-standard categorical anomalies.
- Trains, tunes, and benchmarks four diverse machine learning models: **Logistic Regression**, **Decision Tree**, **Random Forest**, and **Gradient Boosting**.
- Evaluates models with an emphasis on **Recall** to minimize costly False Negatives (unidentified defaulters).
- Deploys the champion **Random Forest** model into a responsive, real-time **Streamlit web application** for interactive credit officer inference.

---

## 🎯 Problem Statement & Banking Context

### The Core Problem
Given historical demographic information, repayment delay records, bill amounts, and previous payment sums over a six-month observation window (April–September), determine whether a customer will default on their credit card payment in the upcoming month ($Y \in \{0, 1\}$).

### The Asymmetric Cost Matrix in Credit Risk
Standard classification accuracy is misleading in credit risk due to significant class imbalance (~78% non-default vs. ~22% default) and unequal error costs:

| Prediction vs Actual | Actual: Non-Defaulter ($0$) | Actual: Defaulter ($1$) |
| :--- | :--- | :--- |
| **Predicted: Non-Defaulter ($0$)** | **True Negative (TN):** Healthy customer maintains active revolving credit. Institution earns interest/interchange fees. | **False Negative (FN) — CRITICAL HAZARD:** Institution fails to detect a defaulter. Direct financial loss (unrecovered principal, interest, high legal recovery costs). |
| **Predicted: Defaulter ($1$)** | **False Positive (FP) — MINOR INCONVENIENCE:** Good customer flagged for scrutiny. May require manual underwriting or credit line freeze. | **True Positive (TP):** High-risk customer correctly intercepted. Proactive limit reduction or collateral enforcement is triggered. |

> **Key Takeaway:** Missing an actual defaulter (**False Negative**) is vastly more expensive to a financial institution than falsely flagging a safe customer (**False Positive**). Therefore, **Recall** (Sensitivity) and **ROC-AUC** are prioritized during model selection over simple Accuracy.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Raw UCI Credit Card Dataset\n30,000 Records x 24 Features] --> B[Data Cleaning & Validation]
    B --> B1[Drop Non-Predictive IDs]
    B --> B2[Consolidate Categorical Anomalies\nEDUCATION: 0,5,6 -> 4 | MARRIAGE: 0 -> 3]
    B --> B3[Verify Nulls & Deduplication]
    
    B1 & B2 & B3 --> C[Exploratory Data Analysis\nDistributions, Correlations & Risk Trends]
    
    C --> D[Data Partitioning & Scaling]
    D --> D1[Stratified 80/20 Train-Test Split]
    D --> D2[StandardScaler Standardization\nZero Mean, Unit Variance]
    
    D1 & D2 --> E[Model Training & Benchmarking]
    E --> E1[Logistic Regression\nCost-Sensitive Balanced]
    E --> E2[Decision Tree\nDepth-Constrained max_depth=6]
    E --> E3[Random Forest Classifier\n200 Estimators, max_depth=10, Balanced]
    E --> E4[Gradient Boosting Classifier\n200 Estimators, lr=0.1, max_depth=3]
    
    E1 & E2 & E3 & E4 --> F[Model Evaluation & Selection]
    F --> F1[Metric Comparison: Accuracy, Precision, Recall, F1]
    F --> F2[ROC Curve & AUC Score Analysis]
    F --> F3[Confusion Matrices & Cost Optimization]
    
    F --> G[Model Serialization]
    G --> G1[random_forest_model.pkl]
    G --> G2[scaler.pkl]
    
    G1 & G2 --> H[Streamlit Web Application\napp.py]
    H --> I[Live Credit Officer Decision Dashboard]
```

---

## 📊 Dataset Description

The system utilizes the **Default of Credit Card Clients Dataset** from the **UCI Machine Learning Repository**, containing transactional data of credit card clients in Taiwan.

* **Sample Size:** 30,000 client records
* **Target Variable:** `default` (Binary: `1` = Default in next month, `0` = No default)
* **Class Imbalance:** ~77.88% Non-Default ($23,364$) vs ~22.12% Default ($6,636$)

### Feature Dictionary

| Feature Name | Type | Description | Values / Scale |
| :--- | :--- | :--- | :--- |
| `LIMIT_BAL` | Continuous | Amount of given credit | NT$ 10,000 to NT$ 1,000,000 |
| `SEX` | Categorical | Gender | `1` = Male, `2` = Female |
| `EDUCATION` | Categorical | Educational attainment | `1` = Graduate School, `2` = University, `3` = High School, `4` = Others |
| `MARRIAGE` | Categorical | Marital status | `1` = Married, `2` = Single, `3` = Others |
| `AGE` | Continuous | Age of client | 21 to 79 years |
| `PAY_0` | Ordinal | Repayment status in September 2005 (Most recent) | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `PAY_2` | Ordinal | Repayment status in August 2005 | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `PAY_3` | Ordinal | Repayment status in July 2005 | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `PAY_4` | Ordinal | Repayment status in June 2005 | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `PAY_5` | Ordinal | Repayment status in May 2005 | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `PAY_6` | Ordinal | Repayment status in April 2005 | `-1` = Paid duly, `0` = Revolving, `1` to `6` = Months delay |
| `BILL_AMT1` to `BILL_AMT6` | Continuous | Bill statement amount from September to April (NT$) | Numerical values (can include negative for credit balance) |
| `PAY_AMT1` to `PAY_AMT6` | Continuous | Amount of previous payment from September to April (NT$) | Numerical values ($\ge 0$) |

---

## 🧹 Data Cleaning & Preprocessing Pipeline

1. **Header & Index Normalization:**
   - Stripped redundant identification columns (`ID` / `Unnamed: 0`) to prevent data leakage and artificial memorization.
   - Standardized target label naming to `default`.
2. **Categorical Anomaly Remediation:**
   - In the raw dataset, `EDUCATION` contains undocumented values `0`, `5`, and `6`. These were logically mapped into Category `4` (`Other`).
   - `MARRIAGE` contains undocumented value `0`, which was mapped into Category `3` (`Other`).
3. **Missing Values & Integrity Verification:**
   - Rigorous validation confirmed zero null values and eliminated potential duplicate observations.
4. **Stratified Splitting:**
   - 80% Training ($24,000$ samples) and 20% Testing ($6,000$ samples) using `stratify=y` with `random_state=42`. This ensures both train and test splits retain the identical ~22% minority default ratio.
5. **Feature Standardization:**
   - Applied `StandardScaler` ($\mu=0, \sigma=1$) to prevent features with wide monetary scales (`LIMIT_BAL`, `BILL_AMT`, `PAY_AMT`) from overpowering gradient steps in scale-sensitive algorithms like Logistic Regression.
   - Scaler was fitted strictly on the training set to eliminate data leakage.

---

## 🔍 Exploratory Data Analysis (EDA) Insights

* **Imbalance Profile:** 78% of the customer base reliably repays, while 22% default. Training algorithms without class weighting results in high accuracy but dismal recall.
* **The Power of Recent Repayment (`PAY_0`):** Customers who delayed payment by 2 or more months in September (`PAY_0 >= 2`) exhibited an immediate default probability exceeding 65%. It is the single highest correlated predictor.
* **Credit Limit Disparities:** Defaulted customers consistently exhibited lower median credit limits (`LIMIT_BAL`) compared to non-defaulters.
* **Multicollinearity in Billing:** Consecutive bill amounts (`BILL_AMT1` through `BILL_AMT6`) showed strong cross-correlation ($r > 0.80$). Tree-based ensembles handle this natural collinearity effectively without requiring aggressive dimensionality reduction.

---

## 🤖 Machine Learning Modeling & Benchmark

Four distinct models representing linear, decision tree, bagging, and boosting paradigms were trained and systematically benchmarked on the test set ($N=6,000$):

| Model | Accuracy | Precision | Recall (Priority) | F1-Score | Key Hyperparameters / Settings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 68.3% | 38.4% | **65.1%** | 48.3% | `class_weight='balanced'`, `max_iter=1000` |
| **Decision Tree** | 77.2% | 48.6% | 58.7% | 53.2% | `max_depth=6`, `class_weight='balanced'` |
| **Random Forest (Champion)** | **79.6%** | **52.8%** | **61.4%** | **56.8%** | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'` |
| **Gradient Boosting** | 81.9% | 67.1% | 36.2% | 47.0% | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3` |

### 🏆 Model Selection Rationale
* While standard Gradient Boosting achieved the highest raw accuracy (81.9%), its **Recall was unacceptably low (36.2%)**, missing nearly two out of every three actual defaulters.
* Logistic Regression achieved high recall (65.1%) but suffered from excessive false positives (precision of only 38.4%).
* **Random Forest** was selected as the **Champion Model** because it provides the optimal balance: an impressive **61.4% Recall** combined with **79.6% Accuracy** and the highest overall **F1-Score (0.568)**, significantly outperforming unweighted classifiers in risk mitigation.

---

## 📈 Feature Importance & Risk Drivers

Extracted from the trained 200-tree Random Forest classifier:

```
PAY_0       [########################################] 24.58%
PAY_2       [###############                         ]  9.26%
PAY_4       [##########                              ]  6.25%
PAY_3       [#########                               ]  5.54%
LIMIT_BAL   [########                                ]  4.65%
PAY_AMT1    [#######                                 ]  4.47%
PAY_AMT2    [#######                                 ]  4.26%
BILL_AMT1   [######                                  ]  3.96%
PAY_6       [######                                  ]  3.81%
PAY_5       [#####                                   ]  3.49%
```

### Business Interpretation:
1. **Recent Repayment History Rules:** `PAY_0` through `PAY_4` account for over **45%** of the model's total predictive power. A customer's behavioral discipline in recent months is far more informative than static demographic variables.
2. **Credit Limit Sensitivity:** `LIMIT_BAL` is the 5th most significant driver; low credit limits often correlate with thinner financial buffers against emergencies.
3. **Demographics Play a Secondary Role:** Gender, Education, and Marital status provide marginal independent predictive signal compared to transactional payment flow.

---

## 🖥️ Streamlit Web Application

The interactive web application (`app.py`) provides an operational interface for loan underwriters and credit analysts:

* **Real-Time Scoring:** Evaluates all 23 input features and returns the default classification within milliseconds.
* **Probabilistic Risk Gauge:** Visual progress bar and percentage breakdown showing both Non-Default and Default probabilities.
* **Dynamic Visual Alerts:**
  - 🟢 **LOW RISK - UNLIKELY TO DEFAULT**: Highlighted in green for low-probability profiles.
  - 🔴 **HIGH RISK - LIKELY TO DEFAULT**: Clear alert triggered when default risk exceeds threshold.
* **Optimized Architecture:** Employs `@st.cache_resource` for instant model loading and verified column alignment matching the scikit-learn pipeline.

---

## 🔒 Security, Data Privacy & Breach Prevention

To ensure strict compliance and prevent any security or data breaches:

1. **Zero Personally Identifiable Information (PII):**
   - The dataset contains zero client names, social security numbers, tax identifiers, account numbers, physical addresses, or phone numbers.
   - All client identifiers are purely synthetic or omitted.
2. **Zero-Breach Credential Hardening (`.gitignore`):**
   - Configured with strict rules to prevent committing `.env`, API keys, `.streamlit/secrets.toml`, credentials JSON files, private keys, or certificates.
   - Sensitive local files and logs are explicitly ignored.
3. **Input Sanitization & Boundary Validation:**
   - The Streamlit interface enforces strict minimum and maximum bounds (`min_value`, `max_value`) on all numerical inputs, preventing buffer overflows, invalid negative numbers, or injection vectors.
4. **Safe Artifact Deserialization:**
   - Serialized pickle files (`random_forest_model.pkl`, `scaler.pkl`) are validated for expected scikit-learn classes and feature order schemas, mitigating unpickling vulnerabilities.

---

## 📁 Repository Structure

```
Customer_default_prediction/
├── .gitignore                          # Zero-breach git ignore rules (secrets, cache, virtualenvs)
├── Customer_Default_Prediction.ipynb  # End-to-end research, EDA, cleaning & training notebook
├── README.md                           # Comprehensive project documentation
├── app.py                              # Interactive Streamlit web application
├── random_forest_model.pkl             # Trained Random Forest classifier artifact
├── requirements.txt                    # Production dependencies for local run and Streamlit Cloud
├── requirement.txt                     # Fallback duplicate for automated package installers
└── scaler.pkl                          # Fitted StandardScaler preprocessing artifact
```

---

## ⚙️ Installation & Local Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/harshmishra21/credit-default-predictor.git
cd credit-default-predictor
```

### 2. Create and Activate a Virtual Environment
* **On macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Your default browser will automatically open:
```
Local URL: http://localhost:8501
Network URL: http://<your-ip>:8501
```

---

## 🚀 Deployment Guide (Streamlit Cloud & GitHub)

### Pushing to GitHub

1. Initialize git and check status:
   ```bash
   git init
   git status
   ```
2. Add all project files:
   ```bash
   git add .
   ```
3. Commit changes:
   ```bash
   git commit -m "feat: complete credit default prediction pipeline, streamlit app, and documentation"
   ```
4. Set branch to main, link your GitHub remote, and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/harshmishra21/credit-default-predictor.git
   git push -u origin main
   ```

### Deploying to Streamlit Community Cloud (Free)

1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `harshmishra21/credit-default-predictor`.
4. Set the branch to: `main`.
5. Set the Main file path to: `app.py`.
6. Click **"Deploy!"**.
7. Streamlit Cloud will read `requirements.txt`, install dependencies, load the serialized model, and publish your live app with a public URL!

---

## 🔮 Future Scope & Roadmap

- [ ] **Advanced Gradient Boosting:** Incorporate XGBoost, LightGBM, and CatBoost with Bayesian Hyperparameter Optimization (Optuna).
- [ ] **Resampling Strategies:** Integrate SMOTE (Synthetic Minority Over-sampling Technique) and Tomek Links to enhance minority-class boundary separation.
- [ ] **Model Explainability (XAI):** Integrate SHAP (SHapley Additive exPlanations) force plots and waterfall charts directly inside the Streamlit UI to explain individual customer risk decisions to underwriting officers.
- [ ] **RESTful API Service:** Package inference endpoints using **FastAPI** with Docker containerization and CI/CD via GitHub Actions.

---

## 📜 License & Acknowledgments

* **License:** This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
* **Dataset Attribution:** Courtesy of the **UCI Machine Learning Repository** and Lichman, M. (2013). *Default of Credit Card Clients Dataset*.
* **Authors:** Built with dedication by **Harsh Mishra** and **Sarthak Tajane**.
