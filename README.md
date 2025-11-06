# Home Credit Default Risk Prediction

Many individuals face challenges in accessing loans due to limited or non-existent credit histories. Home Credit aims to promote financial inclusion by providing fair and responsible lending opportunities to the unbanked population. To achieve this, the company utilizes a wide range of alternative data sources—including telco usage patterns, transactional behavior, and demographic information—to assess clients’ repayment capabilities.

This project focuses on developing an end-to-end machine learning solution that predicts the likelihood of a client defaulting on a loan. By leveraging advanced models and optimization techniques, the goal is to help ensure that creditworthy clients are not rejected while minimizing the risk of default. This contributes to a fairer, data-driven approach to lending and supports Home Credit’s mission of empowering financial inclusion.

## Project Overview
The goal of this project is to **predict the probability of a customer defaulting on a loan** using real-world financial data from Home Credit Group.  
The objective is to help lenders make **responsible and data-driven credit decisions**, ensuring financial inclusion while minimizing default risk.

---

## Problem Statement
Financial institutions often struggle to identify clients who are likely to default on loans.  
This project builds a **binary classification model** that predicts whether a client will **default (1)** or **repay (0)** their loan.

---

## Dataset Description
The dataset consists of multiple related tables containing customer-level, credit-level, and behavioral information:

- `application_train.csv` – main applicant data  
- `bureau.csv`, `bureau_balance.csv` – historical external credit records  
- `previous_application.csv` – previous loan applications  
- `installments_payments.csv` – repayment history  
- `POS_CASH_balance.csv` – active point-of-sale loans  

Total size: **~300,000 applicants** and **hundreds of engineered features** after preprocessing.

---

## Data Preprocessing
- Merged multiple relational tables using applicant IDs.  
- Imputed missing values using **mean**, **median**, and **mode**.  
- Applied **RobustScaler** to handle outliers in financial data.  
- Handled class imbalance using **ADASYN** to oversample minority class (defaulters).  
- Encoded categorical variables using **One-Hot Encoding**, with fitted encoder saved via pickle for deployment.

---

## Feature Engineering
- Aggregated features from bureau and previous application data using mean, sum, and count statistics.  
- Created behavioral ratios such as `Credit_Income_Ratio`, `Payment_Rate`, and `Days_Employed`.  
- Dropped highly correlated features (|r| > 0.85) to reduce redundancy.  
- Selected top predictors using feature importance and variance thresholding.

---

## Exploratory Data Analysis (EDA)
- Visualized income, loan amount, occupation, and education trends vs. default probability.  
- Found higher default rates among low-income and short-employment applicants.  
- Analyzed correlations and patterns using heatmaps, boxplots, and bar charts.

---

## Model Development
Implemented and compared multiple classification algorithms:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- Naïve Bayes  

**XGBoost** achieved the best performance overall.

---

## Model Evaluation
Evaluation metrics used:
- **Accuracy**
- **Precision, Recall, F1-Score**
- **ROC-AUC**
- **Log Loss**
- **Average Precision Score**

### Key Results
| Model | Accuracy | Recall | ROC-AUC | Log Loss |
|:------|:----------|:--------|:----------|:-----------|
| Logistic Regression | 74% | 61% | 0.79 | 0.47 |
| Random Forest | 78% | 68% | 0.82 | 0.39 |
| XGBoost | **80%** | **70%** | **0.83** | **0.36** |

- **Recall prioritized** to minimize false negatives (i.e., undetected defaulters).
- Used **Youden’s J statistic** to determine the optimal probability threshold.

---

## Hyperparameter Tuning
- Conducted tuning using **Grid Search CV**, **Random Search CV**, and **HyperOpt**.
- **HyperOpt** (Bayesian optimization) achieved the best convergence and performance gains.

---

## Final Model
- **Chosen Model:** XGBoost  
- **Test Recall:** ~70%  
- **ROC-AUC:** ~0.83  
- **Key Predictors:** External credit scores (EXT_SOURCE), days employed, and credit-to-income ratio.

---

## Business Impact
- Helps financial institutions **identify high-risk applicants** early in the loan process.  
- Reduces financial loss by improving loan approval accuracy.  
- Enables better **credit scoring and customer risk segmentation**.

---

## Tech Stack
**Languages:** Python  
**Libraries:** pandas, numpy, scikit-learn, xgboost, hyperopt, matplotlib, seaborn  
**Tools:** Jupyter Notebook, Git, VS Code  
**Techniques:** Feature Engineering, Class Balancing, Hyperparameter Tuning, Model Explainability  

---

## Key Learnings
- Handling **multi-table datasets** and **imbalanced classification** in finance.  
- Importance of **recall optimization** for risk prediction.  
- End-to-end pipeline building — from **data ingestion → preprocessing → modeling → evaluation → deployment-ready outputs**.

---

## Tags
`Machine Learning` • `XGBoost` • `Credit Risk Modeling` • `Financial Data Science` • `Python` • `Feature Engineering` • `Imbalanced Classification` • `Hyperparameter Tuning`


Note:- Dataset is big kindly download it directly from kaggle via this link https://www.kaggle.com/c/home-credit-default-risk/
