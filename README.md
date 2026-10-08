<div align="center">

# 📡 Telco Customer Churn Prediction

### Predicting Customer Churn using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-2E7D32?style=for-the-badge)

**A machine-learning solution to identify telecom customers who are likely to leave and support proactive retention.**

</div>

---

## 🎯 Project Objective

Customer churn directly affects telecom revenue. When a customer leaves, the company loses recurring revenue and may spend additional money to acquire a replacement customer.

This project builds a **Customer Churn Prediction System** that:

- 📊 Understands customer information
- 🧹 Cleans and prepares the data
- 🔎 Selects relevant customer features
- 🤖 Trains a Logistic Regression model
- 📈 Estimates churn probability
- 🏷️ Classifies customers as **Likely to Churn** or **Likely to Stay**
- 🧪 Evaluates model performance
- 💼 Converts model results into business insights

---

## 🧠 Machine Learning Approach

| Component | Used in Project |
|---|---|
| Problem Type | Binary Classification |
| Target | `Churn` |
| Model | Logistic Regression |
| Churn | `1` / `Yes` |
| Stay | `0` / `No` |
| Classification Threshold | `0.50` |
| Train/Test Split | `80% / 20%` |
| Random State | `42` |

### Prediction Flow

```text
Customer Data
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Encoding + Scaling
     ↓
Logistic Regression
     ↓
Churn Probability
     ↓
Probability ≥ 0.50 → Likely to Churn
Probability < 0.50 → Likely to Stay

Telco-Customer-Churn-Prediction
│
├── 📁 data
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 📁 outputs
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── customer_predictions.csv
│   └── model_results.txt
│
├── 📁 src
│   └── churn_prediction.py
│
├── 📁 .vscode
│   └── launch.json
│
├── 📄 README.md
└── 📄 requirements.txt
