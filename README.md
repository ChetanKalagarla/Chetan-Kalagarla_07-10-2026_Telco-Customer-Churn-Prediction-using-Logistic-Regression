# 📡 Telco Customer Churn Prediction

## Project Overview
This project predicts whether a telecom customer is **Likely to Churn** or **Likely to Stay**.

The solution uses:
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Business Goal
Identify customers who may leave the company so that the business can provide retention offers before they churn.

## Project Structure
```text
Telco_Customer_Churn_Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── churn_distribution.png
│   └── model_results.txt
│
├── src/
│   └── churn_prediction.py
│
├── requirements.txt
└── README.md
```

## Step 1 — Download the Dataset
Download the **Telco Customer Churn** CSV dataset from Kaggle.

Place the CSV inside:
```text
data/
```

Rename it to:
```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Step 2 — Install Packages
Open the VS Code / Visual Studio terminal in this project folder:

```bash
pip install -r requirements.txt
```

## Step 3 — Run
```bash
python src/churn_prediction.py
```

## What the program does
1. Loads the dataset
2. Performs basic data understanding
3. Cleans the data
4. Selects relevant customer features
5. Encodes categorical data
6. Splits data into training and testing sets
7. Trains a Logistic Regression model
8. Calculates churn probability
9. Classifies customers as Likely to Churn / Likely to Stay
10. Evaluates Accuracy, Precision, Recall and F1-score
11. Displays a confusion matrix
12. Creates predictions for unseen customer records
13. Saves useful outputs

## Business Interpretation
In churn prediction, missing a customer who is actually going to leave can be more costly than wrongly targeting a loyal customer. Therefore, **Recall for the Churn class** is an important metric.

## One Improvement
A useful improvement is to tune the classification threshold based on business cost. Instead of always using 0.50, the company can choose a threshold such as 0.40 if the priority is to catch more potential churn customers.
