"""
============================================================
TELCO CUSTOMER CHURN PREDICTION
============================================================
Author : Chetan Kalagarla
Purpose: Predict whether a telecom customer is likely to
         churn or stay using Machine Learning.

Model   : Logistic Regression
Target  : Churn
============================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 65)
print("        TELCO CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 65)

if not DATA_FILE.exists():
    print("\n❌ Dataset not found!")
    print(f"Expected location:\n{DATA_FILE}")
    print("\nDownload the Telco Customer Churn CSV and place it in the data folder.")
    raise SystemExit

df = pd.read_csv(DATA_FILE)

print("\n[1] DATASET LOADED SUCCESSFULLY")
print("-" * 65)
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. BASIC DATA UNDERSTANDING
# ============================================================

print("\n[2] DATASET OVERVIEW")
print("-" * 65)

print("\nFirst 5 records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

print("\nTarget distribution:")
print(df["Churn"].value_counts())


# ============================================================
# 4. DATA CLEANING
# ============================================================

print("\n[3] DATA CLEANING")
print("-" * 65)

# TotalCharges sometimes contains blank strings.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove customer ID because it is an identifier, not a useful
# predictive feature.
df = df.drop(columns=["customerID"], errors="ignore")

# Convert target to numeric:
# Yes = 1 (Churn)
# No  = 0 (Stay)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("✓ Customer ID removed")
print("✓ TotalCharges converted to numeric")
print("✓ Churn converted to 0/1")


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

print("\n[4] FEATURE SELECTION")
print("-" * 65)

selected_features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

selected_features = [col for col in selected_features if col in df.columns]

X = df[selected_features]
y = df["Churn"]

print("Selected customer attributes:")
for feature in selected_features:
    print(f"  • {feature}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

print("\n[5] TRAIN / TEST SPLIT")
print("-" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"Training records : {len(X_train)}")
print(f"Testing records  : {len(X_test)}")


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 8. BUILD MACHINE LEARNING MODEL
# ============================================================

print("\n[6] BUILDING MACHINE LEARNING MODEL")
print("-" * 65)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

print("✓ Logistic Regression model trained successfully")


# ============================================================
# 9. PREDICTION
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

# Standard classification threshold
THRESHOLD = 0.50

y_pred = (y_probability >= THRESHOLD).astype(int)


# ============================================================
# 10. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)

true_negative, false_positive, false_negative, true_positive = cm.ravel()

print("\n[7] MODEL PERFORMANCE")
print("-" * 65)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Stay", "Churn"],
    zero_division=0,
))

print("\nCONFUSION MATRIX ANALYSIS")
print("-" * 65)
print(f"Correctly identified churn customers : {true_positive}")
print(f"Churn customers missed               : {false_negative}")
print(f"Non-churn customers correctly found  : {true_negative}")
print(f"Non-churn customers wrongly flagged  : {false_positive}")


# ============================================================
# 11. SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted Stay", "Predicted Churn"],
    yticklabels=["Actual Stay", "Actual Churn"],
)

plt.title("Customer Churn - Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.tight_layout()

confusion_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(confusion_path, dpi=200)
plt.close()


# ============================================================
# 12. CHURN DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

df["Churn"].map({0: "Stay", 1: "Churn"}).value_counts().plot(
    kind="bar"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()

distribution_path = OUTPUT_DIR / "churn_distribution.png"
plt.savefig(distribution_path, dpi=200)
plt.close()


# ============================================================
# 13. PREDICT UNSEEN CUSTOMER RECORDS
# ============================================================

print("\n[8] UNSEEN CUSTOMER PREDICTION")
print("-" * 65)

# These are example customer records that were not used for training.
# The model predicts their churn probability.

unseen_customers = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 75.00,
            "TotalCharges": 150.00,
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 80.00,
            "TotalCharges": 4800.00,
        },
    ]
)

# Keep only the model's selected feature order.
unseen_customers = unseen_customers[selected_features]

unseen_probability = model.predict_proba(unseen_customers)[:, 1]

unseen_result = unseen_customers.copy()
unseen_result["Churn Probability"] = unseen_probability
unseen_result["Prediction"] = np.where(
    unseen_probability >= THRESHOLD,
    "Likely to Churn",
    "Likely to Stay",
)

print(unseen_result[["Churn Probability", "Prediction"]].to_string(index=False))


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["Actual Churn"] = y_test.values
test_results["Churn Probability"] = y_probability
test_results["Prediction"] = np.where(
    y_pred == 1,
    "Likely to Churn",
    "Likely to Stay",
)

prediction_path = OUTPUT_DIR / "customer_predictions.csv"
test_results.to_csv(prediction_path, index=False)


# ============================================================
# 15. SAVE MODEL REPORT
# ============================================================

report = f"""
TELCO CUSTOMER CHURN - MODEL REPORT
====================================

Model:
Logistic Regression

Classification threshold:
{THRESHOLD}

Dataset:
{len(df)} customer records

Training records:
{len(X_train)}

Testing records:
{len(X_test)}

Performance:
Accuracy  = {accuracy:.4f}
Precision = {precision:.4f}
Recall    = {recall:.4f}
F1 Score  = {f1:.4f}
ROC-AUC   = {roc_auc:.4f}

Confusion Matrix:
True Negative  = {true_negative}
False Positive = {false_positive}
False Negative = {false_negative}
True Positive  = {true_positive}

Business Interpretation:
The model correctly identified {true_positive} actual churn customers.
It wrongly flagged {false_positive} customers who did not churn.
It missed {false_negative} customers who actually churned.

Business Recommendation:
Missing an actual churn customer can result in lost recurring revenue.
Therefore, churn recall should receive strong attention when selecting
the final model and classification threshold.

Suggested Improvement:
Tune the probability threshold according to the financial cost of
false positives and false negatives. A lower threshold can identify
more potential churn customers for proactive retention campaigns.
"""

report_path = OUTPUT_DIR / "model_results.txt"
report_path.write_text(report, encoding="utf-8")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("                 PROJECT COMPLETED")
print("=" * 65)

print(f"""
✓ Model trained successfully
✓ Churn probability calculated
✓ Customers classified
✓ Model evaluated
✓ Unseen customers predicted
✓ Confusion matrix saved
✓ Customer predictions saved
✓ Model report saved

Output folder:
{OUTPUT_DIR}

Important business metric:
Recall = {recall:.4f}

This measures how many actual churn customers were successfully
identified by the model.
""")

print("=" * 65)
