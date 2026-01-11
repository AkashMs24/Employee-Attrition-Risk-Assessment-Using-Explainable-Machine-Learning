import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("employee_data.csv")

df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "OverTime"
]

df = df[features + ["Attrition"]]

# Encode OverTime
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X = df[features]
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# 5. Model Training
# ---------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------------------
# 6. Evaluation
# ---------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# ---------------------------
# 7. Save Model
# ---------------------------
joblib.dump(model, "rf_model_4_features.pkl")

print("\nModel saved as rf_model_4_features.pkl")


