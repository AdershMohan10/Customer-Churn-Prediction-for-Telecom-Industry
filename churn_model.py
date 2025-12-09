import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pickle

print("Loading data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ----------------------------------------
# CLEANING
# ----------------------------------------
print("Cleaning data...")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)

# ----------------------------------------
# FEATURES
# ----------------------------------------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [col for col in df.columns if col not in numeric_features + ["Churn"]]

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------
# ENCODING
# ----------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

print("Applying encodings...")
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# ----------------------------------------
# CLASS IMBALANCE HANDLING
# ----------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ----------------------------------------
# XGBOOST MODEL (compatible settings)
# ----------------------------------------
print("Training XGBoost 3.1.2...")

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
)

model.fit(X_train, y_train)

# ----------------------------------------
# EVALUATION
# ----------------------------------------
print("\nEvaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------
print("Saving model as churn_xgb_model.pkl...")
with open("churn_xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Saving cleaned_telecom_data.csv...")
df.to_csv("cleaned_telecom_data.csv", index=False)

print("\n🎉 DONE — Model trained successfully!")
