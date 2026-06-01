import pandas as pd
import joblib

# Load model
model = joblib.load("models/credit_risk_model.pkl")

# Load sample data
df = pd.read_csv("data/application_train.csv")

# Remove target and ID
X = df.drop(columns=["TARGET", "SK_ID_CURR"])

# Take first applicant
sample = X.iloc[[0]]

# Predict probability
probability = model.predict_proba(sample)[0][1]

print(f"Default Probability: {probability:.4f}")

# Risk Band
if probability < 0.30:
    risk = "Low Risk"
elif probability < 0.70:
    risk = "Medium Risk"
else:
    risk = "High Risk"

print(f"Risk Band: {risk}")