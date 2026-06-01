import pandas as pd
import joblib

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================
# Load Dataset
# ==========================
print("Loading dataset...")

df = pd.read_csv("data/application_train.csv")

# ==========================
# Features and Target
# ==========================
X = df.drop(columns=["TARGET", "SK_ID_CURR"])
y = df["TARGET"]

# ==========================
# Identify Column Types
# ==========================
num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X.select_dtypes(
    include=["object", "string"]
).columns

print(f"\nNumerical Columns: {len(num_cols)}")
print(f"Categorical Columns: {len(cat_cols)}")

# ==========================
# Numerical Preprocessing
# ==========================
num_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# ==========================
# Categorical Preprocessing
# ==========================
cat_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ==========================
# Combine Preprocessing
# ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# ==========================
# LightGBM Model
# ==========================
model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced",
    random_state=42
)

# ==========================
# Full Pipeline
# ==========================
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==========================
# Train/Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")

# ==========================
# Train Model
# ==========================
pipeline.fit(X_train, y_train)

# ==========================
# Predictions
# ==========================
predictions = pipeline.predict(X_test)

probabilities = pipeline.predict_proba(X_test)[:, 1]

# ==========================
# Evaluation Metrics
# ==========================
accuracy = accuracy_score(y_test, predictions)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================
# Save Model
# ==========================
joblib.dump(
    pipeline,
    "models/credit_risk_model.pkl"
)

print("\nModel saved successfully!")
print("Saved as: models/credit_risk_model.pkl")