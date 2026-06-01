import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/application_train.csv")

# Dataset Overview
print("\nDataset Shape:")
print(df.shape)

print("\nTop 10 Missing Values:")
print(df.isnull().sum().sort_values(ascending=False).head(10))

# Target Distribution
print("\nTarget Distribution:")
print(df["TARGET"].value_counts())

# Graph 1
plt.figure(figsize=(8, 5))
sns.countplot(x="TARGET", data=df)

plt.title("Loan Default Distribution")
plt.xlabel("Default Status")
plt.ylabel("Count")

plt.savefig("images/target_distribution.png")
plt.show()

# Income Statistics
print("\nIncome Statistics:")
print(df["AMT_INCOME_TOTAL"].describe())

# Graph 2 - Original
plt.figure(figsize=(10, 5))
sns.histplot(df["AMT_INCOME_TOTAL"], bins=50)

plt.title("Income Distribution (Including Outliers)")
plt.xlabel("Annual Income")
plt.ylabel("Count")

plt.savefig("images/income_distribution_original.png")
plt.show()

# Graph 3 - Without Extreme Outliers
filtered_income = df[df["AMT_INCOME_TOTAL"] < 500000]

plt.figure(figsize=(10, 5))
sns.histplot(filtered_income["AMT_INCOME_TOTAL"], bins=50)

plt.title("Income Distribution (Income < 500,000)")
plt.xlabel("Annual Income")
plt.ylabel("Count")

plt.savefig("images/income_distribution_clean.png")
plt.show()

default_rate = (
    df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
    .mean()
    .sort_values(ascending=False)
)

print(default_rate)

plt.figure(figsize=(10,5))

default_rate.plot(kind="bar")

plt.title("Default Rate by Education Level")
plt.ylabel("Default Rate")

plt.tight_layout()

plt.savefig("images/default_rate_by_education.png")

plt.show()

income_type_default = (
    df.groupby("NAME_INCOME_TYPE")["TARGET"]
    .mean()
    .sort_values(ascending=False)
)

print(income_type_default)

plt.figure(figsize=(10,5))
income_type_default.plot(kind="bar")

plt.title("Default Rate by Income Type")
plt.ylabel("Default Rate")

plt.tight_layout()

plt.savefig("images/default_rate_by_income_type.png")

plt.show()
print(df["NAME_INCOME_TYPE"].value_counts())

gender_default = (
    df.groupby("CODE_GENDER")["TARGET"]
    .mean()
    .sort_values(ascending=False)
)

print(gender_default)

plt.figure(figsize=(6,4))

gender_default.plot(kind="bar")

plt.title("Default Rate by Gender")
plt.ylabel("Default Rate")

plt.tight_layout()

plt.savefig("images/default_rate_by_gender.png")

plt.show()

