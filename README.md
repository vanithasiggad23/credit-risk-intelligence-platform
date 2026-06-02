DATASET LINK: https://www.kaggle.com/competitions/home-credit-default-risk/data
APPLICATION_TRAIN.CSV
APPLICATION_TEST.CSV

# 1. Dataset Overview

The Home Credit Default Risk dataset was loaded successfully using Pandas for analysis.

## Dataset Dimensions

* Number of records (loan applications): 307,511
* Number of features: 122

This indicates a large-scale dataset containing information related to applicant demographics, financial status, credit history, housing information, and previous loan behavior.

# 2. Initial Data Inspection

The first five records were examined using the `head()` function to understand the structure of the dataset.

### Key Observations

* The target variable is `TARGET`.
* `TARGET = 1` indicates that the client had payment difficulties or defaulted.
* `TARGET = 0` indicates that the client successfully repaid the loan.
* Features include demographic information, loan details, income information, housing attributes, and credit bureau information.

# 3. Data Quality Assessment

A missing value analysis was performed on all features.

## Top Features with Missing Values

| Feature                  | Missing Values |
| ------------------------ | -------------- |
| COMMONAREA_AVG           | 214,865        |
| COMMONAREA_MODE          | 214,865        |
| COMMONAREA_MEDI          | 214,865        |
| NONLIVINGAPARTMENTS_MEDI | 213,514        |
| NONLIVINGAPARTMENTS_MODE | 213,514        |
| NONLIVINGAPARTMENTS_AVG  | 213,514        |
| FONDKAPREMONT_MODE       | 210,295        |
| LIVINGAPARTMENTS_AVG     | 210,199        |
| LIVINGAPARTMENTS_MEDI    | 210,199        |
| LIVINGAPARTMENTS_MODE    | 210,199        |

## Observations

* Several housing-related variables contain a very large number of missing values.
* Many of these features have more than 65–70% missing data.
* These features may require:

  * Removal from the dataset,
  * Missing value imputation, or
  * Further investigation before model training.

# 4. Business Interpretation

The dataset appears to combine multiple sources of customer information, including:

1. Applicant demographics
2. Income and employment details
3. Loan and credit information
4. Housing characteristics
5. Historical credit behavior

The presence of extensive missing values in housing-related attributes suggests that this information may not be available for many applicants and may have limited predictive value unless properly handled.
# 5. Target Variable Analysis

The target variable (`TARGET`) represents whether a customer defaulted on a loan.

## Distribution of Target Variable

| TARGET Value | Meaning    | Count   |
| ------------ | ---------- | ------- |
| 0            | No Default | 282,686 |
| 1            | Default    | 24,825  |

## Percentage Distribution

### Non-Defaulters

[
\frac{282686}{307511} \times 100 \approx 91.93%
]

Approximately 91.93% of applicants successfully repaid their loans.

### Defaulters

[
\frac{24825}{307511} \times 100 \approx 8.07%
]

Approximately 8.07% of applicants defaulted on their loans.

## Key Observation: Class Imbalance

The dataset is highly imbalanced:

* Majority Class (Non-Default): 91.93%
* Minority Class (Default): 8.07%

This imbalance can lead machine learning models to favor the majority class and achieve misleadingly high accuracy.

For example, a model that predicts every applicant as "No Default" would achieve approximately 92% accuracy while providing no real business value.

## Implications for Model Development

To address class imbalance, the following techniques may be considered:

* Class weighting
* Random oversampling
* SMOTE (Synthetic Minority Oversampling Technique)
* Evaluation using Precision, Recall, F1-Score, and ROC-AUC instead of Accuracy alone

## Business Interpretation

Loan defaults are relatively rare events in the dataset. Therefore, identifying high-risk applicants becomes a critical task for the credit risk prediction model. The model should prioritize correctly detecting defaulters while maintaining acceptable false positive rates.
## Business Insight 1: Loan Defaults Are Relatively Rare

The target variable distribution shows that approximately 91.93% of applicants did not default, while only 8.07% defaulted on their loans.

This indicates that loan defaults are relatively rare events. The dataset is highly imbalanced, which makes default prediction a challenging problem.

From a business perspective, the bank must focus on accurately identifying the small group of high-risk applicants while avoiding excessive rejection of low-risk customers.
## Business Insight 2: Most Loan Applicants Belong to the Middle-Income Segment

The income distribution shows that the majority of applicants earn between 100,000 and 200,000 annually.

### Supporting Statistics

* Median Income: 147,150
* 75% of applicants earn less than 202,500
* Very high-income applicants are relatively rare

### Business Interpretation

The bank's primary customer base consists of middle-income individuals. Credit risk strategies and loan products should therefore be optimized for this segment.

The presence of a small number of extremely high-income applicants creates significant outliers that may require special treatment during model development.

## Business Insight 3: Education Level Is Strongly Associated with Credit Risk

Analysis of default rates across education levels reveals a clear relationship between educational attainment and loan repayment behavior.

### Default Rates by Education Level

| Education Level               | Approximate Default Rate |
| ----------------------------- | ------------------------ |
| Lower Secondary               | 10.9%                    |
| Secondary / Secondary Special | 8.9%                     |
| Incomplete Higher             | 8.5%                     |
| Higher Education              | 5.4%                     |
| Academic Degree               | 1.8%                     |

### Observation

Applicants with lower educational qualifications exhibit significantly higher default rates compared to applicants with higher educational attainment.

### Business Interpretation

Education level appears to be a strong indicator of creditworthiness. Applicants with higher education and academic degrees demonstrate substantially lower default rates, suggesting greater financial stability and repayment capacity.

This feature is likely to be an important predictor in the credit risk model.

## Business Insight 4: Income Type Influences Default Risk

Default rates vary significantly across different income sources.

### Observed Default Rates

| Income Type          | Default Rate |
| -------------------- | ------------ |
| Maternity Leave      | 40.0%        |
| Unemployed           | 36.4%        |
| Working              | 9.6%         |
| Commercial Associate | 7.5%         |
| State Servant        | 5.8%         |
| Pensioner            | 5.4%         |

### Observation

Applicants without stable employment income exhibit substantially higher default rates than salaried employees, pensioners, and government workers.

### Business Interpretation

Income source appears to be an important risk indicator. Applicants with stable and predictable income streams generally demonstrate lower default rates, while applicants with irregular or absent income sources show elevated credit risk.

This feature is expected to contribute significantly to credit risk prediction models.

## Business Insight 5: Male Applicants Exhibit Higher Default Rates

Analysis of loan repayment behavior by gender reveals noticeable differences in default rates.

### Observed Default Rates

| Gender | Approximate Default Rate |
| ------ | ------------------------ |
| Male   | 10.1%                    |
| Female | 7.0%                     |

### Observation

Male applicants demonstrate a higher default rate compared to female applicants.

### Business Interpretation

Gender appears to have a measurable relationship with loan repayment behavior in this dataset. Male applicants show a greater likelihood of default, suggesting that demographic characteristics may contribute to credit risk assessment.

However, such variables should be handled carefully to ensure compliance with applicable fairness, ethical, and regulatory requirements.

# Model Comparison

## Random Forest (Baseline)

### Results

* Accuracy: 91.97%
* Recall (Default Class): ~0%
* Defaulters Detected: 10 out of 4,949

### Observation

The model achieved high accuracy but failed to identify default cases. The high accuracy was primarily due to class imbalance.

---

## LightGBM (Improved Model)

### Results

* Accuracy: 70.29%
* Recall (Default Class): 68%
* Defaulters Detected: 3,342 out of 4,949

### Observation

Although overall accuracy decreased, the model became significantly better at identifying high-risk applicants.

### Business Interpretation

For credit risk prediction, detecting potential defaulters is more important than maximizing overall accuracy. Therefore, the LightGBM model provides substantially greater business value.


## Final LightGBM Model Performance

### Evaluation Metrics

* Accuracy: 70.78%
* ROC-AUC Score: 0.7603
* Recall (Default Class): 68%
* Precision (Default Class): 17%

### Confusion Matrix

| Actual / Predicted | Non-Default | Default |
| ------------------ | ----------- | ------- |
| Non-Default        | 40,166      | 16,372  |
| Default            | 1,602       | 3,363   |

### Business Interpretation

The model successfully identifies approximately 68% of default cases.

Although overall accuracy is lower than the baseline Random Forest model, the LightGBM model provides substantially greater business value because it actively detects high-risk applicants rather than classifying nearly all applicants as low risk.

The ROC-AUC score of 0.7603 indicates good discriminatory power between defaulters and non-defaulters.

# Risk Scoring Framework

The trained LightGBM model outputs the probability of loan default for each applicant.

## Risk Band Assignment

| Probability of Default | Risk Category |
| ---------------------- | ------------- |
| 0.00 – 0.29            | Low Risk      |
| 0.30 – 0.69            | Medium Risk   |
| 0.70 – 1.00            | High Risk     |

## Example Prediction

* Default Probability: 0.8698
* Risk Category: High Risk

This risk scoring framework enables business users to interpret model outputs more easily and supports credit decision-making.
