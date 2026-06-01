# Exploratory Data Analysis (EDA) Findings

## Dataset Overview

The Home Credit Default Risk dataset contains:

* Total Records: 307,511
* Total Features: 122

The dataset includes applicant demographics, financial information, employment details, credit history, housing information, and loan-related attributes.

---

## Data Quality Assessment

### Features with Highest Missing Values

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

### Observation

Several housing-related variables contain more than 65–70% missing values. These features may require removal or imputation during preprocessing.

---

# Business Insight 1: Class Imbalance

### Target Distribution

| Target | Meaning    | Count   |
| ------ | ---------- | ------- |
| 0      | No Default | 282,686 |
| 1      | Default    | 24,825  |

### Observation

* Non-default rate: 91.93%
* Default rate: 8.07%

The dataset is highly imbalanced and requires special handling during model training.

---

# Business Insight 2: Income Distribution

### Key Statistics

* Mean Income: 168,798
* Median Income: 147,150
* Maximum Income: 117,000,000

### Observation

The income distribution is highly right-skewed with several extreme outliers.

### Business Interpretation

Most applicants belong to the middle-income segment, while a small number of applicants report exceptionally high incomes.

---

# Business Insight 3: Education and Default Risk

| Education Level               | Default Rate |
| ----------------------------- | ------------ |
| Lower Secondary               | 10.9%        |
| Secondary / Secondary Special | 8.9%         |
| Incomplete Higher             | 8.5%         |
| Higher Education              | 5.4%         |
| Academic Degree               | 1.8%         |

### Observation

Higher educational attainment is associated with lower default rates.

### Business Interpretation

Education level appears to be an important predictor of creditworthiness.

---

# Business Insight 4: Income Type and Default Risk

| Income Type          | Default Rate |
| -------------------- | ------------ |
| Maternity Leave      | 40.0%        |
| Unemployed           | 36.4%        |
| Working              | 9.6%         |
| Commercial Associate | 7.5%         |
| State Servant        | 5.8%         |
| Pensioner            | 5.4%         |

### Observation

Applicants with unstable or no regular income exhibit significantly higher default rates.

### Business Interpretation

Income source is a strong indicator of repayment capability and should be considered in risk assessment.

---

# Business Insight 5: Gender and Default Risk

| Gender | Default Rate |
| ------ | ------------ |
| Male   | 10.1%        |
| Female | 7.0%         |

### Observation

Male applicants exhibit a higher default rate compared to female applicants.

### Business Interpretation

Demographic characteristics show measurable relationships with loan repayment behavior.

---

# Generated Visualizations

1. target_distribution.png
2. income_distribution_original.png
3. income_distribution_clean.png
4. education_vs_default.png
5. default_rate_by_education.png
6. default_rate_by_income_type.png
7. default_rate_by_gender.png

## Initial Baseline Model Results

A Random Forest classifier was trained as the baseline model.

### Results

* Accuracy: 91.97%
* Precision (Default Class): 0.91
* Recall (Default Class): 0.00
* F1 Score (Default Class): 0.00

### Observation

Although the model achieved high overall accuracy, it failed to identify most default cases.

The model correctly classified only 10 out of 4,949 default instances in the test set.

### Interpretation

The high accuracy is misleading due to severe class imbalance in the dataset. Since approximately 92% of applicants do not default, the model learns to favor the majority class.

Additional techniques are required to improve detection of default cases.

## Baseline Model Evaluation

A Random Forest classifier was initially trained as a baseline model.

Although the model achieved approximately 92% accuracy, it failed to effectively identify default cases due to severe class imbalance.

Applying class weighting produced minimal improvement in recall. Therefore, a gradient boosting approach (LightGBM) was selected for further experimentation because boosting models are generally more effective for imbalanced tabular credit-risk datasets.
