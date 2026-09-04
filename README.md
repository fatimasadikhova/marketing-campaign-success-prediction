# 📊 Marketing Campaign Success Prediction

## 📌 Project Overview

This project focuses on developing a **Machine Learning classification solution for predicting marketing campaign outcomes** based on customer demographic, financial, communication, and campaign-related characteristics.

The main objective is to transform historical customer and campaign data into predictive insights that can support **more targeted marketing strategies, better customer prioritization, and data-driven decision-making**.

Two different Machine Learning approaches were developed and compared:

* **Logistic Regression**
* **Random Forest**

The project goes beyond basic model training by implementing **feature engineering, WOE transformation, feature selection, correlation analysis, feature importance analysis, overfitting detection, hyperparameter optimization, cross-validation, and model evaluation**.

---

## 🎯 Business Problem

Marketing campaigns can involve thousands of customers with different demographic, financial, and behavioral characteristics.

Treating all customers equally may lead to:

* inefficient use of marketing resources,
* unnecessary customer contacts,
* lower campaign effectiveness,
* difficulty identifying high-potential customer segments.

A predictive classification model can help estimate campaign outcomes and provide a data-driven basis for **customer targeting and campaign optimization**.

Therefore, the purpose of this project is to build and compare predictive models capable of identifying patterns associated with campaign success.

---

# 📂 Dataset

The dataset contains **12,870 observations** and initially includes **17 variables**.

### Main Features

| Feature     | Description                                    |
| ----------- | ---------------------------------------------- |
| `age`       | Customer age                                   |
| `job`       | Customer occupation                            |
| `marital`   | Marital status                                 |
| `education` | Education level                                |
| `default`   | Credit default status                          |
| `balance`   | Average yearly account balance                 |
| `housing`   | Housing loan status                            |
| `loan`      | Personal loan status                           |
| `contact`   | Contact communication method                   |
| `day`       | Day of contact                                 |
| `month`     | Month of contact                               |
| `campaign`  | Number of contacts during the current campaign |
| `pdays`     | Number of days since previous contact          |
| `previous`  | Number of previous contacts                    |
| `response`  | Outcome of previous campaign                   |
| `result`    | Current campaign outcome / target              |

The target variable was converted into a binary classification format:

```text
yes → 0
no  → 1
```

The resulting target distribution contains:

* **3,967 observations — Class 0**
* **8,903 observations — Class 1**

---

# 🔎 Exploratory Data Analysis

The project starts with an exploratory analysis of the dataset.

The following steps were performed:

* Dataset structure inspection
* Data type analysis
* Descriptive statistics
* Unique value analysis
* Missing value analysis
* Target variable distribution
* Numerical feature inspection
* Categorical feature inspection

The analysis was used to understand the structure and quality of the dataset before model development.

---

# 🧹 Data Preprocessing

Several preprocessing steps were applied before modeling.

### Feature Removal

The following variables were removed:

```text
ID
previous
pdays
```

`ID` was excluded because it represents an identifier rather than predictive information.

The remaining variables were used for the Machine Learning pipelines.

---

# ✂️ Train / Test Split

The dataset was divided into training and testing datasets.

```text
Training set: 80%
Testing set: 20%
```

This resulted in:

```text
Train: 10,296 observations
Test:   2,574 observations
```

A fixed:

```python
random_state = 42
```

was used to ensure reproducibility.

---

# 🤖 Machine Learning Approach

Two separate modeling pipelines were developed:

```text
                    Dataset
                       │
              Data Preprocessing
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
 Logistic Regression           Random Forest
          │                         │
    WOE Transformation       Label Encoding
          │                         │
 Correlation Analysis        Feature Importance
          │                         │
 Feature Selection           Feature Selection
          │                         │
          ▼                         ▼
 Logistic Regression          Hyperparameter
                               Optimization
          │                         │
          └────────────┬────────────┘
                       ▼
                Model Comparison
```

---

# 🔵 Logistic Regression Pipeline

Logistic Regression was implemented as an interpretable statistical classification approach.

## 1. Feature Separation

Features were divided into:

### Numerical Features

```text
age
balance
day
campaign
```

### Categorical Features

```text
job
marital
education
default
housing
loan
contact
month
response
```

---

## 2. Quantile Binning

Numerical variables were transformed into quantile-based bins.

The following percentiles were used:

```text
25th percentile
50th percentile
75th percentile
```

This produced approximately four groups for each numerical variable.

Duplicate bin boundaries were handled using:

```python
duplicates='drop'
```

---

# 📐 Weight of Evidence (WOE)

WOE transformation was applied to the binned numerical variables and categorical variables.

The resulting features included:

```text
age_woe
job_woe
marital_woe
education_woe
default_woe
balance_woe
housing_woe
loan_woe
contact_woe
day_woe
month_woe
campaign_woe
response_woe
```

WOE was used to transform categorical/binned information into numerical representations based on the relationship between feature categories and the target variable.

This approach also improves the interpretability and structure of the Logistic Regression modeling pipeline.

---

# 🔗 Spearman Intercorrelation Analysis

After WOE transformation, correlations between WOE variables were analyzed using **Spearman correlation**.

A threshold of:

```text
|Correlation| ≥ 0.70
```

was used to identify highly correlated variables.

This step helped detect potentially redundant predictors before model development.

---

# 🎯 Univariate Feature Selection

Each WOE feature was evaluated individually using Logistic Regression.

The following criteria were used:

```text
Train Gini > 10%
Test Gini > 10%
Gini Gap < 5%
```

The purpose was to select features that demonstrated both:

* predictive power,
* stability between training and testing data.

### Selected Features

```text
marital_woe
education_woe
balance_woe
housing_woe
contact_woe
month_woe
campaign_woe
response_woe
```

---

# 🟢 Random Forest Pipeline

A separate pipeline was developed using Random Forest to capture **non-linear relationships and interactions** between predictors.

---

## 1. Categorical Encoding

Categorical variables were converted into numerical representations using:

```python
LabelEncoder
```

Separate encoders were maintained for the categorical variables.

---

## 2. Initial Random Forest

The initial model was created using:

```python
RandomForestClassifier(
    n_estimators=300,
    random_state=42
)
```

The initial Random Forest achieved very high training performance but demonstrated a significant difference between training and testing results.

### Initial Results

| Metric    |   Train |   Test |
| --------- | ------: | -----: |
| Gini      | 100.00% | 55.10% |
| Precision | 100.00% | 80.98% |
| Recall    | 100.00% | 90.52% |

The resulting:

```text
Train-Test Gini Gap = 44.90%
```

was a strong indication of **overfitting**.

---

# 🌲 Random Forest Feature Importance

Feature importance was extracted from the initial Random Forest model.

### Feature Importance

| Feature     | Importance |
| ----------- | ---------: |
| `balance`   |     18.83% |
| `age`       |     16.70% |
| `day`       |     14.04% |
| `month`     |     11.50% |
| `job`       |      7.62% |
| `campaign`  |      6.95% |
| `response`  |      6.90% |
| `contact`   |      4.56% |
| `education` |      4.09% |
| `marital`   |      3.51% |
| `housing`   |      3.38% |
| `loan`      |      1.62% |
| `default`   |      0.29% |

Features with importance between:

```text
1% ≤ Importance ≤ 35%
```

were retained for the next stage.

---

# ⚙️ Hyperparameter Optimization

Because the initial Random Forest showed substantial overfitting, hyperparameter optimization was performed using:

```python
RandomizedSearchCV
```

The search included:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
class_weight
```

### Optimization Configuration

```text
Random parameter combinations: 50
Cross-validation: 5-Fold
Scoring metric: ROC-AUC
Total model fits: 250
```

The best cross-validation ROC-AUC was approximately:

```text
0.7861
```

---

# 🏆 Optimized Random Forest

The selected Random Forest configuration was:

```python
RandomForestClassifier(
    n_estimators=700,
    min_samples_split=5,
    min_samples_leaf=10,
    max_features=None,
    max_depth=None,
    class_weight=None,
    n_jobs=-1,
    random_state=42
)
```

The optimization improved test-set performance compared with the initial Random Forest while reducing, although not eliminating, the generalization gap.

---

# 📏 Model Evaluation

The models were evaluated using several metrics.

## Gini

Gini was calculated from ROC-AUC:

```text
Gini = 2 × AUC − 1
```

Gini was used as one of the primary metrics for evaluating model discriminatory power.

---

## Precision

Precision measures the proportion of predicted positive observations that were actually positive.

---

## Recall

Recall measures the proportion of actual positive observations correctly identified by the model.

---

## Train-Test Gini Gap

The difference between training and testing Gini was used as an indicator of:

* model stability,
* generalization,
* overfitting.

A smaller gap indicates more consistent performance between training and unseen data.

---

# 📊 Model Comparison

| Model                                   | Train Gini |  Test Gini |  Gini Gap | Test Precision | Test Recall |
| --------------------------------------- | ---------: | ---------: | --------: | -------------: | ----------: |
| Logistic Regression – Initial           |     53.12% |     50.51% |     2.61% |         78.67% |      93.50% |
| Logistic Regression – Selected Features |     51.53% |     49.52% | **2.01%** |         78.24% |      93.72% |
| Random Forest – Initial                 |    100.00% |     55.10% |    44.90% |         80.98% |      90.52% |
| Random Forest – Feature Selection       |    100.00% |     54.53% |    45.47% |         80.96% |      90.41% |
| Random Forest – Optimized               |     85.33% | **55.76%** |    29.57% |     **81.06%** |      90.80% |

---

# 🔍 Key Insights

The comparison demonstrates an important trade-off between **predictive performance and model stability**.

### Logistic Regression

The selected Logistic Regression model achieved:

```text
Test Gini: 49.52%
Test Precision: 78.24%
Test Recall: 93.72%
Gini Gap: 2.01%
```

Its relatively small Gini gap indicates strong consistency between training and testing performance.

---

### Random Forest

The optimized Random Forest achieved:

```text
Test Gini: 55.76%
Test Precision: 81.06%
Test Recall: 90.80%
Gini Gap: 29.57%
```

It achieved the highest test Gini and precision among the evaluated models.

However, the significantly larger Gini gap indicates that the model still has a considerable generalization gap compared with Logistic Regression.

---

# 💡 Main Learning

One of the main conclusions from this project is that **model performance should not be evaluated using a single metric**.

A model with higher predictive performance is not automatically the best model.

Model selection should consider:

```text
Predictive Power
       +
Generalization
       +
Stability
       +
Interpretability
       +
Business Objective
```

In this project:

* **Random Forest** provided stronger test-set predictive performance.
* **Logistic Regression** demonstrated substantially better stability and a much smaller train-test gap.

This highlights the importance of evaluating both **performance and robustness** during Machine Learning model development.

---

# 🛠️ Technologies

### Programming & Data Analysis

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* RandomizedSearchCV
* Cross-Validation

### Statistical & Feature Engineering Techniques

* Weight of Evidence (WOE)
* Quantile Binning
* Spearman Correlation
* Feature Importance
* Univariate Feature Selection
* ROC-AUC
* Gini
* Precision
* Recall

### Visualization

* Matplotlib
* Seaborn

### Environment

* Jupyter Notebook

---

# 📁 Project Structure

```text
Marketing-Campaign-Success-Prediction/
│
├── Classification - RF and LogReg.ipynb
├── marketing.csv
└── README.md
```

---

# 🚀 End-to-End ML Workflow

The complete project can be summarized as:

```text
Raw Data
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning & Feature Removal
   ↓
Train / Test Split
   ↓
 ┌───────────────────────────────┐
 │                               │
 ▼                               ▼
Logistic Regression          Random Forest
 │                               │
 ▼                               ▼
Quantile Binning             Label Encoding
 │                               │
 ▼                               ▼
WOE Transformation           Feature Importance
 │                               │
 ▼                               ▼
Correlation Analysis          Feature Selection
 │                               │
 ▼                               ▼
Univariate Selection          Hyperparameter Tuning
 │                               │
 ▼                               ▼
Model Evaluation             Model Evaluation
 │                               │
 └───────────────┬───────────────┘
                 ▼
          Model Comparison
                 ↓
       Performance & Stability
                 ↓
          Final Assessment
```

---

# 📌 Conclusion

This project demonstrates a complete Machine Learning classification workflow, from **raw data exploration to model comparison and optimization**.

The project combines statistical modeling and ensemble learning approaches and demonstrates practical Machine Learning concepts such as:

* Feature engineering
* WOE transformation
* Feature selection
* Correlation analysis
* Feature importance
* Overfitting detection
* Hyperparameter optimization
* Cross-validation
* Model evaluation
* Generalization analysis

The final results emphasize that successful Machine Learning is not simply about achieving the highest score. It is about building a model that provides an appropriate balance between **predictive performance, stability, interpretability, and business value**.
