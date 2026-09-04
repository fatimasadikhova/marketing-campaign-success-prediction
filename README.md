# Marketing Campaign Success Prediction

## Project Overview

This project develops a Machine Learning classification solution to predict marketing campaign outcomes using customer demographic, financial, communication, and campaign-related data.

The main goal is to support:

* More targeted customer selection
* Better campaign prioritization
* Data-driven marketing decisions

Two models were developed and compared:

* Logistic Regression
* Random Forest

The project includes data preprocessing, feature engineering, WOE transformation, feature selection, correlation analysis, model optimization, cross-validation, and performance evaluation.

---

## Dataset

The dataset contains **12,870 observations and 17 variables**.

Key variables include:

`age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `response`, `result`

The target variable was converted into binary classification:

```text
yes → 0
no  → 1
```

Target distribution:

* Class 0: 3,967
* Class 1: 8,903

---

## Data Preprocessing

The following steps were performed:

* Exploratory Data Analysis
* Data type and missing-value analysis
* Feature inspection
* Removal of `ID`, `previous`, and `pdays`
* 80/20 train-test split
* `random_state = 42`

Dataset split:

```text
Training: 10,296
Testing:   2,574
```

---

## Logistic Regression

The Logistic Regression pipeline included:

1. Separation of numerical and categorical variables
2. Quantile binning of numerical variables
3. WOE transformation
4. Spearman correlation analysis
5. Univariate feature selection

WOE features were evaluated using:

```text
Train Gini > 10%
Test Gini  > 10%
Gini Gap    < 5%
```

Selected features:

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

## Random Forest

The Random Forest pipeline included:

* Label Encoding
* Feature Importance Analysis
* Feature Selection
* Hyperparameter Optimization
* 5-Fold Cross-Validation

The initial model showed significant overfitting:

```text
Train Gini: 100.00%
Test Gini:   55.10%
Gini Gap:    44.90%
```

Optimization was performed using `RandomizedSearchCV` with:

* 50 parameter combinations
* 5-Fold Cross-Validation
* ROC-AUC scoring
* 250 total model fits

Best cross-validation ROC-AUC:

```text
0.7861
```

---

## Model Comparison

| Model                             | Train Gini |  Test Gini |  Gini Gap |  Precision | Recall |
| --------------------------------- | ---------: | ---------: | --------: | ---------: | -----: |
| Logistic Regression – Initial     |     53.12% |     50.51% |     2.61% |     78.67% | 93.50% |
| Logistic Regression – Selected    |     51.53% |     49.52% | **2.01%** |     78.24% | 93.72% |
| Random Forest – Initial           |    100.00% |     55.10% |    44.90% |     80.98% | 90.52% |
| Random Forest – Feature Selection |    100.00% |     54.53% |    45.47% |     80.96% | 90.41% |
| Random Forest – Optimized         |     85.33% | **55.76%** |    29.57% | **81.06%** | 90.80% |

---

## Key Results

### Logistic Regression

* Test Gini: **49.52%**
* Precision: **78.24%**
* Recall: **93.72%**
* Gini Gap: **2.01%**

The model demonstrated strong stability and a low train-test performance gap.

### Random Forest

* Test Gini: **55.76%**
* Precision: **81.06%**
* Recall: **90.80%**
* Gini Gap: **29.57%**

Random Forest achieved the highest test Gini and precision, but still showed a considerably larger generalization gap.

---

## Conclusion

The results demonstrate that model selection should not rely on a single performance metric.

The evaluation considered:

**Predictive Performance + Stability + Generalization + Interpretability + Business Value**

Random Forest provided stronger predictive performance, while Logistic Regression demonstrated substantially better stability and generalization.

Overall, the project demonstrates an end-to-end Machine Learning workflow covering data preparation, feature engineering, model development, optimization, and business-oriented model evaluation.

---

## Technologies

**Programming & Data Analysis:** Python, Pandas, NumPy

**Machine Learning:** Scikit-learn, Logistic Regression, Random Forest, RandomizedSearchCV, Cross-Validation

**Statistical Techniques:** WOE, Quantile Binning, Spearman Correlation, Feature Selection, ROC-AUC, Gini, Precision, Recall

**Visualization:** Matplotlib, Seaborn

**Environment:** Jupyter Notebook

## Project Structure

```text
Marketing-Campaign-Success-Prediction/
├── Classification - RF and LogReg.ipynb
├── marketing.csv
└── README.md
```
