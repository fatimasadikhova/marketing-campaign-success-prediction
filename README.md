# Marketing Campaign Success Prediction

## Project Overview

This project focuses on predicting the outcome of a marketing campaign using supervised machine learning classification techniques.

The project compares two different machine learning approaches:

- Logistic Regression
- Random Forest Classifier

The analysis includes exploratory data analysis, data preprocessing, feature engineering, Weight of Evidence (WOE) transformation, feature selection, model training, hyperparameter optimization, and model evaluation.

The main objective is not only to build predictive models, but also to compare their predictive performance, generalization ability, stability, and suitability for a business-oriented classification problem.

---

## Business Problem

Marketing campaigns generate large amounts of customer-related data. Understanding which customer characteristics and campaign-related factors are associated with different campaign outcomes can help organizations improve targeting strategies and allocate marketing resources more effectively.

A classification model can be used to estimate the probability of a particular campaign outcome based on customer demographic information, financial characteristics, communication channels, and campaign history.

This project demonstrates how machine learning can transform these variables into predictive insights and support data-driven decision-making.

---

## Dataset

The dataset contains 12,870 observations and initially includes 17 variables.

The main variables include:

| Variable | Description |
|---|---|
| `age` | Customer age |
| `job` | Type of occupation |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Credit default status |
| `balance` | Customer account balance |
| `housing` | Housing loan status |
| `loan` | Personal loan status |
| `contact` | Contact communication type |
| `day` | Day of campaign contact |
| `month` | Month of campaign contact |
| `campaign` | Number of contacts performed during the campaign |
| `pdays` | Number of days since previous campaign contact |
| `previous` | Number of contacts performed before the current campaign |
| `response` | Customer response information |
| `result` | Target variable |

The target variable was transformed from categorical values into numerical classes:

- `yes` → `0`
- `no` → `1`

The target distribution consists of:

- Class 0: 3,967 observations
- Class 1: 8,903 observations

---

## Exploratory Data Analysis

The exploratory analysis was performed to understand the structure and distribution of the dataset before model development.

The analysis included:

- Examination of numerical variables
- Analysis of categorical variables
- Target variable distribution
- Feature distributions
- Identification of potential relationships between variables
- Correlation analysis
- Visualization of relevant variables

This stage was used to better understand the data and identify appropriate preprocessing and modeling strategies.

---

## Data Preprocessing

The following preprocessing steps were performed before model development.

### Removing Unnecessary Variables

The following variables were excluded from the modeling dataset:

- `ID`
- `previous`
- `pdays`

The remaining variables were used for model development and feature engineering.

### Train-Test Split

The dataset was divided into training and testing subsets using an 80/20 split.

```text
Training set: 10,296 observations
Test set:      2,574 observations
Random state:  42
