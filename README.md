# Credit Risk Intelligence & Default Prediction System

## Overview

This project focuses on building an end-to-end credit risk analytics and default prediction system using the LendingClub loan dataset.

The objective is to analyze borrower risk characteristics, identify patterns associated with loan default, and develop predictive models that support data-driven lending and portfolio risk decisions.

The project combines:
- Data engineering
- Exploratory risk analytics
- Financial feature engineering
- Machine learning-based default prediction
- Risk segmentation
- SQL analytics
- Business intelligence reporting

---

# Business Problem

Financial institutions issue large volumes of consumer loans with varying borrower risk profiles.

Ineffective credit risk assessment can result in:
- Increased default rates
- Higher portfolio losses
- Reduced underwriting efficiency
- Elevated financial exposure
- Poor portfolio quality

This project develops a structured analytics system capable of:
- Predicting borrower default probability
- Segmenting borrowers by risk level
- Supporting lending decision workflows
- Monitoring portfolio-level exposure
- Generating actionable risk intelligence

---

# Dataset

## Source
LendingClub Loan Dataset
(https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)

## Dataset Characteristics

The dataset contains historical consumer lending data including:
- Borrower financial information
- Credit history attributes
- Loan application characteristics
- Lending outcomes
- Loan performance labels

### Example Features
- Loan Amount
- Interest Rate
- Annual Income
- Debt-to-Income Ratio
- Employment Length
- Revolving Utilization
- Delinquencies
- Loan Purpose
- Home Ownership
- Credit Grade

---

# Target Variable

Loan Status:
- Fully Paid → Good Loan (0)
- Charged Off → Bad Loan (1)

---

# Project Objectives

- Build a structured credit risk analytics pipeline
- Perform borrower-level risk analysis
- Engineer predictive financial risk features
- Train and evaluate multiple ML models
- Develop borrower risk scoring logic
- Create lending decision-support frameworks
- Build portfolio monitoring dashboards
- Generate operational risk insights

---

# Current Project Status

## Completed

### Data Engineering
- Dataset ingestion and validation
- Missing value analysis
- Data quality auditing
- Target variable creation
- Structured preprocessing pipeline

### Exploratory Data Analysis
- Default trend analysis
- Borrower segmentation analysis
- Portfolio distribution analysis
- Loan purpose risk analysis
- Credit grade default analysis
- Financial risk behavior analysis

### Feature Engineering

Engineered risk-focused features including:
- Income-to-loan ratio
- Installment-to-income ratio
- Credit exposure score
- Employment stability indicators
- Delinquency segmentation
- Inquiry risk segmentation
- Revolving balance exposure metrics

### Predictive Modeling

Implemented and evaluated:
- Logistic Regression
- Random Forest
- XGBoost

### Model Evaluation

Completed:
- ROC-AUC evaluation
- Precision / Recall analysis
- F1-score analysis
- ROC curve comparison
- Feature importance analysis
- Risk band segmentation
- Threshold optimization
- Portfolio decision analysis

### Risk Segmentation Framework

Developed:
- Borrower risk scoring system
- Risk bands:
  - Very Low Risk
  - Low Risk
  - Medium Risk
  - High Risk
  - Very High Risk

### Lending Decision Framework

Implemented operational decision logic:
- Approve
- Manual Review
- Reject

based on probability-of-default thresholds.

### SQL Analytics Layer

Implemented:
- PostgreSQL integration
- Portfolio analytics queries
- Borrower risk analysis queries
- Default segmentation analysis
- Exposure monitoring queries

---

# Model Performance Summary

## Final Selected Model
XGBoost Classifier

### Performance Metrics
- ROC-AUC: ~0.82
- Accuracy: ~84%
- Strong portfolio risk separation capability

### Key Predictive Features

Top drivers of default risk:
- Credit Grade
- Sub Grade
- Interest Rate
- Debt-to-Income Ratio
- Installment Burden
- Credit Exposure Metrics

---

# Risk Segmentation Results

Observed portfolio default rates:

| Risk Band | Approx. Default Rate |
|---|---|
| Very Low Risk | ~11% |
| Low Risk | ~28% |
| Medium Risk | ~48% |
| High Risk | ~69% |
| Very High Risk | ~94% |

The segmentation framework demonstrated strong borrower risk separation capability.

---

# Decision Framework Results

| Decision | Observed Default Rate |
|---|---|
| Approve | ~11% |
| Manual Review | ~31% |
| Reject | ~61% |

The decision framework successfully differentiated:
- low-risk borrowers,
- moderate-risk applicants,
- and high-risk lending candidates.

---

# System Architecture

```text
Raw Lending Data
        ↓
Data Cleaning & Validation
        ↓
Exploratory Risk Analysis
        ↓
Feature Engineering
        ↓
Predictive Modeling
        ↓
Probability of Default Estimation
        ↓
Risk Segmentation
        ↓
Decision Framework
        ↓
SQL Analytics & Dashboard Reporting
```

---

# Technology Stack

## Programming & Analytics
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Database & Querying
- PostgreSQL
- SQLAlchemy
- SQL

## Visualization & Reporting
- Power BI
- Matplotlib
- Seaborn

## Development Tools
- Jupyter Notebook
- Git & GitHub
- VS Code

---

# Repository Structure

```text
Credit-Risk-Intelligence-and-Default-Prediction-System/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│   ├── figures/
│   └── predictions/
│
├── sql/
│
├── src/
│
├── dashboard/
│
├── reports/
│
├── models/
│
├── requirements.txt
│
└── README.md
```

---

# SQL Analytics Implemented

Implemented portfolio analytics queries for:
- Default rate by credit grade
- Loan purpose risk analysis
- Risk band portfolio distribution
- Borrower exposure analysis
- Home ownership risk segmentation
- Portfolio exposure monitoring
- DTI-based risk analysis

---

# Dashboard Development (In Progress)

Planned Power BI dashboards include:
- Executive Portfolio Overview
- Borrower Risk Analytics
- Model Performance Monitoring
- Decision Framework Analytics

---

# Key Business Insights

Key findings from the analysis include:
- Higher credit grades exhibit materially lower default risk
- Interest rate strongly correlates with borrower risk
- High debt-to-income borrowers show elevated default probability
- Risk segmentation effectively separates borrower quality
- Threshold optimization improves operational lending decisions
- Portfolio concentration risk can be monitored through borrower segmentation

---

# Future Enhancements

Planned future improvements:
- Real-time scoring API
- Automated reporting pipelines
- Model explainability integration
- Approval recommendation engine
- Model monitoring & drift tracking
- Ensemble risk scoring framework
- Cloud deployment pipeline

---

# Disclaimer

This project is developed for educational and analytical purposes using publicly available lending data.
It does not represent production lending advice or financial recommendations.