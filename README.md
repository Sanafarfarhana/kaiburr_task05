# Consumer Complaint Classification Task

## 1. Project Overview

This project performs text classification on the Consumer Complaint Dataset to categorize complaints into:
0 - Credit reporting, repair, or other
1 - Debt collection
2 - Consumer Loan
3 - Mortgage

## 2. Steps Followed
- Explanatory Data Analysis and Feature Engineering
- Text Pre-Processing using NLTK (lowercasing, stopword removal, lemmatization)
- TF-IDF Vectorization
- Multi-class Model Selection and Training
- Model Evaluation and Prediction

## 3. Model Accuracy Comparison
Model	Accuracy
Naive Bayes	0.74
Logistic Regression	0.83
Random Forest	0.78
XGBoost	0.82
LightGBM	0.81

## 4. Classification Report (Logistic Regression)
              precision    recall  f1-score
0 (Credit)       0.85       0.84      0.84
1 (Debt)         0.82       0.83      0.82
2 (Loan)         0.79       0.78      0.78
3 (Mortgage)     0.84       0.83      0.83

## 5. Example Predictions
Input:
"They reported a wrong credit amount and did not fix it even after 3 months."
"They are calling me daily to repay a loan I never took."
Predictions:
['Credit reporting, credit repair services, or other personal consumer reports',
'Debt collection']

