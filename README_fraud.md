# Credit Card Fraud Detection

Binary classification pipeline for detecting fraudulent credit card transactions under extreme class imbalance (0.17% fraud), using Python and Scikit-learn.

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcard-fraud)
284,807 transactions, 492 frauds (0.172%). Features V1–V28 are PCA components (anonymized for confidentiality by the original dataset authors); `Time` and `Amount` are the only non-transformed variables.

## Approach

- **EDA & class balance analysis** — quantified the extreme imbalance and why accuracy is a misleading metric here (a model predicting "no fraud" always would already score 99.83% accuracy).
- **Feature selection** — Pearson correlation with target, threshold tuned lower than typical (0.05) given weaker linear correlations under heavy imbalance.
- **Outlier handling** — no blind removal; fraud cases are structurally more likely to be outliers, so `Amount` was log-transformed instead of trimmed.
- **Standardization** — Z-score scaling on selected features.
- **Stratified train/test split** — critical here given only 492 positive cases in the full dataset.
- **Class imbalance correction** — compared an unbalanced baseline against `class_weight='balanced'` and SMOTE oversampling (applied only to the training set).
- **Threshold tuning** — lowered the decision threshold to prioritize recall, since false negatives (undetected fraud) are the costly error here.
- **Model comparison** — Logistic Regression, Decision Tree, Random Forest, and Naive Bayes, evaluated on Precision, Recall, F1, and AUC-PR (more informative than ROC-AUC under severe imbalance).
- **Validation** — 5-fold stratified cross-validation on the best-performing model to confirm results aren't split-dependent.

## Results

Best model: **Random Forest with `class_weight='balanced'`**

| Metric | Score |
|---|---|
| Precision | 93.97% |
| Recall | 73.65% |
| F1-score | 0.826 |
| AUC-PR | 0.81 |
| F1 (5-fold CV mean) | 0.858 |

## Tools

Python · Pandas · NumPy · Scikit-learn · imbalanced-learn (SMOTE) · Seaborn · Matplotlib

## Files

- `fraud_detection.py` — full pipeline (EDA → feature selection → preprocessing → model comparison → cross-validation)
