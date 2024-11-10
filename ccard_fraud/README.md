# README for Credit Card Fraud Detection - XGBoost

This project aims to detect fraudulent credit card transactions using an XGBoost classification model. By leveraging techniques such as SMOTE for handling imbalanced data, and evaluation metrics including Precision, Recall, and AUC-ROC, the model achieves high accuracy and robustness in identifying fraud.

## Project Workflow

### 1. Data Loading and Preprocessing
- **Data Loading**: The dataset is loaded with `pandas` and checked for null values (no missing data was found).
- **Train-Test Split**: Using `train_test_split` from `sklearn`, the data is divided with an 80-20 split, stratifying by the target class (`Class`) to maintain class balance across splits.

### 2. Handling Imbalanced Data with SMOTE
- **SMOTE (Synthetic Minority Over-sampling Technique)**: To address class imbalance, SMOTE is applied to the training data, creating synthetic samples of the minority (fraud) class.

### 3. Model Training with XGBoost
- **XGBoost Classifier**: Configured with `n_estimators=100`, `max_depth=5`, and `learning_rate=0.1`, the model is trained on the resampled dataset to enhance detection of fraudulent transactions.

### 4. Evaluation Metrics
- **Confusion Matrix**: Provides insights into True Positives, True Negatives, False Positives, and False Negatives. Results:
  - **True Non-Fraud**: 56,724
  - **True Fraud**: 86
  - **False Non-Fraud**: 140
  - **False Fraud**: 12
- **Precision and Recall**: Precision of 38% and Recall of 88% for fraud detection, achieving an F1-score of 0.53 for fraud cases.
- **ROC-AUC Score**: The ROC-AUC score of 0.94 indicates strong model performance in distinguishing fraudulent from non-fraudulent transactions.

### Libraries Used
- **pandas**, **NumPy**: Data manipulation
- **scikit-learn**: Train-test split, evaluation metrics, and SMOTE
- **XGBoost**: Model training and predictions
- **matplotlib** and **seaborn**: Visualizations (confusion matrix and ROC curve)
