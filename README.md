# XGBoost Models (Financial Data)

Two XGBoost projects tailored to financial [**regression**](https://en.wikipedia.org/wiki/Regression_analysis) and [**classification**](https://www.ibm.com/think/topics/classification-machine-learning) tasks.

## Projects

### S&P 500 Price Forecasting

* Predicts the next trading day's closing price.
* Uses five years of S&P 500 OHLCV data.
* Set up moving averages, RSI, EMA, MACD, Bollinger Bands, VWAP, lagged values, and time-based features.
* Uses a 80/20 train-test split.
* Trains an XGBoost regressor with regularization, early stopping, and grid search.
* Best test RMSE: **2.67**.

### Credit Card Fraud Detection

* Classifies transactions as fraudulent or legitimate.
* Uses anonymized transaction features and transaction amounts.
* Applies a stratified train-test split.
* Uses SMOTE on the training set to handle class imbalance.
* Trains an [XGBClassifier](https://www.geeksforgeeks.org/machine-learning/xgbclassifier/).
* ![XGB](https://media.geeksforgeeks.org/wp-content/uploads/20250521100554969405/XG-Boost.webp)
* Evaluates confusion matrix, [precision](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [F1-score](https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/), and [ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).
* Reported fraud recall: **0.88**.
* Reported ROC-AUC: **0.938**.
