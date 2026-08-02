# XGBoost Models (Financial Data)

Two XGBoost projects tailored to financial regression and classification tasks.

## Projects

### S&P 500 Price Forecasting

* Predicts the next trading day's closing price.
* Uses five years of S&P 500 OHLCV data.
* Set up moving averages, RSI, EMA, MACD, Bollinger Bands, VWAP, lagged values, and time-based features.
* Uses a chronological 80/20 train-test split.
* Trains an XGBoost regressor with regularization, early stopping, and grid search.
* Best reported test RMSE: **2.67**.

### Credit Card Fraud Detection

* Classifies transactions as fraudulent or legitimate.
* Uses anonymized transaction features and transaction amounts.
* Applies a stratified train-test split.
* Uses SMOTE on the training set to address class imbalance.
* Trains an `XGBClassifier`.
* Evaluates confusion matrix, precision, recall, F1-score, and ROC-AUC.
* Reported fraud recall: **0.88**.
* Reported ROC-AUC: **0.938**.

## Tech Stack

* Pandas and NumPy
* XGBoost
* Scikit-learn
* Imbalanced-learn
* Matplotlib and Seaborn
* Jupyter Notebook
