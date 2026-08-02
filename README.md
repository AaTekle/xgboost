# XGBoost Models (Financial Data)

Two XGBoost models [**ensemble learning**](https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/) tailored to financial [**regression**](https://en.wikipedia.org/wiki/Regression_analysis) and [**classification**](https://www.ibm.com/think/topics/classification-machine-learning) tasks.

## Projects

### S&P 500 Price Forecasting

* Predicts the next trading day's closing price.
* Uses five years of S&P 500 OHLCV data.
* Set up [Moving Averages](https://www.investopedia.com/terms/m/movingaverage.asp), [RSI](https://www.investopedia.com/terms/r/rsi.asp), [EMA](https://www.investopedia.com/terms/e/ema.asp), [MACD](https://www.investopedia.com/terms/m/macd.asp), [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp), [VWAP(Volume-Weighted Average Price)](https://www.investopedia.com/terms/v/vwap.asp), [lagged values](https://www.geeksforgeeks.org/machine-learning/what-is-lag-in-time-series-forecasting/), and [time-based features](https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/index.html).
* Uses a 80/20 train-test split.
* Trains an XGBoost regressor with regularization, early stopping, and grid search.
* Best test RMSE: **2.67**.

### Credit Card Fraud Detection

* Classifies transactions as fraudulent or legitimate.
* Uses anonymized transaction features and transaction amounts.
* Applies a stratified train-test split.
* Uses [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) on the training set to handle class imbalance.
* Trains an [XGBClassifier](https://www.geeksforgeeks.org/machine-learning/xgbclassifier/).
* ![XGB](https://media.geeksforgeeks.org/wp-content/uploads/20250521100554969405/XG-Boost.webp)
* Evaluates [confusion matrix](https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/), [precision](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [F1-score](https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/), and [ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).
* [**Fraud recall:**](https://www.datacamp.com/tutorial/precision-vs-recall) **0.88**.
* [**ROC-AUC:**](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) **0.938**.
