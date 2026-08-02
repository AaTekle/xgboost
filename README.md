# [XGBoost Models](https://machinelearningmastery.com/extreme-gradient-boosting-ensemble-in-python/) (Financial Data)

### Note: Using [Extreme Gradient Boosting](https://www.sciencedirect.com/topics/computer-science/extreme-gradient-boosting)
- fast high-performance ml algo that uses gradient boosted decision trees. works by building trees one after another, where each new tree fixes the errors (residuals) of the previous ones. used for structured data because of its speed, built-in regularization, and parallel processing


Two XGBoost models ([**ensemble learning**])(https://www.geeksforgeeks.org/machine-learning/a-comprehensive-guide-to-ensemble-learning/) tailored to financial [**regression**](https://en.wikipedia.org/wiki/Regression_analysis) and [**classification**](https://www.ibm.com/think/topics/classification-machine-learning) tasks.

## Projects:

### S&P 500 Price Forecasting:

* predicts each stock’s next-day log return and converts it into a next-day closing-price forecast
* uses five years of S&P 500 OHLCV data with 619,029 rows, 505 tickers, and dates from 2013-02-08 to 2018-02-07
* sets up [**Moving Averages**](https://www.investopedia.com/terms/m/movingaverage.asp), [**RSI**](https://www.investopedia.com/terms/r/rsi.asp), [**EMA**](https://www.investopedia.com/terms/e/ema.asp), [**MACD**](https://www.investopedia.com/terms/m/macd.asp), [**Bollinger Bands**](https://www.investopedia.com/terms/b/bollingerbands.asp), [**VWAP(Volume-Weighted Average Price**)](https://www.investopedia.com/terms/v/vwap.asp), [**lagged values**](https://www.geeksforgeeks.org/machine-learning/what-is-lag-in-time-series-forecasting/), and [**time-based features**](https://feature-engine.trainindata.com/en/1.8.x/user_guide/timeseries/forecasting/index.html).
* uses chronological, purged train-validation-test splits: 411,136 training rows, 89,734 validation rows, and 91,414 test rows
* trained XGBoost regressor with native categorical ticker support, regularization, row and feature subsampling, and validation-based early stopping
* tests four parameter configurations; the model uses 18 trees, max_depth=7, min_child_weight=20, subsample=0.75, and colsample_bytree=0.90
* **test results:** [**RMSE**](https://en.wikipedia.org/wiki/Root_mean_square_deviation) 2.3976, [**MAE**](https://en.wikipedia.org/wiki/Mean_absolute_error) 0.9621, [**MAPE**](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) 0.9775%, [**R²**](https://en.wikipedia.org/wiki/Coefficient_of_determination) 0.999663, and 52.19% [**directional accuracy**](https://en.wikipedia.org/wiki/Mean_directional_accuracy)
* **Takeaways:**
    * the model had accurate price-level forecasts, with an average error below 1% of the actual closing price
    * (however) the naive baseline, which uses today’s closing price as tomorrow’s prediction, achieved a lower [**RMSE**](https://en.wikipedia.org/wiki/Root_mean_square_deviation) of 2.3918 and performed 0.242% better than XGBoost
        * this shows that the high [**R²**](https://en.wikipedia.org/wiki/Coefficient_of_determination) reiterates the fact that daily stock prices usually change very little from one day to the next
    * the model’s 52.19% directional accuracy suggests a small (predictive) edge, but not enough to show that it would remain profitable after trading costs, slippage, and market risk
    * the model finds short-term market patterns, but does not perform better than the simple baseline
    * ![XGB Regression](https://miro.medium.com/v2/resize:fit:720/format:webp/1*2UV8DrF8wbE7PIiYiiSW5w.png)

      https://ai.plainenglish.io/xgboost-regression-in-depth-cb2b3f623281

### Credit Card Fraud Detection:

* Classifies transactions as fraudulent or legitimate.
* Uses anonymized transaction features and transaction amounts.
* uses a stratified train-test split.
    - makes sure that the training and testing subsets maintain the exact same proportion of class labels as the original dataset. important pre-processing step within ML workflows, prevents random splits from losing minority classes in either subset
* When handling imbalanced data, random sampling can drop or underrepresent rare minority classes in your subsets
* Uses [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) on the training set to handle class imbalance.
* Trains an [XGBClassifier](https://www.geeksforgeeks.org/machine-learning/xgbclassifier/).
* ![XGB](https://media.geeksforgeeks.org/wp-content/uploads/20250521100554969405/XG-Boost.webp)
* Evaluates [confusion matrix](https://www.geeksforgeeks.org/machine-learning/confusion-matrix-machine-learning/), [precision](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), [F1-score](https://www.geeksforgeeks.org/machine-learning/f1-score-in-machine-learning/), and [ROC-AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).
* [**Fraud recall:**](https://www.datacamp.com/tutorial/precision-vs-recall) **0.88**.
* [**ROC-AUC:**](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) **0.938**.
