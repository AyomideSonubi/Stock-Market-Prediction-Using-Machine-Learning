# Stock-Market-Prediction-Using-Machine-Learning
Build a binary classifier that predicts whether a stock's closing price will increase or decrease relative to the prior trading day using machine learrning
================================================================
Final Project
Predicting Next-Day Direction of U.S. Sector ETFs

Team: Vanessa Oladipo, Adaline Powell, Ayomide Sonubi, Maxim Medvedev
Spring 2026
================================================================


PROJECT OVERVIEW
----------------
This project builds and compares two classification models (Logistic
Regression and Random Forest) to predict whether the next-day adjusted
closing price of a U.S. sector ETF will be higher or lower than today's.

Five ETFs are studied: SPY, QQQ, XLK, XLF, and XLE, covering January
2015 through December 2025 (2,765 trading days per ETF, 13,825 total
ticker-day observations).


FILES INCLUDED
--------------
1. Final project. py
       Python script that trains and evaluates both models.
       Generates all results files, charts, and summary tables.

2. Stock_Market_Prediction.csv
       Prepared dataset containing 13 engineered features and the binary
       target variable. Built from raw daily OHLCV data pulled from
       Yahoo Finance via the yfinance Python library. The first 10
       trading days per ticker were dropped because the 10-day rolling
       features (volatility_10d, price_vs_sma_10d, volume_vs_sma_10d)
       require 10 prior days to compute.

3. Final_Report.docx
       Project report (6 pages, includes figures and analysis).

4. README.txt
       This file.


SYSTEM REQUIREMENTS
-------------------
- Python 3.9 or later (tested on Anaconda 3.9+)
- Operating system: Windows, macOS, or Linux
- ~500 MB free disk space


PYTHON DEPENDENCIES
-------------------
Install the following packages before running the script:

    pip install pandas numpy scikit-learn matplotlib

Or, with conda:

    conda install pandas numpy scikit-learn matplotlib


REPLICATION INSTRUCTIONS
------------------------
1. Place modeling_v2.py and Stock_Market_Prediction.csv in the same
   folder (the working directory).

2. Open a terminal in that folder.

3. Run the modeling script:

       python modeling_v2.py

4. Expected runtime: approximately 30-60 seconds on a standard laptop.

5. The script will print progress and final metrics to the console and
   will save the following files to the same working directory:

       - model_results_summary.csv     (all metrics per model x ETF)
       - rf_feature_importance.csv     (Random Forest feature importance)
       - lr_coefficients.csv           (Logistic Regression coefficients)
       - accuracy_vs_baseline.png      (chart: accuracy vs baseline)
       - roc_curves.png                (chart: ROC curves)


REPRODUCIBILITY NOTES
---------------------
- The script uses random_state=42 throughout for full reproducibility.
- All file paths in the script are relative; no absolute paths are used.
- The script does not require an internet connection at runtime
  (the dataset is included as a CSV).


METHODOLOGY SUMMARY
-------------------
- Time-based 80/20 train/test split per ticker (no shuffling).
- Hyperparameter tuning via GridSearchCV with 5-fold TimeSeriesSplit.
- For Logistic Regression: day_of_week and month one-hot encoded;
  features standardized using StandardScaler fit on the training set.
- For Random Forest: features used as-is (no scaling required).
- Metrics reported per ETF and averaged across ETFs:
  Accuracy, Precision, Recall, F1, AUC-ROC, Error Rate.
- A majority-class baseline ("always predict up") is included as a
  reference point in every comparison.


KEY FINDING
-----------
Neither Logistic Regression nor Random Forest beats the majority-class
baseline on any of the five ETFs. AUC-ROC values cluster near 0.50,
indicating no discriminative power above random chance. The result is
consistent with the weak-form Efficient Market Hypothesis: daily ETF
direction is not predictable from historical price and volume data
alone.
