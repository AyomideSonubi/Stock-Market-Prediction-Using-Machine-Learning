"""
Final Project
Stock Direction Prediction for U.S. Sector ETFs
Logistic Regression vs Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


# CONFIG

DATA_PATH = 'Stock_Market_Prediction.csv'
TICKERS = ['SPY', 'QQQ', 'XLK', 'XLF', 'XLE']
RANDOM_STATE = 42
CV_FOLDS = 5


# CUSTOM F1 FUNCTION
def f1_with_matrix(y, y_pred, ticker, model_name):
    """Compute F1 and print confusion matrix in Chen's Assignment 8 format."""
    y = np.array(y)
    y_pred = np.array(y_pred)
    tp = int(np.sum((y == 1) & (y_pred == 1)))
    fn = int(np.sum((y == 1) & (y_pred == 0)))
    fp = int(np.sum((y == 0) & (y_pred == 1)))
    tn = int(np.sum((y == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    print(f'\n--- {model_name}: {ticker} ---')
    print('----------------------------------')
    print('                 Actual Value')
    print('----------------------------------')
    print(f'            Positive    Negative')
    print(f'Positive    {tp:^8}    {fp:^8}')
    print(f'Negative    {fn:^8}    {tn:^8}')
    print('----------------------------------')
    return f1, tp, fn, fp, tn



# DATA LOADING

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows across {df['ticker'].nunique()} tickers.\n")



# LOGISTIC REG WITH GRID SEARCH

def run_logistic_regression(ticker_df, ticker_name):
    # Target and features
    y = ticker_df['target'].values
    X = ticker_df.drop(columns=['target', 'ticker', 'date'])

    # One-hot encode day_of_week and month for LogReg
    # (RF handles integers fine; LogReg treats integers as ordinal, which is wrong here)
    X = pd.get_dummies(X, columns=['day_of_week', 'month'], drop_first=True)

    # Time-based 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GridSearchCV with TimeSeriesSplit (I did this because it preserves temporal order in CV folds)
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid, cv=tscv, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    f1, tp, fn, fp, tn = f1_with_matrix(y_test, y_pred, ticker_name, "Logistic Regression")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    err = 1 - acc
    baseline = max((y_test == 1).mean(), (y_test == 0).mean())

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy:    {acc:.4f}   |  Baseline: {baseline:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Error Rate:  {err:.4f}")

    return {
        'ticker': ticker_name,
        'model': 'Logistic Regression',
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'error_rate': err, 'baseline': baseline,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'best_params': grid.best_params_,
        'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
        'feature_names': X.columns.tolist(),
        'coefficients': best_model.coef_[0]
    }



# R.F WITH GRID SEARCH

def run_random_forest(ticker_df, ticker_name):
    y = ticker_df['target'].values
    X = ticker_df.drop(columns=['target', 'ticker', 'date'])

    # RF handles integer features fine — no encoding needed
    feature_names = X.columns.tolist()

    # Time-based 80/20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # GridSearchCV with TimeSeriesSplit
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 4],
        'max_features': ['sqrt', 'log2']
    }
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, cv=tscv, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Predict on test
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Metrics
    f1, tp, fn, fp, tn = f1_with_matrix(y_test, y_pred, ticker_name, "Random Forest")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    err = 1 - acc
    baseline = max((y_test == 1).mean(), (y_test == 0).mean())

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy:    {acc:.4f}   |  Baseline: {baseline:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Error Rate:  {err:.4f}")

    return {
        'ticker': ticker_name,
        'model': 'Random Forest',
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc, 'error_rate': err, 'baseline': baseline,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'best_params': grid.best_params_,
        'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba,
        'feature_names': feature_names,
        'feature_importances': best_model.feature_importances_
    }



# MAIN LOOP

all_results = []

for ticker in TICKERS:
    print(f"\n{'='*60}\nProcessing {ticker}\n{'='*60}")
    sub = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    lr_result = run_logistic_regression(sub, ticker)
    rf_result = run_random_forest(sub, ticker)
    all_results.append(lr_result)
    all_results.append(rf_result)



# OUR RESULTS SUMMARY TABLE

print("\n\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

summary_df = pd.DataFrame([
    {
        'Ticker': r['ticker'],
        'Model': r['model'],
        'Accuracy': round(r['accuracy'], 4),
        'Baseline': round(r['baseline'], 4),
        'Beats Baseline': 'Yes' if r['accuracy'] > r['baseline'] else 'No',
        'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4),
        'F1': round(r['f1'], 4),
        'AUC-ROC': round(r['auc'], 4),
        'Error Rate': round(r['error_rate'], 4)
    }
    for r in all_results
])
print(summary_df.to_string(index=False))

# Save summary to CSV for the writeup
summary_df.to_csv('model_results_summary.csv', index=False)
print("\nSaved summary to: model_results_summary.csv")



# MODEL COMPARISON: LogReg vs RF (mean across tickers)

print("\n" + "="*80)
print("MODEL COMPARISON (mean across 5 ETFs)")
print("="*80)
comparison = summary_df.groupby('Model')[['Accuracy', 'F1', 'AUC-ROC']].agg(['mean', 'std']).round(4)
print(comparison)



# FEATURE IMPORTANCE  Section: Random Forest and LogReg

print("\n" + "="*80)
print("FEATURE IMPORTANCE / COEFFICIENTS (averaged across tickers)")
print("="*80)

# RF feature importance averaged across tickers
rf_results = [r for r in all_results if r['model'] == 'Random Forest']
rf_feat_names = rf_results[0]['feature_names']
rf_importance_avg = np.mean([r['feature_importances'] for r in rf_results], axis=0)
rf_imp_df = pd.DataFrame({
    'feature': rf_feat_names,
    'importance': rf_importance_avg
}).sort_values('importance', ascending=False)
print("\nRandom Forest - Top features (averaged across tickers):")
print(rf_imp_df.to_string(index=False))
rf_imp_df.to_csv('rf_feature_importance.csv', index=False)

# LogReg coefficients (only common features — SPY's encoding as reference)
lr_results = [r for r in all_results if r['model'] == 'Logistic Regression']
lr_feat_names = lr_results[0]['feature_names']
lr_coef_avg = np.mean([r['coefficients'] for r in lr_results], axis=0)
lr_coef_df = pd.DataFrame({
    'feature': lr_feat_names,
    'coefficient': lr_coef_avg,
    'abs_coefficient': np.abs(lr_coef_avg)
}).sort_values('abs_coefficient', ascending=False)
print("\nLogistic Regression - Top features by |coefficient| (averaged across tickers):")
print(lr_coef_df.drop(columns='abs_coefficient').to_string(index=False))
lr_coef_df.to_csv('lr_coefficients.csv', index=False)



# Our PLOTS: ROC curves and confusion matrices

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC curves - LogReg
for r in lr_results:
    fpr, tpr, _ = roc_curve(r['y_test'], r['y_proba'])
    axes[0].plot(fpr, tpr, label=f"{r['ticker']} (AUC={r['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Logistic Regression - ROC Curves')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ROC curves - RF
for r in rf_results:
    fpr, tpr, _ = roc_curve(r['y_test'], r['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{r['ticker']} (AUC={r['auc']:.3f})")
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Random Forest - ROC Curves')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
print("\nSaved: roc_curves.png")
plt.close()

# Accuracy vs Baseline bar chart
fig, ax = plt.subplots(figsize=(10, 6))
tickers_list = TICKERS
x = np.arange(len(tickers_list))
width = 0.25

lr_acc = [next(r['accuracy'] for r in lr_results if r['ticker'] == t) for t in tickers_list]
rf_acc = [next(r['accuracy'] for r in rf_results if r['ticker'] == t) for t in tickers_list]
baselines = [next(r['baseline'] for r in lr_results if r['ticker'] == t) for t in tickers_list]

ax.bar(x - width, lr_acc, width, label='Logistic Regression', color='steelblue')
ax.bar(x, rf_acc, width, label='Random Forest', color='darkorange')
ax.bar(x + width, baselines, width, label='Baseline (majority class)', color='gray', alpha=0.7)
ax.set_xlabel('ETF')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy vs Baseline (per ETF)')
ax.set_xticks(x)
ax.set_xticklabels(tickers_list)
ax.legend()
ax.set_ylim(0, 0.75)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('accuracy_vs_baseline.png', dpi=150, bbox_inches='tight')
print("Saved: accuracy_vs_baseline.png")
plt.close()

print("\n" + "="*80)
print("DONE. Files generated:")
print("  - model_results_summary.csv")
print("  - rf_feature_importance.csv")
print("  - lr_coefficients.csv")
print("  - roc_curves.png")
print("  - accuracy_vs_baseline.png")
print("="*80)
