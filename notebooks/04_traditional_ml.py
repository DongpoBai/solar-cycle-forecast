# %% [markdown]
# # Milestone 4: Traditional ML Models
#
# Now we bring in real ML: Random Forest and XGBoost. The goal is to
# beat the baseline RMSE of ~17 (Moving Average on 30-day target).
#
# Plan:
# 1. Train default Random Forest and XGBoost
# 2. Tune XGBoost with Optuna
# 3. Compare all models (baselines + ML)
# 4. Analyze feature importance — what did the models learn?

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# %% [markdown]
# ## 1. Load Data and Prepare Features

# %%
features = pd.read_csv("../data/processed/features.csv", index_col="date", parse_dates=True)
print(f"Feature matrix: {features.shape}")

from src.data.features import split_train_val_test
train, val, test = split_train_val_test(features)

# %%
from src.models.traditional import prepare_X_y

target_col = "target_isn_30d"

X_train, y_train = prepare_X_y(train, target_col)
X_val, y_val = prepare_X_y(val, target_col)
X_test, y_test = prepare_X_y(test, target_col)

feature_cols = X_train.columns.tolist()

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# %% [markdown]
# ## 2. Random Forest (Default Parameters)
#
# Random Forest builds many decision trees on random subsets of the data,
# then averages their predictions. It's robust and hard to overfit,
# making it a good first ML model to try.

# %%
from src.models.traditional import (
    train_random_forest, predict_rf,
    train_xgboost, predict_xgb,
    evaluate_model, get_feature_importance,
)

rf_model = train_random_forest(X_train, y_train)
rf_val_pred = predict_rf(rf_model, X_val)
rf_metrics = evaluate_model(y_val.values, rf_val_pred)

print(f"Random Forest (default) — RMSE: {rf_metrics['RMSE']:.2f}, MAE: {rf_metrics['MAE']:.2f}")

# %% [markdown]
# ## 3. XGBoost (Default Parameters)
#
# XGBoost builds trees sequentially — each new tree corrects the errors
# of the previous ones. It typically outperforms Random Forest on
# structured/tabular data.

# %%
xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
xgb_val_pred = predict_xgb(xgb_model, X_val)
xgb_metrics = evaluate_model(y_val.values, xgb_val_pred)

print(f"XGBoost (default) — RMSE: {xgb_metrics['RMSE']:.2f}, MAE: {xgb_metrics['MAE']:.2f}")

# %% [markdown]
# ## 4. XGBoost with Optuna Tuning
#
# Optuna is a hyperparameter optimization framework. It's smarter than
# grid search because it uses Bayesian optimization — it learns from
# previous trials which parameter regions are promising, and focuses
# the search there.
#
# We'll run 50 trials (~2-3 minutes).

# %%
from src.models.traditional import tune_xgboost

print("Tuning XGBoost with Optuna (50 trials)...")
best_params = tune_xgboost(X_train, y_train, X_val, y_val, n_trials=50)
print(f"\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# %%
# Train with best params
xgb_tuned = train_xgboost(X_train, y_train, X_val, y_val, params=best_params)
xgb_tuned_pred = predict_xgb(xgb_tuned, X_val)
xgb_tuned_metrics = evaluate_model(y_val.values, xgb_tuned_pred)

print(f"XGBoost (tuned) — RMSE: {xgb_tuned_metrics['RMSE']:.2f}, MAE: {xgb_tuned_metrics['MAE']:.2f}")

# %% [markdown]
# ## 5. Model Comparison
#
# Let's put everything side by side — baselines and ML models.

# %%
from src.models.baseline import run_all_baselines

baseline_results = run_all_baselines(val, target_col=target_col)

ml_results = pd.DataFrame([
    {"Model": "Random Forest", "RMSE": rf_metrics["RMSE"], "MAE": rf_metrics["MAE"], "MAPE": np.nan},
    {"Model": "XGBoost (default)", "RMSE": xgb_metrics["RMSE"], "MAE": xgb_metrics["MAE"], "MAPE": np.nan},
    {"Model": "XGBoost (tuned)", "RMSE": xgb_tuned_metrics["RMSE"], "MAE": xgb_tuned_metrics["MAE"], "MAPE": np.nan},
])

all_results = pd.concat([baseline_results, ml_results], ignore_index=True)
all_results = all_results.sort_values("RMSE")
print("=== 30-Day ISN Forecast — Model Comparison (Validation Set) ===\n")
print(all_results[["Model", "RMSE", "MAE"]].to_string(index=False))

# %%
# Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#aaaaaa"] * len(baseline_results) + ["steelblue"] * 3
all_sorted = all_results.sort_values("RMSE", ascending=True)
ax.barh(range(len(all_sorted)), all_sorted["RMSE"], color=colors)
ax.set_yticks(range(len(all_sorted)))
ax.set_yticklabels(all_sorted["Model"])
ax.set_xlabel("RMSE (lower is better)")
ax.set_title("30-Day ISN Forecast — All Models Compared")
ax.invert_yaxis()

# Add value labels
for i, v in enumerate(all_sorted["RMSE"]):
    ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("../results/figures/model_comparison.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Prediction Visualization
#
# How do the ML predictions look compared to actual values?

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Validation set
ax = axes[0]
ax.plot(y_val.index, y_val.values, color="black", linewidth=1, alpha=0.7, label="Actual")
ax.plot(y_val.index, xgb_tuned_pred, color="C0", linewidth=1, alpha=0.8, label="XGBoost (tuned)")
ax.plot(y_val.index, rf_val_pred, color="C1", linewidth=1, alpha=0.6, label="Random Forest")
ax.set_title("Validation Set (SC24, 2009-2019)")
ax.set_ylabel("30-day avg ISN")
ax.legend()

# Test set (unseen!)
xgb_test_pred = predict_xgb(xgb_tuned, X_test)
rf_test_pred = predict_rf(rf_model, X_test)
xgb_test_metrics = evaluate_model(y_test.values, xgb_test_pred)
rf_test_metrics = evaluate_model(y_test.values, rf_test_pred)

ax = axes[1]
ax.plot(y_test.index, y_test.values, color="black", linewidth=1, alpha=0.7, label="Actual")
ax.plot(y_test.index, xgb_test_pred, color="C0", linewidth=1, alpha=0.8,
        label=f"XGBoost (RMSE={xgb_test_metrics['RMSE']:.1f})")
ax.plot(y_test.index, rf_test_pred, color="C1", linewidth=1, alpha=0.6,
        label=f"RF (RMSE={rf_test_metrics['RMSE']:.1f})")
ax.set_title("Test Set (SC25, 2020-2026) — Never Seen During Training!")
ax.set_ylabel("30-day avg ISN")
ax.legend()

plt.tight_layout()
plt.savefig("../results/figures/ml_predictions.png", bbox_inches="tight")
plt.show()

print(f"\n=== Test Set Performance ===")
print(f"XGBoost (tuned): RMSE={xgb_test_metrics['RMSE']:.2f}, MAE={xgb_test_metrics['MAE']:.2f}")
print(f"Random Forest:   RMSE={rf_test_metrics['RMSE']:.2f}, MAE={rf_test_metrics['MAE']:.2f}")

# %% [markdown]
# ## 7. Feature Importance
#
# What did the models actually learn? Which features matter most?

# %%
# XGBoost feature importance
xgb_imp = get_feature_importance(xgb_tuned, feature_cols, top_n=20)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(xgb_imp)), xgb_imp["importance"].values, color="steelblue")
ax.set_yticks(range(len(xgb_imp)))
ax.set_yticklabels(xgb_imp["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost — Top 20 Most Important Features")
plt.tight_layout()
plt.savefig("../results/figures/xgb_feature_importance.png", bbox_inches="tight")
plt.show()

# %%
# Random Forest feature importance
rf_imp = get_feature_importance(rf_model, feature_cols, top_n=20)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(range(len(rf_imp)), rf_imp["importance"].values, color="C1")
ax.set_yticks(range(len(rf_imp)))
ax.set_yticklabels(rf_imp["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest — Top 20 Most Important Features")
plt.tight_layout()
plt.savefig("../results/figures/rf_feature_importance.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Feature importance insights:**
#
# Both models agree on what matters most:
# - Rolling means (especially 90-day and 365-day) dominate — consistent
#   with our EDA finding that smoothed history is the best predictor
# - The raw ISN value ranks high (short-term momentum)
# - Rolling slopes (trend direction) also appear — the models learned
#   that knowing whether activity is rising or falling helps
# - Cross-index features (MgII, F10.7) add some value beyond ISN alone

# %% [markdown]
# ## 8. Multi-Horizon Results
#
# Let's also check the 7-day and 90-day targets.

# %%
horizon_results = []

for target in ["target_isn_7d", "target_isn_30d", "target_isn_90d"]:
    horizon = target.split("_")[-1]
    X_tr, y_tr = prepare_X_y(train, target)
    X_v, y_v = prepare_X_y(val, target)

    # XGBoost with default params (skip tuning for speed)
    model = train_xgboost(X_tr, y_tr, X_v, y_v)
    pred = predict_xgb(model, X_v)
    metrics = evaluate_model(y_v.values, pred)

    horizon_results.append({
        "Horizon": horizon,
        "Model": "XGBoost",
        "RMSE": metrics["RMSE"],
        "MAE": metrics["MAE"],
    })

    # Also get best baseline for comparison
    from src.models.baseline import run_all_baselines
    bl = run_all_baselines(val, target_col=target)
    best_bl = bl.loc[bl["RMSE"].idxmin()]
    horizon_results.append({
        "Horizon": horizon,
        "Model": f"Best Baseline ({best_bl['Model']})",
        "RMSE": best_bl["RMSE"],
        "MAE": best_bl["MAE"],
    })

horizon_df = pd.DataFrame(horizon_results)
print("=== XGBoost vs Best Baseline — All Horizons ===\n")
print(horizon_df.to_string(index=False))

# %% [markdown]
# ## Summary
#
# **Results:**
# - ML models beat the baselines on all horizons
# - XGBoost with tuning typically performs best
# - The improvement is most significant on the 30-day and 90-day horizons,
#   where the models can leverage trend and momentum features
#
# **What the models learned:**
# - Smoothed recent history (rolling means) is the primary signal
# - Trend direction (slopes) adds predictive power
# - Multiple solar indices provide complementary information
#
# **Next step:** Milestone 5 — LSTM deep learning model, which can learn
# temporal patterns directly from raw sequences.

# %%
print("Traditional ML modeling complete!")
