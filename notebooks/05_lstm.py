# %% [markdown]
# # Milestone 5: LSTM Deep Learning Model
#
# The LSTM (Long Short-Term Memory) takes a fundamentally different approach
# from XGBoost. Instead of hand-crafted features, it reads a raw sequence
# of past observations and learns temporal patterns directly.
#
# **Input:** Past 90 days of 6 solar indices (shape: 90 × 6)
# **Output:** Predicted average ISN over the next 30 days (scalar)
#
# Think of it as: "Here are the last 3 months of solar data — what
# happens next month?"

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# %% [markdown]
# ## 1. Load Data

# %%
features = pd.read_csv("../data/processed/features.csv", index_col="date", parse_dates=True)

from src.data.features import split_train_val_test
train, val, test = split_train_val_test(features)

print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

# %% [markdown]
# ## 2. Train LSTM
#
# The LSTM sees 6 raw daily values over a 90-day window:
# - ISN (sunspot number)
# - F10.7 (radio flux)
# - MgII (UV proxy)
# - SEM UV flux
# - TSI (total solar irradiance)
# - SSN monthly (smoothed sunspot number from NOAA)
#
# We normalize all inputs to [0,1] range — neural networks train much
# better with normalized data compared to raw values that span different
# scales (ISN: 0-400, MgII: 0.26-0.29, TSI: 1360-1362).

# %%
from src.models.deep import train_lstm, predict_lstm, LSTMConfig, LSTM_FEATURES, SolarSequenceDataset

config = LSTMConfig(
    lookback=90,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=1e-3,
    batch_size=64,
    epochs=50,
    patience=10,
)

target_col = "target_isn_30d"
feature_cols = [c for c in LSTM_FEATURES if c in train.columns]
print(f"LSTM input features: {feature_cols}")
print(f"Lookback: {config.lookback} days")

model, history = train_lstm(train, val, target_col=target_col, feature_cols=feature_cols, config=config)

# %% [markdown]
# ### Training Curves
#
# The loss curves tell us if the model is learning and whether it's
# overfitting. If val_loss stops decreasing while train_loss keeps
# dropping, that's overfitting.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
epochs = range(1, len(history["train_loss"]) + 1)
ax.plot(epochs, np.sqrt(history["train_loss"]), label="Train RMSE", linewidth=1.5)
ax.plot(epochs, np.sqrt(history["val_loss"]), label="Val RMSE", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.set_title("LSTM Training Curves")
ax.legend()
plt.tight_layout()
plt.savefig("../results/figures/lstm_training_curves.png", bbox_inches="tight")
plt.show()
print("running...")

# %% [markdown]
# ## 3. Evaluate LSTM

# %%
# Get normalization params from training set for consistent prediction
train_ds = SolarSequenceDataset(train, feature_cols, target_col, config.lookback)

# Predict on validation set
val_pred, val_dates = predict_lstm(
    model, val, target_col, feature_cols,
    lookback=config.lookback,
    feat_min=train_ds.feat_min,
    feat_range=train_ds.feat_range,
)

val_actual = val.loc[val_dates, target_col].values

lstm_rmse = float(np.sqrt(np.mean((val_actual - val_pred) ** 2)))
lstm_mae = float(np.mean(np.abs(val_actual - val_pred)))
print(f"LSTM Validation — RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")

# Predict on test set
test_pred, test_dates = predict_lstm(
    model, test, target_col, feature_cols,
    lookback=config.lookback,
    feat_min=train_ds.feat_min,
    feat_range=train_ds.feat_range,
)

test_actual = test.loc[test_dates, target_col].values
lstm_test_rmse = float(np.sqrt(np.mean((test_actual - test_pred) ** 2)))
lstm_test_mae = float(np.mean(np.abs(test_actual - test_pred)))
print(f"LSTM Test      — RMSE: {lstm_test_rmse:.2f}, MAE: {lstm_test_mae:.2f}")

# %% [markdown]
# ## 4. All Models Comparison

# %%
from src.models.baseline import run_all_baselines
from src.models.traditional import (
    prepare_X_y, train_random_forest, predict_rf,
    train_xgboost, predict_xgb, evaluate_model, tune_xgboost,
)

# Re-train XGBoost for fair comparison
X_train, y_train = prepare_X_y(train, target_col)
X_val, y_val = prepare_X_y(val, target_col)
X_test, y_test = prepare_X_y(test, target_col)

xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
xgb_val_pred = predict_xgb(xgb_model, X_val)
xgb_val_metrics = evaluate_model(y_val.values, xgb_val_pred)

xgb_test_pred = predict_xgb(xgb_model, X_test)
xgb_test_metrics = evaluate_model(y_test.values, xgb_test_pred)

# Baselines on val
bl_val = run_all_baselines(val, target_col=target_col)
best_bl_val = bl_val.loc[bl_val["RMSE"].idxmin()]

# Build comparison table
comparison = pd.DataFrame([
    {"Model": "Persistence (baseline)", "Val RMSE": 25.4, "Val MAE": "-", "Test RMSE": "-"},
    {"Model": f"Best Baseline ({best_bl_val['Model']})", "Val RMSE": f"{best_bl_val['RMSE']:.1f}", "Val MAE": f"{best_bl_val['MAE']:.1f}", "Test RMSE": "-"},
    {"Model": "XGBoost", "Val RMSE": f"{xgb_val_metrics['RMSE']:.1f}", "Val MAE": f"{xgb_val_metrics['MAE']:.1f}", "Test RMSE": f"{xgb_test_metrics['RMSE']:.1f}"},
    {"Model": "LSTM", "Val RMSE": f"{lstm_rmse:.1f}", "Val MAE": f"{lstm_mae:.1f}", "Test RMSE": f"{lstm_test_rmse:.1f}"},
])

print("=== 30-Day ISN Forecast — Final Comparison ===\n")
print(comparison.to_string(index=False))

# %% [markdown]
# ## 5. Prediction Visualization

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

# Validation
ax = axes[0]
ax.plot(val.index, val[target_col], color="black", linewidth=1, alpha=0.7, label="Actual")
ax.plot(y_val.index, xgb_val_pred, color="C0", linewidth=1, alpha=0.7, label=f"XGBoost (RMSE={xgb_val_metrics['RMSE']:.1f})")
ax.plot(val_dates, val_pred, color="C3", linewidth=1, alpha=0.7, label=f"LSTM (RMSE={lstm_rmse:.1f})")
ax.set_title("Validation Set (SC24)")
ax.set_ylabel("30-day avg ISN")
ax.legend()

# Test
ax = axes[1]
ax.plot(test.index, test[target_col], color="black", linewidth=1, alpha=0.7, label="Actual")
ax.plot(y_test.index, xgb_test_pred, color="C0", linewidth=1, alpha=0.7, label=f"XGBoost (RMSE={xgb_test_metrics['RMSE']:.1f})")
ax.plot(test_dates, test_pred, color="C3", linewidth=1, alpha=0.7, label=f"LSTM (RMSE={lstm_test_rmse:.1f})")
ax.set_title("Test Set (SC25)")
ax.set_ylabel("30-day avg ISN")
ax.legend()

plt.tight_layout()
plt.savefig("../results/figures/lstm_vs_xgboost.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Summary
#
# | Model | Val RMSE | Test RMSE | Notes |
# |-------|----------|-----------|-------|
# | Best Baseline | ~17 | - | Moving Average, no ML needed |
# | XGBoost | ~14 | ~17 | Best overall, uses hand-crafted features |
# | LSTM | ~?? | ~?? | Learns from raw sequences |
#
# (Fill in from results above)
#
# **Key takeaways:**
#
# - XGBoost with well-engineered features is a strong contender and often
#   hard to beat with deep learning on tabular/time-series data
# - LSTM's advantage (if any) is that it doesn't need manual feature
#   engineering — it learns directly from raw sequences
# - On the test set (SC25), performance degrades for all models because
#   SC25 is more active than anything in the training data
# - This is a fundamental limitation: we're predicting a quasi-periodic
#   process with only a few historical cycles to learn from
#
# **For the portfolio:** The fact that you built BOTH approaches and
# compared them honestly is more impressive than claiming one is
# "better." It shows you understand the tradeoffs.

# %%
print("LSTM modeling complete!")
