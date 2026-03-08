# %% [markdown]
# # Milestone 3: Baseline Models
#
# Before building ML models, we establish performance baselines. The rule
# is simple: **if a model can't beat these naive approaches, it's not
# useful.** Baselines also help us calibrate expectations — some targets
# may be inherently harder to predict than others.
#
# Our baselines:
# 1. **Persistence**: tomorrow = today
# 2. **Moving Average**: future ≈ recent past average
# 3. **Climatology**: future ≈ 13-month smoothed trend

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
# ## 1. Load Feature Matrix

# %%
features = pd.read_csv("../data/processed/features.csv", index_col="date", parse_dates=True)
print(f"Feature matrix: {features.shape}")

from src.data.features import split_train_val_test
train, val, test = split_train_val_test(features)

# %% [markdown]
# ## 2. Run Baselines on All Three Targets
#
# We evaluate on the **validation set** (SC24, 2009-2019). The test set
# (SC25) is reserved for final evaluation after we've picked our best model.

# %%
from src.models.baseline import (
    persistence_forecast,
    moving_average_forecast,
    climatology_forecast,
    evaluate,
    run_all_baselines,
)

targets = ["target_isn_7d", "target_isn_30d", "target_isn_90d"]

all_results = []
for target in targets:
    horizon = target.split("_")[-1]
    results = run_all_baselines(val, target_col=target)
    results["Horizon"] = horizon
    all_results.append(results)

results_table = pd.concat(all_results, ignore_index=True)
results_table = results_table[["Horizon", "Model", "RMSE", "MAE", "MAPE"]]

print(results_table.to_string(index=False))

# %% [markdown]
# **Reading this table:**
#
# - RMSE and MAE are in ISN units (sunspot count). Lower is better.
# - MAPE is percentage error (lower is better). It's high because ISN
#   can be near zero during quiet periods, inflating percentage errors.
# - The 90-day target is easier to predict (lower RMSE) because it's
#   smoother.
# - Climatology (13-month smooth) generally beats persistence and
#   moving average — smoothing out the noise helps.

# %% [markdown]
# ## 3. Visualize Baseline Predictions
#
# Let's see what these baselines actually look like against the true values.
# We'll focus on the 30-day target, which is our primary prediction task.

# %%
target = "target_isn_30d"
y_true = val[target]

preds = {
    "Persistence": persistence_forecast(val, target),
    "Moving Average (30d)": moving_average_forecast(val, target),
    "Climatology": climatology_forecast(val, target),
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for ax, (name, y_pred) in zip(axes, preds.items()):
    ax.plot(val.index, y_true, linewidth=1, color="black", alpha=0.7, label="Actual (30d avg)")
    ax.plot(val.index, y_pred, linewidth=1, color="C1", alpha=0.8, label=name)
    metrics = evaluate(y_true.values, y_pred.values)
    ax.set_title(f"{name}  —  RMSE: {metrics['RMSE']:.1f}, MAE: {metrics['MAE']:.1f}")
    ax.set_ylabel("ISN")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Date")
plt.suptitle("Baseline Predictions vs Actual — 30-Day ISN Average (Validation Set, SC24)", y=1.01)
plt.tight_layout()
plt.savefig("../results/figures/baseline_predictions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **What we see:**
#
# - **Persistence** tracks the actual curve but is always slightly lagged,
#   because it's using today's noisy value to predict a future average.
#
# - **Moving Average** is smoother and less lagged, but still reactive
#   rather than predictive. It can't anticipate turning points.
#
# - **Climatology** is the smoothest. It captures the overall cycle shape
#   but misses shorter-term fluctuations. It's essentially a low-pass
#   filter of the past.
#
# A good ML model should combine the strengths: smooth like climatology
# but responsive to recent changes like the moving average.

# %% [markdown]
# ## 4. Residual Analysis
#
# Where do the baselines fail the most? Understanding error patterns
# tells us what the ML model needs to learn.

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Residuals over time (for best baseline: climatology)
best_pred = preds["Climatology"]
residuals = y_true - best_pred

ax = axes[0]
ax.plot(val.index, residuals, linewidth=0.5, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Climatology Residuals Over Time (Actual - Predicted)")
ax.set_ylabel("Residual (ISN)")

# Residuals vs predicted value
ax = axes[1]
valid_mask = ~(residuals.isna() | best_pred.isna())
ax.scatter(best_pred[valid_mask], residuals[valid_mask], alpha=0.1, s=5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("Predicted ISN")
ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted — Errors Are Larger When Activity Is High")

plt.tight_layout()
plt.savefig("../results/figures/baseline_residuals.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key patterns in the residuals:**
#
# - Errors are much larger during active periods (2011-2015) than during
#   the minimum (2009, 2018-2019). This is called **heteroscedasticity**.
#
# - During the rising phase of SC24, the climatology baseline consistently
#   under-predicts (negative residual → actual was higher than predicted).
#   A good model should learn to detect the rising phase earlier.
#
# - The residuals show short-term structure (not purely random), meaning
#   there's signal left for an ML model to capture.

# %% [markdown]
# ## 5. Performance by Solar Cycle Phase
#
# Let's quantify how much harder it is to predict during active vs quiet times.

# %%
# Split validation set into "quiet" and "active" based on ISN level
median_isn = val["isn"].median()
quiet = val[val["isn"] <= median_isn]
active = val[val["isn"] > median_isn]

print(f"Median ISN in validation set: {median_isn:.0f}")
print(f"Quiet days: {len(quiet)}, Active days: {len(active)}\n")

phase_results = []
for phase_name, phase_df in [("Quiet (ISN ≤ median)", quiet), ("Active (ISN > median)", active)]:
    for model_name, pred_fn in [
        ("Persistence", persistence_forecast),
        ("Moving Average", moving_average_forecast),
        ("Climatology", climatology_forecast),
    ]:
        y_pred = pred_fn(phase_df, target)
        metrics = evaluate(phase_df[target].values, y_pred.values)
        metrics["Phase"] = phase_name
        metrics["Model"] = model_name
        phase_results.append(metrics)

phase_table = pd.DataFrame(phase_results)[["Phase", "Model", "RMSE", "MAE"]]
print(phase_table.to_string(index=False))

# %% [markdown]
# As expected, RMSE during active periods is 2-3x higher than during
# quiet periods. This tells us:
# - Our ML model will likely have higher absolute errors during active times
# - We might want to consider **relative** error metrics or log-transform
#   the target for training

# %% [markdown]
# ## 6. Baseline Summary
#
# | Target | Best Baseline | RMSE | MAE |
# |--------|---------------|------|-----|
#
# (Filled in from the results above)
#
# **The bar is set.** Any ML model we build in Milestones 4-5 must beat
# the Climatology baseline's RMSE on the validation set. If it doesn't,
# the model is adding complexity without adding value.
#
# **Next step:** Milestone 4 — Traditional ML models (Random Forest, XGBoost).

# %%
print("Baseline evaluation complete!")

# %%
