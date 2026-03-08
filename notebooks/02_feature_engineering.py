# %% [markdown]
# # Milestone 2: Data Preprocessing & Feature Engineering
#
# This notebook transforms raw solar activity time series into a feature
# matrix suitable for ML models. The key challenge is **avoiding information
# leakage**: every feature must be computed using only past data, never
# future values.
#
# We'll cover:
# 1. Merging and aligning all datasets
# 2. Feature engineering (rolling stats, differences, cyclical encoding)
# 3. Target variable construction
# 4. Train / Validation / Test split
# 5. Feature quality checks

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# %% [markdown]
# ## 1. Merge and Align Datasets
#
# Our datasets have different start dates and some have gaps. The
# preprocessing step creates a single DataFrame on a daily grid, with
# short gaps (≤3 days) interpolated linearly.

# %%
from src.data.preprocess import build_daily_dataframe

daily = build_daily_dataframe(start="1996-01-01")
print(f"Shape: {daily.shape}")
print(f"Date range: {daily.index.min().date()} to {daily.index.max().date()}")
print(f"\nNull counts:")
print(daily.isnull().sum())
print(f"\nFirst 5 rows:")
daily.head()

# %% [markdown]
# Notice that F10.7 daily has many NaN values before 2004 (when Space
# Weather Canada's data starts), and TSI before 2003. This is expected —
# the `f107_monthly` column from NOAA fills the gap with monthly data for
# the longer history.

# %% [markdown]
# ## 2. Feature Engineering
#
# We engineer features in 4 categories. The key principle: **all features
# use a `shift(1)` before the rolling window**, so today's prediction only
# uses data up to yesterday.
#
# ### 2.1 Rolling Window Statistics
#
# For each index, we compute over past 7, 30, 90, and 365 days:
# - **Mean**: smoothed level of activity
# - **Std**: volatility (high during active periods)
# - **Min/Max**: recent extremes
# - **Slope**: trend direction and strength (OLS fit over window)
#
# ### 2.2 Difference Features
# - **1st diff**: day-to-day change (direction)
# - **2nd diff**: acceleration (is the rise speeding up or slowing down?)
# - **Percent change**: relative change (scale-independent)
#
# ### 2.3 Cyclical Time Encoding
# - Month and day-of-year encoded as sin/cos so the model knows
#   December and January are neighbors
#
# ### 2.4 Cross-Index Features
# - ISN/F10.7 ratio: deviations from the typical relationship
# - MgII anomaly: MgII minus its 1-year rolling mean

# %%
from src.data.features import build_feature_matrix

features = build_feature_matrix(daily)
print(f"Feature matrix: {features.shape}")

# Separate features and targets for clarity
target_cols = [c for c in features.columns if c.startswith("target_")]
feature_cols = [c for c in features.columns if not c.startswith("target_")]
print(f"Features: {len(feature_cols)}, Targets: {len(target_cols)}")
print(f"\nTarget columns: {target_cols}")

# %% [markdown]
# ### Feature Example: ISN with Rolling Statistics
#
# Let's visualize what these features look like for ISN during a recent
# active period, to build intuition.

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

plot_range = ("2010-01-01", "2016-01-01")
plot_data = features.loc[plot_range[0]:plot_range[1]]

# Raw ISN + rolling means
ax = axes[0]
ax.plot(plot_data.index, plot_data["isn"], alpha=0.4, linewidth=0.5, label="ISN daily")
ax.plot(plot_data.index, plot_data["isn_30d_mean"], linewidth=1.5, label="30-day mean")
ax.plot(plot_data.index, plot_data["isn_90d_mean"], linewidth=1.5, label="90-day mean")
ax.plot(plot_data.index, plot_data["isn_365d_mean"], linewidth=2, label="365-day mean")
ax.set_ylabel("ISN")
ax.set_title("ISN Rolling Means — Each Window Captures a Different Time Scale")
ax.legend()

# Rolling std (volatility)
ax = axes[1]
ax.plot(plot_data.index, plot_data["isn_30d_std"], label="30-day std")
ax.plot(plot_data.index, plot_data["isn_90d_std"], label="90-day std")
ax.set_ylabel("Std Dev")
ax.set_title("ISN Rolling Volatility — Higher During Active Periods")
ax.legend()

# Slope (trend)
ax = axes[2]
ax.plot(plot_data.index, plot_data["isn_30d_slope"], label="30-day slope")
ax.plot(plot_data.index, plot_data["isn_90d_slope"], label="90-day slope")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Slope")
ax.set_title("ISN Rolling Slope — Positive During Rising Phase, Negative During Declining")
ax.legend()

plt.tight_layout()
plt.savefig("../results/figures/feature_example_isn.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Reading this plot:**
#
# - Top: The 365-day mean shows the smooth cycle envelope, while 30-day
#   mean tracks shorter-term fluctuations. A model can use the gap between
#   short and long means as a "momentum" signal.
#
# - Middle: Volatility is low during the minimum (~2009) and high during
#   the maximum (~2014). This is a feature the model can exploit.
#
# - Bottom: The slope is positive during the rising phase of SC24
#   (2010-2014) and negative during the declining phase (2014-2019).
#   The 90-day slope is smoother and changes sign less often.

# %% [markdown]
# ## 3. Target Variables
#
# Our targets are the **future average ISN** over 3 horizons:
#
# | Target | Meaning | Use case |
# |--------|---------|----------|
# | `target_isn_7d` | Average ISN over next 7 days | Short-term forecast |
# | `target_isn_30d` | Average ISN over next 30 days | Medium-term forecast |
# | `target_isn_90d` | Average ISN over next 90 days | Seasonal forecast |
#
# Why use future *average* instead of future *value on day N*? Because
# daily ISN is very noisy. Predicting the 30-day average is a more
# meaningful and achievable task than predicting the exact value 30 days out.

# %%
fig, ax = plt.subplots(figsize=(14, 5))

plot_data = features.loc["2015-01-01":"2022-01-01"]
ax.plot(plot_data.index, plot_data["isn"], alpha=0.3, linewidth=0.5, label="ISN daily")
for target in target_cols:
    horizon = target.split("_")[-1]
    ax.plot(plot_data.index, plot_data[target], linewidth=1.5, label=f"Target: {horizon} avg")

ax.set_ylabel("ISN")
ax.set_title("Prediction Targets — Smoothed Future ISN at Different Horizons")
ax.legend()
plt.tight_layout()
plt.savefig("../results/figures/target_visualization.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# The 90-day target is the smoothest — easier to predict but less useful
# for short-term decisions. The 7-day target is noisier but more
# actionable. We'll train models for all three and compare.

# %% [markdown]
# ## 4. Train / Validation / Test Split
#
# We split by **solar cycle boundaries** (not randomly!):
#
# | Set | Period | Solar Cycle | Purpose |
# |-----|--------|-------------|---------|
# | Train | 1996-2008 | SC23 | Learn patterns |
# | Validation | 2009-2019 | SC24 | Tune hyperparameters |
# | Test | 2020-2026 | SC25 | Final evaluation |
#
# This ensures the model is always evaluated on a cycle it has never
# seen during training.

# %%
from src.data.features import split_train_val_test

train, val, test = split_train_val_test(features)

print(f"Train: {train.shape}  ({train.index.min().date()} to {train.index.max().date()})")
print(f"Val:   {val.shape}  ({val.index.min().date()} to {val.index.max().date()})")
print(f"Test:  {test.shape}  ({test.index.min().date()} to {test.index.max().date()})")

# Visualize the split
fig, ax = plt.subplots(figsize=(14, 4))
for df_part, label, color in [
    (train, "Train (SC23)", "C0"),
    (val, "Validation (SC24)", "C1"),
    (test, "Test (SC25)", "C2"),
]:
    isn_monthly = df_part["isn"].resample("MS").mean()
    ax.plot(isn_monthly.index, isn_monthly, color=color, linewidth=1.5, label=label)

ax.axvline(pd.Timestamp("2009-01-01"), color="gray", linestyle="--", alpha=0.7)
ax.axvline(pd.Timestamp("2020-01-01"), color="gray", linestyle="--", alpha=0.7)
ax.set_ylabel("ISN (monthly mean)")
ax.set_title("Train / Validation / Test Split by Solar Cycle")
ax.legend()
plt.tight_layout()
plt.savefig("../results/figures/train_val_test_split.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Why this split makes sense:**
#
# - SC23 (train) was a strong cycle — the model learns what high activity looks like
# - SC24 (validation) was notably weak — tests whether the model generalizes
#   to different activity levels
# - SC25 (test) is ongoing and appears to be intermediate — a fair final test

# %% [markdown]
# ## 5. Feature Quality Checks
#
# ### 5.1 Missing Values After Feature Engineering

# %%
# Check NaN counts in the training set (where we actually fit models)
# The first ~365 rows will have NaN from the 365-day rolling window
train_nans = train[feature_cols].isnull().sum().sort_values(ascending=False)
print("Top 10 features by NaN count in training set:")
print(train_nans.head(10))
print(f"\nRows with zero NaN in all features: {train.dropna(subset=feature_cols).shape[0]} / {train.shape[0]}")

# %% [markdown]
# The NaN values come from two sources:
# 1. **Rolling windows**: the first N days don't have enough history
#    (e.g., 365-day features need a year of warmup)
# 2. **Missing source data**: F10.7 daily starts in 2004, TSI in 2003
#
# For tree-based models (Random Forest, XGBoost), NaN is handled natively.
# For LSTM, we'll need to either drop these rows or impute.

# %% [markdown]
# ### 5.2 Feature Distributions

# %%
# Look at distributions of key features in the training set
key_features = ["isn", "isn_30d_mean", "isn_90d_slope", "isn_30d_std",
                "f107_adj", "mgii", "isn_diff1", "isn_pct_change"]
available = [f for f in key_features if f in train.columns]

fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, feat in zip(axes.flatten(), available):
    train[feat].dropna().hist(bins=50, ax=ax, alpha=0.7, edgecolor="none")
    ax.set_title(feat, fontsize=9)
    ax.set_ylabel("")

plt.suptitle("Feature Distributions (Training Set)", y=1.02)
plt.tight_layout()
plt.savefig("../results/figures/feature_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Observations:**
# - ISN and F10.7 are right-skewed (many quiet days, fewer active days)
# - Rolling slopes are roughly centered around zero
# - Percent change has heavy tails (extreme daily swings)
#
# Tree-based models handle these distributions fine without scaling.
# For neural networks, we'll need to normalize.

# %% [markdown]
# ### 5.3 Feature-Target Correlations
#
# Which features are most predictive of our 30-day target?

# %%
# Correlation of each feature with the 30-day target
target = "target_isn_30d"
corr_with_target = train[feature_cols].corrwith(train[target]).dropna().abs().sort_values(ascending=False)

print("Top 20 features by |correlation| with 30-day ISN target:")
print(corr_with_target.head(20).to_string())

# %%
fig, ax = plt.subplots(figsize=(12, 6))
top20 = corr_with_target.head(20)
ax.barh(range(len(top20)), top20.values, color="steelblue")
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("|Correlation| with target_isn_30d")
ax.set_title("Top 20 Most Correlated Features with 30-Day ISN Target")
plt.tight_layout()
plt.savefig("../results/figures/feature_target_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key insight:** The rolling means (especially 90-day and 365-day) are
# the strongest predictors. This makes sense — if the Sun has been active
# for the past 3 months, it's likely to remain active for the next month.
# The rolling slope features also rank high, confirming that trend
# information is valuable.
#
# The raw daily ISN is actually LESS predictive than its smoothed versions,
# because daily noise obscures the signal. This validates our feature
# engineering approach.

# %% [markdown]
# ## 6. Save Processed Data
#
# Save the feature matrix so later notebooks can load it directly.

# %%
output_path = "../data/processed/features.csv"
features.to_csv(output_path)
print(f"Saved feature matrix to {output_path}")
print(f"Shape: {features.shape}")

# %% [markdown]
# ## Summary
#
# **What we built:**
# - 103 features from 6 solar activity indices
# - 3 target variables (7-day, 30-day, 90-day future ISN average)
# - Clean train/val/test split along solar cycle boundaries
#
# **What we learned:**
# - Rolling statistics (especially 90-day and 365-day means) are the
#   strongest predictors — the solar cycle is smooth enough that recent
#   history is highly informative
# - Trend (slope) features add value beyond simple level features
# - Daily noise makes raw values poor predictors; smoothing helps
#
# **Next step:** Milestone 3 — Baseline models to establish a performance
# floor that our ML models must beat.

# %%
print("Feature engineering complete!")

# %%
