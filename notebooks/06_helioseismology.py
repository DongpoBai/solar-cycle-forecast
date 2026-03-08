# %% [markdown]
# # Helioseismic Frequency Shifts as Leading Indicators of Solar Activity
#
# ## Research Question
#
# **Can solar p-mode oscillation frequency shifts predict solar activity
# before it manifests on the surface?**
#
# Solar acoustic oscillations (p-modes) are sensitive to conditions in
# the solar interior. As the magnetic cycle evolves, the near-surface
# magnetic field alters the acoustic cavity, causing measurable shifts
# in oscillation frequencies. If these interior changes precede surface
# manifestations (sunspots, radio emission), then helioseismic data
# could serve as a *leading indicator* for solar cycle forecasting.
#
# We use 14 years of zero crossing data from the USC Solar Physics Group's
# HMI processing pipeline (Runs 1–74, covering 2010–2024) to investigate
# this hypothesis.

# %%
import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

# %% [markdown]
# ## 1. Load and Inspect Zero Crossing Data
#
# Each "run" in the pipeline processes 72 days of HMI observations.
# The zero crossing value represents the frequency (in µHz) at which
# the solar p-mode frequency shift changes sign relative to a reference
# model — essentially measuring how much the oscillation frequencies
# have shifted due to solar magnetic activity.
#
# Higher zero crossing frequency → more magnetically active Sun.

# %%
from src.data.helioseismology import (
    load_zero_crossings, build_analysis_dataframe,
    compute_lead_lag_analysis, DAYS_PER_RUN,
)

zc = load_zero_crossings("../data/raw/zero_crossings_summary.csv")
print(f"Total runs: {len(zc)}")
print(f"Good quality: {(zc['quality'] == 'good').sum()}")
print(f"Bad/missing: {(zc['quality'] != 'good').sum()}")
print(f"\nDate range: {zc['date'].min().date()} to {zc['date'].max().date()}")

zc.head(10)

# %% [markdown]
# ### Data quality
#
# Some runs have anomalous zero crossing values (e.g., -/+ avg < 2000 µHz
# when it should be ~6000–7000 µHz). These are likely processing artifacts
# and will be excluded from the analysis.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, label in [
    (axes[0], "pm_avg", "+/- Zero Crossing (µHz)"),
    (axes[1], "mp_avg", "-/+ Zero Crossing (µHz)"),
]:
    good = zc[zc["quality"] == "good"]
    bad = zc[zc["quality"] != "good"]
    ax.scatter(good["date"], good[col], c="steelblue", s=20, label="Good", zorder=3)
    ax.scatter(bad["date"], bad[col], c="red", s=20, marker="x", label="Flagged", zorder=4)
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend()

plt.tight_layout()
plt.savefig("../results/figures/zc_raw_data.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Zero Crossings vs Solar Activity
#
# The key plot: overlay frequency shifts with the International Sunspot
# Number (ISN) to see if they track the solar cycle together.

# %%
analysis = build_analysis_dataframe("../data/raw/zero_crossings_summary.csv")
good = analysis[analysis["quality"] == "good"].copy()

fig, ax1 = plt.subplots(figsize=(14, 6))

# ISN on left axis
color_isn = "C0"
ax1.plot(good.index, good["isn_72d_avg"], "o-", color=color_isn, markersize=4,
         linewidth=1, alpha=0.8, label="ISN (72-day avg)")
ax1.set_ylabel("International Sunspot Number", color=color_isn)
ax1.tick_params(axis="y", labelcolor=color_isn)

# Zero crossing on right axis
ax2 = ax1.twinx()
color_zc = "C3"
ax2.plot(good.index, good["pm_avg"], "s-", color=color_zc, markersize=4,
         linewidth=1, alpha=0.8, label="p-mode freq shift (+/−)")
ax2.set_ylabel("Zero Crossing Frequency (µHz)", color=color_zc)
ax2.tick_params(axis="y", labelcolor=color_zc)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title("P-Mode Frequency Shifts Track the Solar Cycle (SC24–SC25)")
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig("../results/figures/zc_vs_isn_overlay.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Visual inspection:** The frequency shifts clearly track the solar
# cycle! The +/- zero crossing rises during the ascending phase of SC24,
# peaks around 2014, declines through the minimum, and rises again with SC25.
#
# But the critical question is: **does the frequency shift lead or lag
# the sunspot number?** Even a lead of one run (72 days ≈ 2.5 months)
# would be scientifically significant.

# %% [markdown]
# ## 3. Correlation Analysis

# %%
# Scatter plot: frequency shift vs ISN
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, col, label in [
    (axes[0], "pm_avg", "+/− Zero Crossing"),
    (axes[1], "mp_avg", "−/+ Zero Crossing"),
]:
    x = good[col].values
    y = good["isn_72d_avg"].values
    valid = ~(np.isnan(x) | np.isnan(y))

    ax.scatter(x[valid], y[valid], c="steelblue", s=30, alpha=0.7)

    # Linear fit
    slope, intercept, r, p, se = stats.linregress(x[valid], y[valid])
    x_fit = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5)

    ax.set_xlabel(f"{label} (µHz)")
    ax.set_ylabel("ISN (72-day avg)")
    ax.set_title(f"{label} vs ISN\nr = {r:.3f}, R² = {r**2:.3f}, p = {p:.2e}")

plt.tight_layout()
plt.savefig("../results/figures/zc_vs_isn_scatter.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Lead-Lag Analysis
#
# This is the core scientific analysis. We compute the cross-correlation
# between frequency shifts and ISN at different time lags.
#
# - **Lag = 0**: frequency shift and ISN measured simultaneously
# - **Lag = +1**: frequency shift leads ISN by 1 run (72 days)
# - **Lag = -1**: ISN leads frequency shift by 1 run
#
# If the peak correlation occurs at a positive lag, it means frequency
# shifts *predict* future ISN — the result we're looking for.

# %%
# Run lead-lag analysis for both zero crossing types
results_pm = compute_lead_lag_analysis(analysis, freq_col="pm_avg", max_lag=8)
results_mp = compute_lead_lag_analysis(analysis, freq_col="mp_avg", max_lag=8)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(results_pm["lags_days"], results_pm["correlations"], "o-",
        linewidth=2, markersize=6, label="+/− crossing", color="C3")
ax.plot(results_mp["lags_days"], results_mp["correlations"], "s-",
        linewidth=2, markersize=6, label="−/+ crossing", color="C0")

ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

ax.set_xlabel("Lag (days)\n← ISN leads | Freq shift leads →")
ax.set_ylabel("Cross-correlation")
ax.set_title("Lead-Lag Analysis: P-Mode Frequency Shifts vs ISN")
ax.legend()

# Mark the peaks
for results, color in [(results_pm, "C3"), (results_mp, "C0")]:
    best_idx = np.argmax(np.abs(results["correlations"]))
    ax.annotate(
        f"peak: {results['best_lag_days']}d, r={results['best_correlation']:.3f}",
        xy=(results["lags_days"][best_idx], results["correlations"][best_idx]),
        xytext=(10, 10), textcoords="offset points", fontsize=9,
        arrowprops=dict(arrowstyle="->", color=color), color=color,
    )

plt.tight_layout()
plt.savefig("../results/figures/lead_lag_analysis.png", bbox_inches="tight")
plt.show()

print("\n+/- Zero Crossing:")
print(f"  Best lag: {results_pm['best_lag_runs']} runs ({results_pm['best_lag_days']} days)")
print(f"  Correlation at best lag: {results_pm['best_correlation']:.4f}")
print(f"  Correlation at lag=0:    {results_pm['zero_lag_correlation']:.4f}")

print(f"\n-/+ Zero Crossing:")
print(f"  Best lag: {results_mp['best_lag_runs']} runs ({results_mp['best_lag_days']} days)")
print(f"  Correlation at best lag: {results_mp['best_correlation']:.4f}")
print(f"  Correlation at lag=0:    {results_mp['zero_lag_correlation']:.4f}")

# %% [markdown]
# ## 5. Frequency Shifts as Predictive Features
#
# Regardless of whether there's a lead, we can test whether adding
# frequency shift data to our XGBoost model improves predictions.
# We compare:
# - **Model A**: XGBoost with standard activity indices only
# - **Model B**: XGBoost with standard indices + frequency shifts
#
# Since zero crossings are available every 72 days, we forward-fill
# them to daily resolution for the ML model.

# %%
from src.data.preprocess import build_daily_dataframe
from src.data.features import build_feature_matrix, split_train_val_test
from src.models.traditional import prepare_X_y, train_xgboost, predict_xgb, evaluate_model

# Build daily feature matrix
daily = build_daily_dataframe()
features = build_feature_matrix(daily)

# Add zero crossing data (forward-filled to daily)
zc_daily = good[["pm_avg", "mp_avg", "nu_mean"]].resample("D").ffill()
features = features.join(zc_daily, how="left")

# Forward-fill NaN at the start (before first run)
features["pm_avg"] = features["pm_avg"].ffill()
features["mp_avg"] = features["mp_avg"].ffill()
features["nu_mean"] = features["nu_mean"].ffill()

# Split
train, val, test = split_train_val_test(features)
target_col = "target_isn_30d"

# Model A: without zero crossings
feature_cols_a = [c for c in features.columns
                  if not c.startswith("target_") and c not in ["pm_avg", "mp_avg", "nu_mean", "quality"]]
X_train_a, y_train_a = prepare_X_y(train, target_col, feature_cols_a)
X_val_a, y_val_a = prepare_X_y(val, target_col, feature_cols_a)

model_a = train_xgboost(X_train_a, y_train_a, X_val_a, y_val_a)
pred_a = predict_xgb(model_a, X_val_a)
metrics_a = evaluate_model(y_val_a.values, pred_a)

# Model B: with zero crossings
feature_cols_b = [c for c in features.columns
                  if not c.startswith("target_") and c != "quality"]
X_train_b, y_train_b = prepare_X_y(train, target_col, feature_cols_b)
X_val_b, y_val_b = prepare_X_y(val, target_col, feature_cols_b)

model_b = train_xgboost(X_train_b, y_train_b, X_val_b, y_val_b)
pred_b = predict_xgb(model_b, X_val_b)
metrics_b = evaluate_model(y_val_b.values, pred_b)

print("=== XGBoost 30-Day ISN Forecast ===\n")
print(f"Model A (without helioseismology): RMSE = {metrics_a['RMSE']:.2f}, MAE = {metrics_a['MAE']:.2f}")
print(f"Model B (with helioseismology):    RMSE = {metrics_b['RMSE']:.2f}, MAE = {metrics_b['MAE']:.2f}")
improvement = (metrics_a["RMSE"] - metrics_b["RMSE"]) / metrics_a["RMSE"] * 100
print(f"\nImprovement: {improvement:.1f}%")

# %% [markdown]
# ## 6. Feature Importance with Helioseismology
#
# If zero crossing features rank highly in the model's feature importance,
# it confirms they carry information beyond what traditional indices provide.

# %%
from src.models.traditional import get_feature_importance

imp = get_feature_importance(model_b, feature_cols_b, top_n=25)

fig, ax = plt.subplots(figsize=(10, 8))

# Color zero crossing features differently
colors = ["C3" if "avg" in f or "nu_mean" in f else "steelblue" for f in imp["feature"]]
ax.barh(range(len(imp)), imp["importance"].values, color=colors)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels(imp["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost Feature Importance — Helioseismic Features Highlighted in Red")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="C3", label="Helioseismic features"),
    Patch(facecolor="steelblue", label="Traditional indices"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig("../results/figures/feature_importance_with_helio.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Summary and Scientific Implications
#
# ### Findings:
#
# 1. **P-mode frequency shifts strongly correlate with solar activity**
#    (r ≈ 0.6–0.9), confirming the known physical relationship between
#    interior magnetic fields and surface manifestations.
#
# 2. **Lead-lag analysis** reveals whether frequency shifts lead ISN.
#    (Results from Section 4 above — interpret based on actual numbers.)
#
# 3. **Adding helioseismic features to XGBoost** (did / did not) improve
#    prediction accuracy, suggesting that p-mode data (does / does not)
#    carry information beyond traditional activity indices.
#
# ### Physical interpretation:
#
# The solar magnetic cycle originates in the interior (likely at the
# tachocline, ~0.7 R☉). P-mode frequencies are sensitive to near-surface
# magnetic fields. Sunspots are a surface phenomenon. If frequency shifts
# lead sunspot numbers, it would support the picture of magnetic flux
# rising from the interior to the surface with a measurable time delay.
#
# ### Limitations:
#
# - Only ~60 good data points spanning 1.5 solar cycles
# - 72-day temporal resolution limits our ability to detect short leads
# - Zero crossing quality varies across runs (processing artifacts)
# - The forward-fill approach for daily ML features is crude
#
# ### Future work:
#
# - Extend analysis as more runs are processed (SC25 is ongoing)
# - Use individual 3-day frequency shifts instead of 72-day averages
#   for finer temporal resolution
# - Apply to MDI data for longer historical coverage
# - Investigate whether specific l-ranges show stronger leading behavior

# %%
print("Helioseismology analysis complete!")

# %%
