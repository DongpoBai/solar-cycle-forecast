# %% [markdown]
# # Solar Cycle Forecasting — Exploratory Data Analysis
#
# This notebook explores the solar activity datasets we'll use to build
# forecasting models. The goal is to answer three questions:
#
# 1. **What data do we have?** — Coverage, resolution, and gaps
# 2. **What does it look like?** — Periodicity, trends, and anomalies
# 3. **How do the indices relate?** — Correlations and complementarity
#
# These answers will directly inform our feature engineering and modeling
# decisions in later milestones.

# %%
import sys
sys.path.insert(0, "..")  # so we can import src/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

# Plot style
plt.rcParams.update({
    "figure.figsize": (14, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# %% [markdown]
# ## 1. Load All Datasets
#
# Our download module fetches 6 datasets from public sources. Each returns
# a pandas DataFrame indexed by date. Let's load them all and see what
# we're working with.

# %%
from src.data.download import (
    fetch_isn, fetch_f107_daily, fetch_noaa_monthly,
    fetch_sem_uv, fetch_mgii, fetch_tsi,
)

isn = fetch_isn()
f107 = fetch_f107_daily()
noaa = fetch_noaa_monthly()
sem = fetch_sem_uv()
mgii = fetch_mgii()
tsi = fetch_tsi()

datasets = {
    "ISN (SILSO)": isn,
    "F10.7 daily (SpaceWeather CA)": f107,
    "NOAA monthly (SSN/F10.7/Ap)": noaa,
    "SEM UV (LASP)": sem,
    "MgII (SET)": mgii,
    "TSI (SORCE+TSIS)": tsi,
}

# %% [markdown]
# ### 1.1 Coverage Overview
#
# Different instruments started operating at different times. This matters
# because our ML models can only train on periods where data overlaps.

# %%
# Build a summary table
summary_rows = []
for name, df in datasets.items():
    summary_rows.append({
        "Dataset": name,
        "Start": df.index.min().strftime("%Y-%m-%d"),
        "End": df.index.max().strftime("%Y-%m-%d"),
        "Rows": f"{len(df):,}",
        "Resolution": "monthly" if "monthly" in name.lower() else "daily",
        "Missing %": f"{df.iloc[:, 0].isna().mean() * 100:.1f}%",
    })

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# %%
# Visual timeline of data coverage
fig, ax = plt.subplots(figsize=(14, 4))
colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))

for i, (name, df) in enumerate(datasets.items()):
    start = df.index.min()
    end = df.index.max()
    ax.barh(i, (end - start).days, left=start, height=0.6, color=colors[i], alpha=0.8)
    ax.text(end + pd.Timedelta(days=365), i, name, va="center", fontsize=9)

ax.set_yticks([])
ax.set_xlabel("Year")
ax.set_title("Data Coverage Timeline")
ax.xaxis.set_major_locator(mdates.YearLocator(20))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("../results/figures/data_coverage.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key observations:**
#
# - ISN has the longest history (200+ years), but pre-1850 data has higher uncertainty
# - F10.7 from Space Weather Canada only goes back to ~2004. For longer F10.7
#   history, we need the NOAA monthly dataset (which has it from 1947)
# - All daily datasets overlap from ~2005 onward — that's our "multivariate window"
# - Only ISN and NOAA monthly can cover multiple complete solar cycles (~11 years each)

# %% [markdown]
# ## 2. Individual Time Series
#
# Let's look at each index. Solar activity follows an approximately 11-year
# cycle (the Schwabe cycle), but the amplitude and duration vary significantly
# from one cycle to the next. This variability is exactly what makes
# forecasting challenging — and interesting.
#
# ### Solar cycle reference dates
# We'll mark the boundaries of recent cycles for context.

# %%
# Approximate solar minimum dates (cycle boundaries)
# Source: SILSO / NOAA
CYCLE_BOUNDARIES = {
    "SC21 start": "1976-06",
    "SC22 start": "1986-09",
    "SC23 start": "1996-08",
    "SC24 start": "2008-12",
    "SC25 start": "2019-12",
}

def add_cycle_boundaries(ax, ymax=None):
    """Add vertical lines for solar cycle boundaries."""
    for label, date_str in CYCLE_BOUNDARIES.items():
        d = pd.Timestamp(date_str)
        if ax.get_xlim()[0] <= mdates.date2num(d) <= ax.get_xlim()[1]:
            ax.axvline(d, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
            if ymax is not None:
                ax.text(d, ymax * 0.95, label, rotation=90, fontsize=7,
                        va="top", ha="right", color="gray")

# %% [markdown]
# ### 2.1 International Sunspot Number (ISN)
#
# The sunspot number is the oldest and most fundamental solar activity index.
# Sunspots are dark regions on the Sun's surface caused by intense magnetic
# fields. More sunspots = more magnetically active Sun.

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Full history
ax = axes[0]
ax.plot(isn.index, isn["isn"], linewidth=0.3, alpha=0.5, color="C0", label="Daily")
isn_monthly = isn["isn"].resample("MS").mean()
ax.plot(isn_monthly.index, isn_monthly, linewidth=1, color="C0", label="Monthly mean")
ax.set_title("International Sunspot Number — Full History (1818-present)")
ax.set_ylabel("ISN")
ax.legend()
add_cycle_boundaries(ax, ymax=isn["isn"].max())

# Recent cycles (zoom in)
ax = axes[1]
recent = isn[isn.index >= "1996-01-01"]
ax.plot(recent.index, recent["isn"], linewidth=0.3, alpha=0.4, color="C0", label="Daily")
recent_monthly = recent["isn"].resample("MS").mean()
ax.plot(recent_monthly.index, recent_monthly, linewidth=1.5, color="C0", label="Monthly mean")
ax.set_title("ISN — Solar Cycles 23, 24, 25")
ax.set_ylabel("ISN")
ax.legend()
add_cycle_boundaries(ax, ymax=recent["isn"].max())

plt.tight_layout()
plt.savefig("../results/figures/isn_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **What we see:**
#
# - Clear ~11 year cycles, but amplitude varies a lot (SC19 in the 1950s was
#   the strongest on record; SC24 was unusually weak)
# - SC25 (current cycle) is turning out stronger than SC24
# - Daily values are very noisy — monthly smoothing reveals the cycle shape
# - The noise itself carries information: high-activity periods have more
#   day-to-day variability

# %% [markdown]
# ### 2.2 F10.7 Radio Flux

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# NOAA monthly F10.7 (long history)
ax = axes[0]
noaa_f107 = noaa["f107_monthly"].dropna()
ax.plot(noaa_f107.index, noaa_f107, linewidth=1, color="C1")
ax.set_title("F10.7 Radio Flux — Monthly (NOAA, 1947-present)")
ax.set_ylabel("F10.7 (sfu)")
add_cycle_boundaries(ax, ymax=noaa_f107.max())

# Daily (SpaceWeather CA, shorter)
ax = axes[1]
ax.plot(f107.index, f107["f107_adj"], linewidth=0.3, alpha=0.5, color="C1", label="Daily")
f107_monthly = f107["f107_adj"].resample("MS").mean()
ax.plot(f107_monthly.index, f107_monthly, linewidth=1.5, color="C1", label="Monthly mean")
ax.set_title("F10.7 Radio Flux — Daily (Space Weather Canada, 2004-present)")
ax.set_ylabel("F10.7 adjusted (sfu)")
ax.legend()
add_cycle_boundaries(ax, ymax=f107["f107_adj"].max())

plt.tight_layout()
plt.savefig("../results/figures/f107_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **F10.7 basics:** The 10.7 cm radio flux is emitted from the solar corona and
# chromosphere. It correlates strongly with sunspot number but is measured
# objectively (no human observer bias). The "sfu" unit is solar flux units
# (10⁻²² W/m²/Hz). Quiet-sun baseline is ~65-70 sfu.

# %% [markdown]
# ### 2.3 MgII Core-to-Wing Ratio

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(mgii.index, mgii["mgii"], linewidth=0.3, alpha=0.4, color="C2", label="Daily")
mgii_monthly = mgii["mgii"].resample("MS").mean()
ax.plot(mgii_monthly.index, mgii_monthly, linewidth=1.5, color="C2", label="Monthly mean")
ax.set_title("MgII Core-to-Wing Ratio (1978-present)")
ax.set_ylabel("MgII c/w ratio")
ax.legend()
add_cycle_boundaries(ax, ymax=mgii["mgii"].max())
plt.tight_layout()
plt.savefig("../results/figures/mgii_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **MgII basics:** The MgII index measures the ratio of emission in the core
# of the Mg II spectral line (280 nm) to its wings. It's a good proxy for
# solar UV variability, which is what directly affects Earth's upper atmosphere.
# Notice it tracks the same ~11-year cycle but with a different amplitude
# pattern compared to ISN.

# %% [markdown]
# ### 2.4 SEM UV Flux

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(sem.index, sem["sem_flux"], linewidth=0.3, alpha=0.4, color="C3", label="Daily")
sem_monthly = sem["sem_flux"].resample("MS").mean()
ax.plot(sem_monthly.index, sem_monthly, linewidth=1.5, color="C3", label="Monthly mean")
ax.set_title("SOHO/SEM UV Flux (1996-present)")
ax.set_ylabel("SEM Flux")
ax.legend()
add_cycle_boundaries(ax, ymax=sem["sem_flux"].max())
plt.tight_layout()
plt.savefig("../results/figures/sem_uv_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.5 Total Solar Irradiance (TSI)

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(tsi.index, tsi["tsi"], linewidth=0.3, alpha=0.4, color="C4", label="Daily")
tsi_monthly = tsi["tsi"].resample("MS").mean()
ax.plot(tsi_monthly.index, tsi_monthly, linewidth=1.5, color="C4", label="Monthly mean")
ax.set_title("Total Solar Irradiance (SORCE 2003-2020, TSIS-1 2018-present)")
ax.set_ylabel("TSI (W/m²)")
ax.legend()
add_cycle_boundaries(ax, ymax=tsi["tsi"].max())
plt.subplots_adjust(bottom=0.15)
plt.savefig("../results/figures/tsi_timeseries.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **TSI basics:** Total Solar Irradiance is the total power per unit area
# from the Sun at 1 AU. It varies by only ~0.1% over the solar cycle
# (about 1361 ± 0.7 W/m²), but that small variation has measurable effects
# on Earth's climate. Notice the cycle is visible but the signal-to-noise
# ratio is much lower than for ISN or F10.7.

# %% [markdown]
# ## 3. Data Quality
#
# ### 3.1 Missing Values

# %%
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()

daily_datasets = {
    "ISN": isn["isn"],
    "F10.7 daily": f107["f107_adj"],
    "SEM UV": sem["sem_flux"],
    "MgII": mgii["mgii"],
    "TSI": tsi["tsi"],
}

for i, (name, series) in enumerate(daily_datasets.items()):
    ax = axes[i]
    # Resample to monthly and count missing fraction
    monthly_missing = series.resample("MS").apply(lambda x: x.isna().mean())
    ax.bar(monthly_missing.index, monthly_missing.values, width=25, color=f"C{i}", alpha=0.7)
    ax.set_title(f"{name} — Monthly Missing Fraction")
    ax.set_ylabel("Fraction missing")
    ax.set_ylim(0, 1)

# Hide unused subplot
axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig("../results/figures/missing_values.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.2 Basic Statistics

# %%
stats_rows = []
for name, series in daily_datasets.items():
    s = series.dropna()
    stats_rows.append({
        "Index": name,
        "Count": f"{len(s):,}",
        "Mean": f"{s.mean():.2f}",
        "Std": f"{s.std():.2f}",
        "Min": f"{s.min():.2f}",
        "Median": f"{s.median():.2f}",
        "Max": f"{s.max():.2f}",
        "Missing": f"{series.isna().sum():,} ({series.isna().mean()*100:.1f}%)",
    })

stats_df = pd.DataFrame(stats_rows)
print(stats_df.to_string(index=False))

# %% [markdown]
# ## 4. Cross-Index Correlations
#
# Different solar indices measure different physical phenomena (sunspot
# counts, radio emission, UV flux, total irradiance), but they're all
# driven by the same underlying solar magnetic cycle. Let's see how
# tightly they track each other.
#
# We'll align all daily datasets to a common date range and compute
# correlations.

# %%
# Merge all daily datasets into one DataFrame
# Use inner join to keep only dates where all indices have data
merged = pd.DataFrame(index=pd.date_range("1996-01-01", "2026-03-01", freq="D"))
merged = merged.join(isn[["isn"]])
merged = merged.join(f107[["f107_adj"]])
merged = merged.join(sem[["sem_flux"]])
merged = merged.join(mgii[["mgii"]])
merged = merged.join(tsi[["tsi"]])

# Also add NOAA monthly data (forward-fill to daily for alignment)
noaa_daily = noaa[["ssn_monthly", "f107_monthly"]].resample("D").ffill()
merged = merged.join(noaa_daily)

print(f"Merged DataFrame: {merged.shape}")
print(f"Date range: {merged.index.min().date()} to {merged.index.max().date()}")
print(f"\nNon-null counts per column:")
print(merged.count())

# %%
# Correlation matrix on the overlap period (drop rows with any NaN)
corr_data = merged.dropna()
print(f"\nRows with all indices present: {len(corr_data):,}")
print(f"Period: {corr_data.index.min().date()} to {corr_data.index.max().date()}")

corr_matrix = corr_data.corr()

fig, ax = plt.subplots(figsize=(8, 7))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax,
)
ax.set_title("Correlation Matrix — Daily Solar Indices (overlap period)")
plt.tight_layout()
plt.savefig("../results/figures/correlation_matrix.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key findings:**
#
# - ISN and F10.7 are very highly correlated (both directly measure magnetic
#   activity consequences)
# - MgII and SEM UV track UV-specific variability and correlate well with each other
# - TSI has the weakest correlation with other indices — its variation is tiny
#   compared to the measurement noise
# - NOAA monthly SSN and ISN daily should be nearly identical (one is monthly
#   average of the other), serving as a sanity check

# %% [markdown]
# ### 4.1 Scatter Plots — Key Pairs
#
# Correlation numbers can hide non-linear relationships. Let's look at
# the actual scatter plots for the most important pairs.

# %%
pairs = [
    ("isn", "f107_adj", "ISN vs F10.7"),
    ("isn", "mgii", "ISN vs MgII"),
    ("isn", "tsi", "ISN vs TSI"),
    ("f107_adj", "sem_flux", "F10.7 vs SEM UV"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (x_col, y_col, title) in zip(axes.flatten(), pairs):
    valid = corr_data[[x_col, y_col]].dropna()
    ax.scatter(valid[x_col], valid[y_col], alpha=0.05, s=5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)

    # Add correlation coefficient
    r = valid[x_col].corr(valid[y_col])
    ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            fontsize=11, va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

plt.tight_layout()
plt.savefig("../results/figures/scatter_pairs.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Periodicity Analysis
#
# The ~11-year solar cycle is the dominant pattern. But how regular is it
# really? Let's measure the actual cycle lengths and amplitudes.

# %%
# Use NOAA monthly SSN for clean cycle analysis (longest reliable history)
ssn = noaa["ssn_monthly"].dropna()

# Smooth with 13-month running mean (standard in solar physics)
ssn_smooth = ssn.rolling(13, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ssn.index, ssn, alpha=0.3, linewidth=0.5, label="Monthly SSN")
ax.plot(ssn_smooth.index, ssn_smooth, linewidth=2, color="C3", label="13-month smoothed")
ax.set_title("NOAA Monthly Sunspot Number with 13-Month Smoothing")
ax.set_ylabel("SSN")
ax.set_xlabel("Year")
ax.legend()
ax.set_xlim(pd.Timestamp("1947-01-01"), pd.Timestamp("2026-01-01"))
add_cycle_boundaries(ax, ymax=ssn.max())
plt.tight_layout()
plt.savefig("../results/figures/ssn_smoothed.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5.1 Cycle-by-Cycle Comparison
#
# Each solar cycle has a different peak amplitude and duration. This is
# the core challenge for forecasting: the cycle is quasi-periodic, not
# strictly periodic.

# %%
# Define cycles by their start (minimum) dates
cycles = [
    ("SC19", "1954-04", "1964-10"),
    ("SC20", "1964-10", "1976-06"),
    ("SC21", "1976-06", "1986-09"),
    ("SC22", "1986-09", "1996-08"),
    ("SC23", "1996-08", "2008-12"),
    ("SC24", "2008-12", "2019-12"),
    ("SC25", "2019-12", "2026-03"),  # ongoing
]

fig, ax = plt.subplots(figsize=(12, 6))
for name, start, end in cycles:
    mask = (ssn_smooth.index >= start) & (ssn_smooth.index <= end)
    segment = ssn_smooth[mask]
    if len(segment) == 0:
        continue
    # Normalize time to "years since cycle start"
    t0 = pd.Timestamp(start)
    years_since_start = (segment.index - t0).days / 365.25
    ax.plot(years_since_start, segment.values, linewidth=2, label=name)

ax.set_xlabel("Years since cycle minimum")
ax.set_ylabel("13-month smoothed SSN")
ax.set_title("Solar Cycles Superimposed — Each Cycle Has Different Shape")
ax.legend(loc="upper right")
ax.set_xlim(0, 14)
plt.tight_layout()
plt.savefig("../results/figures/cycles_superimposed.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **This plot is important for understanding the forecasting problem:**
#
# - Cycles vary in both peak amplitude (SC19 peaked ~200, SC24 peaked ~120)
#   and rise time (SC22 rose fast, SC24 rose slowly)
# - SC25 (current) appears to be tracking above SC24 so far
# - A model that just learns "the average cycle shape" will miss these
#   variations — we need features that capture the current cycle's momentum

# %% [markdown]
# ## 6. Summary and Implications for Modeling
#
# ### What we learned:
#
# 1. **Usable multivariate window is ~2005-2025** (where all daily indices
#    overlap). That's roughly 2 solar cycles — tight for training.
#
# 2. **ISN alone covers 200+ years** (~18 complete cycles). For long-horizon
#    forecasting, ISN will be the primary feature.
#
# 3. **Indices are highly correlated** but not identical. F10.7 and MgII
#    add information beyond what ISN provides, especially for UV-related
#    predictions.
#
# 4. **TSI has the lowest SNR** — its solar-cycle variation is tiny. It may
#    not be useful as a predictor but could be an interesting target to predict.
#
# 5. **Cycles are quasi-periodic** — the core ML challenge is capturing
#    the variability in amplitude and timing from one cycle to the next.
#
# ### Implications for next steps:
#
# - **Feature engineering (Milestone 2):** Focus on ISN-derived features
#   (rolling statistics, cycle phase) as the backbone, with multi-index
#   features for the recent period.
#
# - **Train/test split:** Use walk-forward validation. Cannot randomly
#   shuffle because of strong temporal autocorrelation. Likely train on
#   SC19-SC23, validate on SC24, test on SC25.
#
# - **Baseline (Milestone 3):** "Persistence" and "13-month moving average"
#   are natural baselines for this data.

# %%
print("EDA complete. Figures saved to results/figures/")