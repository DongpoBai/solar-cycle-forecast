"""Helioseismology frequency shift analysis.

This module combines p-mode frequency shift data (zero crossings from
the Rhodes group HMI pipeline) with solar activity indices to investigate
whether helioseismic signals lead surface activity indicators.

Physical background:
    Solar p-mode oscillation frequencies shift with the solar magnetic
    cycle. The frequency increases when solar activity is high because
    the near-surface magnetic field alters the acoustic cavity. If
    changes in the interior magnetic field (detected via p-modes)
    precede changes in surface manifestations (sunspots, radio flux),
    then frequency shifts could serve as a LEADING indicator for
    solar cycle forecasting.

Data:
    - Zero crossings from /hmi3dayall13zerocrossings/ (Runs 1-74)
    - Each run covers 72 days of HMI observations
    - Two zero crossing values per run: +/- and -/+ (ascending and
      descending crossings of the regression line)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import signal, stats

logger = logging.getLogger(__name__)

# MDI day epoch
MDI_EPOCH = date(1993, 1, 1)

# Run parameters
FIRST_RUN_MDI_DAY = 6328  # Run 01 start
DAYS_PER_RUN = 72         # 24 × 3-day sets


def mdi_day_to_date(mdi_day: int) -> date:
    """Convert MDI day number to calendar date."""
    return MDI_EPOCH + timedelta(days=mdi_day)


def run_to_date(run_num: int) -> date:
    """Convert run number to the midpoint calendar date.

    Each run covers 72 days. We use the midpoint (day 36) as the
    representative date for that run's measurements.
    """
    start_mdi = FIRST_RUN_MDI_DAY + (run_num - 1) * DAYS_PER_RUN
    mid_mdi = start_mdi + DAYS_PER_RUN // 2
    return mdi_day_to_date(mid_mdi)


def load_zero_crossings(csv_path: str) -> pd.DataFrame:
    """Load zero crossing summary CSV and add calendar dates.

    Args:
        csv_path: Path to zero_crossings_summary.csv

    Returns:
        DataFrame with columns: run, date, pm_avg, mp_avg, nu_mean,
        with known bad runs flagged.
    """
    df = pd.read_csv(csv_path)

    # Parse run numbers (may have leading zeros)
    df["run"] = df["run"].astype(int)

    # Add calendar dates
    df["date"] = df["run"].apply(run_to_date)
    df["date"] = pd.to_datetime(df["date"])

    # Rename columns
    df = df.rename(columns={
        "plus_minus_avg": "pm_avg",
        "minus_plus_avg": "mp_avg",
    })

    # Flag known anomalous runs
    # Runs 31,33,36,37 have mp_avg < 2000 (should be ~6000-7000)
    # Run 43 has both values anomalously low
    bad_mask = (df["mp_avg"] < 3000) | (df["pm_avg"] < 4000)
    df["quality"] = "good"
    df.loc[bad_mask, "quality"] = "bad"
    df.loc[df["pm_avg"].isna() | df["mp_avg"].isna(), "quality"] = "missing"

    n_bad = (df["quality"] != "good").sum()
    logger.info("Loaded %d runs, %d flagged as bad/missing", len(df), n_bad)

    # Compute mean of +/- and -/+ (standard practice for zero crossings)
    # The mean of the two crossings is a more robust estimate
    df["nu_mean"] = (df["pm_avg"] + df["mp_avg"]) / 2

    return df


def build_analysis_dataframe(
    zc_path: str,
    isn_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a combined DataFrame with zero crossings and activity indices.

    Aligns the 72-day zero crossing measurements with corresponding
    72-day averages of solar activity indices.

    Args:
        zc_path: Path to zero_crossings_summary.csv
        isn_df: Daily ISN DataFrame (from fetch_isn). If None, fetched automatically.

    Returns:
        DataFrame indexed by run date with zero crossings and activity indices.
    """
    zc = load_zero_crossings(zc_path)

    # Get ISN data
    if isn_df is None:
        from src.data.download import fetch_isn
        isn_df = fetch_isn()

    # For each run, compute the 72-day average ISN centered on the run midpoint
    isn_avgs = []
    for _, row in zc.iterrows():
        mid = row["date"]
        start = mid - timedelta(days=DAYS_PER_RUN // 2)
        end = mid + timedelta(days=DAYS_PER_RUN // 2)
        mask = (isn_df.index >= pd.Timestamp(start)) & (isn_df.index <= pd.Timestamp(end))
        avg_isn = isn_df.loc[mask, "isn"].mean()
        isn_avgs.append(avg_isn)

    zc["isn_72d_avg"] = isn_avgs

    # Also get monthly smoothed ISN (13-month running mean)
    isn_monthly = isn_df["isn"].resample("MS").mean()
    isn_smooth = isn_monthly.rolling(13, center=True).mean()

    smooth_vals = []
    for _, row in zc.iterrows():
        mid = pd.Timestamp(row["date"])
        # Find closest month
        idx = isn_smooth.index.searchsorted(mid)
        if idx < len(isn_smooth):
            smooth_vals.append(isn_smooth.iloc[min(idx, len(isn_smooth) - 1)])
        else:
            smooth_vals.append(np.nan)

    zc["isn_13m_smooth"] = smooth_vals

    zc = zc.set_index("date").sort_index()
    return zc


def compute_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized cross-correlation between two series.

    Positive lag means x leads y (x shifts to the right relative to y).

    Args:
        x: First series (e.g., frequency shift).
        y: Second series (e.g., ISN).
        max_lag: Maximum lag in data points (each point = 72 days).

    Returns:
        Tuple of (lags array, correlation values).
    """
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]

    # Normalize
    x_norm = (x_clean - np.mean(x_clean)) / np.std(x_clean)
    y_norm = (y_clean - np.mean(y_clean)) / np.std(y_clean)

    n = len(x_norm)
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = []

    for lag in lags:
        if lag >= 0:
            corr = np.corrcoef(x_norm[:n - lag], y_norm[lag:])[0, 1]
        else:
            corr = np.corrcoef(x_norm[-lag:], y_norm[:n + lag])[0, 1]
        correlations.append(corr)

    return lags, np.array(correlations)


def compute_lead_lag_analysis(
    df: pd.DataFrame,
    freq_col: str = "pm_avg",
    activity_col: str = "isn_72d_avg",
    max_lag: int = 8,
) -> dict:
    """Analyze lead/lag relationship between frequency shifts and activity.

    Args:
        df: Analysis DataFrame from build_analysis_dataframe().
        freq_col: Column with frequency shift data.
        activity_col: Column with activity index.
        max_lag: Maximum lag in run units (1 run ≈ 72 days).

    Returns:
        Dict with analysis results including optimal lag and correlation.
    """
    good = df[df["quality"] == "good"].copy()

    x = good[freq_col].values
    y = good[activity_col].values

    lags, corrs = compute_cross_correlation(x, y, max_lag=max_lag)

    # Convert lags to days
    lags_days = lags * DAYS_PER_RUN

    # Find optimal lag
    best_idx = np.argmax(np.abs(corrs))
    best_lag = lags[best_idx]
    best_corr = corrs[best_idx]

    # Simultaneous correlation (lag=0)
    zero_idx = np.where(lags == 0)[0][0]
    zero_corr = corrs[zero_idx]

    # Linear regression at zero lag
    valid = ~(np.isnan(x) | np.isnan(y))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[valid], y[valid])

    results = {
        "lags": lags,
        "lags_days": lags_days,
        "correlations": corrs,
        "best_lag_runs": best_lag,
        "best_lag_days": best_lag * DAYS_PER_RUN,
        "best_correlation": best_corr,
        "zero_lag_correlation": zero_corr,
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r_squared": r_value ** 2,
        "regression_p_value": p_value,
        "n_good_runs": len(good),
    }

    logger.info(
        "Lead-lag analysis: best lag = %d runs (%d days), r = %.3f, "
        "zero-lag r = %.3f, R² = %.3f, p = %.2e",
        best_lag, best_lag * DAYS_PER_RUN, best_corr,
        zero_corr, r_value ** 2, p_value,
    )

    return results
