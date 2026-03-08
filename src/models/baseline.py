"""Baseline models for solar activity forecasting.

These naive models establish a performance floor. Any ML model must
beat these baselines to be considered useful.

Baselines:
    1. Persistence: predict future = current value
    2. Moving Average: predict future = mean of past N days
    3. Climatology: predict future = historical average for this
       phase of the solar cycle
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Excludes cases where y_true is near zero to avoid division issues.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (np.abs(y_true) > 1.0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all metrics at once."""
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Baseline 1: Persistence
# ---------------------------------------------------------------------------

def persistence_forecast(df: pd.DataFrame, target_col: str) -> pd.Series:
    """Persistence baseline: predict future = today's ISN value.

    The simplest possible forecast. If ISN is 120 today, predict the
    7/30/90-day future average will also be 120.

    Args:
        df: Feature DataFrame with 'isn' column.
        target_col: Target column name (e.g. 'target_isn_30d').

    Returns:
        Series of predictions aligned with df's index.
    """
    return df["isn"].copy()


# ---------------------------------------------------------------------------
# Baseline 2: Moving Average
# ---------------------------------------------------------------------------

def moving_average_forecast(
    df: pd.DataFrame, target_col: str, window: int | None = None
) -> pd.Series:
    """Moving average baseline: predict future = mean of past N days.

    The window is matched to the target horizon by default. For example,
    if predicting the 30-day future average, use the past 30-day average.

    Args:
        df: Feature DataFrame.
        target_col: Target column name.
        window: Number of past days to average. If None, inferred from
                target name (e.g. 'target_isn_30d' -> window=30).

    Returns:
        Series of predictions.
    """
    if window is None:
        # Extract horizon from target name like 'target_isn_30d'
        parts = target_col.split("_")
        window = int(parts[-1].replace("d", ""))

    return df["isn"].shift(1).rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Baseline 3: Climatology (historical cycle average)
# ---------------------------------------------------------------------------

def climatology_forecast(
    df: pd.DataFrame, target_col: str, noaa_monthly: pd.DataFrame | None = None
) -> pd.Series:
    """Climatology baseline: predict using 13-month smoothed historical average.

    Uses the 13-month running mean of ISN as the "expected" activity
    level. This is essentially what solar physicists use as a baseline:
    the smoothed trend.

    Args:
        df: Feature DataFrame.
        target_col: Target column name (unused, kept for API consistency).
        noaa_monthly: Optional NOAA monthly data for longer history.

    Returns:
        Series of predictions.
    """
    # 13-month smoothed ISN (standard in solar physics)
    # Use a 390-day window (≈13 months) on daily data
    smoothed = df["isn"].shift(1).rolling(window=390, min_periods=180).mean()
    return smoothed


# ---------------------------------------------------------------------------
# Run all baselines
# ---------------------------------------------------------------------------

def run_all_baselines(
    df: pd.DataFrame,
    target_col: str = "target_isn_30d",
) -> pd.DataFrame:
    """Run all baseline models and return a comparison table.

    Args:
        df: Feature DataFrame with target columns.
        target_col: Which target to evaluate against.

    Returns:
        DataFrame with metrics for each baseline.
    """
    y_true = df[target_col].values

    baselines = {
        "Persistence": persistence_forecast(df, target_col),
        "Moving Average": moving_average_forecast(df, target_col),
        "Climatology (13-mo smooth)": climatology_forecast(df, target_col),
    }

    results = []
    for name, y_pred_series in baselines.items():
        y_pred = y_pred_series.values
        metrics = evaluate(y_true, y_pred)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results)[["Model", "RMSE", "MAE", "MAPE"]]
    return results_df
