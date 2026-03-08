"""Feature engineering for solar cycle forecasting.

Transforms the preprocessed daily/monthly DataFrames into feature matrices
suitable for ML models. All features are computed using ONLY past data
(no future leakage).

Feature categories:
    1. Rolling window statistics (mean, std, min, max, slope)
    2. Difference features (1st and 2nd order)
    3. Cyclical time encoding (month sin/cos, day-of-year)
    4. Cross-index features (ratios between indices)
    5. Target variables (future ISN at various horizons)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Windows for rolling statistics (in days)
WINDOWS = [7, 30, 90, 365]


# ---------------------------------------------------------------------------
# 1. Rolling window statistics
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, column: str, windows: list[int] | None = None) -> pd.DataFrame:
    """Add rolling window statistics for a given column.

    All windows use a TRAILING window (past data only, excluding current
    row) to prevent information leakage.

    Features added per window:
        - {col}_mean_{w}d:  rolling mean
        - {col}_std_{w}d:   rolling std (captures volatility)
        - {col}_min_{w}d:   rolling min
        - {col}_max_{w}d:   rolling max
        - {col}_slope_{w}d: linear trend (OLS slope over window)

    Args:
        df: Input DataFrame with DatetimeIndex.
        column: Column name to compute features for.
        windows: List of window sizes in days. Defaults to WINDOWS.

    Returns:
        DataFrame with new feature columns added.
    """
    if windows is None:
        windows = WINDOWS

    series = df[column]

    for w in windows:
        prefix = f"{column}_{w}d"

        # shift(1) ensures we only use PAST data (not the current row)
        rolling = series.shift(1).rolling(window=w, min_periods=max(1, w // 2))

        df[f"{prefix}_mean"] = rolling.mean()
        df[f"{prefix}_std"] = rolling.std()
        df[f"{prefix}_min"] = rolling.min()
        df[f"{prefix}_max"] = rolling.max()

        # Slope: linear regression coefficient over the window
        # This is more robust than simple difference for capturing trend
        df[f"{prefix}_slope"] = _rolling_slope(series.shift(1), w)

    return df


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear regression slope.

    Uses the closed-form OLS formula for speed:
        slope = (n * sum(x*y) - sum(x)*sum(y)) / (n * sum(x^2) - sum(x)^2)
    where x = [0, 1, 2, ..., n-1] (time index within window).

    Args:
        series: Input time series (already shifted to avoid leakage).
        window: Window size.

    Returns:
        Series of slope values.
    """
    # For a window of size n with x = 0,1,...,n-1:
    # sum(x) = n*(n-1)/2, sum(x^2) = n*(n-1)*(2n-1)/6
    # denominator = n * sum(x^2) - sum(x)^2 is constant for fixed n
    n = window
    sum_x = n * (n - 1) / 2
    sum_x2 = n * (n - 1) * (2 * n - 1) / 6
    denom = n * sum_x2 - sum_x ** 2

    if denom == 0:
        return pd.Series(0.0, index=series.index)

    # sum(x * y) using rolling window
    x_weights = np.arange(n, dtype=float)

    # We need sum(x_i * y_i) over the window. Use a weighted rolling sum.
    # x_weights applied to reversed window (oldest=0, newest=n-1)
    sum_xy = series.rolling(window=n, min_periods=max(1, n // 2)).apply(
        lambda vals: np.sum(np.arange(len(vals)) * vals) if len(vals) == n else np.nan,
        raw=True,
    )
    sum_y = series.rolling(window=n, min_periods=max(1, n // 2)).sum()

    slope = (n * sum_xy - sum_x * sum_y) / denom
    return slope


# ---------------------------------------------------------------------------
# 2. Difference features
# ---------------------------------------------------------------------------

def add_diff_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add first and second order difference features.

    - 1st diff: change from yesterday (direction of movement)
    - 2nd diff: change of the change (acceleration/deceleration)
    - Percent change: relative change (useful for comparing across scales)

    Args:
        df: Input DataFrame.
        column: Column to differentiate.

    Returns:
        DataFrame with new columns added.
    """
    df[f"{column}_diff1"] = df[column].diff(1)
    df[f"{column}_diff2"] = df[column].diff(1).diff(1)
    df[f"{column}_pct_change"] = df[column].pct_change(1)

    return df


# ---------------------------------------------------------------------------
# 3. Cyclical time encoding
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encoding features.

    Uses sin/cos encoding so the model knows that December and January
    are adjacent (which simple month numbers don't convey).

    Features:
        - month_sin, month_cos: annual cycle
        - doy_sin, doy_cos: finer annual cycle (day of year)

    Args:
        df: Input DataFrame with DatetimeIndex.

    Returns:
        DataFrame with time features added.
    """
    day_of_year = df.index.dayofyear
    days_in_year = 365.25
    month = df.index.month

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    return df


# ---------------------------------------------------------------------------
# 4. Cross-index features
# ---------------------------------------------------------------------------

def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that combine multiple solar indices.

    These capture relationships between indices that a model might not
    learn on its own (especially tree-based models which can't easily
    learn ratios).

    Args:
        df: Input DataFrame with ISN and F10.7 columns.

    Returns:
        DataFrame with cross-features added.
    """
    # ISN / F10.7 ratio — deviations from the typical relationship
    # may signal unusual solar behavior
    if "isn" in df.columns and "f107_adj" in df.columns:
        # Add small epsilon to avoid division by zero
        df["isn_f107_ratio"] = df["isn"] / (df["f107_adj"] + 1e-6)

    # MgII normalized by its rolling mean — captures relative activity
    if "mgii" in df.columns:
        mgii_baseline = df["mgii"].shift(1).rolling(365, min_periods=180).mean()
        df["mgii_anomaly"] = df["mgii"] - mgii_baseline

    return df


# ---------------------------------------------------------------------------
# 5. Target variables
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame, column: str = "isn", horizons: list[int] | None = None) -> pd.DataFrame:
    """Add prediction target columns.

    Target is the FUTURE mean of the specified column over various
    horizons. For instance, target_isn_30d is the average ISN over the
    next 30 days starting from tomorrow.

    Args:
        df: Input DataFrame.
        column: Column to predict.
        horizons: Prediction horizons in days. Default: [7, 30, 90].

    Returns:
        DataFrame with target columns added.
    """
    if horizons is None:
        horizons = [7, 30, 90]

    for h in horizons:
        # shift(-h) looks into the future, rolling mean gives average
        # over the horizon. We shift by -1 first so "future" starts tomorrow.
        future = df[column].shift(-1)
        df[f"target_{column}_{h}d"] = future.rolling(window=h, min_periods=h).mean().shift(-(h - 1))

    return df


# ---------------------------------------------------------------------------
# 6. Train / Validation / Test split
# ---------------------------------------------------------------------------

# Solar cycle boundary dates for splitting
SPLIT_DATES = {
    "train_end": "2008-12-31",       # End of SC23
    "val_end": "2019-12-31",          # End of SC24
    # Everything after val_end is test (SC25)
}


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = SPLIT_DATES["train_end"],
    val_end: str = SPLIT_DATES["val_end"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/validation/test sets by time.

    Default boundaries follow solar cycle minima:
        Train: everything up to 2008-12-31 (SC23 and earlier)
        Val:   2009-01-01 to 2019-12-31 (SC24)
        Test:  2020-01-01 onward (SC25)

    Args:
        df: Feature DataFrame with DatetimeIndex.
        train_end: Last date in training set.
        val_end: Last date in validation set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train = df[df.index <= train_end].copy()
    val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df[df.index > val_end].copy()

    logger.info("Train: %d rows (%s to %s)", len(train), train.index.min().date(), train.index.max().date())
    logger.info("Val:   %d rows (%s to %s)", len(val), val.index.min().date(), val.index.max().date())
    logger.info("Test:  %d rows (%s to %s)", len(test), test.index.min().date(), test.index.max().date())

    return train, val, test


# ---------------------------------------------------------------------------
# 7. Full pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    daily_df: pd.DataFrame,
    target_column: str = "isn",
    target_horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Takes a preprocessed daily DataFrame and produces a feature matrix
    ready for ML training.

    Args:
        daily_df: Output of preprocess.build_daily_dataframe().
        target_column: Column to predict.
        target_horizons: Prediction horizons in days.

    Returns:
        Feature matrix with all engineered features and targets.
    """
    df = daily_df.copy()

    logger.info("Building feature matrix...")

    # Rolling features for primary indices
    for col in ["isn", "f107_adj", "mgii"]:
        if col in df.columns:
            df = add_rolling_features(df, col)
            df = add_diff_features(df, col)

    # Rolling features for secondary indices (fewer windows to keep it manageable)
    for col in ["sem_flux", "tsi"]:
        if col in df.columns:
            df = add_rolling_features(df, col, windows=[30, 90])

    # Time and cross features
    df = add_time_features(df)
    df = add_cross_features(df)

    # Targets
    df = add_targets(df, column=target_column, horizons=target_horizons)

    # Count features
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    logger.info("Features: %d, Targets: %d", len(feature_cols), len(target_cols))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from src.data.preprocess import build_daily_dataframe

    daily = build_daily_dataframe()
    features = build_feature_matrix(daily)

    print(f"\nFeature matrix: {features.shape}")
    print(f"\nColumns ({len(features.columns)}):")
    for col in sorted(features.columns):
        print(f"  {col}")

    # Split and show sizes
    train, val, test = split_train_val_test(features)
    print(f"\nTrain: {train.shape}, Val: {val.shape}, Test: {test.shape}")
