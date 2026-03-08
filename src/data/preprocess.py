"""Preprocessing: merge, align, and clean solar activity datasets.

This module takes the individual DataFrames from download.py and produces
two clean DataFrames ready for feature engineering:

    1. daily_df:   All daily indices aligned on a common date axis
    2. monthly_df: Monthly averages (including NOAA long-history data)

Missing value strategy:
    - Short gaps (≤ 3 days): linear interpolation
    - Longer gaps: left as NaN (will be handled per-model later)
    - NOAA monthly data: forward-filled to daily for alignment only
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.download import (
    fetch_f107_daily,
    fetch_isn,
    fetch_mgii,
    fetch_noaa_monthly,
    fetch_sem_uv,
    fetch_tsi,
)

logger = logging.getLogger(__name__)


def build_daily_dataframe(start: str = "1996-01-01", end: str | None = None) -> pd.DataFrame:
    """Merge all daily datasets into a single DataFrame.

    Uses an outer join so no data is lost; columns will have NaN where
    a particular index doesn't have coverage for that date.

    Args:
        start: Earliest date to include (default: 1996, when SEM starts).
        end: Latest date (default: today).

    Returns:
        DataFrame with columns: isn, f107_adj, sem_flux, mgii, tsi,
        plus forward-filled monthly columns: ssn_monthly, f107_monthly, ap_monthly.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    logger.info("Building daily DataFrame from %s to %s", start, end)

    # Load all datasets (from cache if available)
    isn = fetch_isn()
    f107 = fetch_f107_daily()
    sem = fetch_sem_uv()
    mgii = fetch_mgii()
    tsi = fetch_tsi()
    noaa = fetch_noaa_monthly()

    # Create a complete daily date index
    idx = pd.date_range(start, end, freq="D")
    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    # Join daily datasets (keep only the primary column from each)
    df = df.join(isn[["isn"]])
    df = df.join(f107[["f107_adj"]])
    df = df.join(sem[["sem_flux"]])
    df = df.join(mgii[["mgii"]])
    df = df.join(tsi[["tsi"]])

    # Add NOAA monthly data, forward-filled to daily resolution
    # This gives us long-history F10.7 (from 1947) on a daily grid
    noaa_daily = noaa[["ssn_monthly", "f107_monthly", "ap_monthly"]].resample("D").ffill()
    df = df.join(noaa_daily)

    # Interpolate short gaps (≤ 3 days) linearly
    daily_cols = ["isn", "f107_adj", "sem_flux", "mgii", "tsi"]
    for col in daily_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=3)

    logger.info(
        "Daily DataFrame: %d rows, %d columns, %s to %s",
        len(df), len(df.columns), df.index.min().date(), df.index.max().date(),
    )
    return df


def build_monthly_dataframe(start: str = "1749-01-01") -> pd.DataFrame:
    """Build a monthly DataFrame for long-history analysis.

    Combines NOAA monthly data with monthly averages of daily indices.
    This is useful for models that need many solar cycles for training.

    Args:
        start: Earliest date to include.

    Returns:
        DataFrame at monthly resolution with all available indices.
    """
    logger.info("Building monthly DataFrame from %s", start)

    # NOAA monthly is the backbone (longest history)
    noaa = fetch_noaa_monthly()

    # Compute monthly means from daily datasets
    isn = fetch_isn()
    isn_monthly = isn[["isn"]].resample("MS").mean().rename(columns={"isn": "isn_daily_mean"})

    mgii = fetch_mgii()
    mgii_monthly = mgii[["mgii"]].resample("MS").mean()

    sem = fetch_sem_uv()
    sem_monthly = sem[["sem_flux"]].resample("MS").mean()

    tsi = fetch_tsi()
    tsi_monthly = tsi[["tsi"]].resample("MS").mean()

    f107 = fetch_f107_daily()
    f107_monthly = f107[["f107_adj"]].resample("MS").mean()

    # Merge all
    df = noaa.copy()
    df = df.join(isn_monthly)
    df = df.join(mgii_monthly)
    df = df.join(sem_monthly)
    df = df.join(tsi_monthly)
    df = df.join(f107_monthly)

    df = df[df.index >= start]

    logger.info(
        "Monthly DataFrame: %d rows, %d columns, %s to %s",
        len(df), len(df.columns), df.index.min().date(), df.index.max().date(),
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    daily = build_daily_dataframe()
    print(f"\nDaily: {daily.shape}")
    print(daily.info())

    monthly = build_monthly_dataframe()
    print(f"\nMonthly: {monthly.shape}")
    print(monthly.info())
