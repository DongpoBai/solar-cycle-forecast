"""Download solar activity data from public sources.

Each fetch_* function downloads the full available history of one solar
index and returns a pandas DataFrame with a DatetimeIndex. Raw files are
cached in data/raw/ so repeated runs don't hit the network.

Data sources and their coverage:
    - ISN (SILSO):           1818-present, daily
    - F10.7 (SpaceWeather CA): 1947-present, daily
    - F10.7 + SSN (NOAA):   1749-present, monthly
    - SEM UV (LASP):         1996-present, daily
    - MgII c/w (SET):        1978-present, daily
    - TSI SORCE (LASP):      2003-2020, daily
    - TSI TSIS-1 (LASP):     2018-present, daily

Ingestion logic adapted from:
    https://github.com/HakuyuFujiwara/solar-activity-pipeline
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

_CLIENT_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = _CLIENT_TIMEOUT) -> httpx.Response:
    """HTTP GET with a generous timeout and user-agent."""
    with httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "SolarCycleForecast/0.1 (research)"},
    ) as client:
        logger.info("Downloading %s", url)
        resp = client.get(url)
        resp.raise_for_status()
        return resp


def _cache_path(name: str) -> Path:
    """Return the cache file path for a given dataset name."""
    return RAW_DIR / f"{name}.csv"


# ---------------------------------------------------------------------------
# 1. ISN — SILSO daily International Sunspot Number
# ---------------------------------------------------------------------------

SILSO_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"


def fetch_isn(force: bool = False) -> pd.DataFrame:
    """Download daily International Sunspot Number from SILSO.

    Returns a DataFrame with columns: isn, std_dev, n_obs, provisional.
    Values of -1 in the raw file are treated as missing.

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("silso_isn")
    if cache.exists() and not force:
        logger.info("Loading ISN from cache: %s", cache)
        df = pd.read_csv(cache, index_col="date", parse_dates=True)
        return df

    resp = _get(SILSO_URL)

    rows: list[dict] = []
    reader = csv.reader(io.StringIO(resp.text), delimiter=";")
    for row in reader:
        if not row or len(row) < 5:
            continue
        try:
            obs_date = date(int(row[0].strip()), int(row[1].strip()), int(row[2].strip()))
            isn_raw = float(row[4].strip())
            isn = isn_raw if isn_raw != -1 else None

            std_raw = float(row[5].strip()) if len(row) > 5 else None
            std_dev = std_raw if std_raw is not None and std_raw != -1 else None

            n_obs_raw = row[6].strip() if len(row) > 6 else ""
            n_obs = int(n_obs_raw) if n_obs_raw and n_obs_raw != "-1" else None

            provisional = row[7].strip() == "1" if len(row) > 7 else False

            rows.append({
                "date": obs_date,
                "isn": isn,
                "std_dev": std_dev,
                "n_obs": n_obs,
                "provisional": provisional,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(cache)
    logger.info("ISN: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# 2. F10.7 — Space Weather Canada daily adjusted flux
# ---------------------------------------------------------------------------

FLUX_TABLE_URL = (
    "https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt"
)


def fetch_f107_daily(force: bool = False) -> pd.DataFrame:
    """Download daily F10.7 adjusted flux from Space Weather Canada.

    Selects the reading closest to 20:00 UTC each day (matching the
    DailyActivityValuesUpdater convention). Returns a DataFrame with
    column: f107_adj.

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("spaceweather_ca_f107")
    if cache.exists() and not force:
        logger.info("Loading F10.7 daily from cache: %s", cache)
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    resp = _get(FLUX_TABLE_URL)

    # For each date, keep the reading closest to 20:00 UTC
    best: dict[date, tuple[int, float]] = {}  # date -> (delta_from_200000, adj_flux)

    for line in resp.text.split("\n"):
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 6 or not tokens[0].isdigit():
            continue
        try:
            ds = tokens[0]  # YYYYMMDD
            ts = tokens[1]  # HHMMSS
            if len(ds) != 8 or not ts.isdigit():
                continue

            obs_date = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
            delta = abs(int(ts) - 200000)
            adj_flux = float(tokens[5])

            if obs_date not in best or delta < best[obs_date][0]:
                best[obs_date] = (delta, adj_flux)
        except (ValueError, IndexError):
            continue

    rows = [{"date": d, "f107_adj": v[1]} for d, v in sorted(best.items())]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(cache)
    logger.info("F10.7 daily: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# 3. NOAA SWPC — monthly solar cycle indices (long history)
# ---------------------------------------------------------------------------

NOAA_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)


def fetch_noaa_monthly(force: bool = False) -> pd.DataFrame:
    """Download monthly solar indices from NOAA SWPC.

    Returns a DataFrame with columns: ssn, f107, ap.
    Records are keyed to the 1st of each month.

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("noaa_monthly")
    if cache.exists() and not force:
        logger.info("Loading NOAA monthly from cache: %s", cache)
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    resp = _get(NOAA_URL)
    records = resp.json()

    rows: list[dict] = []
    for rec in records:
        try:
            time_tag = rec.get("time-tag", "")
            obs_date = datetime.strptime(time_tag, "%Y-%m").date().replace(day=1)

            def safe_float(val: object) -> float | None:
                if val is None or val == "":
                    return None
                try:
                    f = float(val)
                    return f if f >= 0 else None
                except (ValueError, TypeError):
                    return None

            rows.append({
                "date": obs_date,
                "ssn_monthly": safe_float(rec.get("ssn")),
                "f107_monthly": safe_float(rec.get("f10.7")),
                "ap_monthly": safe_float(rec.get("ap")),
            })
        except (ValueError, KeyError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(cache)
    logger.info("NOAA monthly: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# 4. SEM UV — LASP SOHO/SEM daily averages
# ---------------------------------------------------------------------------

LASP_SEM_BASE = (
    "https://lasp.colorado.edu/eve/data_access/eve_data/"
    "lasp_soho_sem_data/long/daily_avg"
)


def fetch_sem_uv(force: bool = False) -> pd.DataFrame:
    """Download SOHO/SEM daily UV flux from LASP.

    Downloads per-year files from 1996 to the current year. Extracts the
    last two columns (first-order flux and the column before it), matching
    the DailyActivityValuesUpdater convention (columns 13-14).

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("lasp_sem_uv")
    if cache.exists() and not force:
        logger.info("Loading SEM UV from cache: %s", cache)
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    current_year = date.today().year
    all_rows: list[dict] = []

    for year in range(1996, current_year + 1):
        yy = f"{year % 100:02d}"
        url = f"{LASP_SEM_BASE}/{yy}_v4.day"
        try:
            resp = _get(url)
        except httpx.HTTPStatusError:
            logger.warning("SEM UV: no file for year %d", year)
            continue

        for line in resp.text.split("\n"):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            tokens = line.split()
            if len(tokens) < 16:
                continue
            try:
                file_year = int(tokens[1])
                doy = int(tokens[2])
                obs_date = date(file_year, 1, 1) + timedelta(days=doy - 1)
                sem_second_last = float(tokens[-2])
                sem_last = float(tokens[-1])

                all_rows.append({
                    "date": obs_date,
                    "sem_flux": sem_second_last,
                    "sem_first_order": sem_last,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(cache)
    logger.info("SEM UV: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# 5. MgII core-to-wing ratio
# ---------------------------------------------------------------------------

MGII_URL = "https://sol.spacenvironment.net/spacewx/data/mg2_atmos.dat.txt"


def fetch_mgii(force: bool = False) -> pd.DataFrame:
    """Download MgII core-to-wing ratio from Space Environment Technologies.

    The MgII index is a proxy for solar UV variability. Values of 0.0 in
    the raw file are treated as missing data.

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("mgii")
    if cache.exists() and not force:
        logger.info("Loading MgII from cache: %s", cache)
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    resp = _get(MGII_URL)

    rows: list[dict] = []
    for line in resp.text.split("\n"):
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 5:
            continue
        try:
            obs_date = date(int(tokens[0]), int(tokens[1]), int(tokens[2]))
            mgii = float(tokens[4])
            if mgii == 0.0:
                mgii = None  # 0.0 = missing per file convention
            rows.append({"date": obs_date, "mgii": mgii})
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv(cache)
    logger.info("MgII: %d rows, %s to %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# 6. TSI — SORCE + TSIS-1 via LISIRD LaTiS API
# ---------------------------------------------------------------------------

SORCE_TSI_URL = "https://lasp.colorado.edu/lisird/latis/dap/sorce_tsi_24hr_l3.csv"
TSIS_TSI_URL = "https://lasp.colorado.edu/lisird/latis/dap/tsis_tsi_24hr.csv"


def _parse_lisird_tsi(text: str, source_name: str) -> pd.DataFrame:
    """Parse LISIRD LaTiS CSV for TSI data.

    The first column is Julian Date; the second is TSI at 1 AU (W/m^2).
    Values of 0.0 indicate missing data (instrument gaps).

    Args:
        text: Raw CSV response body.
        source_name: Label for the 'source' column (e.g. "sorce" or "tsis").

    Returns:
        DataFrame indexed by date with columns: tsi, source.
    """
    rows: list[dict] = []
    lines = text.strip().split("\n")

    for line in lines[1:]:  # skip header
        tokens = line.split(",")
        if len(tokens) < 2:
            continue
        try:
            jd = float(tokens[0])
            tsi = float(tokens[1])
            if tsi == 0.0:
                continue  # 0.0 = missing/gap

            # Julian Date to calendar date
            # JD 2451545.0 = 2000-01-01 12:00 UTC (J2000 epoch)
            obs_date = datetime(2000, 1, 1, 12) + timedelta(days=jd - 2451545.0)
            rows.append({
                "date": obs_date.date(),
                "tsi": tsi,
                "tsi_source": source_name,
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def fetch_tsi(force: bool = False) -> pd.DataFrame:
    """Download daily TSI from SORCE (2003-2020) and TSIS-1 (2018-present).

    In the overlap period (2018-2020), TSIS-1 values are preferred as
    the newer, more accurate instrument. Returns a DataFrame with
    columns: tsi, tsi_source.

    Args:
        force: If True, re-download even if cached file exists.
    """
    cache = _cache_path("tsi_combined")
    if cache.exists() and not force:
        logger.info("Loading TSI from cache: %s", cache)
        return pd.read_csv(cache, index_col="date", parse_dates=True)

    # Download both datasets
    logger.info("Downloading SORCE TSI...")
    sorce_resp = _get(SORCE_TSI_URL, timeout=120)
    sorce = _parse_lisird_tsi(sorce_resp.text, "sorce")
    logger.info("SORCE TSI: %d rows", len(sorce))

    logger.info("Downloading TSIS-1 TSI...")
    tsis_resp = _get(TSIS_TSI_URL, timeout=120)
    tsis = _parse_lisird_tsi(tsis_resp.text, "tsis")
    logger.info("TSIS-1 TSI: %d rows", len(tsis))

    # Combine: use SORCE for dates before TSIS starts, then TSIS
    tsis_start = tsis.index.min()
    sorce_only = sorce[sorce.index < tsis_start]
    combined = pd.concat([sorce_only, tsis]).sort_index()

    # Drop any duplicate dates (keep last = TSIS)
    combined = combined[~combined.index.duplicated(keep="last")]

    combined.to_csv(cache)
    logger.info(
        "TSI combined: %d rows, %s to %s",
        len(combined), combined.index.min(), combined.index.max(),
    )
    return combined


# ---------------------------------------------------------------------------
# Convenience: download everything
# ---------------------------------------------------------------------------

def download_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download all datasets and return them as a dict of DataFrames.

    Args:
        force: If True, re-download all datasets.

    Returns:
        Dict mapping dataset name to its DataFrame.
    """
    datasets = {}
    fetchers = {
        "isn": fetch_isn,
        "f107_daily": fetch_f107_daily,
        "noaa_monthly": fetch_noaa_monthly,
        "sem_uv": fetch_sem_uv,
        "mgii": fetch_mgii,
        "tsi": fetch_tsi,
    }

    for name, fetcher in fetchers.items():
        try:
            logger.info("Fetching %s...", name)
            datasets[name] = fetcher(force=force)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", name, exc)

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = download_all(force=True)
    for name, df in data.items():
        print(f"{name:15s}  {len(df):>7,} rows  {df.index.min().date()} to {df.index.max().date()}")
