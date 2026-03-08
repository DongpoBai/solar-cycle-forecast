# Data

Raw data is not tracked in git. To download all datasets:

```bash
python -m src.data.download
```

This fetches data from:
- **SILSO** — International Sunspot Number (ISN), daily, 1818-present
- **Space Weather Canada** — F10.7 adjusted radio flux, daily, 2004-present
- **NOAA SWPC** — Monthly SSN, F10.7, Ap index, 1749-present
- **LASP SEM** — SOHO/SEM UV flux, daily, 1996-present
- **Space Environment Technologies** — MgII core-to-wing ratio, daily, 1978-present
- **LASP LISIRD** — TSI (SORCE 2003-2020 + TSIS-1 2018-present), daily

Downloaded CSVs are cached in `data/raw/`. Subsequent runs load from cache
unless `force=True` is passed.
