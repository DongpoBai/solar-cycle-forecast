# Solar Cycle Forecasting with Machine Learning and Helioseismology

Predicting solar activity using multi-source solar indices and helioseismic frequency shift data from the USC Solar Physics Group's HMI processing pipeline.

## Why This Matters

Solar activity directly affects satellite communications, GPS accuracy, power grids, aviation radiation exposure, and astronaut safety. Accurate forecasting of the ~11-year solar cycle helps NASA, NOAA, and the space weather community plan missions and mitigate risks. This project investigates whether machine learning — and specifically, helioseismic data from the solar interior — can improve upon traditional forecasting methods.

## Key Results

| Model | Val RMSE | Test RMSE | Notes |
|-------|----------|-----------|-------|
| Persistence (baseline) | 25.4 | — | Predict future = today's value |
| Moving Average (baseline) | 16.9 | — | Past 30-day average |
| **XGBoost (tuned)** | **13.5** | **17.4** | **Best performer, 20% below baseline** |
| Random Forest | 17.2 | 20.4 | On par with baseline |
| LSTM | 42.1 | 74.5 | Insufficient data for deep learning |

**Target:** 30-day average International Sunspot Number (ISN). Validation on Solar Cycle 24 (2009–2019), test on Solar Cycle 25 (2020–2026).

### Helioseismology Analysis

We investigated whether solar p-mode oscillation frequency shifts — measured by the USC group's HMI pipeline — can serve as leading indicators of solar activity. Preliminary findings at 72-day resolution:

- Frequency shifts correlate with ISN at r = 0.58 (p = 3.4×10⁻⁷)
- No statistically significant lead detected at 72-day resolution
- Adding helioseismic features to XGBoost improved RMSE by 0.7% (not significant)
- Finer temporal resolution (3-day) is needed to detect potential short-timescale leads (see [Future Work](#future-work))

## Project Structure

```
solar-cycle-forecast/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.py                    # Exploratory data analysis
│   ├── 02_feature_engineering.py    # Feature construction and train/test split
│   ├── 03_baseline.py              # Naive baseline models
│   ├── 04_traditional_ml.py        # Random Forest and XGBoost
│   ├── 05_lstm.py                  # LSTM deep learning
│   └── 06_helioseismology.py       # Frequency shift analysis
├── src/
│   ├── data/
│   │   ├── download.py             # Fetch all 6 public data sources
│   │   ├── preprocess.py           # Merge, align, clean
│   │   ├── features.py             # 103 engineered features + targets
│   │   └── helioseismology.py      # Zero crossing analysis
│   └── models/
│       ├── baseline.py             # Persistence, moving average, climatology
│       ├── traditional.py          # Random Forest, XGBoost, Optuna tuning
│       └── deep.py                 # LSTM with PyTorch
├── data/
│   ├── raw/                        # Downloaded CSVs (gitignored)
│   ├── processed/                  # Feature matrices
│   └── README.md                   # Data source documentation
├── results/
│   └── figures/                    # All visualizations
└── tests/
```

## Quick Start

```bash
git clone https://github.com/HakuyuFujiwara/solar-cycle-forecast.git
cd solar-cycle-forecast
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install scikit-learn xgboost optuna torch --index-url https://download.pytorch.org/whl/cpu

# Download all datasets (~15 seconds)
python -m src.data.download

# Open notebooks in VSCode (each .py file uses # %% cell markers)
```

## Data Sources

| Source | Index | Resolution | Coverage | URL |
|--------|-------|------------|----------|-----|
| SILSO | International Sunspot Number | Daily | 1818–present | sidc.be |
| Space Weather Canada | F10.7 radio flux | Daily | 2004–present | spaceweather.gc.ca |
| NOAA SWPC | SSN, F10.7, Ap | Monthly | 1749–present | swpc.noaa.gov |
| LASP SEM | EUV flux | Daily | 1996–present | lasp.colorado.edu |
| Space Env. Technologies | MgII core-to-wing | Daily | 1978–present | spacenvironment.net |
| LASP LISIRD | TSI (SORCE + TSIS-1) | Daily | 2003–present | lasp.colorado.edu/lisird |
| USC Solar Physics Group | p-mode zero crossings | 72-day | 2010–2024 | HMI pipeline (internal) |

## Methodology

### Feature Engineering

103 features derived from 6 solar indices, including rolling window statistics (7/30/90/365-day mean, std, min, max, slope), first and second order differences, cyclical time encoding (sin/cos of month and day-of-year), and cross-index features (ISN/F10.7 ratio, MgII anomaly). All features use strictly past data to prevent information leakage.

### Train/Test Split

Split by solar cycle boundaries (not random) to evaluate generalization to unseen cycles:
- **Train:** 1996–2008 (Solar Cycle 23)
- **Validation:** 2009–2019 (Solar Cycle 24)
- **Test:** 2020–2026 (Solar Cycle 25, ongoing)

### Models

- **Baselines:** Persistence, moving average, 13-month climatological smoothing
- **XGBoost:** Gradient-boosted trees with Optuna hyperparameter tuning. Best performer.
- **Random Forest:** 200-tree ensemble. Matched baseline performance.
- **LSTM:** 2-layer LSTM with 90-day lookback on raw sequences. Underfitted due to limited training data (~4600 sequences), confirming the known advantage of tree-based models on small tabular datasets.

### Helioseismology

Solar p-mode oscillation frequencies shift with the magnetic cycle. We extracted zero crossing frequencies from 66 pipeline runs (2010–2024) and performed cross-correlation analysis against ISN to test whether interior changes precede surface activity. At 72-day resolution, no lead was detected — finer resolution analysis is needed.

## Key Insights

1. **Feature engineering matters more than model complexity.** XGBoost with 103 hand-crafted features beat LSTM on raw sequences by a large margin. On small structured datasets, domain knowledge wins.

2. **Rolling means dominate feature importance.** The 90-day and 365-day ISN means were the top predictors — the solar cycle is smooth enough that recent history is highly informative.

3. **Multi-source data helps.** MgII and F10.7 features appeared in the top-20 importance for both tree models, confirming that UV and radio proxies carry information beyond sunspot counts.

4. **Prediction is harder during active periods.** RMSE during solar maximum is 2–3× higher than during minimum. The models struggle most when it matters most.

5. **Test set performance degrades.** SC25 is more active than SC24 (the validation set), so all models show higher test RMSE. This highlights the fundamental challenge: each solar cycle behaves differently.

## Future Work

- **3-day resolution helioseismology:** Extract per-3-day frequency shifts from the pipeline's Part 4 intermediate products to achieve 24× finer temporal resolution for lead-lag analysis. Literature suggests p-mode shifts may lead surface activity by days to weeks — a timescale invisible at 72-day resolution.
- **Solar cycle peak prediction:** Instead of short-term forecasting, predict the peak amplitude and timing of a new cycle during its early ascending phase, using the 200-year ISN record.
- **Walk-forward cross-validation:** Implement expanding-window validation across multiple historical cycles to get more robust performance estimates.
- **MDI extension:** Apply the helioseismology analysis to MDI data (1996–2010) to extend coverage to a full additional solar cycle.

## Related Project

[solar-activity-pipeline](https://github.com/HakuyuFujiwara/solar-activity-pipeline) — Automated data engineering pipeline for solar activity indices, used in the USC Solar Physics Group's helioseismology research. That project handles data collection; this project uses the same data for ML forecasting.

## Background

This project was developed as part of undergraduate research in the [Solar Physics Group](https://dornsife.usc.edu/physics-and-astronomy/) at the University of Southern California, under the supervision of Prof. Edward Rhodes. The HMI pipeline processes observations from NASA's Solar Dynamics Observatory to study solar oscillations and their relationship to the solar magnetic cycle.

## License

MIT