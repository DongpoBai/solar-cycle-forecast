"""Traditional ML models for solar activity forecasting.

Implements Random Forest and XGBoost regressors with Optuna
hyperparameter tuning. Both models handle NaN natively, which is
important because our feature matrix has missing values (especially
for indices that start later than 1996).

Why these two models:
    - Random Forest: robust, hard to overfit, good baseline for tree methods
    - XGBoost: usually wins on tabular data, handles missing values,
      fast training, industry standard
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

logger = logging.getLogger(__name__)


def prepare_X_y(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y, dropping rows where target is NaN.

    Args:
        df: Feature DataFrame.
        target_col: Name of the target column.
        feature_cols: List of feature column names. If None, uses all
                      columns except those starting with 'target_'.

    Returns:
        Tuple of (X, y) with aligned indices and no NaN in y.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if not c.startswith("target_")]

    valid_mask = df[target_col].notna()
    X = df.loc[valid_mask, feature_cols].copy()
    y = df.loc[valid_mask, target_col]

    # Replace inf/-inf with NaN (caused by pct_change when ISN ≈ 0)
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> RandomForestRegressor:
    """Train a Random Forest regressor.

    Args:
        X_train: Training features.
        y_train: Training targets.
        params: Hyperparameters. Uses sensible defaults if None.

    Returns:
        Fitted RandomForestRegressor.
    """
    default_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        default_params.update(params)

    # Random Forest cannot handle NaN in sklearn, so we fill with -999
    # (tree models learn to split around this sentinel value)
    X_filled = X_train.fillna(-999)

    model = RandomForestRegressor(**default_params)
    model.fit(X_filled, y_train)
    logger.info("Random Forest trained: %d trees, %d features", model.n_estimators, X_filled.shape[1])
    return model


def predict_rf(model: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict using a trained Random Forest."""
    return model.predict(X.fillna(-999))


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict[str, Any] | None = None,
) -> xgb.XGBRegressor:
    """Train an XGBoost regressor.

    XGBoost handles NaN natively (learns optimal split direction for
    missing values), so no imputation needed.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features (for early stopping).
        y_val: Validation targets.
        params: Hyperparameters.

    Returns:
        Fitted XGBRegressor.
    """
    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)

    fit_params: dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["verbose"] = False

    model.fit(X_train, y_train, **fit_params)
    logger.info(
        "XGBoost trained: %d rounds, %d features",
        model.best_iteration if hasattr(model, "best_iteration") and model.best_iteration else default_params["n_estimators"],
        X_train.shape[1],
    )
    return model


def predict_xgb(model: xgb.XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict using a trained XGBoost model."""
    return model.predict(X)


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
) -> dict[str, Any]:
    """Tune XGBoost hyperparameters using Optuna.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        n_trials: Number of Optuna trials.

    Returns:
        Best hyperparameters dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, y_pred)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best RMSE: %.2f", study.best_value)
    logger.info("Best params: %s", study.best_params)

    # Merge with fixed params
    best = study.best_params.copy()
    best["n_estimators"] = 500
    best["random_state"] = 42
    best["n_jobs"] = -1
    return best


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: RandomForestRegressor | xgb.XGBRegressor,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract and sort feature importances.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names: List of feature names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns: feature, importance (sorted descending).
    """
    importances = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)
    return df


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE and MAE."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }
