"""LSTM model for solar activity forecasting.

Unlike tree-based models that operate on hand-crafted features, the LSTM
takes a raw sequence of past observations as input and learns temporal
patterns directly. This is the sequence-to-value approach:

    Input:  past L days of solar indices  (shape: L × num_features)
    Output: predicted future ISN average  (shape: 1)

Key concepts:
    - Lookback window (L): how many past days the model sees
    - The LSTM's hidden state accumulates information across the sequence
    - We normalize inputs to [0,1] range for stable training
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LSTMConfig:
    """Hyperparameters for the LSTM model."""
    lookback: int = 90          # Days of history to feed the model
    hidden_size: int = 64       # LSTM hidden state dimension
    num_layers: int = 2         # Stacked LSTM layers
    dropout: float = 0.2        # Dropout between LSTM layers
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10          # Early stopping patience


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Features to feed the LSTM (raw daily values, no hand-crafted features)
LSTM_FEATURES = ["isn", "f107_adj", "mgii", "sem_flux", "tsi", "ssn_monthly"]


class SolarSequenceDataset(Dataset):
    """PyTorch Dataset that creates (sequence, target) pairs.

    For each valid index t, the input is the feature matrix from
    t-lookback to t-1, and the target is the future ISN average
    starting from t.

    Args:
        df: DataFrame with features and target columns.
        feature_cols: Columns to use as LSTM input features.
        target_col: Target column name.
        lookback: Number of past days in each input sequence.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        lookback: int = 90,
    ):
        self.lookback = lookback
        self.target_col = target_col

        # Extract and fill NaN for LSTM (can't handle NaN)
        features = df[feature_cols].fillna(0).values.astype(np.float32)
        targets = df[target_col].values.astype(np.float32)

        # Normalize features to [0, 1] using min-max scaling
        # Store params for inverse transform later
        self.feat_min = features.min(axis=0)
        self.feat_max = features.max(axis=0)
        self.feat_range = self.feat_max - self.feat_min
        self.feat_range[self.feat_range == 0] = 1  # avoid division by zero

        features_norm = (features - self.feat_min) / self.feat_range

        # Build valid (sequence, target) pairs
        self.sequences: list[np.ndarray] = []
        self.targets: list[float] = []

        for i in range(lookback, len(features)):
            if np.isnan(targets[i]):
                continue
            seq = features_norm[i - lookback:i]  # shape: (lookback, n_features)
            self.sequences.append(seq)
            self.targets.append(targets[i])

        logger.info(
            "Dataset: %d sequences, lookback=%d, features=%d",
            len(self.sequences), lookback, len(feature_cols),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SolarLSTM(nn.Module):
    """LSTM model for solar activity forecasting.

    Architecture:
        Input (lookback × n_features)
        → LSTM (hidden_size, num_layers, dropout)
        → Take last hidden state
        → Linear (hidden_size → 32)
        → ReLU
        → Linear (32 → 1)
        → Output (scalar prediction)
    """

    def __init__(self, n_features: int, config: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, n_features).

        Returns:
            Predictions of shape (batch,).
        """
        # lstm_out shape: (batch, lookback, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step's output
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.head(last_hidden)      # (batch, 1)
        return out.squeeze(-1)            # (batch,)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str = "target_isn_30d",
    feature_cols: list[str] | None = None,
    config: LSTMConfig | None = None,
) -> tuple[SolarLSTM, dict[str, list[float]]]:
    """Train the LSTM model with early stopping.

    Args:
        train_df: Training data.
        val_df: Validation data.
        target_col: Target column.
        feature_cols: Input feature columns.
        config: Model configuration.

    Returns:
        Tuple of (trained model, training history dict with 'train_loss' and 'val_loss').
    """
    if config is None:
        config = LSTMConfig()
    if feature_cols is None:
        feature_cols = [c for c in LSTM_FEATURES if c in train_df.columns]

    # Create datasets
    # Combine train+val for normalization consistency, then split
    combined = pd.concat([train_df, val_df])
    train_ds = SolarSequenceDataset(train_df, feature_cols, target_col, config.lookback)
    val_ds = SolarSequenceDataset(val_df, feature_cols, target_col, config.lookback)

    # Use train set's normalization params for val set
    val_ds.feat_min = train_ds.feat_min
    val_ds.feat_max = train_ds.feat_max
    val_ds.feat_range = train_ds.feat_range

    # Rebuild val sequences with train normalization
    val_features = val_df[feature_cols].fillna(0).values.astype(np.float32)
    val_features_norm = (val_features - train_ds.feat_min) / train_ds.feat_range
    val_targets = val_df[target_col].values.astype(np.float32)

    val_ds.sequences = []
    val_ds.targets = []
    for i in range(config.lookback, len(val_features)):
        if np.isnan(val_targets[i]):
            continue
        val_ds.sequences.append(val_features_norm[i - config.lookback:i])
        val_ds.targets.append(val_targets[i])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarLSTM(n_features=len(feature_cols), config=config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    logger.info("Training LSTM on %s, %d train / %d val sequences", device, len(train_ds), len(val_ds))

    # Training loop with early stopping
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # RMSE for logging
        val_rmse = np.sqrt(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d — train_loss: %.1f, val_loss: %.1f (RMSE: %.1f)",
                epoch + 1, config.epochs, train_loss, val_loss, val_rmse,
            )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_lstm(
    model: SolarLSTM,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
    train_ds: SolarSequenceDataset | None = None,
    lookback: int = 90,
    feat_min: np.ndarray | None = None,
    feat_range: np.ndarray | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Generate predictions for a DataFrame.

    Args:
        model: Trained LSTM model.
        df: DataFrame to predict on.
        target_col: Target column (for alignment).
        feature_cols: Feature columns.
        lookback: Lookback window size.
        feat_min: Normalization min from training set.
        feat_range: Normalization range from training set.

    Returns:
        Tuple of (predictions array, corresponding date index).
    """
    if feature_cols is None:
        feature_cols = [c for c in LSTM_FEATURES if c in df.columns]

    device = next(model.parameters()).device
    features = df[feature_cols].fillna(0).values.astype(np.float32)
    targets = df[target_col].values

    # Normalize using training set params
    if feat_min is not None and feat_range is not None:
        features_norm = (features - feat_min) / feat_range
    else:
        features_norm = features  # fallback, not ideal

    predictions = []
    dates = []

    model.eval()
    with torch.no_grad():
        for i in range(lookback, len(features)):
            if np.isnan(targets[i]):
                continue
            seq = torch.tensor(
                features_norm[i - lookback:i], dtype=torch.float32
            ).unsqueeze(0).to(device)
            pred = model(seq).item()
            predictions.append(pred)
            dates.append(df.index[i])

    return np.array(predictions), pd.DatetimeIndex(dates)
