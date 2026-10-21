"""Causal persistence, seasonal persistence, and per-location ridge regression."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from greenload.config import Config
from greenload.data.dataset import SignalScaler, calendar


def persistence(raw, origins, horizon, period=None):
    steps = np.arange(1, horizon + 1)
    if period is None:
        indices = np.broadcast_to(origins[:, None], (len(origins), horizon))
    else:
        indices = origins[:, None] + (steps - 1) % period + 1 - period
    if indices.min() < 0:
        raise ValueError("Insufficient seasonal history")
    return raw[indices].transpose(0, 2, 1).copy()


def ridge_features(normalized, origins, horizon):
    # The latest day plus 48h and weekly lags; all feature timestamps <= origin.
    lags = np.r_[np.arange(24), 47, 167]
    history = normalized[origins[:, None] - lags].transpose(0, 2, 1)
    times = calendar(origins)
    future = calendar(origins[:, None] + np.arange(1, horizon + 1)).reshape(len(origins), -1)
    known = np.concatenate([times, future, np.ones((len(origins), 1))], axis=1)
    known = np.broadcast_to(known[:, None], (*history.shape[:2], known.shape[-1]))
    return np.concatenate([history, known], axis=2)


def ridge_predict(raw, origins, cfg: Config):
    """Select one alpha per signal using validation error only."""
    scaler = SignalScaler.fit(raw[: cfg.train_hours], "per_location")
    normalized = scaler.normalize(raw)
    train_start, train_end = cfg.bounds("train")
    val_start, val_end = cfg.bounds("val")
    train_origins = np.arange(train_start - 1, train_end - cfg.window_size)
    val_origins = np.arange(val_start - 1, val_end - cfg.window_size)
    train_x = ridge_features(normalized, train_origins, cfg.window_size)
    val_x = ridge_features(normalized, val_origins, cfg.window_size)
    train_y = normalized[train_origins[:, None] + np.arange(1, cfg.window_size + 1)].transpose(
        0, 2, 1
    )
    val_y = normalized[val_origins[:, None] + np.arange(1, cfg.window_size + 1)].transpose(0, 2, 1)
    predict_x = ridge_features(normalized, origins, cfg.window_size)
    best_loss, best_prediction, best_alpha = float("inf"), None, None
    losses = {}
    for alpha in (0.01, 1.0, 100.0):
        val_predictions, predictions = [], []
        for loc in range(raw.shape[1]):
            x, y = train_x[:, loc], train_y[:, loc]
            regularizer = np.eye(x.shape[1]) * alpha
            regularizer[-1, -1] = 0  # intercept
            weights = solve(x.T @ x + regularizer, x.T @ y, assume_a="pos")
            val_predictions.append(val_x[:, loc] @ weights)
            predictions.append(predict_x[:, loc] @ weights)
        loss = float(np.mean((np.stack(val_predictions, axis=1) - val_y) ** 2))
        losses[str(alpha)] = loss
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
            best_prediction = np.stack(predictions, axis=1)
    return scaler.predictions_to_raw(best_prediction), {
        "selected_alpha": best_alpha,
        "validation_normalized_mse": losses,
        "lags": [*range(24), 47, 167],
        "normalization": scaler.to_dict(),
    }
