"""Causal history windows, disjoint target periods, and training-only scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from greenload.config import DEFAULT, Config


@dataclass
class SignalScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray, normalization: str) -> SignalScaler:
        axis = 0 if normalization == "per_location" else None
        mean = np.asarray(training.mean(axis=axis))
        std = np.maximum(np.asarray(training.std(axis=axis)), 1e-8)
        return cls(mean, std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def predictions_to_raw(self, x: np.ndarray) -> np.ndarray:
        """x has shape (origins, locations, horizon)."""
        return x * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> SignalScaler:
        return cls(np.asarray(value["mean"]), np.asarray(value["std"]))


@dataclass
class WindowedSplit:
    source: torch.Tensor  # (windows * locations, history, features)
    target: torch.Tensor  # (windows * locations, 1, horizon)
    future: torch.Tensor  # known calendar covariates only
    origins: np.ndarray  # timestamp of the last observed hour
    num_windows: int
    num_loc: int


@dataclass
class SignalSplits:
    train: WindowedSplit
    val: WindowedSplit
    test: WindowedSplit
    scaler: SignalScaler


def calendar(hours: np.ndarray) -> np.ndarray:
    hours = np.asarray(hours)
    return np.stack(
        [f(2 * np.pi * hours / period) for period in (24, 168) for f in (np.sin, np.cos)],
        axis=-1,
    ).astype(np.float32)


def inputs_at(
    normalized: np.ndarray,
    origins: np.ndarray,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct histories using only timestamps <= each origin."""
    origins = np.asarray(origins)
    indices = origins[:, None] + np.arange(1 - cfg.history_hours, 1)
    if indices.min() < 0 or origins.max() >= len(normalized):
        raise ValueError("History outside the available observations")
    history = normalized[indices].transpose(0, 2, 1)[..., None]
    n, locations, length, _ = history.shape
    future_times = origins[:, None] + np.arange(1, cfg.window_size + 1)
    future = np.broadcast_to(
        calendar(future_times)[:, None], (n, locations, cfg.window_size, 4)
    ).copy()
    if cfg.calendar_features:
        past = np.broadcast_to(calendar(indices)[:, None], (n, locations, length, 4))
        history = np.concatenate([history, past], axis=-1)
    else:
        future.fill(0)
    return (
        torch.tensor(history.reshape(-1, length, cfg.input_size), dtype=torch.float32),
        torch.tensor(future.reshape(-1, cfg.window_size, 4), dtype=torch.float32),
    )


def build_signal_splits(
    raw_signal: np.ndarray,
    cfg: Config = DEFAULT,
    scaler: SignalScaler | None = None,
) -> SignalSplits:
    scaler = scaler or SignalScaler.fit(raw_signal[: cfg.train_hours], cfg.normalization)
    normalized = scaler.normalize(raw_signal)
    locations = raw_signal.shape[1]
    splits = {}
    for name in ("train", "val", "test"):
        start, end = cfg.bounds(name)
        origins = np.arange(start - 1, end - cfg.window_size)
        source, future = inputs_at(normalized, origins, cfg)
        target = normalized[origins[:, None] + np.arange(1, cfg.window_size + 1)]
        target = torch.tensor(
            target.transpose(0, 2, 1).reshape(-1, 1, cfg.window_size), dtype=torch.float32
        )
        splits[name] = WindowedSplit(source, target, future, origins, len(origins), locations)
    return SignalSplits(**splits, scaler=scaler)


def build_all_splits(raw_series: dict[str, np.ndarray], cfg: Config = DEFAULT):
    return {name: build_signal_splits(series, cfg) for name, series in raw_series.items()}


def iter_batches(
    split: WindowedSplit,
    batch_size: int,
    num_loc: int,
    shuffle: bool,
    generator: torch.Generator,
):
    order = (
        torch.randperm(split.num_windows, generator=generator)
        if shuffle
        else torch.arange(split.num_windows)
    )
    for start in range(0, split.num_windows, batch_size):
        window_idx = order[start : start + batch_size]
        rows = (window_idx[:, None] * num_loc + torch.arange(num_loc)).reshape(-1)
        yield split.source[rows], split.target[rows], split.future[rows]
